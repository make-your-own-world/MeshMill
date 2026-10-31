"""Experimental direct offset of externally visible source triangles."""
import hashlib
import time

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import mutable, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkStaticCellLocator
from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData,
    vtkFeatureEdges,
    vtkPolyDataNormals,
    vtkTriangleFilter,
)


def exposed_surface(poly, offset_mm, cancel=None, backend="auto", gpu_preference="system",
                    progress=lambda *args: None, surface_seeds=()):
    started = time.perf_counter()
    progress("Preparing surface normals", 0, 1000)
    triangle = vtkTriangleFilter()
    triangle.SetInputData(poly)
    clean = vtkCleanPolyData()
    clean.SetInputConnection(triangle.GetOutputPort())
    clean.SetTolerance(0.0)
    clean.Update()
    normal_input = vtkPolyData()
    normal_input.ShallowCopy(clean.GetOutput())
    # Existing STL/scan normals can disagree with repaired triangle winding.
    # Remove copied attributes so both cell and vertex normals are rebuilt.
    normal_input.GetPointData().SetNormals(None)
    normal_input.GetCellData().SetNormals(None)
    boundaries = vtkFeatureEdges()
    boundaries.SetInputData(normal_input)
    boundaries.BoundaryEdgesOn()
    boundaries.NonManifoldEdgesOn()
    boundaries.FeatureEdgesOff()
    boundaries.ManifoldEdgesOff()
    boundaries.Update()
    closed_input = boundaries.GetOutput().GetNumberOfCells() == 0
    normals = vtkPolyDataNormals()
    normals.SetInputData(normal_input)
    normals.SplittingOff()
    normals.SetConsistency(closed_input)
    normals.SetAutoOrientNormals(closed_input)
    normals.ComputeCellNormalsOn()
    normals.ComputePointNormalsOn()
    normals.AddObserver("ProgressEvent", lambda obj, _event: progress(
        "Preparing surface normals", round(obj.GetProgress() * 1000), 1000))
    normals.Update()
    mesh = normals.GetOutput()
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    cell_normals = vtk_to_numpy(mesh.GetCellData().GetNormals())
    from surface_sheets import surface_sheets
    progress("Detecting supported surface sheets", 0, 1)
    supported, sheet_signs, seed_faces, sheet_report, sheet_data = surface_sheets(mesh, surface_seeds)
    cell_normals = cell_normals * sheet_signs[:, None]
    progress("Detecting supported surface sheets", 1, 1)
    from wall_thickness import analyze_wall_thickness
    supported, thickness_report = analyze_wall_thickness(mesh, cell_normals, sheet_data, supported, cancel, progress)
    span = np.linalg.norm(np.ptp(points, axis=0))
    epsilon = max(span * 1e-7, 1e-8)
    length = span * 3
    keep = np.zeros(len(faces), bool)
    orientation = np.zeros(len(faces), dtype=np.int8)
    t, sub, cell = mutable(0.0), mutable(0), mutable(0)
    hit, param = [0.0] * 3, [0.0] * 3
    centers = points[faces].mean(axis=1)
    occluder = mesh
    occluder_points = vtk_to_numpy(occluder.GetPoints().GetData()).astype(float)
    occluder_faces = vtk_to_numpy(occluder.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    directions = np.array([
        [x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
        if x or y or z
    ], dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    report = {"device": "CPU"}
    if backend != "cpu":
        try:
            from imprint_gpu import visibility_gpu
            orientation, report = visibility_gpu(occluder_points, occluder_faces, centers, cell_normals, directions,
                                          epsilon, length, gpu_preference, cancel, progress)
            keep = orientation != 0
        except Exception as exc:
            if backend == "gpu" or (cancel is not None and cancel.is_set()):
                raise
            report = {"device": "CPU", "gpu_fallback": str(exc)}
    if report["device"] == "CPU":
        locator = vtkStaticCellLocator()
        locator.SetDataSet(occluder)
        locator.AddObserver("ProgressEvent", lambda obj, _event: progress(
            "Building CPU ray index", round(obj.GetProgress() * 1000), 1000))
        locator.BuildLocator()
    for i in range(len(faces)) if report["device"] == "CPU" else ():
        center, normal = centers[i], cell_normals[i]
        if i % 256 == 0 and cancel is not None and cancel.is_set():
            raise RuntimeError("Surface extraction cancelled.")
        if i % 256 == 0:
            progress("Surface probes: CPU", i, len(faces))
        # Start beyond this triangle to avoid self-intersection. A hidden layer
        # meets another surface before reaching the exterior along this probe.
        start = center + normal * epsilon
        keep[i] = not locator.IntersectWithLine(start, center + normal * length,
                                               epsilon * 0.01, t, hit, param, sub, cell)
        if keep[i]:
            orientation[i] = 1
    if not keep.any():
        raise ValueError("No exposed surface was found. Check scan normals.")
    keep &= supported
    keep[seed_faces] = True
    orientation[seed_faces] = 1
    if not keep.any():
        raise ValueError("No supported exterior sheet was found. Pick a known surface to guide detection.")
    progress("Surface probes: " + report["device"], len(faces), len(faces))
    progress("Creating offset mesh", 0, 3)
    selected_faces = faces[keep].copy()
    reversed_faces = (orientation[keep] * sheet_signs[keep]) < 0
    selected_faces[reversed_faces] = selected_faces[reversed_faces][:, [0, 2, 1]]
    outward = cell_normals[keep] * orientation[keep, None]
    point_normals = np.zeros_like(points)
    np.add.at(point_normals, selected_faces.ravel(), np.repeat(outward, 3, axis=0))
    point_normals /= np.maximum(np.linalg.norm(point_normals, axis=1)[:, None], 1e-12)
    used, inverse = np.unique(selected_faces, return_inverse=True)
    progress("Creating offset mesh", 1, 3)
    inverse = inverse.ravel()
    displaced = np.ascontiguousarray(points[used] + offset_mm * point_normals[used])
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(displaced, deep=True))
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, len(inverse) + 1, 3, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(inverse.astype(np.int64), deep=True))
    result = vtkPolyData()
    result.SetPoints(vtk_points)
    result.SetPolys(cells)
    from imprint_cleanup import clean_generated_mesh
    result, cleanup = clean_generated_mesh(result, closed=False)
    report["cleanup"] = cleanup
    progress("Creating offset mesh", 3, 3)
    from geometry_workload import record
    record("CPU", points, faces)
    if report["device"] != "CPU":
        record("GPU", occluder_points, occluder_faces)
    return result, {**report, **sheet_report, **thickness_report, "source": len(faces), "kept": int(keep.sum()),
                    "removed": int((~keep).sum()), "seconds": time.perf_counter() - started,
                    "selection_sha256": hashlib.sha256(keep.tobytes()).hexdigest()}
