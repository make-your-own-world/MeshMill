"""Boundary filling and optional intersection repair using MeshFix.

Implementation reference: https://github.com/pyvista/pymeshfix
The optional backend is GPL-3.0 and uses its upstream license and attribution.
"""
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData


def _repair_arrays(points, faces, repair_intersections, connection, gpu_preference=None):
    import io
    import sys

    # GUI workers launched by pythonw have no console streams. Native meshing
    # wrappers still write status messages even when their quiet flag is set.
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()

    def progress(stage, done, total):
        connection.send((stage, done, total))
    cancel = None
    try:
        from pymeshfix import PyTMesh
    except ImportError as error:
        raise ValueError("The surface repair runtime is unavailable in this build.") from error
    if repair_intersections:
        from solid_surface_prepare import complete_scan_volume
        output_points, output_faces, report = complete_scan_volume(points, faces, gpu_preference, progress)
        check = PyTMesh()
        check.set_quiet(True)
        check.load_array(output_points.astype(float), output_faces.astype(np.int32))
        if check.n_boundaries or len(check.select_intersecting_triangles()):
            raise ValueError("Generated solid failed the final intersection and boundary checks.")
        report["self_intersections"] = 0
        return output_points, output_faces, report
    mesh = PyTMesh()
    mesh.set_quiet(True)
    mesh.load_array(points, faces)
    loops = mesh.n_boundaries
    progress("Filling boundary holes", 0, max(loops, 1))
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Surface repair cancelled.")
    if loops:
        mesh.fill_small_boundaries(refine=True)
    progress("Filling boundary holes", max(loops, 1), max(loops, 1))
    intersections = len(mesh.select_intersecting_triangles())
    if intersections and not repair_intersections:
        raise ValueError("The boundary patches cross existing surfaces. Use a smaller selection, or choose Surface repair, which can rebuild surrounding geometry.")
    for iteration in range(10 if repair_intersections else 0):
        if not intersections:
            break
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Surface repair cancelled.")
        progress("Repairing surface intersections", iteration, 10)
        mesh.clean(max_iters=1, inner_loops=3)
        intersections = len(mesh.select_intersecting_triangles())
    if mesh.n_boundaries or intersections:
        raise ValueError("Surface repair could not produce a closed surface without intersections. Try a smaller selection.")
    progress("Checking repaired surface", 0, 1)
    output_points, output_faces = mesh.return_arrays()
    # Validate at the precision actually stored in binary STL.
    output_points = output_points.astype(np.float32)
    _, inverse = np.unique(np.vstack((points.astype(np.float32), output_points)),
                                   axis=0, return_inverse=True)
    original_faces = np.ascontiguousarray(np.sort(inverse[:len(points)][faces], axis=1))
    repaired_faces = np.ascontiguousarray(np.sort(inverse[len(points):][output_faces], axis=1))
    row_type = np.dtype((np.void, original_faces.dtype.itemsize * 3))
    retained = len(np.intersect1d(original_faces.view(row_type).ravel(), repaired_faces.view(row_type).ravel()))
    preservation = {}
    if retained != len(faces):
        if not repair_intersections:
            raise ValueError("Boundary filling would alter existing surfaces.")
        from surface_support_repair import restore_surface_support
        missing = ~np.isin(original_faces.view(row_type).ravel(), repaired_faces.view(row_type).ravel())
        output_points, output_faces, preservation = restore_surface_support(
            points, faces, output_points, output_faces, missing, progress)

    check = PyTMesh()
    check.set_quiet(True)
    check.load_array(output_points.astype(float), output_faces.astype(np.int32))
    final_intersections = len(check.select_intersecting_triangles())
    if check.n_boundaries or final_intersections:
        raise ValueError(f"The repaired surface failed validation at STL precision: {check.n_boundaries} open boundaries, {final_intersections} intersecting faces.")
    progress("Checking repaired surface", 1, 1)
    return output_points, output_faces, {"filled_boundaries": loops, "self_intersections": 0,
                                       "may_rebuild_existing_surface": bool(repair_intersections), **preservation}


def repair_surface(source, repair_intersections=False, cancel=None, progress=lambda *args: None, gpu_preference=None):
    import multiprocessing
    from concurrent.futures import TimeoutError

    from worker_lifecycle import OwnedProcessPool
    points = vtk_to_numpy(source.GetPoints().GetData()).astype(np.float64)
    faces = vtk_to_numpy(source.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int32)
    points, inverse = np.unique(points, axis=0, return_inverse=True)
    faces = inverse[faces].astype(np.int32)
    receiver, sender = multiprocessing.get_context("spawn").Pipe(duplex=False)
    try:
        with OwnedProcessPool(max_workers=1) as pool:
            future = pool.submit(_repair_arrays, points, faces, repair_intersections, sender, gpu_preference)
            while True:
                while receiver.poll():
                    progress(*receiver.recv())
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Surface repair cancelled.")
                try:
                    output_points, output_faces, report = future.result(timeout=.05)
                    break
                except TimeoutError:
                    continue
            while receiver.poll():
                progress(*receiver.recv())
    finally:
        receiver.close()
        sender.close()
    vertices = vtkPoints()
    vertices.SetData(numpy_to_vtk(output_points, deep=True))
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, output_faces.size + 1, 3, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(output_faces.ravel().astype(np.int64), deep=True))
    output = vtkPolyData()
    output.SetPoints(vertices)
    output.SetPolys(cells)
    return output, report
