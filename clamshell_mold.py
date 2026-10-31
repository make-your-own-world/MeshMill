"""Clamshell construction from extracted surface geometry."""
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import (
    vtkImageData,
    vtkPlane,
    vtkPolyData,
)
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData,
    vtkFlyingEdges3D,
    vtkHull,
)
from vtkmodules.vtkFiltersGeneral import vtkTransformFilter
from vtkmodules.vtkFiltersSources import vtkCubeSource
from vtkmodules.vtkImagingMorphological import (
    vtkImageConnectivityFilter,
    vtkImageDilateErode3D,
    vtkImageOpenClose3D,
)
from vtkmodules.vtkImagingStencil import vtkImageStencilToImage, vtkPolyDataToImageStencil


def rotated_split_normal(x_degrees=0, y_degrees=0, z_degrees=0):
    """Rotate the Z-normal split plane about world X, then Y, then Z."""
    x, y, z = np.radians([x_degrees, y_degrees, z_degrees])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx @ np.array([0., 0., 1.])


def split_rim_lines(half, normal, origin, tolerance):
    """Outline the cut-face rim, including the cavity opening, without diagonals."""
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    from vtkmodules.vtkFiltersCore import vtkCutter
    origin = np.asarray(origin, dtype=float)
    sign = 1 if np.dot(np.asarray(half.GetCenter()) - origin, normal) >= 0 else -1
    shift = normal * sign * tolerance * 10
    plane = vtkPlane()
    plane.SetOrigin(*(origin + shift))
    plane.SetNormal(*normal)
    cutter = vtkCutter()
    cutter.SetInputData(half)
    cutter.SetCutFunction(plane)
    cutter.Update()
    clean = vtkCleanPolyData()
    clean.SetInputConnection(cutter.GetOutputPort())
    clean.SetTolerance(0)
    clean.Update()
    result = vtkPolyData()
    result.DeepCopy(clean.GetOutput())
    points = vtk_to_numpy(result.GetPoints().GetData())
    points -= shift
    result.GetPoints().Modified()
    return result


def separated_mold_halves(halves):
    """Translate matching halves apart without changing their shape or scale."""
    if len(halves) != 2:
        return halves
    displacement = (np.asarray(halves[0].GetCenter()) - np.asarray(halves[1].GetCenter())) * .2
    output = []
    for half, sign in zip(halves, (1, -1), strict=True):
        transform = vtkTransform()
        transform.Translate(*(displacement * sign))
        moved = vtkTransformFilter()
        moved.SetInputData(half)
        moved.SetTransform(transform)
        moved.Update()
        result = vtkPolyData()
        result.DeepCopy(moved.GetOutput())
        output.append(result)
    return output


def mold_sampling_plan(bounds, wall_mm=0, bridge_mm=None, pitch_mm=None, memory_budget_bytes=512 * 1024**2,
                       detail=1):
    bounds = np.asarray(bounds, dtype=float).reshape(3, 2)
    span = np.ptp(bounds, axis=1)
    scale = float(span.max())
    if not np.isfinite(bounds).all() or scale <= 0:
        raise ValueError("A nonempty mesh is required.")
    wall = float(wall_mm) if wall_mm > 0 else scale * .025
    bridge = scale * .015 if bridge_mm is None else max(0., float(bridge_mm))
    requested = float(pitch_mm) if pitch_mm else min(scale / (192 * detail), wall / 4)
    pitch = requested
    # Covers scalar images, label arrays, morphology buffers and output mesh.
    # Coarsen to fit available memory instead of rejecting a fixed voxel count.
    cell_budget = max(1000, int(memory_budget_bytes // 128))
    while True:
        pad = wall + bridge + 4 * pitch
        lower, upper = bounds[:, 0] - pad, bounds[:, 1] + pad
        dims = np.ceil((upper - lower) / pitch).astype(np.int64) + 1
        cells = int(np.prod(dims.astype(float)))
        if cells <= cell_budget:
            break
        pitch *= max(1.02, (cells / cell_budget) ** (1 / 3) * 1.02)
    return {"pitch_mm": pitch, "requested_pitch_mm": requested, "wall_mm": wall,
            "bridge_mm": bridge, "lower": lower, "dims": dims, "estimated_bytes": cells * 128,
            "memory_adjusted": pitch > requested * 1.001}


def clamshell_mold(surface, wall_mm=3, bridge_mm=2, pitch_mm=.5, split_normal=(0, 0, 1), shape="Blob",
                   cancel=None, progress=lambda *args: None, memory_budget_bytes=512 * 1024**2, split_origin=None):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    from geometry_workload import record
    record("CPU", vtk_to_numpy(surface.GetPoints().GetData()),
           vtk_to_numpy(surface.GetPolys().GetConnectivityArray()).reshape(-1, 3))
    bounds = np.asarray(surface.GetBounds()).reshape(3, 2)
    plan = mold_sampling_plan(bounds, wall_mm, bridge_mm, pitch_mm, memory_budget_bytes)
    pitch, wall_mm, bridge_mm = plan["pitch_mm"], plan["wall_mm"], plan["bridge_mm"]
    lower, dims = plan["lower"], plan["dims"]

    def run(filter_, stage):
        def update(obj, _event):
            if cancel is not None and cancel.is_set():
                obj.AbortExecuteOn()
            progress(stage, round(obj.GetProgress() * 1000), 1000)
        filter_.AddObserver("ProgressEvent", update)
        progress(stage, 0, 1000)
        filter_.Update()
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Clamshell construction cancelled.")
        progress(stage, 1000, 1000)

    band_width = pitch * .9
    image = vtkImageData()
    image.SetDimensions(*map(int, dims))
    image.SetOrigin(*lower)
    image.SetSpacing(pitch, pitch, pitch)
    from partial_scan_cavity import partial_scan_cavity
    cavity, depth_report = partial_scan_cavity(surface, image, cancel, progress)
    radius = max(0, int(np.ceil(bridge_mm / (2 * pitch))))
    image.GetPointData().SetScalars(numpy_to_vtk(cavity.astype(np.uint8), deep=True))
    cavity_closing = vtkImageOpenClose3D()
    cavity_closing.SetInputData(image)
    cavity_closing.SetKernelSize(*([2 * radius + 1] * 3))
    cavity_closing.SetCloseValue(1)
    cavity_closing.SetOpenValue(0)
    run(cavity_closing, "Bridging cavity volume")
    cavity = vtk_to_numpy(cavity_closing.GetOutput().GetPointData().GetScalars()).astype(bool)
    image.GetPointData().SetScalars(numpy_to_vtk((~cavity).astype(np.uint8), deep=True))
    fill = vtkImageConnectivityFilter()
    fill.SetInputData(image)
    fill.SetScalarRange(1, 1)
    fill.SetExtractionModeToAllRegions()
    fill.SetLabelScalarTypeToInt()
    run(fill, "Removing interior islands")
    void_labels = vtk_to_numpy(fill.GetOutput().GetPointData().GetScalars())
    cavity = void_labels != void_labels[0]

    hull = vtkHull() if shape == "Blob" else vtkCubeSource()
    if shape == "Blob":
        hull.SetInputData(surface)
        hull.AddRecursiveSpherePlanes(2)
    elif shape == "Box":
        expanded = bounds + np.array([-1, 1]) * (wall_mm + band_width)
        hull.SetBounds(*expanded.ravel())
    else:
        raise ValueError("Outer shape must be Blob or Box.")
    run(hull, "Building outer blob")
    stencil = vtkPolyDataToImageStencil()
    stencil.SetInputConnection(hull.GetOutputPort())
    stencil.SetOutputOrigin(*image.GetOrigin())
    stencil.SetOutputSpacing(*image.GetSpacing())
    stencil.SetOutputWholeExtent(image.GetExtent())
    raster = vtkImageStencilToImage()
    raster.SetInputConnection(stencil.GetOutputPort())
    raster.SetInsideValue(1)
    raster.SetOutsideValue(0)
    raster.SetOutputScalarTypeToUnsignedChar()
    run(raster, "Sampling outer blob")
    dilation = vtkImageDilateErode3D()
    dilation.SetInputConnection(raster.GetOutputPort())
    wall_radius = max(1, int(np.ceil((wall_mm + band_width) / pitch)))
    dilation.SetKernelSize(*([2 * wall_radius + 1] * 3 if shape == "Blob" else [1, 1, 1]))
    dilation.SetDilateValue(1)
    dilation.SetErodeValue(0)
    run(dilation, "Adding clamshell walls")
    outside = vtk_to_numpy(dilation.GetOutput().GetPointData().GetScalars()).astype(bool)
    if np.any(cavity & ~outside):
        raise ValueError("Cavity extends outside the blob; increase wall thickness.")
    material = outside & ~cavity
    image.GetPointData().SetScalars(numpy_to_vtk(material.astype(np.uint8), deep=True))
    connected = vtkImageConnectivityFilter()
    connected.SetInputData(image)
    connected.SetScalarRange(1, 1)
    connected.SetExtractionModeToLargestRegion()
    run(connected, "Removing detached material")
    material = vtk_to_numpy(connected.GetOutput().GetPointData().GetScalars()) != 0
    image.GetPointData().SetScalars(numpy_to_vtk(material.astype(np.float32), deep=True))
    contour = vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, .5)
    run(contour, "Creating hollow clamshell")
    sample_ids = np.flatnonzero(cavity)
    sample_ids = sample_ids[np.linspace(0, len(sample_ids) - 1, min(512, len(sample_ids))).astype(int)]
    iz, iy, ix = np.unravel_index(sample_ids, tuple(dims[::-1]))
    cavity_checks = lower + np.column_stack((ix, iy, iz)) * pitch
    report = {**plan, **depth_report, "cavity_voxels": int(cavity.sum()), "mold_voxels": int(material.sum()),
              "cavity_test_points": cavity_checks}
    if split_normal is None:
        full = vtkPolyData()
        full.DeepCopy(contour.GetOutput())
        from imprint_cleanup import clean_generated_mesh
        full, cleanup = clean_generated_mesh(full, closed=True)
        report["cleanup"] = [cleanup]
        return [full], report
    normal = np.asarray(split_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    origin = bounds.mean(axis=1) if split_origin is None else np.asarray(split_origin)
    x = (lower[0] + np.arange(dims[0]) * pitch - origin[0]) * normal[0]
    y = (lower[1] + np.arange(dims[1]) * pitch - origin[1]) * normal[1]
    z = (lower[2] + np.arange(dims[2]) * pitch - origin[2]) * normal[2]
    plane_distance = (z[:, None, None] + y[None, :, None] + x[None, None, :]).astype(np.float32)
    solid_field = np.where(material.reshape(tuple(dims[::-1])), -pitch / 2, pitch / 2).astype(np.float32)
    halves = []
    for sign in (1, -1):
        # Intersection of solid material and the split half-space. The cavity
        # stays outside the material field, including across the cut surface.
        half_field = np.maximum(solid_field, -sign * plane_distance)
        # FlyingEdges orients toward decreasing scalar values. Keep material
        # positive, as in the unsplit binary volume, so wall normals face out.
        image.GetPointData().SetScalars(numpy_to_vtk((-half_field).ravel(), deep=True))
        split = vtkFlyingEdges3D()
        split.SetInputData(image)
        split.SetValue(0, 0)
        run(split, f"Splitting clamshell half {len(halves) + 1}")
        half = vtkPolyData()
        half.DeepCopy(split.GetOutput())
        if not half.GetNumberOfCells():
            raise ValueError("Split plane did not produce two clamshell halves.")
        from imprint_cleanup import clean_generated_mesh
        progress("Cleaning generated geometry", len(halves), 2)
        half, cleanup = clean_generated_mesh(half, closed=True)
        report.setdefault("cleanup", []).append(cleanup)
        halves.append(half)
    return halves, report
