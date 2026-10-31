"""Complete unknown scan regions; carve only measured, front-facing free space."""
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import mutable
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
from vtkmodules.vtkFiltersCore import vtkHull
from vtkmodules.vtkImagingStencil import vtkImageStencilToImage, vtkPolyDataToImageStencil


def partial_scan_cavity(surface, image, cancel=None, progress=lambda *args: None, reconcile=False, dominant_only=False):
    marker = surface.GetFieldData().GetArray("ImprintClosedOutput")
    if marker is not None and marker.GetValue(0):
        stencil = vtkPolyDataToImageStencil()
        stencil.SetInputData(surface)
        stencil.SetOutputOrigin(*image.GetOrigin())
        stencil.SetOutputSpacing(*image.GetSpacing())
        stencil.SetOutputWholeExtent(image.GetExtent())
        raster = vtkImageStencilToImage()
        raster.SetInputConnection(stencil.GetOutputPort())
        raster.SetInsideValue(1)
        raster.SetOutsideValue(0)
        raster.SetOutputScalarTypeToUnsignedChar()
        def update(obj, _event):
            if cancel is not None and cancel.is_set():
                obj.AbortExecuteOn()
            progress("Sampling prepared solid", round(obj.GetProgress() * 1000), 1000)
        raster.AddObserver("ProgressEvent", update)
        raster.Update()
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Cavity sampling cancelled.")
        cavity = vtk_to_numpy(raster.GetOutput().GetPointData().GetScalars()).astype(bool)
        if not cavity.any():
            raise ValueError("The prepared solid contains no cavity volume at this resolution.")
        return cavity, {"prepared_solid": True, "unknown_rays": 0}
    hull = vtkHull()
    hull.SetInputData(surface)
    hull.AddRecursiveSpherePlanes(3)
    hull.Update()
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
    raster.Update()
    dims = np.array(image.GetDimensions())
    cavity = vtk_to_numpy(raster.GetOutput().GetPointData().GetScalars()).astype(bool).reshape(tuple(dims[::-1]))
    prior = cavity.copy() if reconcile else None
    points = vtk_to_numpy(surface.GetPoints().GetData())
    faces = vtk_to_numpy(surface.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    normals = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
    locator = vtkStaticCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    lower = np.array(image.GetOrigin())
    pitch = float(image.GetSpacing()[0])
    upper = lower + (dims - 1) * pitch
    t, sub, cell = mutable(0.), mutable(0), mutable(0)
    hit, param = [0.] * 3, [0.] * 3
    total = int(2 * sum(dims[(axis + 1) % 3] * dims[(axis + 2) % 3] for axis in range(3)))
    done = 0
    measured = 0
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        coordinates = lower[axis] + np.arange(dims[axis]) * pitch
        for i in range(dims[other[0]]):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Cavity measurement cancelled.")
            for j in range(dims[other[1]]):
                start, end = lower.copy(), lower.copy()
                start[other[0]] = end[other[0]] = lower[other[0]] + i * pitch
                start[other[1]] = end[other[1]] = lower[other[1]] + j * pitch
                end[axis] = upper[axis]
                index = [slice(None)] * 3
                index[2 - other[0]], index[2 - other[1]] = i, j
                for sign in (1, -1):
                    origin, destination = (start, end) if sign == 1 else (end, start)
                    done += 1
                    # A miss or a back-facing hit is unknown, not evidence of
                    # free space on the other side of an incomplete scan.
                    if not locator.IntersectWithLine(origin, destination, pitch * 1e-7, t, hit, param, sub, cell):
                        continue
                    if normals[int(cell), axis] * sign >= -.1:
                        continue
                    if dominant_only and abs(normals[int(cell), axis]) + 1e-6 < np.max(np.abs(normals[int(cell)])):
                        continue
                    free = coordinates < hit[axis] if sign == 1 else coordinates > hit[axis]
                    cavity[tuple(index)] &= ~free
                    measured += 1
            progress("Measuring exterior depth", done, total)
    restored = 0
    if reconcile:
        from vtkmodules.util.numpy_support import numpy_to_vtk
        from vtkmodules.vtkCommonCore import vtkDoubleArray
        from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance

        distance = vtkImplicitPolyDataDistance()
        distance.SetInput(surface)
        candidates = np.flatnonzero(prior.ravel() & ~cavity.ravel())
        progress("Checking cavity surface support", 0, len(candidates))
        for start_index in range(0, len(candidates), 32768):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Cavity measurement cancelled.")
            ids = candidates[start_index:start_index + 32768]
            z, y, x = np.unravel_index(ids, tuple(dims[::-1]))
            coordinates = lower + np.column_stack((x, y, z)) * pitch
            values = vtkDoubleArray()
            distance.EvaluateFunction(numpy_to_vtk(coordinates, deep=True), values)
            inside = vtk_to_numpy(values) < -pitch * .1
            cavity.ravel()[ids[inside]] = True
            restored += int(inside.sum())
            progress("Checking cavity surface support", min(start_index + len(ids), len(candidates)), len(candidates))
    if not cavity.any():
        raise ValueError("No cavity volume could be inferred from this selection.")
    return cavity.ravel(), {"depth_rays": done, "measured_rays": measured,
                            "unknown_rays": done - measured, "restored_cavity_voxels": restored}


def sealed_scan_cavity(surface, image, cancel=None, progress=lambda *args: None):
    """Build a temporary closed proxy, retaining the source scan unchanged."""
    from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkPolyDataNormals
    from vtkmodules.vtkFiltersModeling import vtkFillHolesFilter

    def run(filter_, stage):
        def update(obj, _event):
            if cancel is not None and cancel.is_set():
                obj.AbortExecuteOn()
            progress(stage, round(obj.GetProgress() * 1000), 1000)
        filter_.AddObserver("ProgressEvent", update)
        progress(stage, 0, 1000)
        filter_.Update()
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Cavity reconstruction cancelled.")
        progress(stage, 1000, 1000)

    clean = vtkCleanPolyData()
    clean.SetInputData(surface)
    clean.SetTolerance(0)
    run(clean, "Connecting cavity surface")
    fill = vtkFillHolesFilter()
    fill.SetInputConnection(clean.GetOutputPort())
    fill.SetHoleSize(float(np.linalg.norm(np.ptp(np.asarray(surface.GetBounds()).reshape(3, 2), axis=1))) * 2)
    run(fill, "Bridging scan boundaries")
    normals = vtkPolyDataNormals()
    normals.SetInputConnection(fill.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    run(normals, "Orienting cavity surface")
    from imprint_cleanup import clean_generated_mesh
    proxy, report = clean_generated_mesh(normals.GetOutput(), closed=True)
    stencil = vtkPolyDataToImageStencil()
    stencil.SetInputData(proxy)
    stencil.SetOutputOrigin(*image.GetOrigin())
    stencil.SetOutputSpacing(*image.GetSpacing())
    stencil.SetOutputWholeExtent(image.GetExtent())
    raster = vtkImageStencilToImage()
    raster.SetInputConnection(stencil.GetOutputPort())
    raster.SetInsideValue(1)
    raster.SetOutsideValue(0)
    raster.SetOutputScalarTypeToUnsignedChar()
    run(raster, "Sampling completed cavity")
    cavity = vtk_to_numpy(raster.GetOutput().GetPointData().GetScalars()).astype(bool)
    if not cavity.any():
        raise ValueError("The completed scan did not enclose a cavity.")
    return cavity, {"cavity_proxy": report, "unknown_rays": 0,
                    "bridged_faces": proxy.GetNumberOfPolys() - surface.GetNumberOfPolys()}


def reconstructed_scan_cavity(surface, image, cancel=None, progress=lambda *args: None):
    from vtkmodules.vtkImagingCore import vtkImageReslice
    from vtkmodules.vtkImagingHybrid import vtkSurfaceReconstructionFilter

    reconstruction = vtkSurfaceReconstructionFilter()
    reconstruction.SetInputData(surface)
    reconstruction.SetSampleSpacing(image.GetSpacing()[0])
    reconstruction.SetNeighborhoodSize(20)
    reconstruction.AddObserver("ProgressEvent", lambda obj, event: progress(
        "Reconstructing scan gaps", round(obj.GetProgress() * 1000), 1000))
    reconstruction.Update()
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Cavity reconstruction cancelled.")
    field = vtk_to_numpy(reconstruction.GetOutput().GetPointData().GetScalars())
    dims = reconstruction.GetOutput().GetDimensions()
    field3 = field.reshape(tuple(dims[::-1]))
    border = np.concatenate([field3[0].ravel(), field3[-1].ravel(),
                             field3[:, 0].ravel(), field3[:, -1].ravel(),
                             field3[:, :, 0].ravel(), field3[:, :, -1].ravel()])
    sign = 1 if np.median(border) > 0 else -1
    sample = vtkImageReslice()
    sample.SetInputConnection(reconstruction.GetOutputPort())
    sample.SetOutputOrigin(*image.GetOrigin())
    sample.SetOutputSpacing(*image.GetSpacing())
    sample.SetOutputExtent(*image.GetExtent())
    sample.SetInterpolationModeToLinear()
    sample.SetBackgroundLevel(sign * float(np.max(np.abs(field))))
    sample.Update()
    cavity = vtk_to_numpy(sample.GetOutput().GetPointData().GetScalars()) * sign < 0
    return cavity, {"unknown_rays": 0, "reconstructed_surface_points": surface.GetNumberOfPoints()}
