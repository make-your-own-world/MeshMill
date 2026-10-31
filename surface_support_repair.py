"""Restore scanned surface support after intersection repair closes a hole.

The repair solid supplies gap closure. Oriented, narrow-band source support
restores surface regions removed by that repair. This is a bounded-resolution
reconstruction, not a guarantee that the original triangle tessellation survives.
"""
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import (
    vtkFlyingEdges3D,
    vtkImplicitPolyDataDistance,
    vtkWindowedSincPolyDataFilter,
)
from vtkmodules.vtkFiltersHybrid import vtkImplicitModeller
from vtkmodules.vtkImagingMorphological import vtkImageContinuousDilate3D, vtkImageContinuousErode3D
from vtkmodules.vtkImagingStencil import vtkImageStencilToImage, vtkPolyDataToImageStencil

from imprint_cleanup import clean_generated_mesh


def _poly(points, faces):
    from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    vertices = vtkPoints()
    vertices.SetData(numpy_to_vtk(points, deep=True))
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, faces.size + 1, 3, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(faces.ravel().astype(np.int64), deep=True))
    result = vtkPolyData()
    result.SetPoints(vertices)
    result.SetPolys(cells)
    return result


def restore_surface_support(points, faces, repaired_points, repaired_faces, missing, progress):
    bounds = np.ptp(points, axis=0)
    diagonal = float(np.linalg.norm(bounds))
    if not diagonal:
        raise ValueError("Surface repair requires a nonzero model size.")
    pitch = diagonal / 430
    # Bound dense field storage independently of input triangle count.
    while np.prod(np.ceil((bounds + 16 * pitch) / pitch) + 1) > 24_000_000:
        pitch *= 1.05
    lower = points.min(axis=0) - 8 * pitch
    dims = np.ceil((bounds + 16 * pitch) / pitch).astype(int) + 1
    upper = lower + (dims - 1) * pitch
    image = vtkImageData()
    image.SetDimensions(*map(int, dims))
    image.SetOrigin(*lower)
    image.SetSpacing(pitch, pitch, pitch)
    repaired = _poly(repaired_points, repaired_faces)
    removed = _poly(points, faces[missing])

    def run(filter_, stage):
        filter_.AddObserver("ProgressEvent", lambda obj, event:
                            progress(stage, int(obj.GetProgress() * 1000), 1000))
        progress(stage, 0, 1000)
        filter_.Update()
        progress(stage, 1000, 1000)
        return filter_.GetOutput()

    stencil = vtkPolyDataToImageStencil()
    stencil.SetInputData(repaired)
    stencil.SetOutputOrigin(*lower)
    stencil.SetOutputSpacing(pitch, pitch, pitch)
    stencil.SetOutputWholeExtent(image.GetExtent())
    raster = vtkImageStencilToImage()
    raster.SetInputConnection(stencil.GetOutputPort())
    raster.SetInsideValue(1)
    raster.SetOutsideValue(0)
    raster.SetOutputScalarTypeToUnsignedChar()
    occupied = vtk_to_numpy(run(raster, "Sampling repaired interior").GetPointData().GetScalars()).astype(bool)

    def distances(mesh, stage):
        distance = vtkImplicitModeller()
        distance.SetInputData(mesh)
        distance.SetSampleDimensions(*map(int, dims))
        distance.SetModelBounds(*np.column_stack((lower, upper)).ravel())
        distance.AdjustBoundsOff()
        distance.CappingOff()
        distance.SetProcessModeToPerCell()
        distance.SetMaximumDistance(4 * pitch / np.linalg.norm(upper - lower))
        return vtk_to_numpy(run(distance, stage).GetPointData().GetScalars()).copy()

    band = distances(removed, "Restoring scanned surfaces")
    field = distances(repaired, "Joining repaired surfaces")
    field[occupied] *= -1
    del occupied
    nearby = np.flatnonzero(band < 3.2 * pitch)
    del band
    signed = vtkImplicitPolyDataDistance()
    signed.SetInput(removed)
    for begin in range(0, len(nearby), 100_000):
        indices = nearby[begin:begin + 100_000]
        z, y, x = np.unravel_index(indices, tuple(dims[::-1]))
        xyz = lower + np.column_stack((x, y, z)) * pitch
        values = vtkDoubleArray()
        signed.EvaluateFunction(numpy_to_vtk(xyz, deep=True), values)
        distance = vtk_to_numpy(values)
        # Extend inward from the original oriented surface, not around a hull.
        support = np.maximum(distance, -distance - 2.4 * pitch)
        field[indices] = np.minimum(field[indices], support)
        progress("Joining source support", min(begin + 100_000, len(nearby)), len(nearby))
    image.GetPointData().SetScalars(numpy_to_vtk(field.astype(np.float32), deep=True))
    fused = fuse_surface_seams(image, run)
    contour = vtkFlyingEdges3D()
    contour.SetInputData(fused)
    contour.SetValue(0, pitch * .01)
    run(contour, "Building repaired surface")
    smooth = vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(contour.GetOutputPort())
    smooth.SetNumberOfIterations(3)
    smooth.SetPassBand(.3)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NormalizeCoordinatesOn()
    raw = run(smooth, "Finishing repaired surface")
    from vtkmodules.vtkFiltersCore import vtkPolyDataConnectivityFilter
    original_regions = vtkPolyDataConnectivityFilter()
    original_regions.SetInputData(_poly(points, faces))
    original_regions.SetExtractionModeToAllRegions()
    original_regions.Update()
    if original_regions.GetNumberOfExtractedRegions() == 1:
        connected = vtkPolyDataConnectivityFilter()
        connected.SetInputData(raw)
        connected.SetExtractionModeToLargestRegion()
        raw = run(connected, "Checking connected surface")
    result, _ = clean_generated_mesh(raw, closed=True)
    out_points = vtk_to_numpy(result.GetPoints().GetData()).astype(np.float32)
    out_faces = vtk_to_numpy(result.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int32)
    distance = vtkImplicitPolyDataDistance()
    distance.SetInput(result)
    samples = np.vstack((points, points[faces].mean(axis=1)))
    errors = []
    for begin in range(0, len(samples), 100_000):
        values = vtkDoubleArray()
        distance.EvaluateFunction(numpy_to_vtk(samples[begin:begin + 100_000], deep=True), values)
        errors.append(np.abs(vtk_to_numpy(values)))
        progress("Checking source preservation", min(begin + 100_000, len(samples)), len(samples))
    error = np.concatenate(errors)
    if np.percentile(error, 95) > pitch or error.max() > pitch * 4:
        raise ValueError("Surface repair could not retain the scanned surfaces within its reconstruction resolution.")
    return out_points, out_faces, {
        "reconstruction_pitch": pitch,
        "surface_rms": float(np.sqrt(np.mean(error ** 2))),
        "surface_p95": float(np.percentile(error, 95)),
        "surface_max": float(error.max()),
    }



def fuse_surface_seams(image, run):
    """Close gaps up to four sample intervals, retaining wider recesses."""
    # Material is negative, so minimum followed by maximum closes its gaps.
    expand = vtkImageContinuousErode3D()
    expand.SetInputData(image)
    expand.SetKernelSize(5, 5, 5)
    run(expand, "Fusing surface seams")
    contract = vtkImageContinuousDilate3D()
    contract.SetInputConnection(expand.GetOutputPort())
    contract.SetKernelSize(5, 5, 5)
    result = vtkImageData()
    result.DeepCopy(run(contract, "Finishing seam joins"))
    return result
