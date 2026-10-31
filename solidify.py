"""Source solidification shared by Solidify and Cast."""
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData,
    vtkDelaunay3D,
    vtkFlyingEdges3D,
    vtkPolyDataNormals,
    vtkTriangleFilter,
)
from vtkmodules.vtkFiltersHybrid import vtkImplicitModeller
from vtkmodules.vtkImagingMorphological import vtkImageConnectivityFilter, vtkImageOpenClose3D

from imprint_cleanup import clean_generated_mesh

METHODS = ("Surface repair",)


def needs_solidification(source):
    from vtkmodules.vtkFiltersCore import vtkFeatureEdges
    weld = vtkCleanPolyData()
    weld.SetInputData(source)
    weld.SetTolerance(0)
    edges = vtkFeatureEdges()
    edges.SetInputConnection(weld.GetOutputPort())
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    return bool(edges.GetOutput().GetNumberOfCells())


def solidify_mesh(source, method, gap_mm, pitch_mm, memory_budget_bytes, cancel=None, progress=lambda *args: None, gpu_preference=None):
    from clamshell_mold import mold_sampling_plan
    from geometry_workload import record
    if not source.GetNumberOfPoints() or not source.GetNumberOfPolys():
        raise ValueError("Select geometry or open a mesh before solidifying.")
    points = vtk_to_numpy(source.GetPoints().GetData())
    faces = vtk_to_numpy(source.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    record("CPU", points, faces)
    plan = mold_sampling_plan(source.GetBounds(), wall_mm=pitch_mm * 2, bridge_mm=gap_mm,
                              pitch_mm=pitch_mm, memory_budget_bytes=memory_budget_bytes)
    pitch = plan["pitch_mm"]
    warnings = []
    repair_report = {}

    def run(filter_, stage):
        errors = []
        def update(obj, _event):
            if cancel is not None and cancel.is_set():
                obj.AbortExecuteOn()
            progress(stage, round(obj.GetProgress() * 1000), 1000)
        filter_.AddObserver("ProgressEvent", update)
        filter_.AddObserver("ErrorEvent", lambda obj, event: errors.append(stage))
        filter_.AddObserver("WarningEvent", lambda obj, event: warnings.append(stage))
        progress(stage, 0, 1000)
        filter_.Update()
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Solidification cancelled.")
        if errors:
            raise ValueError(f"Solidify failed during {stage.lower()}. Try another method or gap size.")
        progress(stage, 1000, 1000)
        return filter_.GetOutput()

    if method == "Convex envelope":
        from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
        hull = vtkDelaunay3D()
        hull.SetInputData(source)
        hull.SetTolerance(1e-6)
        run(hull, "Building convex solid")
        boundary = vtkDataSetSurfaceFilter()
        boundary.SetInputConnection(hull.GetOutputPort())
        raw = run(boundary, "Extracting convex skin")
    elif method in {"Boundary patches", "Surface repair"}:
        from surface_repair import repair_surface
        raw, repair_report = repair_surface(source, method == "Surface repair", cancel, progress, gpu_preference)
    elif method == "Voxel gap closing":
        lower, dims = plan["lower"], plan["dims"]
        upper = lower + (dims - 1) * pitch
        distance = vtkImplicitModeller()
        distance.SetInputData(source)
        distance.SetSampleDimensions(*map(int, dims))
        distance.SetModelBounds(*np.column_stack((lower, upper)).ravel())
        distance.AdjustBoundsOff()
        distance.CappingOff()
        distance.SetProcessModeToPerCell()
        distance.SetMaximumDistance(min(1., max(pitch * 2, gap_mm) / np.linalg.norm(upper - lower)))
        field = run(distance, "Sampling solid surface")
        band = vtk_to_numpy(field.GetPointData().GetScalars()) <= pitch * .8
        image = vtkImageData()
        image.DeepCopy(field)
        image.GetPointData().SetScalars(numpy_to_vtk(band.astype(np.uint8), deep=True))
        closing = vtkImageOpenClose3D()
        closing.SetInputData(image)
        radius = max(0, int(np.ceil(gap_mm / (2 * pitch))))
        closing.SetKernelSize(*([2 * radius + 1] * 3))
        closing.SetCloseValue(1)
        closing.SetOpenValue(0)
        shell = vtk_to_numpy(run(closing, "Closing solid gaps").GetPointData().GetScalars()).astype(bool)
        image.GetPointData().SetScalars(numpy_to_vtk((~shell).astype(np.uint8), deep=True))
        regions = vtkImageConnectivityFilter()
        regions.SetInputData(image)
        regions.SetScalarRange(1, 1)
        regions.SetExtractionModeToAllRegions()
        regions.SetLabelScalarTypeToInt()
        labels = vtk_to_numpy(run(regions, "Filling solid interior").GetPointData().GetScalars())
        enclosed = (~shell) & (labels != labels[0])
        if not enclosed.any():
            raise ValueError("The gaps remain open at this gap size. Increase gap bridging or choose another Solidify method.")
        solid = labels != labels[0]
        image.GetPointData().SetScalars(numpy_to_vtk(solid.astype(np.float32), deep=True))
        contour = vtkFlyingEdges3D()
        contour.SetInputData(image)
        contour.SetValue(0, .5)
        raw = run(contour, "Creating solid mesh")
    else:
        raise ValueError("Unknown Solidify method")
    triangles = vtkTriangleFilter()
    triangles.SetInputData(raw)
    run(triangles, "Triangulating solid")
    weld = vtkCleanPolyData()
    weld.SetInputConnection(triangles.GetOutputPort())
    weld.SetTolerance(0)
    run(weld, "Welding solid vertices")
    normals = vtkPolyDataNormals()
    normals.SetInputConnection(weld.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    raw = run(normals, "Orienting solid faces")
    progress("Validating solid", 0, 1)
    result, report = clean_generated_mesh(raw, closed=True)
    progress("Validating solid", 1, 1)
    from solidify_metrics import completion_metrics
    output_points = vtk_to_numpy(result.GetPoints().GetData())
    output_faces = vtk_to_numpy(result.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    progress("Checking solid completion", 0, 1)
    completion = completion_metrics(points, faces, output_points, output_faces)
    if completion["open_edges"]:
        raise ValueError(f"Solidify left {completion['open_edges']} open edges. No validated solid was produced.")
    if (completion["added_faces"] == 0 and completion["retained_faces"] == len(faces)
            and len(output_faces) == len(faces)):
        raise ValueError("The input is already closed; no surface was added or changed.")
    progress("Checking solid completion", 1, 1)
    return result, {"method": method, "pitch_mm": pitch, "topology": report,
                    "backend_warning_stages": warnings, "completion": completion,
                    "surface_repair": repair_report}


def combine_selected_solid(points, faces, mask, solid):
    """Keep unselected triangles unchanged; no implicit Boolean union."""
    if mask is None or mask.all():
        return solid
    from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkAppendPolyData
    vertices = vtkPoints()
    vertices.SetData(numpy_to_vtk(points, deep=True))
    kept = np.ascontiguousarray(faces[~mask], dtype=np.int64)
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, kept.size + 1, 3, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(kept.ravel(), deep=True))
    remaining = vtkPolyData()
    remaining.SetPoints(vertices)
    remaining.SetPolys(cells)
    combined = vtkAppendPolyData()
    combined.AddInputData(remaining)
    combined.AddInputData(solid)
    combined.Update()
    output = vtkPolyData()
    output.DeepCopy(combined.GetOutput())
    return output


def offset_solid(source, clearance):
    """Offset the complete prepared skin instead of deleting its concave faces."""
    from vtkmodules.vtkFiltersGeneral import vtkWarpVector
    normals = vtkPolyDataNormals()
    normals.SetInputData(source)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    mesh = normals.GetOutput()
    mesh.GetPointData().SetVectors(mesh.GetPointData().GetNormals())
    offset = vtkWarpVector()
    offset.SetInputData(mesh)
    offset.SetScaleFactor(clearance)
    offset.Update()
    result, report = clean_generated_mesh(offset.GetOutput(), closed=True)
    return result, {"device": "CPU", "kept": result.GetNumberOfPolys(), "removed": 0,
                    "cleanup": report}
