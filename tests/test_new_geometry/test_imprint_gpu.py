"""Regression checks for CPU and GPU exterior-surface agreement."""
import threading

import numpy as np
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersCore import vtkAppendPolyData, vtkReverseSense, vtkTriangleFilter
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkCylinderSource, vtkSphereSource

from exposed_surface import exposed_surface


@pytest.mark.parametrize("shape", ["nested", "overlap", "cube"])
def test_gpu_matches_cpu(shape):
    outer = vtkSphereSource() if shape != "cube" else vtkCubeSource()
    if shape != "cube":
        outer.SetRadius(10)
        outer.SetThetaResolution(24)
        outer.SetPhiResolution(24)
    outer.Update()
    inner = vtkSphereSource()
    inner.SetRadius(4)
    if shape == "overlap":
        inner.SetCenter(9, 0, 0)
    inner.Update()
    combined = vtkAppendPolyData()
    combined.AddInputData(outer.GetOutput())
    if shape != "cube":
        combined.AddInputData(inner.GetOutput())
    combined.Update()
    source = combined.GetOutput()
    original = vtk_to_numpy(source.GetPoints().GetData()).copy()
    cpu_mesh, cpu = exposed_surface(source, 1, backend="cpu")
    updates = []
    gpu_mesh, gpu = exposed_surface(source, 1, backend="gpu", progress=lambda *args: updates.append(args))
    assert cpu["selection_sha256"] == gpu["selection_sha256"]
    assert np.array_equal(vtk_to_numpy(cpu_mesh.GetPoints().GetData()), vtk_to_numpy(gpu_mesh.GetPoints().GetData()))
    assert np.array_equal(vtk_to_numpy(cpu_mesh.GetPolys().GetConnectivityArray()),
                          vtk_to_numpy(gpu_mesh.GetPolys().GetConnectivityArray()))
    assert np.array_equal(original, vtk_to_numpy(source.GetPoints().GetData()))
    assert any(stage.startswith("Surface probes: GPU") and done == total for stage, done, total in updates)


def test_cancel_before_gpu_work():
    source = vtkSphereSource()
    source.Update()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        exposed_surface(source.GetOutput(), 1, backend="gpu", cancel=cancel)


def test_gpu_failure_falls_back_to_cpu(monkeypatch):
    import imprint_gpu

    def unavailable(*args, **kwargs):
        raise RuntimeError("Synthetic unavailable GPU")

    monkeypatch.setattr(imprint_gpu, "visibility_gpu", unavailable)
    source = vtkSphereSource()
    source.Update()
    result, report = exposed_surface(source.GetOutput(), 1)
    assert report["device"] == "CPU"
    assert "Synthetic" in report["gpu_fallback"]
    assert result.GetNumberOfCells() == source.GetOutput().GetNumberOfCells()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_open_tube_keeps_outer_wall_and_excludes_inner_layer(backend):
    surfaces = vtkAppendPolyData()
    for radius in (10, 8):
        tube = vtkCylinderSource()
        tube.SetRadius(radius)
        tube.SetHeight(20)
        tube.SetResolution(64)
        tube.CappingOff()
        tube.Update()
        surfaces.AddInputData(tube.GetOutput())
    surfaces.Update()
    result, report = exposed_surface(surfaces.GetOutput(), .5, backend=backend)
    points = vtk_to_numpy(result.GetPoints().GetData())
    radii = np.linalg.norm(points[:, [0, 2]], axis=1)
    assert report["kept"] == 128
    assert report["surface_sheets"] == 2
    assert report["supported_sheets"] == 1
    assert np.allclose(radii, 10.5, atol=.02)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_positive_clearance_moves_outward_even_with_reversed_input_normals(backend):
    source = vtkSphereSource()
    source.SetRadius(10)
    source.SetThetaResolution(32)
    source.SetPhiResolution(32)
    reverse = vtkReverseSense()
    reverse.SetInputConnection(source.GetOutputPort())
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.Update()
    original = vtk_to_numpy(reverse.GetOutput().GetPointData().GetNormals()).copy()
    result, _ = exposed_surface(reverse.GetOutput(), .5, backend=backend)
    radii = np.linalg.norm(vtk_to_numpy(result.GetPoints().GetData()), axis=1)
    assert np.allclose(radii, 10.5, atol=.02)
    assert np.array_equal(original, vtk_to_numpy(reverse.GetOutput().GetPointData().GetNormals()))


def test_manual_seed_guides_a_whole_incomplete_surface_sheet():
    tube = vtkCylinderSource()
    tube.SetRadius(10)
    tube.SetHeight(20)
    tube.SetResolution(64)
    tube.CappingOff()
    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(tube.GetOutputPort())
    reverse = vtkReverseSense()
    reverse.SetInputConnection(triangles.GetOutputPort())
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.Update()
    source = reverse.GetOutput()
    vertices = vtk_to_numpy(source.GetPoints().GetData())
    faces = vtk_to_numpy(source.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    centers = vertices[faces].mean(axis=1)
    point = centers[np.argmax(centers[:, 0])]
    outward = np.array([point[0], 0, point[2]])
    outward /= np.linalg.norm(outward)
    seeds = [{"point": point, "normal": outward}]
    result, report = exposed_surface(source, .5, backend="cpu", surface_seeds=seeds)
    assert report["kept"] == 128
    assert report["manual_surface_picks"] == 1
    radii = np.linalg.norm(vtk_to_numpy(result.GetPoints().GetData())[:, [0, 2]], axis=1)
    assert np.allclose(radii, 10.5, atol=.02)


def test_true_tube_wall_thickness_and_solid_not_wall():
    surfaces = vtkAppendPolyData()
    for radius in (10, 8):
        tube = vtkCylinderSource()
        tube.SetRadius(radius)
        tube.SetHeight(20)
        tube.SetResolution(64)
        tube.CappingOff()
        tube.Update()
        if radius == 8:
            reverse = vtkReverseSense()
            reverse.SetInputData(tube.GetOutput())
            reverse.ReverseCellsOn()
            reverse.ReverseNormalsOn()
            reverse.Update()
            surfaces.AddInputData(reverse.GetOutput())
        else:
            surfaces.AddInputData(tube.GetOutput())
    surfaces.Update()
    _, report = exposed_surface(surfaces.GetOutput(), .5, backend="cpu")
    assert report["wall_pairs"] >= 1
    assert report["wall_thickness_mm"] == pytest.approx(2, abs=.05)
    sphere = vtkSphereSource()
    sphere.SetThetaResolution(32)
    sphere.SetPhiResolution(32)
    sphere.Update()
    _, report = exposed_surface(sphere.GetOutput(), .1, backend="cpu")
    assert report["wall_thickness_mm"] is None
