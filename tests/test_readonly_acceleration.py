import numpy as np
import pytest
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.util.numpy_support import vtk_to_numpy

import meshmill as mm
import depth_adaptive_qem as daq
from test_operation_progress import sphere


def test_source_measurements_preserve_targets_and_metrics(monkeypatch):
    points, faces = sphere()
    measurements = mm.source_geometry_measurements(points, faces)
    for quality in ("Draft", "Balanced", "Fine"):
        assert mm.smart_target(points, faces, quality) == mm.smart_target(
            points, faces, quality, measurements)
    output_points, output_faces = mm.simplify_arrays(points, faces, 1000, "Density balanced")
    baseline = mm.mesh_quality_metrics(points, faces, output_points, output_faces)
    observed = mm.mesh_quality_metrics(points, faces, output_points, output_faces,
                                       source_measurements=measurements)
    assert baseline == observed
    def fail(*args, **kwargs):
        raise AssertionError("Cached target remeasured the source")
    monkeypatch.setattr(mm, "mesh_surface_area", fail)
    mm.smart_target(points, faces, "Balanced", measurements)


def test_viewport_preparation_preserves_geometry_and_normals():
    points, faces = sphere()
    poly = mm.arrays_polydata(points, faces)
    baseline = vtkPolyDataNormals()
    baseline.SetInputData(poly)
    baseline.ComputePointNormalsOn()
    baseline.ComputeCellNormalsOff()
    baseline.SplittingOff()
    baseline.ConsistencyOn()
    baseline.Update()
    prepared = mm.prepare_viewport_poly(poly)
    assert np.array_equal(vtk_to_numpy(prepared.GetPointData().GetNormals()),
                          vtk_to_numpy(baseline.GetOutput().GetPointData().GetNormals()))
    assert all(np.array_equal(a, b) for a, b in zip(mm.polydata_arrays(poly), (points, faces)))
    assert poly.GetPointData().GetNormals() is None


def test_gpu_count_unavailable_falls_back(monkeypatch):
    points, faces = sphere()
    monkeypatch.setattr(daq, "cp", None)
    assert daq.readonly_gpu_edge_counts(faces) is None
    baseline = mm.mesh_quality_metrics(points, faces, points, faces)
    observed = mm.mesh_quality_metrics(points, faces, points, faces, gpu_preference="system")
    assert baseline == observed


def test_cached_measurements_are_bound_to_array_identity():
    class CacheOwner:
        pass
    owner = CacheOwner()
    points, faces = sphere()
    measurements = mm.source_geometry_measurements(points, faces)
    owner.source_measurement_cache = (points, faces, measurements)
    assert mm.MainWindow._cached_source_measurements(owner, points, faces) is measurements
    assert mm.MainWindow._cached_source_measurements(owner, points.copy(), faces) is None


def test_packed_edge_counts_exact_and_input_unchanged(monkeypatch):
    # Exercise the same integer operations on CPU in CI, including reversed edges,
    # degeneracies, shared faces, and IDs close to the signed 32-bit maximum.
    faces = np.tile(np.array([[0, 1, 2], [2, 1, 0], [0, 0, 2147483647],
                              [2147483647, 2, 1]], dtype=np.int32), (65536, 1))
    original = faces.copy()
    fake = SimpleNamespace(
        cuda=SimpleNamespace(Device=lambda _: nullcontext(),
                             runtime=SimpleNamespace(memGetInfo=lambda: (2**40, 2**40))),
        asarray=np.asarray, minimum=np.minimum, maximum=np.maximum, uint64=np.uint64,
        concatenate=np.concatenate, unique=np.unique, asnumpy=np.asarray,
    )
    monkeypatch.setattr(daq, "cp", fake)
    monkeypatch.setattr(daq, "select_cuda_device", lambda _: 0)
    counts = daq.readonly_gpu_edge_counts(faces)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    _, baseline = np.unique(edges, axis=0, return_counts=True)
    assert np.array_equal(counts, baseline)
    assert np.array_equal(faces, original)
    def fail(_):
        raise RuntimeError("Synthetic backend failure")
    monkeypatch.setattr(daq, "select_cuda_device", fail)
    assert daq.readonly_gpu_edge_counts(faces) is None


def test_viewport_preparation_reports_real_progress():
    points, faces = sphere()
    poly = mm.arrays_polydata(points, faces)
    events = []
    mm.prepare_viewport_poly(poly, lambda *event: events.append(event))
    assert events[0] == ("Preparing viewport normals", 0, len(points))
    assert events[-1] == ("Preparing viewport normals", len(points), len(points))


@pytest.mark.parametrize("faces", [
    np.empty((0, 3), dtype=np.int32),
    np.array([[0, 1, 2], [2, 1, 0], [0, 0, 0], [0, 1, 3]], dtype=np.int32),
    np.array([[0, 4294967295, 2], [2, 4294967295, 0]], dtype=np.int64),
    np.array([[0, 4294967296, 2], [2, 4294967296, 0]], dtype=np.int64),
    np.array([[-1, 0, 1], [1, 0, -1]], dtype=np.int64),
    np.random.default_rng(42).integers(0, 10000, (10000, 3), dtype=np.int32),
])
def test_cpu_packed_edges_match_legacy(faces):
    original = faces.copy()
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    _, expected = np.unique(edges, axis=0, return_counts=True)
    assert np.array_equal(mm.readonly_cpu_edge_counts(faces), expected)
    assert np.array_equal(faces, original)


def test_cpu_edge_count_cancellation():
    event = threading.Event()
    event.set()
    with pytest.raises(mm.OperationCancelled):
        mm.readonly_cpu_edge_counts(np.zeros((100, 3), dtype=np.int32), cancel_event=event)


def test_density_gpu_routing_preserves_palette_and_inputs(monkeypatch):
    points, faces = sphere()
    original_points, original_faces = points.copy(), faces.copy()
    field = mm._cpu_density_field(points, faces)
    calls = []
    def accelerated(p, f, device, progress, cancel):
        calls.append(device)
        return field
    monkeypatch.setattr(daq, "readonly_gpu_density_field", accelerated)
    for base, dense in [((11, 27, 52), None), ((40, 80, 40), (255, 0, 0))]:
        expected = mm.point_density_colors(points, faces, base_color=base, dense_color=dense)
        actual = mm.point_density_colors(points, faces, base_color=base, dense_color=dense,
                                         gpu_preference="selected-device")
        assert np.array_equal(actual, expected)
    assert calls == ["selected-device", "selected-device"]
    assert np.array_equal(points, original_points)
    assert np.array_equal(faces, original_faces)


def test_density_gpu_fallback_and_cancellation(monkeypatch):
    points, faces = sphere()
    expected = mm.point_density_colors(points, faces)
    monkeypatch.setattr(daq, "cp", None)
    assert np.array_equal(mm.point_density_colors(points, faces, gpu_preference="system"), expected)
    cancelled = threading.Event()
    def cancel(*args):
        cancelled.set()
        return None
    monkeypatch.setattr(daq, "readonly_gpu_density_field", cancel)
    with pytest.raises(mm.OperationCancelled):
        mm.point_density_colors(points, faces, cancel_event=cancelled, gpu_preference="system")


def test_gpu_density_backend_preserves_chunk_arithmetic(monkeypatch):
    points, faces = sphere()
    faces = np.tile(faces, (max(1, 262144 // len(faces) + 1), 1))
    expected, valid = mm._cpu_density_field(points, faces)
    fake = SimpleNamespace(**{name: getattr(np, name) for name in (
        "asarray", "bincount", "float32", "zeros", "cross", "repeat", "log", "maximum")},
        linalg=np.linalg, asnumpy=np.asarray,
        cuda=SimpleNamespace(Device=lambda _: nullcontext(),
            get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
            runtime=SimpleNamespace(memGetInfo=lambda: (2**40, 2**40))))
    monkeypatch.setattr(daq, "cp", fake)
    monkeypatch.setattr(daq, "select_cuda_device", lambda _: 0)
    observed, observed_valid = daq.readonly_gpu_density_field(points, faces)
    assert np.array_equal(observed, expected)
    assert np.array_equal(observed_valid, valid)
