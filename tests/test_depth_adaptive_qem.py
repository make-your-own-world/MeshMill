from __future__ import annotations

import numpy as np
import pytest

import depth_adaptive_qem as daq
import meshmill as mm


def tetrahedra_fixture(count: int = 400) -> tuple[np.ndarray, np.ndarray]:
    base_points = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    base_faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32
    )
    offsets = np.arange(count, dtype=np.float64)[:, None, None] * np.asarray([1.5, 0, 0])
    points = (base_points[None, :, :] + offsets).reshape(-1, 3)
    face_offsets = (np.arange(count, dtype=np.int32) * 4)[:, None, None]
    faces = (base_faces[None, :, :] + face_offsets).reshape(-1, 3)
    return points, faces


def test_algorithm_order_prioritizes_quality_modes():
    assert mm.ALGORITHMS[:4] == [
        mm.DEPTH_DENSITY_ADAPTIVE_LABEL,
        mm.DEPTH_ADAPTIVE_LABEL,
        "Density balanced",
        "Fast QEM",
    ]
    assert mm.CLI_ALGORITHMS["fast"] == "Fast QEM"


def test_cursor_orbit_rotation_preserves_radius_about_anchor():
    vector = np.asarray([3.0, -2.0, 5.0])
    rotated = mm.MeshInteractorStyle._rotate_vector(
        vector, np.asarray([0.0, 1.0, 0.0]), 37.0
    )
    assert np.linalg.norm(rotated) == pytest.approx(np.linalg.norm(vector))
    assert rotated[0] != pytest.approx(vector[0])


def test_default_algorithm_tracks_gpu_support(monkeypatch):
    monkeypatch.setattr(mm, "gpu_backend_supported", lambda preference="system": True)
    assert mm.default_algorithm() == mm.DEPTH_DENSITY_ADAPTIVE_LABEL
    monkeypatch.setattr(mm, "gpu_backend_supported", lambda preference="system": False)
    assert mm.default_algorithm() == "Density balanced"


def test_startup_algorithm_remembers_supported_last_choice(monkeypatch):
    monkeypatch.setattr(mm, "gpu_backend_supported", lambda preference="system": True)
    assert (
        mm.resolve_startup_algorithm(
            "last", mm.DEPTH_ADAPTIVE_LABEL, "selected-gpu"
        )
        == mm.DEPTH_ADAPTIVE_LABEL
    )


def test_startup_algorithm_migrates_previous_quality_labels(monkeypatch):
    monkeypatch.setattr(mm, "gpu_backend_supported", lambda preference="system": True)
    assert (
        mm.resolve_startup_algorithm(
            "last", "Depth + Density Adaptive QEM (GPU / Highest Quality)"
        )
        == mm.DEPTH_DENSITY_ADAPTIVE_LABEL
    )


def test_startup_algorithm_falls_back_from_unavailable_gpu_choice(monkeypatch):
    monkeypatch.setattr(mm, "gpu_backend_supported", lambda preference="system": False)
    assert (
        mm.resolve_startup_algorithm(
            "last", mm.DEPTH_DENSITY_ADAPTIVE_LABEL, "system"
        )
        == "Density balanced"
    )
    assert mm.resolve_startup_algorithm("auto", "Fast QEM") == "Density balanced"


def test_gpu_algorithms_are_registered_when_backend_is_supported():
    if daq.gpu_backend_supported():
        assert mm.CLI_ALGORITHMS["depth-adaptive"] == mm.DEPTH_ADAPTIVE_LABEL
        assert (
            mm.CLI_ALGORITHMS["depth-density-adaptive"]
            == mm.DEPTH_DENSITY_ADAPTIVE_LABEL
        )
    else:
        assert "depth-adaptive" not in mm.CLI_ALGORITHMS
        assert "depth-density-adaptive" not in mm.CLI_ALGORITHMS


def test_cuda_device_preferences_use_stable_pci_ids():
    devices = daq.cuda_devices()
    if not devices:
        return
    assert len({device["pci_bus_id"] for device in devices}) == len(devices)
    assert all(device["name"] and device["total_memory"] > 0 for device in devices)
    try:
        selected = daq.select_cuda_device(devices[-1]["pci_bus_id"])
        assert selected == devices[-1]["id"]
    finally:
        assert daq.select_cuda_device("system") in {device["id"] for device in devices}


def test_depth_adaptive_falls_back_cleanly_without_cuda(monkeypatch):
    points, faces = tetrahedra_fixture()
    monkeypatch.setattr(daq, "cp", None)

    output_points, output_faces, report = daq.simplify_depth_adaptive_qem(
        points, faces, 1_000
    )

    assert len(output_faces) <= 1_000
    assert len(output_points) > 0
    assert report["backend"] == "Fast QEM fallback"


def test_gpu_preparation_is_a_noop_without_supported_backend(monkeypatch):
    monkeypatch.setattr(daq, "gpu_backend_supported", lambda preference="system": False)
    report = daq.prepare_gpu_backend()
    assert report["backend"] == "unavailable"
    assert report["seconds"] >= 0.0


def test_depth_density_falls_back_cleanly_without_cuda(monkeypatch):
    points, faces = tetrahedra_fixture()
    monkeypatch.setattr(daq, "cp", None)

    output_points, output_faces, report = daq.simplify_depth_density_adaptive_qem(
        points, faces, 1_000
    )

    assert len(output_faces) <= 1_000
    assert len(output_points) > 0
    assert report["backend"] == "Fast QEM fallback"
    assert "surface_finishing" in report
    assert "topology_finishing" in report


def test_nonmanifold_finishing_removes_only_surplus_faces():
    points = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int32)

    output_faces, report = daq._remove_nonmanifold_surplus(points, faces)

    assert len(output_faces) == 2
    assert report == {"nonmanifold_edges_before": 1, "removed_faces": 1}


def test_surface_finishing_matches_area_and_bounds():
    source_points = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32)
    output_points = source_points.copy()
    output_points[4, 2] = 0.05

    finished, report = daq._match_surface_area(
        source_points, faces, output_points, faces, maximum_alpha=0.5
    )

    assert report["final_area_ratio"] == pytest.approx(1.0, abs=1e-4)
    assert np.ptp(finished, axis=0) == pytest.approx(
        np.ptp(source_points, axis=0), abs=2e-6
    )


def test_physical_probe_dimensions_are_mesh_derived():
    points, faces = tetrahedra_fixture()
    dimensions = daq.physical_probe_dimensions(points, faces, 1_000)

    assert dimensions["diameter"] > 0
    assert dimensions["tip_radius"] == dimensions["diameter"] * 0.5
    assert dimensions["minimum_pitch"] >= dimensions["diameter"]


def test_quality_metrics_are_percentages_with_expected_keys():
    points, faces = tetrahedra_fixture()
    output_points, output_faces = mm.simplify_arrays(points, faces, 1_000, "Fast QEM")

    metrics = mm.mesh_quality_metrics(points, faces, output_points, output_faces)

    assert set(metrics) == {
        "reduction_percent",
        "rms_percent",
        "p95_percent",
        "drift_percent",
        "area_percent",
        "nonmanifold_percent",
    }
    assert all(np.isfinite(value) and value >= 0 for value in metrics.values())


def test_quality_baseline_projects_settings_without_mesh_scale_analysis():
    points, faces = tetrahedra_fixture(count=1)
    fast = mm.mesh_quality_baseline(
        points, faces, estimated_triangles=2, algorithm="Fast QEM", quality="Balanced"
    )
    adaptive = mm.mesh_quality_baseline(
        points,
        faces,
        estimated_triangles=2,
        algorithm="Depth Adaptive QEM (GPU / Faster)",
        quality="Fine",
    )

    assert fast["reduction_percent"] == 50.0
    assert all(np.isfinite(value) and value >= 0 for value in fast.values())
    assert adaptive["rms_percent"] < fast["rms_percent"]
    assert adaptive["p95_percent"] < fast["p95_percent"]


def test_simplify_worker_returns_quality_metrics():
    points, faces = tetrahedra_fixture()

    output_points, output_faces, metrics, error = mm.simplify_process_job(
        points, faces, 1_000, "Fast QEM", "system"
    )

    assert error is None
    assert len(output_points) > 0
    assert len(output_faces) <= 1_000
    assert metrics is not None and "rms_percent" in metrics
