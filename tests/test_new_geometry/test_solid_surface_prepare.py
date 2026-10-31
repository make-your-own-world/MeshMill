import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pymeshfix import PyTMesh
from test_solidify import open_cube
from vtkmodules.util.numpy_support import vtk_to_numpy

from meshmill import arrays_polydata
from solid_surface_prepare import _intersecting_source_faces, complete_scan_volume


def test_intersection_binding_prefix_maps_reordered_geometry():
    points = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0),
                       (0, 0, 1), (1, 1, 1)], dtype=float)
    faces = np.array([(0, 1, 2), (0, 2, 3), (1, 3, 4)])

    class Mesh:
        def select_intersecting_triangles(self):
            # 0.18.1 writes two face IDs into an allocated (2, 3) array.
            return np.array([[2, 0, 900], [901, 902, 903]])

        def return_arrays(self):
            return points[::-1], (len(points) - 1 - faces)[[1, 2, 0]]

    assert np.array_equal(_intersecting_source_faces(points, faces, Mesh()),
                          [True, True, False])


def test_small_crossing_scan_fragments_are_repaired_without_moving_walls():
    source = open_cube()
    points = vtk_to_numpy(source.GetPoints().GetData()).astype(float)
    faces = vtk_to_numpy(source.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    fragments = np.array([(-.1, -.1, 0), (.1, -.1, 0), (0, .1, 0),
                          (0, -.05, -.1), (0, -.05, .1), (0, .1, 0)])
    fragment_faces = len(points) + np.array([(0, 1, 2), (3, 4, 5)])
    input_points = np.vstack((points, fragments))
    input_faces = np.vstack((faces, fragment_faces)).astype(np.int32)
    original_points, original_faces = input_points.copy(), input_faces.copy()
    result_points, result_faces, report = complete_scan_volume(input_points, input_faces)
    assert report['intersecting_input_faces'] == 2
    assert report['replaced_input_faces'] == 2
    assert report['topology']['boundary_edges'] == 0
    assert report['topology']['nonmanifold_vertices'] == 0
    assert np.array_equal(input_points, original_points)
    assert np.array_equal(input_faces, original_faces)

    def keys(p, f):
        return {tuple(sorted(map(tuple, p[face].astype(np.float32)))) for face in f}

    assert keys(points, faces).issubset(keys(result_points, result_faces))
    check = PyTMesh()
    check.set_quiet(True)
    check.load_array(result_points.astype(float), result_faces.astype(np.int32))
    assert check.n_boundaries == 0
    assert len(check.select_intersecting_triangles()) == 0
    assert arrays_polydata(result_points, result_faces).GetNumberOfCells() > 0


@pytest.mark.parametrize("failure", ["unavailable", "setup", "query"])
def test_solid_completion_falls_back_when_cuda_is_unusable(monkeypatch, failure):
    import depth_adaptive_qem

    source = open_cube()
    points = vtk_to_numpy(source.GetPoints().GetData()).astype(float)
    faces = vtk_to_numpy(source.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    expected_points, expected_faces, _ = complete_scan_volume(points, faces)
    calls = []

    class BrokenGPU:
        def __init__(self, *args):
            calls.append("setup")
            if failure == "setup":
                raise RuntimeError("CUDA driver unavailable")

        def evaluate(self, queries):
            calls.append("query")
            raise RuntimeError("CUDA device unavailable")

    monkeypatch.setattr(depth_adaptive_qem, "gpu_backend_supported",
                        lambda preference: failure != "unavailable")
    monkeypatch.setitem(sys.modules, "surface_distance_gpu",
                        SimpleNamespace(GPUSurfaceDistance=BrokenGPU))
    result_points, result_faces, report = complete_scan_volume(
        points, faces, gpu_preference="system")
    assert np.array_equal(result_points, expected_points)
    assert np.array_equal(result_faces, expected_faces)
    assert report["topology"]["boundary_edges"] == 0
    assert calls == {"unavailable": [], "setup": ["setup"],
                     "query": ["setup", "query"]}[failure]
