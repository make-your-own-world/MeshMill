import numpy as np
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy

from mesh_export import calibrated_export_mesh, rigid_export_matrix, write_polygon_mesh
from mesh_workflow import run_workflow
from meshmill import arrays_polydata, load_stl, save_stl


def fixture():
    points = np.array([(0, 0, 0), (2, 0, 0), (0, 3, 0), (0, 0, 7)], dtype=float)
    faces = np.array([(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)])
    a, b = np.radians([23, -31])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    frame = np.eye(4)
    frame[:3, :3] = rx @ ry
    frame[:3, 3] = [23, -41, 17]
    source = points @ frame[:3, :3].T + frame[:3, 3]
    return points, source, faces, np.linalg.inv(frame)


@pytest.mark.parametrize('suffix', ['.stl', '.dae', '.obj', '.ply', '.glb'])
def test_calibrated_axes_origin_and_shape_survive_format_roundtrip(tmp_path, suffix):
    expected, source, faces, matrix = fixture()
    original = source.copy()
    poly = arrays_polydata(source, faces)
    calibrated = calibrated_export_mesh(poly, matrix)
    points = vtk_to_numpy(calibrated.GetPoints().GetData())
    assert np.allclose(points, expected, atol=1e-12)
    assert np.array_equal(vtk_to_numpy(poly.GetPoints().GetData()), original)
    path = tmp_path / ('aligned' + suffix)
    if suffix == '.stl':
        save_stl(calibrated, path)
    else:
        write_polygon_mesh(points, faces, path, merge=False)
    _, loaded, loaded_faces = load_stl(path)
    assert len(loaded_faces) == len(faces)
    assert np.allclose(np.sort(loaded, axis=0), np.sort(expected, axis=0), atol=2e-6)
    assert np.allclose(loaded.min(0), (0, 0, 0), atol=2e-6)
    assert np.allclose(loaded.max(0), (2, 3, 7), atol=2e-6)


def test_headless_export_uses_same_finalized_calibration(tmp_path):
    expected, source, faces, matrix = fixture()
    input_path, output_path = tmp_path / 'source.ply', tmp_path / 'aligned.dae'
    write_polygon_mesh(source, faces, input_path, merge=False)
    run_workflow(input_path, output_path, [{'kind': 'convert', 'export_transform': matrix.tolist()}], progress=lambda _: None)
    _, points, _ = load_stl(output_path)
    assert np.allclose(np.sort(points, axis=0), np.sort(expected, axis=0), atol=1e-10)


def test_export_calibration_rejects_scaling_and_reflections():
    for diagonal in [(2, 1, 1, 1), (-1, 1, 1, 1)]:
        with pytest.raises(ValueError, match='rigid'):
            rigid_export_matrix(np.diag(diagonal))
