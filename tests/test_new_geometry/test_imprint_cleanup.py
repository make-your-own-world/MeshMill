import numpy as np
import pytest
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter

from imprint_cleanup import clean_generated_mesh
from meshmill import arrays_polydata, polydata_arrays


def test_split_edge_repair_closes_t_junction_without_moving_points():
    from vtkmodules.util.numpy_support import vtk_to_numpy
    points = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.],
                       [0., 0., 2.], [1., 0., 0.]])
    faces = np.array([[0, 2, 4], [4, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    result, report = clean_generated_mesh(arrays_polydata(points, faces), closed=True)
    assert report["repaired_split_edges"] == 1
    assert report["boundary_edges"] == 0
    assert result.GetNumberOfPolys() == 6
    assert set(map(tuple, vtk_to_numpy(result.GetPoints().GetData()))) == set(map(tuple, points))


def test_stl_precision_welding_duplicates_and_degenerate_faces(tmp_path):
    sphere = vtkSphereSource()
    sphere.Update()
    points, faces = polydata_arrays(sphere.GetOutput())
    p = np.vstack([points, points])
    f = np.vstack([faces, faces + len(points), [[0, 0, 1]]])
    clean, report = clean_generated_mesh(arrays_polydata(p, f), closed=True)
    assert report["removed_duplicate_faces"] == len(faces)
    assert report["removed_degenerate_faces"] == 1
    assert report["boundary_edges"] == 0
    assert clean.GetNumberOfCells() == len(faces)
    writer = vtkSTLWriter()
    writer.SetInputData(clean)
    writer.SetFileName(str(tmp_path / "clean.stl"))
    writer.SetFileTypeToBinary()
    assert writer.Write()
    reader = vtkSTLReader()
    reader.SetFileName(str(tmp_path / "clean.stl"))
    reader.Update()
    _, roundtrip = clean_generated_mesh(reader.GetOutput(), closed=True)
    assert roundtrip["boundary_edges"] == 0
    assert roundtrip["nonmanifold_edges"] == 0


def test_open_surface_is_reported_and_not_falsely_called_solid():
    poly = arrays_polydata(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]), np.array([[0,1,2]]))
    _, report = clean_generated_mesh(poly)
    assert report["boundary_edges"] == 3
    with pytest.raises(ValueError, match="3 open edges"):
        clean_generated_mesh(poly, closed=True)


def test_standard_save_rechecks_generated_geometry(tmp_path):
    from meshmill import save_stl
    sphere = vtkSphereSource()
    sphere.Update()
    generated, _ = clean_generated_mesh(sphere.GetOutput(), closed=True)
    destination = tmp_path / "automatic-cleanup.stl"
    save_stl(generated, destination)
    reader = vtkSTLReader()
    reader.SetFileName(str(destination))
    reader.Update()
    _, report = clean_generated_mesh(reader.GetOutput(), closed=True)
    assert report["boundary_edges"] == 0
    assert report["nonmanifold_edges"] == 0
    assert report["inconsistent_edges"] == 0
