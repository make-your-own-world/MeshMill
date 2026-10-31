"""Local-only negative mold geometry checks."""
import numpy as np
import pytest
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import (
    vtkAppendPolyData,
    vtkFeatureEdges,
    vtkPolyDataConnectivityFilter,
    vtkTriangleFilter,
)
from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkCylinderSource, vtkSphereSource
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter

from clamshell_mold import (
    clamshell_mold,
    mold_sampling_plan,
    rotated_split_normal,
    separated_mold_halves,
    split_rim_lines,
)


def inside(mesh, point):
    points = vtkPoints()
    points.InsertNextPoint(*point)
    query = vtkPolyData()
    query.SetPoints(points)
    check = vtkSelectEnclosedPoints()
    check.SetInputData(query)
    check.SetSurfaceData(mesh)
    check.Update()
    return bool(check.IsInside(0))


def points_inside(mesh, samples):
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(samples, dtype=float), deep=True))
    query = vtkPolyData()
    query.SetPoints(points)
    check = vtkSelectEnclosedPoints()
    check.SetInputData(query)
    check.SetSurfaceData(mesh)
    check.Update()
    return vtk_to_numpy(check.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)


@pytest.mark.parametrize("shape", ["Blob", "Box"])
@pytest.mark.parametrize("split", [None, (0, 0, 1)])
def test_closed_mold_contains_real_cavity(shape, split):
    sphere = vtkSphereSource()
    sphere.SetRadius(10)
    sphere.SetThetaResolution(24)
    sphere.SetPhiResolution(24)
    sphere.Update()
    meshes, report = clamshell_mold(sphere.GetOutput(), shape=shape, split_normal=split)
    assert len(meshes) == (1 if split is None else 2)
    assert report["cavity_voxels"] > 10000
    for mesh in meshes:
        edges = vtkFeatureEdges()
        edges.SetInputData(mesh)
        edges.BoundaryEdgesOn()
        edges.NonManifoldEdgesOn()
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.Update()
        assert edges.GetOutput().GetNumberOfCells() == 0
        assert not inside(mesh, (0, 0, 2))
        assert not inside(mesh, (20, 0, 2))
        assert not points_inside(mesh, report["cavity_test_points"]).any()
        vertices = vtk_to_numpy(mesh.GetPoints().GetData())
        faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        for edge in ((0, 1), (1, 2), (2, 0)):
            lengths = np.linalg.norm(vertices[faces[:, edge[0]]] - vertices[faces[:, edge[1]]], axis=1)
            # A cell-local split cannot create long cap triangles across the cavity.
            assert lengths.max() <= np.sqrt(3) * report["pitch_mm"] * 1.01
    assert sum(inside(mesh, (12, 0, 2)) for mesh in meshes) == 1
    assert np.isfinite(meshes[0].GetBounds()).all()
    if split is not None:
        for mesh in meshes:
            points = vtk_to_numpy(mesh.GetPoints().GetData())
            faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
            triangles = points[faces]
            cut = triangles[np.all(np.abs(triangles[:, :, 2]) < 1e-5, axis=1)]
            assert len(cut) > 0
            # Split caps form a rim around the cavity, not a disk across it.
            assert np.all(np.linalg.norm(cut.mean(axis=1)[:, :2], axis=1) > 9)


def test_fine_requested_pitch_adapts_to_memory_instead_of_failing():
    budget = 64 * 1024**2
    plan = mold_sampling_plan([0, 140, 0, 110, 0, 115], 3, 2, .001, budget)
    assert plan["memory_adjusted"]
    assert plan["pitch_mm"] > .001
    assert plan["estimated_bytes"] <= budget


def test_auto_defaults_scale_with_geometry():
    small = mold_sampling_plan([0, 10, 0, 20, 0, 15])
    large = mold_sampling_plan([0, 100, 0, 200, 0, 150])
    for key in ("wall_mm", "bridge_mm", "pitch_mm"):
        assert large[key] == pytest.approx(small[key] * 10)


def test_separated_halves_survive_combined_stl_round_trip(tmp_path):
    source = vtkSphereSource()
    source.SetRadius(10)
    source.Update()
    halves, _ = clamshell_mold(source.GetOutput())
    separated = separated_mold_halves(halves)
    joined = vtkAppendPolyData()
    for original, moved in zip(halves, separated, strict=True):
        delta = vtk_to_numpy(moved.GetPoints().GetData()) - vtk_to_numpy(original.GetPoints().GetData())
        assert np.allclose(delta, delta[0], atol=1e-5)
        joined.AddInputData(moved)
    joined.Update()
    path = tmp_path / "separated.stl"
    writer = vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(joined.GetOutput())
    writer.SetFileTypeToBinary()
    assert writer.Write()
    reader = vtkSTLReader()
    reader.SetFileName(str(path))
    reader.MergingOn()
    reader.Update()
    connected = vtkPolyDataConnectivityFilter()
    connected.SetInputData(reader.GetOutput())
    connected.SetExtractionModeToAllRegions()
    connected.Update()
    assert connected.GetNumberOfExtractedRegions() == 2


def test_custom_split_rotations():
    assert np.allclose(rotated_split_normal(), [0, 0, 1])
    assert np.allclose(rotated_split_normal(0, 90, 0), [1, 0, 0])
    assert np.allclose(rotated_split_normal(90, 0, 0), [0, -1, 0])
    normal = rotated_split_normal(23, 37, 12)
    assert np.isclose(np.linalg.norm(normal), 1)
    source = vtkSphereSource()
    source.SetRadius(10)
    source.Update()
    halves, _ = clamshell_mold(source.GetOutput(), split_normal=normal)
    for half in halves:
        projected = vtk_to_numpy(half.GetPoints().GetData()) @ normal
        assert projected.min() >= -1e-4 or projected.max() <= 1e-4
        edges = vtkFeatureEdges()
        edges.SetInputData(half)
        edges.BoundaryEdgesOn()
        edges.NonManifoldEdgesOn()
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.Update()
        assert edges.GetOutput().GetNumberOfCells() == 0


def test_split_rim_outlines_outer_edge_and_cavity():
    source = vtkSphereSource()
    source.SetRadius(10)
    source.Update()
    halves, _ = clamshell_mold(source.GetOutput())
    for half in halves:
        rim = split_rim_lines(half, [0, 0, 20], [0, 0, 0], 1e-4)
        assert rim.GetNumberOfLines() > 0
        assert np.allclose(vtk_to_numpy(rim.GetPoints().GetData())[:, 2], 0)
        segments = vtk_to_numpy(rim.GetLines().GetConnectivityArray()).reshape(-1, 2)
        adjacency = {}
        for first, second in segments:
            adjacency.setdefault(int(first), set()).add(int(second))
            adjacency.setdefault(int(second), set()).add(int(first))
        # Grid-aligned cuts can have touching rim segments at stair-step vertices.
        assert all(len(neighbors) >= 2 and len(neighbors) % 2 == 0 for neighbors in adjacency.values())
        remaining = set(adjacency)
        loops = 0
        while remaining:
            pending = [remaining.pop()]
            loops += 1
            while pending:
                for neighbor in adjacency[pending.pop()]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        pending.append(neighbor)
        assert loops >= 2


def test_split_origin_can_move_off_center():
    source = vtkSphereSource()
    source.SetRadius(10)
    source.Update()
    halves, _ = clamshell_mold(source.GetOutput(), split_normal=[0, 0, 1], split_origin=[0, 0, 2])
    extents = []
    for half in halves:
        z = vtk_to_numpy(half.GetPoints().GetData())[:, 2]
        assert z.min() >= 2 - 1e-4 or z.max() <= 2 + 1e-4
        extents.append(float(z.max() - z.min()))
    assert abs(extents[0] - extents[1]) > 3


@pytest.mark.parametrize("shape", ["Blob", "Box"])
def test_missing_cube_faces_do_not_create_interior_walls(shape):
    cube = vtkCubeSource()
    cube.SetBounds(-10, 10, -10, 10, -10, 10)
    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(cube.GetOutputPort())
    triangles.Update()
    poly = triangles.GetOutput()
    points = vtk_to_numpy(poly.GetPoints().GetData())
    faces = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    centers = points[faces].mean(axis=1)
    faces = faces[(centers[:, 0] > -9) & (centers[:, 2] < 9)]
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, faces.size + 1, 3, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(faces.astype(np.int64).ravel(), deep=True))
    partial = vtkPolyData()
    partial.SetPoints(poly.GetPoints())
    partial.SetPolys(cells)
    from exposed_surface import exposed_surface
    skin, _ = exposed_surface(partial, .5, backend="cpu")
    halves, report = clamshell_mold(skin, shape=shape, split_normal=(.4, .7, .2))
    known_empty = np.array([[x, y, z] for x in (-6, 0, 6) for y in (-6, 0, 6) for z in (-6, 0, 6)])
    assert report["unknown_rays"] > 0
    for half in halves:
        assert not points_inside(half, known_empty).any()


def test_open_tube_cavity_keeps_center_empty():
    tube = vtkCylinderSource()
    tube.SetRadius(10)
    tube.SetHeight(20)
    tube.SetResolution(48)
    tube.CappingOff()
    tube.Update()
    from exposed_surface import exposed_surface
    skin, _ = exposed_surface(tube.GetOutput(), .5, backend="cpu")
    halves, _ = clamshell_mold(skin, shape="Box", split_normal=(0, 1, 0))
    known_empty = np.array([[x, y, z] for x in (-4, 0, 4) for y in (-7, 0, 7) for z in (-4, 0, 4)])
    for half in halves:
        assert not points_inside(half, known_empty).any()


def test_split_wall_normals_face_out_of_material():
    source = vtkSphereSource()
    source.SetRadius(10)
    source.Update()
    halves, _ = clamshell_mold(source.GetOutput())
    for half in halves:
        points = vtk_to_numpy(half.GetPoints().GetData())
        normals = vtk_to_numpy(half.GetPointData().GetNormals())
        radius = np.linalg.norm(points, axis=1)
        dot = np.sum(points * normals, axis=1)
        away_from_cut = np.abs(points[:, 2]) > 1
        assert np.median(dot[(radius > 12) & away_from_cut]) > 0
        assert np.median(dot[(radius < 11) & away_from_cut]) < 0
