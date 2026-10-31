from collections import Counter
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersCore import vtkTriangleFilter
from vtkmodules.vtkFiltersSources import vtkCubeSource
from vtkmodules.vtkIOGeometry import vtkOBJReader
from vtkmodules.vtkIOPLY import vtkPLYReader

from mesh_export import collada_face_groups, coplanar_polygons, write_polygon_mesh
from mesh_formats import load_geometry


def cube():
    source = vtkCubeSource()
    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(source.GetOutputPort())
    triangles.Update()
    poly = triangles.GetOutput()
    points, inverse = np.unique(
        vtk_to_numpy(poly.GetPoints().GetData()), axis=0, return_inverse=True
    )
    faces = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return points, inverse[faces]


def polygon_area(points, polygons):
    return sum(
        np.linalg.norm(
            np.cross(points[p] - points[p[0]], np.roll(points[p], -1, axis=0) - points[p[0]]).sum(
                axis=0
            )
        )
        / 2
        for p in polygons
    )


def test_merge_cube_preserves_closed_boundary_area_and_winding():
    points, faces = cube()
    before_p, before_f = points.copy(), faces.copy()
    polygons, report = coplanar_polygons(points, faces)
    assert len(polygons) == 6 and all(len(p) == 4 for p in polygons)
    assert report["removed_edges"] == 6
    assert polygon_area(points, polygons) == pytest.approx(6)
    edges = Counter((int(a), int(b)) for p in polygons for a, b in zip(p, p[1:] + p[:1]))
    assert all(edges[(b, a)] == count for (a, b), count in edges.items())
    assert np.array_equal(before_p, points) and np.array_equal(before_f, faces)


def test_holes_and_curved_faces_are_not_filled_or_flattened():
    points = np.array(
        [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0], [1, 1, 0], [3, 1, 0], [3, 3, 0], [1, 3, 0]],
        dtype=float,
    )
    faces = np.array(
        [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    )
    polygons, _ = coplanar_polygons(points, faces)
    assert len(polygons) == len(faces)
    assert polygon_area(points, polygons) == pytest.approx(12)
    curved = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.001]])
    assert len(coplanar_polygons(curved, np.array([[0, 1, 2], [1, 3, 2]]))[0]) == 2


@pytest.mark.parametrize("suffix,reader_type", [(".obj", vtkOBJReader), (".ply", vtkPLYReader)])
def test_indexed_polygon_round_trip(tmp_path, suffix, reader_type):
    points, faces = cube()
    path = tmp_path / ("mesh" + suffix)
    write_polygon_mesh(points, faces, path)
    reader = reader_type()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    assert poly.GetNumberOfPolys() == 6
    assert np.array_equal(np.unique(vtk_to_numpy(poly.GetPoints().GetData()), axis=0), points)
    assert poly.GetBounds() == (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)


def test_collada_units_indices_and_no_local_metadata(tmp_path):
    points, faces = cube()
    path = tmp_path / "mesh.dae"
    write_polygon_mesh(points, faces, path)
    root = ET.parse(path).getroot()
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    assert root.find("c:asset/c:unit", ns).get("meter") == "0.001"
    lists = root.findall(".//c:polylist", ns)
    assert sum(int(p.get("count")) for p in lists) == 6
    assert all(int(n) == 4 for p in lists for n in p.find("c:vcount", ns).text.split())
    indices = np.concatenate([np.fromstring(p.find("c:p", ns).text, sep=" ", dtype=int) for p in lists])
    assert len(indices) == 24 and indices.min() == 0 and indices.max() == 7
    assert root.find(".//c:instance_geometry", ns).get("url") == "#mesh"
    assert root.find(".//c:instance_node", ns).get("url") == "#mesh-definition"
    assert root.find(".//c:visual_scene/c:node/c:scale", ns).text == "0.01 0.01 0.01"
    stored = np.fromstring(root.find(".//c:float_array", ns).text, sep=" ").reshape(-1, 3)
    assert np.array_equal(stored, points * 100)
    assert str(tmp_path) not in path.read_text()


def test_collada_material_boundaries_preserve_edge_adjacency(tmp_path):
    points, faces = cube()
    polygons, _ = coplanar_polygons(points, faces)
    groups = collada_face_groups(polygons)
    assert sorted(tuple(p) for group in groups for p in group) == sorted(map(tuple, polygons))
    for group in groups:
        edges = [tuple(sorted((a, b))) for p in group for a, b in zip(p, p[1:] + p[:1])]
        assert max(Counter(edges).values()) == 1
    path = tmp_path / 'mesh.dae'
    write_polygon_mesh(points, faces, path)
    root = ET.parse(path).getroot()
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    colors = [n.text for n in root.findall('.//c:diffuse/c:color', ns)]
    assert len(set(colors)) == 1
    bindings = {n.get('symbol') for n in root.findall('.//c:instance_material', ns)}
    assert bindings == {n.get('material') for n in root.findall('.//c:polylist', ns)}


def test_collada_instancing_rejects_cycles_and_external_nodes(tmp_path):
    points, faces = cube()
    path = tmp_path / 'mesh.dae'
    write_polygon_mesh(points, faces, path)
    tree = ET.parse(path)
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    definition = tree.find('.//c:library_nodes/c:node', ns)
    instance = ET.SubElement(definition, '{'+ns['c']+'}instance_node', url='#mesh-definition')
    tree.write(path)
    with pytest.raises(ValueError, match='cycle'):
        load_geometry(path)
    instance.set('url', 'external.dae#object')
    tree.write(path)
    with pytest.raises(ValueError, match='external or missing'):
        load_geometry(path)


def test_disabled_merge_and_cancel_leave_existing_output_intact(tmp_path):
    points, faces = cube()
    path = tmp_path / "mesh.obj"
    report = write_polygon_mesh(points, faces, path, merge=False)
    assert report["polygons"] == 12
    previous = path.read_bytes()

    def cancel(stage, done, total):
        if stage == "Writing geometry" and done == total:
            raise RuntimeError("Cancelled")

    with pytest.raises(RuntimeError, match="Cancelled"):
        write_polygon_mesh(points, faces, path, progress=cancel)
    assert path.read_bytes() == previous
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mesh.obj"]


@pytest.mark.parametrize("suffix", [".dae", ".glb", ".obj", ".ply"])
def test_supported_imports_preserve_surface_scale_and_closedness(tmp_path, suffix):
    points, faces = cube()
    points = points * [31, 47, 63] + [100, 200, 300]
    path = tmp_path / ("mesh" + suffix)
    write_polygon_mesh(points, faces, path)
    poly, output, triangles = load_geometry(path)
    assert len(triangles) == 12
    assert np.allclose(np.sort(output, axis=0), np.sort(points, axis=0), atol=3e-5, rtol=0)
    assert np.allclose(poly.GetBounds(), (84.5, 115.5, 176.5, 223.5, 268.5, 331.5), atol=3e-5)
    edges = np.sort(
        np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1
    )
    assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)


def test_collada_scene_transform_and_reflection(tmp_path):
    points, faces = cube()
    path = tmp_path / "mesh.dae"
    write_polygon_mesh(points, faces, path)
    tree = ET.parse(path)
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    node = tree.find(".//c:visual_scene/c:node", ns)
    translate = ET.Element("{" + ns["c"] + "}translate")
    translate.text = "10 20 30"
    node.insert(0, translate)
    scale = ET.Element("{" + ns["c"] + "}scale")
    scale.text = "-2 3 4"
    node.insert(1, scale)
    tree.write(path)
    _, output, triangles = load_geometry(path)
    assert np.allclose(output.min(axis=0), [9, 18.5, 28])
    assert np.allclose(output.max(axis=0), [11, 21.5, 32])
    centered = output - output.mean(axis=0)
    signed = (
        np.einsum(
            "ij,ij->i",
            centered[triangles[:, 0]],
            np.cross(centered[triangles[:, 1]], centered[triangles[:, 2]]),
        ).sum()
        / 6
    )
    assert signed == pytest.approx(24)


def test_ply_large_planar_boundary_remains_compatible(tmp_path):
    angle = np.arange(300) * 2 * np.pi / 300
    points = np.vstack(([0, 0, 0], np.column_stack((np.cos(angle), np.sin(angle), np.zeros(300)))))
    faces = np.array([[0, i + 1, (i + 1) % 300 + 1] for i in range(300)])
    path = tmp_path / "disk.ply"
    report = write_polygon_mesh(points, faces, path)
    assert report["polygons"] == 300
    reader = vtkPLYReader()
    reader.SetFileName(str(path))
    reader.Update()
    assert reader.GetOutput().GetNumberOfPolys() == 300
    # A legacy uint-count PLY with a 300-vertex face is also read without truncation.
    import struct

    with path.open("wb") as stream:
        stream.write(
            b"ply\nformat binary_little_endian 1.0\nelement vertex 300\nproperty double x\nproperty double y\nproperty double z\nelement face 1\nproperty list uint int vertex_indices\nend_header\n"
        )
        stream.write(points[1:].astype("<f8").tobytes())
        stream.write(struct.pack("<I", 300))
        stream.write(np.arange(300, dtype="<i4").tobytes())
    _, output, triangles = load_geometry(path)
    assert len(triangles) == 298
    assert polygon_area(output, triangles.tolist()) == pytest.approx(
        300 * np.sin(2 * np.pi / 300) / 2
    )


@pytest.mark.parametrize("encoding", ["ascii", "binary_big_endian"])
def test_ply_encodings_and_ignored_attributes(tmp_path, encoding):
    import struct

    path = tmp_path / "mesh.ply"
    header = f"ply\nformat {encoding} 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nelement face 1\nproperty list uchar int vertex_indices\nproperty list uchar float texcoord\nend_header\n"
    with path.open("wb") as stream:
        stream.write(header.encode())
        if encoding == "ascii":
            stream.write(b"0 0 0 255\n1 0 0 255\n0 1 0 255\n3 0 1 2 6 0.1 0.2 0.3 0.4 0.5 0.6\n")
        else:
            for p in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]:
                stream.write(struct.pack(">fffB", *p, 255))
            stream.write(struct.pack(">BiiiBffffff", 3, 0, 1, 2, 6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    _, points, faces = load_geometry(path)
    assert len(points) == 3 and len(faces) == 1


def test_collada_definition_origin_follows_distant_geometry(tmp_path):
    points, faces = cube()
    points = points * [31, 47, 63] + [1200, -3400, 5600]
    path = tmp_path / "distant.dae"
    write_polygon_mesh(points, faces, path)
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    tree = ET.parse(path)
    local = np.fromstring(tree.find(".//c:float_array", ns).text, sep=" ").reshape(-1, 3)
    assert np.allclose(local.min(0) + local.max(0), 0)
    translation = tree.find(".//c:visual_scene/c:node/c:translate", ns)
    assert np.allclose(np.fromstring(translation.text, sep=" "), [1200, -3400, 5600])
    _, output, triangles = load_geometry(path)
    assert np.allclose(np.sort(output, axis=0), np.sort(points, axis=0), atol=1e-10, rtol=0)
    assert len(triangles) == len(faces)
