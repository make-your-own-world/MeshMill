"""Indexed polygon exports with conservative coplanar face consolidation."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

FORMATS = {".stl": "STL mesh (*.stl)", ".dae": "SketchUp / COLLADA (*.dae)",
           ".glb": "Binary glTF (*.glb)", ".obj": "Wavefront OBJ (*.obj)", ".ply": "Polygon PLY (*.ply)"}


def rigid_export_matrix(matrix):
    value = np.asarray(matrix, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], (0, 0, 0, 1))
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-8)):
        raise ValueError("Calibration requires a finite, rigid, right-handed transform.")
    return value.copy()


def calibrated_export_mesh(poly, matrix):
    """Transform an isolated export copy, including normals and closed markers."""
    if matrix is None:
        return poly
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersGeneral import vtkTransformFilter
    value = rigid_export_matrix(matrix)
    vtk_matrix = vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            vtk_matrix.SetElement(row, column, value[row, column])
    transform = vtkTransform()
    transform.SetMatrix(vtk_matrix)
    result = vtkTransformFilter()
    result.SetInputData(poly)
    result.SetTransform(transform)
    result.SetOutputPointsPrecision(1)
    result.Update()
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    output = vtkPolyData()
    output.DeepCopy(result.GetOutput())
    return output


def collada_face_groups(polygons):
    """Give edge-adjacent faces distinct material identities, with equal colors.

    SketchUp's import-time coplanar merging is less conservative than our
    plane checks and can erase valid shallow surfaces. Material boundaries
    preserve the exported polygons without introducing separate mesh parts.
    """
    owners = {}
    neighbors = [set() for _ in polygons]
    for index, polygon in enumerate(polygons):
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            edge = (min(a, b), max(a, b))
            for other in owners.setdefault(edge, []):
                neighbors[index].add(other)
                neighbors[other].add(index)
            owners[edge].append(index)
    colors = []
    for adjacent in neighbors:
        used = {colors[j] for j in adjacent if j < len(colors)}
        color = 0
        while color in used:
            color += 1
        colors.append(color)
    groups = [[] for _ in range(max(colors, default=-1) + 1)]
    for polygon, color in zip(polygons, colors):
        groups[color].append(polygon)
    return groups


def coplanar_polygons(points, faces, progress=None, max_vertices=None):
    """Merge manifold, consistently wound planar disks; retain holes as triangles.

    Every group is checked against one plane and its original area. Boundary
    vertices stay in place, including collinear vertices shared with other faces.
    """
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    count = len(faces)
    if not count:
        return [], {"triangles": 0, "polygons": 0, "removed_edges": 0}
    notify = progress or (lambda *_: None)
    notify("Finding planar faces", 0, count)
    tri = points[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(lengths[:, None], np.finfo(float).tiny)
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    tolerance = max(diagonal * 1e-9, np.max(np.abs(points)) * np.finfo(float).eps * 64, 1e-12)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    owners = np.tile(np.arange(count), 3)
    ordered = np.sort(edges, axis=1)
    order = np.lexsort((ordered[:, 1], ordered[:, 0]))
    ordered, owners = ordered[order], owners[order]
    starts = np.r_[0, np.flatnonzero(np.any(ordered[1:] != ordered[:-1], axis=1)) + 1]
    sizes = np.diff(np.r_[starts, len(ordered)])
    pairs = starts[sizes == 2]
    a, b = owners[pairs], owners[pairs + 1]
    aligned = np.einsum('ij,ij->i', normals[a], normals[b]) >= 1 - 1e-12
    residual = np.max(np.abs(np.einsum('ijk,ik->ij', tri[b] - tri[a, :1], normals[a])), axis=1)
    pairs = np.column_stack((a[aligned & (residual <= tolerance)], b[aligned & (residual <= tolerance)]))
    parent = np.arange(count)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in pairs:
        a, b = root(a), root(b)
        if a != b:
            parent[b] = a
    groups = {}
    for i in range(count):
        groups.setdefault(root(i), []).append(i)
    notify("Merging planar faces", 0, count)
    polygons, processed = [], 0
    for group in groups.values():
        group_faces = faces[group]
        merged = None
        if len(group) > 1:
            ids = np.unique(group_faces)
            normal = normals[group[0]]
            planar = np.max(np.abs((points[ids] - tri[group[0], 0]) @ normal)) <= tolerance
            if planar:
                directed = np.concatenate((group_faces[:, [0, 1]], group_faces[:, [1, 2]], group_faces[:, [2, 0]]))
                _, inverse, edge_counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
                boundary = directed[edge_counts[inverse] == 1]
                outgoing = {int(a): int(b) for a, b in boundary}
                valid = (len(boundary) >= 3 and len(outgoing) == len(boundary)
                         and len(set(outgoing.values())) == len(boundary)
                         and set(outgoing) == set(outgoing.values())
                         and np.all(edge_counts <= 2))
                if valid:
                    loop = [int(boundary[0, 0])]
                    while outgoing[loop[-1]] != loop[0] and len(loop) <= len(boundary):
                        loop.append(outgoing[loop[-1]])
                    if len(loop) == len(boundary) and (max_vertices is None or len(loop) <= max_vertices):
                        v = points[loop] - points[loop[0]]
                        area = np.dot(np.cross(v, np.roll(v, -1, axis=0)).sum(axis=0), normal) / 2
                        expected = lengths[group].sum() / 2
                        if abs(area - expected) <= max(expected * 1e-10, tolerance * diagonal):
                            merged = loop
        polygons.extend([merged] if merged is not None else group_faces.tolist())
        processed += len(group)
        if processed % 1024 < len(group) or processed == count:
            notify("Merging planar faces", processed, count)
    polygon_edges = np.concatenate([np.column_stack((p, np.roll(p, -1))) for p in polygons])
    remaining_edges = len(np.unique(np.sort(polygon_edges, axis=1), axis=0))
    return polygons, {"triangles": count, "polygons": len(polygons),
                      "removed_edges": len(starts) - remaining_edges, "plane_tolerance": tolerance}


def write_polygon_mesh(points, faces, path, *, merge=True, unit_meter=.001, progress=None):
    """Write a standalone, metadata-free OBJ, binary PLY or COLLADA file atomically."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in ('.obj', '.ply', '.dae', '.glb'):
        raise ValueError('Choose STL, COLLADA (.dae), GLB, OBJ or PLY.')
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if (not np.isfinite(points).all() or faces.ndim != 2 or faces.shape[1] != 3
            or not len(faces) or faces.min() < 0 or faces.max() >= len(points)):
        raise ValueError('Mesh contains invalid coordinates or face indices.')
    notify = progress or (lambda *_: None)
    if merge and suffix != '.glb':
        polygons, report = coplanar_polygons(points, faces, notify, max_vertices=255 if suffix == '.ply' else None)
    else:
        polygons = faces.tolist()
        report = {"triangles": len(faces), "polygons": len(faces), "removed_edges": 0}
    used = np.unique(np.concatenate(polygons))
    mapping = np.full(len(points), -1, dtype=np.int64)
    mapping[used] = np.arange(len(used))
    points = points[used]
    polygons = [mapping[p].tolist() for p in polygons]
    fd, temporary = tempfile.mkstemp(prefix='.meshmill-export-', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        notify('Writing geometry', 0, len(polygons))
        if suffix == '.glb':
            from mesh_formats import glb_bytes
            Path(temporary).write_bytes(glb_bytes(points, np.asarray(polygons)))
        elif suffix == '.obj':
            with open(temporary, 'w', encoding='utf-8', newline='\n') as stream:
                stream.write('# MeshMill\n')
                np.savetxt(stream, points, fmt='v %.17g %.17g %.17g')
                for i, polygon in enumerate(polygons):
                    stream.write('f ' + ' '.join(str(v + 1) for v in polygon) + '\n')
                    if i % 4096 == 0:
                        notify('Writing geometry', i, len(polygons))
        elif suffix == '.ply':
            import struct
            with open(temporary, 'wb') as stream:
                header = (f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n'
                          'property double x\nproperty double y\nproperty double z\n'
                          f'element face {len(polygons)}\nproperty list uchar int vertex_indices\nend_header\n')
                stream.write(header.encode('ascii'))
                stream.write(points.astype('<f8').tobytes())
                for i, polygon in enumerate(polygons):
                    stream.write(struct.pack('<B', len(polygon)))
                    stream.write(np.asarray(polygon, dtype='<i4').tobytes())
                    if i % 4096 == 0:
                        notify('Writing geometry', i, len(polygons))
        else:
            ns = 'http://www.collada.org/2005/11/COLLADASchema'
            ET.register_namespace('', ns)
            def element(parent, tag, text=None, **attrs):
                node = ET.SubElement(parent, '{'+ns+'}'+tag, attrs)
                node.text = text
                return node
            root = ET.Element('{'+ns+'}COLLADA', version='1.4.1')
            asset = element(root, 'asset')
            # Required schema fields use a fixed value, not local activity metadata.
            element(asset, 'created', '2000-01-01T00:00:00Z')
            element(asset, 'modified', '2000-01-01T00:00:00Z')
            element(asset, 'unit', name='model-unit', meter=format(unit_meter, '.17g'))
            element(asset, 'up_axis', 'Z_UP')
            groups = collada_face_groups(polygons)
            effects = element(root, 'library_effects')
            materials = element(root, 'library_materials')
            for i in range(len(groups)):
                effect = element(effects, 'effect', id=f'effect-{i}')
                profile = element(effect, 'profile_COMMON')
                technique = element(profile, 'technique', sid='common')
                diffuse = element(element(technique, 'lambert'), 'diffuse')
                element(diffuse, 'color', '0.8 0.8 0.8 1')
                material = element(materials, 'material', id=f'material-{i}', name=f'Surface {i + 1}')
                element(material, 'instance_effect', url=f'#effect-{i}')
            library = element(root, 'library_geometries')
            geometry = element(library, 'geometry', id='mesh', name='Mesh')
            mesh = element(geometry, 'mesh')
            source = element(mesh, 'source', id='positions')
            # Build tiny faces at a safe local size. An instanced definition
            # retains this local scale in SketchUp; a direct node scale is
            # baked before face creation and does not prevent missing faces.
            definition_scale = 100.0
            definition_origin = (points.min(axis=0) + points.max(axis=0)) * 0.5
            local_points = (points - definition_origin) * definition_scale
            element(source, 'float_array', ' '.join(format(x, '.17g') for x in local_points.flat),
                    id='positions-array', count=str(points.size))
            common = element(source, 'technique_common')
            accessor = element(common, 'accessor', source='#positions-array', count=str(len(points)), stride='3')
            for axis in 'XYZ':
                element(accessor, 'param', name=axis, type='float')
            vertices = element(mesh, 'vertices', id='vertices')
            element(vertices, 'input', semantic='POSITION', source='#positions')
            for i, group in enumerate(groups):
                polylist = element(mesh, 'polylist', count=str(len(group)), material=f'surface-{i}')
                element(polylist, 'input', semantic='VERTEX', source='#vertices', offset='0')
                element(polylist, 'vcount', ' '.join(str(len(p)) for p in group))
                element(polylist, 'p', ' '.join(str(v) for p in group for v in p))
            definitions = element(root, 'library_nodes')
            definition = element(definitions, 'node', id='mesh-definition', name='Mesh', type='NODE')
            instance = element(definition, 'instance_geometry', url='#mesh')
            bindings = element(element(instance, 'bind_material'), 'technique_common')
            for i in range(len(groups)):
                element(bindings, 'instance_material', symbol=f'surface-{i}', target=f'#material-{i}')
            scenes = element(root, 'library_visual_scenes')
            scene = element(scenes, 'visual_scene', id='scene', name='Scene')
            node = element(scene, 'node', id='object', name='Mesh', type='NODE')
            element(node, 'translate', ' '.join(format(x, '.17g') for x in definition_origin))
            element(node, 'scale', '0.01 0.01 0.01')
            element(node, 'instance_node', url='#mesh-definition')
            element(element(root, 'scene'), 'instance_visual_scene', url='#scene')
            with open(temporary, 'wb') as stream:
                ET.ElementTree(root).write(stream, encoding='utf-8', xml_declaration=True)
        notify('Writing geometry', len(polygons), len(polygons))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return report
