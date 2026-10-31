"""Geometry-only interchange, with explicit scene transforms and unit conversion."""

import json
import struct
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


def glb_bytes(points, faces):
    points = np.asarray(points, dtype=float)
    # MeshMill uses millimeters/Z-up; glTF uses meters/Y-up.
    positions = np.ascontiguousarray(points[:, [0, 2, 1]] * [1, 1, -1] * 0.001, dtype="<f4")
    indices = np.ascontiguousarray(faces, dtype="<u4")
    binary = positions.tobytes() + indices.tobytes()
    document = {
        "asset": {"version": "2.0", "generator": "MeshMill"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes, "target": 34962},
            {
                "buffer": 0,
                "byteOffset": positions.nbytes,
                "byteLength": indices.nbytes,
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(points),
                "type": "VEC3",
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist(),
            },
            {"bufferView": 1, "componentType": 5125, "count": indices.size, "type": "SCALAR"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\0" * (-len(binary) % 4)
    return (
        struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary))
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def _transform(points, matrix):
    return (np.column_stack((points, np.ones(len(points)))) @ matrix.T)[:, :3]


def _node_matrix(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=float).reshape(4, 4).T
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    ) @ np.diag(node.get("scale", [1, 1, 1]))
    matrix[:3, 3] = node.get("translation", [0, 0, 0])
    return matrix


def read_glb(path, progress):
    data = Path(path).read_bytes()
    if len(data) < 20 or struct.unpack_from("<III", data) != (0x46546C67, 2, len(data)):
        raise ValueError("Invalid GLB 2.0 file.")
    chunks, offset = {}, 12
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        if offset + 8 + length > len(data):
            raise ValueError("Truncated GLB chunk.")
        chunks[kind] = data[offset + 8 : offset + 8 + length]
        offset += 8 + length
    doc = json.loads(chunks[0x4E4F534A])
    binary = chunks.get(0x004E4942, b"")
    if any(b.get("uri") for b in doc.get("buffers", [])):
        raise ValueError("External GLB buffers are not supported. Export a self-contained GLB.")

    def accessor(index):
        a = doc["accessors"][index]
        if "sparse" in a or "bufferView" not in a:
            raise ValueError("Sparse GLB geometry is not supported. Export dense accessors.")
        view = doc["bufferViews"][a["bufferView"]]
        if view.get("extensions") or view.get("buffer", 0) != 0:
            raise ValueError(
                "Compressed or external GLB geometry is not supported. Export uncompressed GLB."
            )
        dtype = np.dtype({5121: "<u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]])
        width = {"SCALAR": 1, "VEC3": 3}[a["type"]]
        stride = view.get("byteStride", dtype.itemsize * width)
        start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        end = start + max(a["count"] - 1, 0) * stride + dtype.itemsize * width
        if (
            stride < dtype.itemsize * width
            or start < 0
            or end > len(binary)
            or end > view.get("byteOffset", 0) + view["byteLength"]
        ):
            raise ValueError("GLB accessor exceeds its geometry buffer.")
        return np.ndarray(
            (a["count"], width),
            dtype=dtype,
            buffer=binary,
            offset=start,
            strides=(stride, dtype.itemsize),
        ).copy()

    output = []

    def visit(index, transform, ancestors):
        if index in ancestors:
            raise ValueError("Cyclic GLB scene graph.")
        node = doc["nodes"][index]
        transform = transform @ _node_matrix(node)
        if "skin" in node:
            raise ValueError("Skinned GLB meshes require a static mesh export.")
        if "mesh" in node:
            for primitive in doc["meshes"][node["mesh"]]["primitives"]:
                if primitive.get("extensions") or primitive.get("targets"):
                    raise ValueError("Export a static, uncompressed GLB mesh.")
                if primitive.get("mode", 4) != 4:
                    raise ValueError("GLB mesh primitives must be triangle lists.")
                a = doc["accessors"][primitive["attributes"]["POSITION"]]
                if a["componentType"] != 5126:
                    raise ValueError(
                        "Quantized GLB positions are not supported. Export float positions."
                    )
                points = _transform(accessor(primitive["attributes"]["POSITION"]), transform)
                faces = (
                    accessor(primitive["indices"]).ravel()
                    if "indices" in primitive
                    else np.arange(len(points))
                ).reshape(-1, 3)
                if np.linalg.det(transform[:3, :3]) < 0:
                    faces = faces[:, [0, 2, 1]]
                points = points[:, [0, 2, 1]] * [1, -1, 1] * 1000
                output.append((points, faces))
        for child in node.get("children", []):
            visit(child, transform, ancestors | {index})
        progress("Reading mesh", len(output), max(len(doc.get("meshes", [])), len(output), 1))

    if doc.get("scenes"):
        roots = doc["scenes"][doc.get("scene", 0)].get("nodes", [])
    else:
        children = {child for n in doc.get("nodes", []) for child in n.get("children", [])}
        roots = [i for i in range(len(doc.get("nodes", []))) if i not in children]
    for root in roots:
        visit(root, np.eye(4), set())
    return output


def read_dae(path, progress):
    raw = Path(path).read_bytes()
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("COLLADA entity declarations are not supported.")
    root = ET.fromstring(raw)
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    geometries = {}
    for geometry in root.findall("c:library_geometries/c:geometry", ns):
        mesh = geometry.find("c:mesh", ns)
        if mesh is None:
            continue
        sources = {}
        for source in mesh.findall("c:source", ns):
            array = source.find("c:float_array", ns)
            access = source.find("c:technique_common/c:accessor", ns)
            if array is not None and access is not None:
                values = np.fromstring(array.text or "", sep=" ")
                stride = int(access.get("stride", "1"))
                start, count = int(access.get("offset", "0")), int(access.get("count"))
                sources["#" + source.get("id")] = values[start : start + count * stride].reshape(
                    count, stride
                )[:, :3]
        vertices = {
            "#" + v.get("id"): v.find("c:input[@semantic='POSITION']", ns).get("source")
            for v in mesh.findall("c:vertices", ns)
        }
        pieces = []
        for primitive in mesh:
            kind = primitive.tag.rsplit("}", 1)[-1]
            if kind not in ("triangles", "polylist", "polygons"):
                if kind in ("tristrips", "trifans"):
                    raise ValueError("Export COLLADA with triangles or polygon faces.")
                continue
            inputs = primitive.findall("c:input", ns)
            position = next(i for i in inputs if i.get("semantic") in ("VERTEX", "POSITION"))
            source = position.get("source")
            points = sources[vertices.get(source, source)]
            stride = max(int(i.get("offset", "0")) for i in inputs) + 1
            offset = int(position.get("offset", "0"))
            if primitive.find("c:ph", ns) is not None:
                raise ValueError("COLLADA polygons with holes require triangulation before import.")
            lists = [
                np.fromstring(p.text or "", sep=" ", dtype=np.int64).reshape(-1, stride)[:, offset]
                for p in primitive.findall("c:p", ns)
            ]
            if kind == "polygons":
                polygons = lists
            else:
                indices = np.concatenate(lists) if lists else np.array([], dtype=np.int64)
                sizes = (
                    [3] * int(primitive.get("count", "0"))
                    if kind == "triangles"
                    else np.fromstring(
                        primitive.find("c:vcount", ns).text or "", sep=" ", dtype=int
                    )
                )
                if sum(sizes) != len(indices):
                    raise ValueError("COLLADA polygon indices do not match face sizes.")
                polygons = np.split(indices, np.cumsum(sizes)[:-1]) if len(sizes) else []
            pieces.append((points, polygons))
        geometries["#" + geometry.get("id")] = pieces
    output = []

    nodes = {"#" + node.get("id"): node for node in root.iter("{" + ns["c"] + "}node") if node.get("id")}

    def visit(node, transform, ancestors=frozenset()):
        identity = id(node)
        if identity in ancestors:
            raise ValueError("COLLADA node references contain a cycle.")
        ancestors = ancestors | {identity}
        local = np.eye(4)
        for item in node:
            name = item.tag.rsplit("}", 1)[-1]
            if name not in ("matrix", "translate", "rotate", "scale"):
                continue
            values = np.fromstring(item.text or "", sep=" ")
            operation = np.eye(4)
            if name == "matrix":
                operation = values.reshape(4, 4).T
            elif name == "translate":
                operation[:3, 3] = values
            elif name == "scale":
                operation[:3, :3] = np.diag(values)
            else:
                axis = values[:3] / np.linalg.norm(values[:3])
                angle = np.deg2rad(values[3])
                x, y, z = axis
                cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                operation[:3, :3] = (
                    np.eye(3) * np.cos(angle)
                    + (1 - np.cos(angle)) * np.outer(axis, axis)
                    + np.sin(angle) * cross
                )
            local = local @ operation
        transform = transform @ local
        for instance in node.findall("c:instance_geometry", ns):
            url = instance.get("url", "")
            if not url.startswith("#"):
                raise ValueError("External COLLADA geometry is not supported.")
            for points, polygons in geometries[url]:
                if np.linalg.det(transform[:3, :3]) < 0:
                    polygons = [p[::-1] for p in polygons]
                output.append((_transform(points, transform), polygons))
        if node.find("c:instance_controller", ns) is not None:
            raise ValueError("Export a static COLLADA scene with geometry nodes.")
        for instance in node.findall("c:instance_node", ns):
            url = instance.get("url", "")
            if not url.startswith("#") or url not in nodes:
                raise ValueError("COLLADA node reference is external or missing.")
            visit(nodes[url], transform, ancestors)
        for child in node.findall("c:node", ns):
            visit(child, transform, ancestors)

    scene_ref = root.find("c:scene/c:instance_visual_scene", ns)
    if scene_ref is None:
        output = [piece for pieces in geometries.values() for piece in pieces]
    else:
        scene = next(
            (
                s
                for s in root.findall("c:library_visual_scenes/c:visual_scene", ns)
                if "#" + s.get("id") == scene_ref.get("url")
            ),
            None,
        )
        if scene is None:
            raise ValueError("COLLADA scene is missing.")
        for node in scene.findall("c:node", ns):
            visit(node, np.eye(4))
    unit = root.find("c:asset/c:unit", ns)
    scale = float(unit.get("meter", "1")) * 1000 if unit is not None else 1000
    up = root.findtext("c:asset/c:up_axis", "Y_UP", ns)
    converted = []
    for points, polygons in output:
        points = points * scale
        if up == "Y_UP":
            points = points[:, [0, 2, 1]] * [1, -1, 1]
        elif up == "X_UP":
            points = points[:, [2, 1, 0]] * [-1, 1, 1]
        converted.append((points, polygons))
    progress("Reading mesh", len(raw), len(raw))
    return converted


def read_ply(path, progress):
    file_size = Path(path).stat().st_size
    types = dict(
        zip(
            (
                "char",
                "uchar",
                "short",
                "ushort",
                "int",
                "uint",
                "float",
                "double",
                "int8",
                "uint8",
                "int16",
                "uint16",
                "int32",
                "uint32",
                "float32",
                "float64",
            ),
            ("i1", "u1", "i2", "u2", "i4", "u4", "f4", "f8") * 2,
        )
    )
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Invalid PLY header.")
        elements = []
        encoding = None
        while stream.tell() < 1024 * 1024:
            line = stream.readline().decode("ascii").strip().split()
            if not line:
                raise ValueError("Incomplete PLY header.")
            if line[0] == "end_header":
                break
            if line[0] == "format":
                encoding = line[1]
            elif line[0] == "element":
                elements.append((line[1], int(line[2]), []))
            elif line[0] == "property":
                elements[-1][2].append(
                    (line[4], line[2], line[3]) if line[1] == "list" else (line[2], None, line[1])
                )
        else:
            raise ValueError("PLY header is too large.")
        if encoding not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise ValueError("Unsupported PLY encoding.")
        endian = ">" if encoding == "binary_big_endian" else "<"
        points = None
        polygons = []
        for name, count, properties in elements:
            if count < 0:
                raise ValueError("Invalid PLY element count.")
            if name == "vertex" and not set("xyz").issubset(p[0] for p in properties):
                raise ValueError("PLY vertices require x, y and z coordinates.")
            if name == "vertex" and encoding != "ascii" and all(p[1] is None for p in properties):
                dtype = np.dtype([(p[0], endian + types[p[2]]) for p in properties])
                values = np.fromfile(stream, dtype=dtype, count=count)
                if len(values) != count:
                    raise ValueError("Truncated PLY vertices.")
                points = np.column_stack([values[axis] for axis in "xyz"]).astype(float)
                progress("Reading mesh", stream.tell(), file_size)
                continue
            vertex_rows = []
            for row in range(count):
                tokens = iter(stream.readline().split()) if encoding == "ascii" else None
                values = {}
                for prop, list_type, value_type in properties:
                    if list_type is None:
                        if tokens is not None:
                            value = float(next(tokens))
                        else:
                            dtype = np.dtype(endian + types[value_type])
                            value = np.frombuffer(stream.read(dtype.itemsize), dtype=dtype)[0]
                    else:
                        if tokens is not None:
                            length = int(next(tokens))
                            value = np.array(
                                [float(next(tokens)) for _ in range(length)],
                                dtype=np.dtype(types[value_type]),
                            )
                        else:
                            dtype = np.dtype(endian + types[list_type])
                            length = int(np.frombuffer(stream.read(dtype.itemsize), dtype=dtype)[0])
                            item_type = np.dtype(endian + types[value_type])
                            if (
                                length < 0
                                or length * item_type.itemsize > file_size - stream.tell()
                            ):
                                raise ValueError("Invalid PLY face length.")
                            value = np.frombuffer(
                                stream.read(length * item_type.itemsize), dtype=item_type
                            ).copy()
                    values[prop] = value
                if name == "vertex":
                    vertex_rows.append([values[axis] for axis in "xyz"])
                elif name == "face":
                    polygon = values.get("vertex_indices", values.get("vertex_index"))
                    if polygon is None or len(polygon) < 3:
                        raise ValueError("PLY face does not contain a polygon.")
                    polygons.append(np.asarray(polygon, dtype=np.int64))
                if row % 4096 == 0:
                    progress("Reading mesh", stream.tell(), file_size)
            if name == "vertex":
                points = np.asarray(vertex_rows, dtype=float)
        if points is None:
            raise ValueError("PLY has no vertex positions.")
        return [(points, polygons)]


def load_geometry(path, progress=None):
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkAppendPolyData, vtkCleanPolyData, vtkTriangleFilter
    from vtkmodules.vtkIOGeometry import vtkOBJReader

    notify = progress or (lambda *_: None)
    suffix = Path(path).suffix.lower()
    notify("Reading mesh", 0, Path(path).stat().st_size)
    if suffix == ".obj":
        reader = vtkOBJReader()
        reader.SetFileName(str(path))
        size = Path(path).stat().st_size
        reader.AddObserver(
            "ProgressEvent",
            lambda source, _event: notify("Reading mesh", round(source.GetProgress() * size), size),
        )
        reader.Update()
        pieces = [reader.GetOutput()]
    else:
        raw = (
            read_ply(path, notify)
            if suffix == ".ply"
            else (
                read_glb(path, notify)
                if suffix == ".glb"
                else read_dae(path, notify)
                if suffix == ".dae"
                else None
            )
        )
        if raw is None:
            raise ValueError("Supported mesh formats: STL, COLLADA, GLB, OBJ and PLY.")
        pieces = []
        for points, polygons in raw:
            points = np.asarray(points, dtype=np.float64)
            if not np.isfinite(points).all():
                raise ValueError("Geometry contains non-finite coordinates.")
            if not len(polygons):
                continue
            indices = np.concatenate(polygons).astype(np.int64)
            if indices.min() < 0 or indices.max() >= len(points):
                raise ValueError("Face indices exceed the vertex array.")
            poly = vtkPolyData()
            vertices = vtkPoints()
            vertices.SetData(numpy_to_vtk(points, deep=True))
            poly.SetPoints(vertices)
            cells = vtkCellArray()
            offsets = np.r_[0, np.cumsum([len(p) for p in polygons])].astype(np.int64)
            cells.SetData(
                numpy_to_vtkIdTypeArray(offsets, deep=True),
                numpy_to_vtkIdTypeArray(indices, deep=True),
            )
            poly.SetPolys(cells)
            pieces.append(poly)
    append = vtkAppendPolyData()
    for poly in pieces:
        geometry = vtkPolyData()
        geometry.SetPoints(poly.GetPoints())
        geometry.SetPolys(poly.GetPolys())
        append.AddInputData(geometry)
    clean = vtkCleanPolyData()
    clean.SetInputConnection(append.GetOutputPort())
    clean.SetTolerance(0)
    triangle = vtkTriangleFilter()
    triangle.SetInputConnection(clean.GetOutputPort())
    triangle.PassLinesOff()
    triangle.PassVertsOff()
    triangle.AddObserver(
        "ProgressEvent",
        lambda source, _event: notify(
            "Preparing triangles", round(source.GetProgress() * 1000), 1000
        ),
    )
    triangle.Update()
    poly = vtkPolyData()
    poly.DeepCopy(triangle.GetOutput())
    if not poly.GetNumberOfPolys():
        raise ValueError(
            "No polygon surfaces were found. Point clouds require reconstruction before opening."
        )
    points = vtk_to_numpy(poly.GetPoints().GetData()).copy()
    faces = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3).copy()
    if not np.isfinite(points).all():
        raise ValueError("Geometry contains non-finite coordinates.")
    return poly, points, faces
