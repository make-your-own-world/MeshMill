"""Measured open boundaries and surface contributions for Solidify."""
import numpy as np


def surface_area(points, faces):
    area = 0.0
    for start in range(0, len(faces), 100_000):
        triangles = np.asarray(points[faces[start:start + 100_000]], dtype=float)
        area += float(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                             triangles[:, 2] - triangles[:, 0]), axis=1).sum() * .5)
    return area


def input_metrics(points, faces):
    if not len(faces):
        return {"open_edges": 0, "open_rims": 0, "closed_loops": 0, "rim_length": 0.0,
                "nonmanifold_edges": 0, "surface_area": 0.0}
    used, indices = np.unique(faces, return_inverse=True)
    points, weld = np.unique(np.asarray(points[used], dtype=np.float32), axis=0, return_inverse=True)
    faces = weld[indices.reshape(-1, 3)]
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    boundary = edges[counts == 1]
    parent = {}
    def root(vertex):
        parent.setdefault(vertex, vertex)
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex
    degree = {}
    for first, second in boundary:
        first, second = int(first), int(second)
        a, b = root(first), root(second)
        parent[a] = b
        degree[first] = degree.get(first, 0) + 1
        degree[second] = degree.get(second, 0) + 1
    groups = {}
    for vertex, count in degree.items():
        key = root(vertex)
        groups[key] = groups.get(key, True) and count == 2
    length = float(np.linalg.norm(points[boundary[:, 0]].astype(float)
                                 - points[boundary[:, 1]], axis=1).sum()) if len(boundary) else 0.0
    return {"open_edges": len(boundary), "open_rims": len(groups),
            "closed_loops": sum(groups.values()), "rim_length": length,
            "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
            "surface_area": surface_area(points, faces)}


def completion_metrics(source_points, source_faces, result_points, result_faces):
    _, inverse = np.unique(np.vstack((np.asarray(source_points, dtype=np.float32),
                                     np.asarray(result_points, dtype=np.float32))), axis=0, return_inverse=True)
    original = np.ascontiguousarray(np.sort(inverse[:len(source_points)][source_faces], axis=1))
    result = np.ascontiguousarray(np.sort(inverse[len(source_points):][result_faces], axis=1))
    row_type = np.dtype((np.void, original.dtype.itemsize * 3))
    old_keys, new_keys = original.view(row_type).ravel(), result.view(row_type).ravel()
    added = ~np.isin(new_keys, old_keys)
    retained = np.isin(old_keys, new_keys)
    kept = int(np.count_nonzero(retained))
    source_area = surface_area(source_points, source_faces)
    retained_area = surface_area(source_points, source_faces[retained])
    area = surface_area(result_points, result_faces)
    added_area = surface_area(result_points, result_faces[added])
    return {"source_faces": len(source_faces), "retained_faces": kept,
            "retained_percent": 100.0 * retained_area / source_area if source_area else 0.0,
            "added_faces": int(added.sum()), "added_area": added_area,
            "rebuilt_percent": 100.0 * added_area / area if area else 0.0,
            "open_edges": input_metrics(result_points, result_faces)["open_edges"]}
