"""Keep reduction output unchanged while observing native work when available."""
import fast_simplification
import numpy as np

try:
    import _meshmill_qem
except ImportError:
    _meshmill_qem = None


def simplify(points, faces, *, progress=None, **options):
    if progress is None or _meshmill_qem is None:
        return fast_simplification.simplify(points, faces, **options)
    target = options.get("target_count")
    if target is None:
        target = int((1.0 - options["target_reduction"]) * len(faces))
    vertices = np.ascontiguousarray(points, dtype=np.float64)
    triangles = np.ascontiguousarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Points must have shape (n, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Faces must have shape (n, 3)")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError("Triangle vertex index is outside the mesh")
    output_points, output_faces = _meshmill_qem.simplify(
        vertices, triangles, int(target), float(options.get("agg", 7)),
        int(options.get("preserve_border", False)), progress,
    )
    return (np.frombuffer(output_points, np.float64).reshape(-1, 3).copy(),
            np.frombuffer(output_faces, np.int32).reshape(-1, 3).copy())
