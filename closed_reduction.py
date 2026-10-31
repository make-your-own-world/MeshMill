"""Preserve closed-solid topology through reduction and STL serialization."""

import numpy as np


def is_closed_mesh(points, faces):
    if not len(faces):
        return False
    _, inverse = np.unique(np.asarray(points, dtype=np.float32), axis=0, return_inverse=True)
    welded = inverse[faces]
    if np.any(
        (welded[:, 0] == welded[:, 1])
        | (welded[:, 1] == welded[:, 2])
        | (welded[:, 2] == welded[:, 0])
    ):
        return False
    edges = np.concatenate((welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]))
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return bool(len(counts) and np.all(counts == 2))


def validate_closed_reduction(points, faces, progress=None):
    """Repair local reduction defects, then validate at binary STL precision.

    This is for a reduced closed solid, not completion of an open source scan.
    No component removal, global remeshing, smoothing or resampling is requested.
    """
    from pymeshfix import PyTMesh
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

    from constrained_solid import _nonmanifold_vertex_count
    from imprint_cleanup import clean_generated_mesh

    def report(stage, done, total):
        if progress is not None:
            progress(stage, done, total)

    # Validate the same coordinates the STL writer stores, not higher-precision
    # coordinates which may merge only after the file is written.
    points, inverse = np.unique(np.asarray(points, dtype=np.float32), axis=0, return_inverse=True)
    faces = inverse[faces].astype(np.int32)
    mesh = PyTMesh()
    mesh.set_quiet(True)
    mesh.load_array(points.astype(float), faces)
    report("Validating reduced solid", 0, 3)
    if mesh.n_boundaries:
        report("Repairing reduction seams", 0, 4)
        mesh.fill_small_boundaries(refine=False)
    intersections = len(mesh.select_intersecting_triangles())
    report("Validating reduced solid", 1, 3)
    for iteration in range(4):
        if not mesh.n_boundaries and not intersections:
            break
        report("Repairing reduction seams", iteration, 4)
        mesh.clean(max_iters=1, inner_loops=3)
        intersections = len(mesh.select_intersecting_triangles())
    if mesh.n_boundaries or intersections:
        raise ValueError(
            "The reduced solid still has open borders or intersecting faces. Use a higher triangle target."
        )
    p, f = mesh.return_arrays()
    vertices = vtkPoints()
    vertices.SetData(numpy_to_vtk(p, deep=True))
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, f.size + 1, 3, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.asarray(f, dtype=np.int64).ravel(), deep=True),
    )
    poly = vtkPolyData()
    poly.SetPoints(vertices)
    poly.SetPolys(cells)
    clean, _ = clean_generated_mesh(poly, closed=True)
    p = vtk_to_numpy(clean.GetPoints().GetData()).copy()
    f = vtk_to_numpy(clean.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int32)
    # Float32 welding can introduce a contact even when double precision passed.
    check = PyTMesh()
    check.set_quiet(True)
    check.load_array(p.astype(float), f)
    report("Validating reduced solid", 2, 3)
    if (
        check.n_boundaries
        or len(check.select_intersecting_triangles())
        or _nonmanifold_vertex_count(f)
    ):
        raise ValueError("The reduced solid did not pass final STL-precision validation.")
    report("Validating reduced solid", 3, 3)
    return p, f
