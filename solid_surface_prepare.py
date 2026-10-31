"""Local scan-defect preparation for constrained solid completion.

Intersection selection: PyMeshFix 0.18.1 src/_meshfix.cpp.
Only diagnosed intersection neighborhoods and invalid point contacts are opened;
the constrained completion retains all remaining measured triangles.
"""
import numpy as np

from constrained_solid import SurfaceContactConflict, complete_volume


def _row_keys(rows):
    values = np.ascontiguousarray(rows)
    return values.view(np.dtype((np.void, values.dtype.itemsize * values.shape[1]))).ravel()


def _intersecting_source_faces(points, faces, mesh):
    selected = np.asarray(mesh.select_intersecting_triangles())
    if not len(selected):
        return np.zeros(len(faces), dtype=bool)
    # The pinned 0.18.1 binding allocates (count, 3), but writes count face IDs
    # into its flat prefix. The unused tail is not triangle connectivity.
    indices = selected.ravel()[:len(selected)].astype(np.int64)
    mesh_points, mesh_faces = mesh.return_arrays()
    if np.any(indices < 0) or np.any(indices >= len(mesh_faces)):
        raise ValueError("Intersection detection returned invalid face identifiers.")
    _, inverse = np.unique(np.vstack((points, mesh_points)), axis=0, return_inverse=True)
    source = np.sort(inverse[:len(points)][faces], axis=1)
    affected = np.sort(inverse[len(points):][mesh_faces[indices]], axis=1)
    mask = np.isin(_row_keys(source), _row_keys(affected))
    if not mask.any():
        raise ValueError("Intersection detection could not be mapped to the measured surface.")
    return mask


def complete_scan_volume(points, faces, gpu_preference=None, progress=lambda *args: None):
    """Resolve local defects in the normal Solidify operation, without file edits."""
    from pymeshfix import PyTMesh

    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)
    points, inverse = np.unique(points, axis=0, return_inverse=True)
    faces = inverse[faces].astype(np.int32)
    progress("Checking surface intersections", 0, len(faces))
    check = PyTMesh()
    check.set_quiet(True)
    check.load_array(points, faces)
    affected = _intersecting_source_faces(points, faces, check)
    progress("Checking surface intersections", len(faces), len(faces))
    keep = np.ones(len(faces), dtype=bool)
    if affected.any():
        # One vertex-connected ring removes boundary slivers at the crossing.
        keep &= ~np.isin(faces, np.unique(faces[affected])).any(axis=1)
    tri = points[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total_area = float(area.sum())
    while True:
        replaced = int((~keep).sum())
        fraction = float(area[~keep].sum() / max(total_area, 1e-30))
        if not keep.any() or fraction > .01:
            raise ValueError("Scan defects affect more than 1% of the measured surface. Select the intended surface region before solidifying.")
        progress("Repairing local surface defects", replaced, len(faces))
        used, remap = np.unique(faces[keep], return_inverse=True)
        try:
            output_points, output_faces, report = complete_volume(
                points[used], remap.reshape(-1, 3).astype(np.int32), gpu_preference, progress)
        except SurfaceContactConflict as error:
            contact = np.isin(_row_keys(points), _row_keys(error.points))
            revised = keep & ~contact[faces].any(axis=1)
            if np.array_equal(revised, keep):
                raise ValueError("Solidify could not resolve a local surface contact.") from error
            keep = revised
            continue
        report.update(intersecting_input_faces=int(affected.sum()),
                      replaced_input_faces=replaced, replaced_surface_percent=100 * fraction,
                      may_rebuild_existing_surface=bool(replaced))
        return output_points, output_faces, report
