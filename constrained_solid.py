"""Constrained gap-only solid completion preserving measured surface facets.

Related architecture: https://users.cs.utah.edu/~ladislav/jacobson13robust/jacobson13robust.html
TetGen preserves facets; graph segmentation labels the volume. Only generated
cap geometry is adjusted to resolve new vertex contacts.
"""

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkIdList
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator, vtkTriangle
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance, vtkPolyDataConnectivityFilter

from imprint_cleanup import clean_generated_mesh
from surface_support_repair import _poly


class SurfaceContactConflict(ValueError):
    """Measured boundary contacts that require a local replacement patch."""

    def __init__(self, points):
        super().__init__("Measured surfaces meet at an invalid point connection.")
        self.points = np.asarray(points, dtype=float)


def complete_volume(points, faces, gpu_preference=None, progress=lambda *args: None):
    import maxflow
    import tetgen

    sp = np.asarray(points, dtype=float)
    sf = np.asarray(faces, dtype=np.int32)
    progress("Building constrained volume", 0, len(sf))
    mesh = tetgen.TetGen(sp, sf, np.arange(1, len(sf) + 1, dtype=np.int32))
    mesh.tetrahedralize(switches="pcQfnz", nomergefacet=True, nomergevertex=True)
    p = mesh.node
    t = mesh.elem
    if not np.array_equal(p[: len(sp)], sp):
        raise ValueError("Volume meshing changed measured vertices.")
    progress("Building constrained volume", len(sf), len(sf))
    local = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    af = t[:, local].reshape(-1, 3)
    parent = np.repeat(np.arange(len(t)), 4)
    unique, first, inv, counts = np.unique(
        np.sort(af, axis=1), axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    assert counts.max() == 2
    owner0 = parent[first]
    owner1 = np.full(len(unique), -1, dtype=int)
    others = np.ones(len(af), dtype=bool)
    others[first] = False
    owner1[inv[others]] = parent[others]
    record = np.dtype([("a", np.int64), ("b", np.int64), ("c", np.int64)])
    keys = np.ascontiguousarray(unique, dtype=np.int64).view(record).ravel()
    sourcekeys = np.ascontiguousarray(np.sort(sf, axis=1), dtype=np.int64).view(record).ravel()
    where = np.searchsorted(keys, sourcekeys)
    assert np.array_equal(unique[where], np.sort(sf, axis=1))
    centers = p[t].mean(1)
    normals = np.cross(sp[sf[:, 1]] - sp[sf[:, 0]], sp[sf[:, 2]] - sp[sf[:, 0]])
    facecenter = sp[sf].mean(1)
    hardin = np.zeros(len(t), dtype=bool)
    hardout = np.zeros(len(t), dtype=bool)
    for owners in (owner0[where], owner1[where]):
        valid = owners >= 0
        ids = owners[valid]
        side = np.einsum("ij,ij->i", centers[ids] - facecenter[valid], normals[valid])
        hardin[ids[side < 0]] = True
        hardout[ids[side > 0]] = True
    progress("Classifying volume", 0, len(centers))
    sampler = None
    if gpu_preference is not None:
        try:
            from depth_adaptive_qem import gpu_backend_supported

            if gpu_backend_supported(gpu_preference):
                from surface_distance_gpu import GPUSurfaceDistance

                sampler = GPUSurfaceDistance(sp, sf, gpu_preference)
        except Exception:  # noqa: BLE001 - optional CUDA backends expose several exception types
            sampler = None
    distance = None
    signed = np.empty(len(centers), dtype=float)
    for begin in range(0, len(centers), 32768):
        end = min(begin + 32768, len(centers))
        if sampler is not None:
            try:
                signed[begin:end] = sampler.evaluate(centers[begin:end])
            except Exception:  # noqa: BLE001 - a lost device must not prevent CPU completion
                sampler = None
        if sampler is None:
            if distance is None:
                distance = vtkImplicitPolyDataDistance()
                distance.SetInput(_poly(sp, sf))
                # Restart classification so every value uses the same distance backend.
                begin = 0
            values = vtkDoubleArray()
            distance.EvaluateFunction(numpy_to_vtk(centers[begin:end], deep=True), values)
            signed[begin:end] = vtk_to_numpy(values)
        progress("Classifying volume", end, len(centers))
    progress("Classifying volume", len(centers), len(centers))
    vol = (
        np.abs(
            np.einsum(
                "ij,ij->i",
                p[t[:, 1]] - p[t[:, 0]],
                np.cross(p[t[:, 2]] - p[t[:, 0]], p[t[:, 3]] - p[t[:, 0]]),
            )
        )
        / 6
    )
    area = 0.5 * np.linalg.norm(
        np.cross(p[unique[:, 1]] - p[unique[:, 0]], p[unique[:, 2]] - p[unique[:, 0]]), axis=1
    )
    scale = area[where].sum() / vol.sum()
    unary = vol * scale
    G = maxflow.Graph[float](len(t), len(unique))
    nodes = G.add_nodes(len(t))
    costout = np.where(signed < 0, unary, 0.0)
    costin = np.where(signed >= 0, unary, 0.0)
    bound = owner1 < 0
    np.add.at(costin, owner0[bound], area[bound])
    infinity = (area.sum() + unary.sum()) * 10
    costout[hardin] = infinity
    costin[hardout] = infinity
    G.add_grid_tedges(nodes, costout, costin)
    interior = owner1 >= 0
    G.add_edges(owner0[interior], owner1[interior], area[interior], area[interior])
    G.maxflow()
    inside = ~G.get_grid_segments(nodes)
    assert inside[hardin].all() and not inside[hardout].any()
    # Enforce a manifold cut in a small neighborhood of ambiguous edges.
    from pysat.card import CardEnc
    from pysat.examples.rc2 import RC2
    from pysat.formula import WCNF, IDPool

    labels1 = np.zeros(len(unique), dtype=bool)
    labels1[interior] = inside[owner1[interior]]
    boundary = inside[owner0] != labels1
    se = np.concatenate(
        [unique[boundary][:, [0, 1]], unique[boundary][:, [1, 2]], unique[boundary][:, [2, 0]]]
    )
    ue, ec = np.unique(np.sort(se, axis=1), axis=0, return_counts=True)
    bad = ue[ec > 2]
    if len(bad):
        seedvertices = np.unique(bad)
        patch = np.any(np.isin(t, seedvertices), axis=1)
        # One adjacent tetrahedral layer gives the cut room to resolve pinches.
        neighbors = np.unique(
            unique[np.isin(owner0, np.flatnonzero(patch)) | np.isin(owner1, np.flatnonzero(patch))]
        )
        patch = np.any(np.isin(t, neighbors), axis=1)
        ids = np.flatnonzero(patch)
        pool = IDPool()
        variables = {int(i): pool.id(("tet", int(i))) for i in ids}
        formula = WCNF()
        for i, var in variables.items():
            if hardin[i]:
                formula.append([var])
            elif hardout[i]:
                formula.append([-var])
            else:
                formula.append([var if inside[i] else -var], weight=max(1, int(vol[i] * 1e6)))
        cf = np.flatnonzero(
            patch[owner0] | np.where(owner1 >= 0, patch[np.maximum(owner1, 0)], False)
        )
        cuts = {}
        for face in cf:
            a, b = int(owner0[face]), int(owner1[face])
            va = variables.get(a)
            vb = variables.get(b)
            cut = pool.id(("face", int(face)))
            cuts[int(face)] = cut
            if va is None and vb is None:
                formula.append([cut if inside[a] != (inside[b] if b >= 0 else False) else -cut])
            elif va is None:
                if inside[a]:
                    formula.extend([[-cut, -vb], [cut, vb]])
                else:
                    formula.extend([[-cut, vb], [cut, -vb]])
            elif vb is None:
                if b >= 0 and inside[b]:
                    formula.extend([[-cut, -va], [cut, va]])
                else:
                    formula.extend([[-cut, va], [cut, -va]])
            else:
                formula.extend([[-va, -vb, -cut], [va, vb, -cut], [va, -vb, cut], [-va, vb, cut]])
        # Every touched edge gets at most two boundary faces. Untouched faces remain fixed.
        all_edges = np.concatenate([unique[:, [0, 1]], unique[:, [1, 2]], unique[:, [2, 0]]])
        edge_face = np.tile(np.arange(len(unique)), 3)
        edge_keys = np.sort(all_edges.astype(np.int64), axis=1)
        packed = edge_keys[:, 0] * len(p) + edge_keys[:, 1]
        order = np.argsort(packed)
        packed = packed[order]
        edge_face = edge_face[order]
        touched = np.unique(
            np.sort(
                np.concatenate(
                    [unique[cf][:, [0, 1]], unique[cf][:, [1, 2]], unique[cf][:, [2, 0]]]
                ),
                axis=1,
            ),
            axis=0,
        )
        for a, b in touched:
            key = int(a) * len(p) + int(b)
            lo, hi = np.searchsorted(packed, [key, key], side="left")
            hi = np.searchsorted(packed, key, side="right")
            incident = edge_face[lo:hi]
            lits = [cuts[int(i)] for i in incident if int(i) in cuts]
            fixed = sum(bool(boundary[i]) for i in incident if int(i) not in cuts)
            bound = 2 - fixed
            if bound < 0:
                raise ValueError("Outside patch already non-manifold")
            if len(lits) > bound:
                formula.extend(CardEnc.atmost(lits=lits, bound=bound, vpool=pool).clauses)
        with RC2(formula) as solver:
            model = solver.compute()
            if model is None:
                raise ValueError(
                    "The measured surface has no valid completion in this volume mesh."
                )
        positive = {i for i in model if i > 0}
        for i, var in variables.items():
            inside[i] = var in positive
    labels1 = np.zeros(len(unique), dtype=bool)
    labels1[interior] = inside[owner1[interior]]
    boundary = inside[owner0] != labels1
    outfaces = af[first[boundary]].copy()
    tri = p[outfaces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    toward0 = np.einsum("ij,ij->i", normal, centers[owner0[boundary]] - tri[:, 0])
    flip = (toward0 > 0) == inside[owner0[boundary]]
    outfaces[flip] = outfaces[flip][:, [0, 2, 1]]
    assert boundary[where].all()
    return _clean_new_cap_contacts(p, outfaces, sf, t, inside, progress)


def _clean_new_cap_contacts(p, f, sf, t, inside, progress):
    row = np.dtype([("a", np.int64), ("b", np.int64), ("c", np.int64)])
    source_keys = np.ascontiguousarray(np.sort(sf, 1), dtype=np.int64).view(row).ravel()
    keys = np.ascontiguousarray(np.sort(f, 1), dtype=np.int64).view(row).ravel()
    known = np.isin(keys, source_keys)
    if known.sum() != len(sf):
        raise ValueError("Gap assembly lost measured faces.")
    vertices = f.ravel()
    order = np.argsort(vertices)
    vs, starts, counts = np.unique(vertices[order], return_index=True, return_counts=True)
    points = p.tolist()
    moved = []
    conflicts = []
    progress("Resolving surface contacts", 0, len(vs))
    for processed, (v, start, count) in enumerate(zip(vs, starts, counts), 1):
        if processed % 512 == 0:
            progress("Resolving surface contacts", processed, len(vs))
        faces = np.unique(order[start : start + count] // 3)
        adj = {int(i): set() for i in faces}
        by_neighbor = {}
        for face in faces:
            for neighbor in f[face]:
                if neighbor != v:
                    by_neighbor.setdefault(int(neighbor), []).append(int(face))
        for faces_at_edge in by_neighbor.values():
            for a in faces_at_edge:
                adj[a].update(faces_at_edge)
        remaining = set(adj)
        fans = []
        while remaining:
            fan = set()
            pending = [next(iter(remaining))]
            while pending:
                a = pending.pop()
                if a not in remaining:
                    continue
                remaining.remove(a)
                fan.add(a)
                pending.extend(adj[a])
            fans.append(np.array(sorted(fan), dtype=int))
        if len(fans) < 2:
            continue
        keep = max(fans, key=lambda ids: (known[ids].sum(), len(ids)))
        for fan in fans:
            if fan is keep:
                continue
            if known[fan].any():
                conflicts.append(p[v])
                continue
            new = len(points)
            points.append(p[v].tolist())
            f[fan] = np.where(f[fan] == v, new, f[fan])
            moved.append((new, fan))
    if conflicts:
        raise SurfaceContactConflict(np.unique(conflicts, axis=0))
    progress("Resolving surface contacts", len(vs), len(vs))
    pp = np.array(points, dtype=float)
    conn = vtkPolyDataConnectivityFilter()
    used, remapped = np.unique(f, return_inverse=True)
    raw = _poly(pp[used], remapped.reshape(-1, 3))
    pointids = numpy_to_vtk(used.astype(np.int64), deep=True)
    pointids.SetName("OriginalPointId")
    raw.GetPointData().AddArray(pointids)
    conn.SetInputData(raw)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    out = conn.GetOutput()
    labels = vtk_to_numpy(
        out.GetPointData().GetArray("RegionId")
    )  # determine retained components by original geometry, not component size
    originalids = vtk_to_numpy(out.GetPointData().GetArray("OriginalPointId"))
    mapped = np.zeros(len(pp), dtype=int)
    mapped[originalids] = labels
    region = mapped[f[:, 0]]
    keepregions = np.unique(region[known])
    use = np.isin(region, keepregions)
    f = f[use]
    known = known[use]
    T = t
    labels = inside
    centers = p[T].mean(1)
    for new, fan in moved:
        incident = np.flatnonzero(np.any(f == new, axis=1))
        if not len(incident):
            continue
        old = pp[new].copy()
        originalid = int(np.flatnonzero(np.all(p == old, axis=1))[0])
        star = np.flatnonzero(np.any(T == originalid, axis=1) & labels)
        material = centers[star] - old
        tri = pp[f[incident]]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(0)
        n /= max(np.linalg.norm(n), 1e-30)
        directions = [material.mean(0), *material, -n * 0.1, n * 0.1]
        valid = False
        locator = vtkStaticCellLocator()
        locator.SetDataSet(_poly(pp, f))
        locator.BuildLocator()
        hits = vtkIdList()
        bounds = np.column_stack(
            (tri.reshape(-1, 3).min(0) - 0.2, tri.reshape(-1, 3).max(0) + 0.2)
        ).ravel()
        locator.FindCellsWithinBounds(bounds, hits)
        near = np.array([hits.GetId(i) for i in range(hits.GetNumberOfIds())])
        neighbors = np.setdiff1d(near, incident)
        neartri = pp[f[neighbors]]
        nearmin = neartri.min(1)
        nearmax = neartri.max(1)
        for fraction in (0.01, 0.05, 0.1, 0.25):
            for direction in directions:
                if np.linalg.norm(direction) < 1e-10:
                    continue
                pp[new] = (old + direction * fraction).astype(np.float32).astype(float)
                if np.array_equal(pp[new], old):
                    continue
                collision = False
                for face in incident:
                    tr = pp[f[face]]
                    candidates = neighbors[
                        np.all(nearmax >= tr.min(0), axis=1)
                        & np.all(nearmin <= tr.max(0), axis=1)
                        & ~np.isin(f[neighbors], f[face]).any(axis=1)
                    ]
                    for other in candidates:
                        ot = pp[f[other]]
                        if np.any(np.all(tr[:, None, :] == ot[None, :, :], axis=2)):
                            continue
                        if (
                            np.all(tr.max(0) >= ot.min(0))
                            and np.all(ot.max(0) >= tr.min(0))
                            and vtkTriangle.TrianglesIntersect(*tr, *ot)
                        ):
                            collision = True
                            break
                    if collision:
                        break
                if not collision:
                    valid = True
                    break
            if valid:
                break
        if not valid:
            raise ValueError("No collision-free new-cap vertex position")
    result, topology = clean_generated_mesh(_poly(pp, f), closed=True)
    output_points = vtk_to_numpy(result.GetPoints().GetData()).astype(np.float32)
    output_faces = (
        vtk_to_numpy(result.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int32)
    )
    invalid_vertices = _nonmanifold_vertex_count(output_faces)
    if invalid_vertices:
        raise ValueError(f"Generated solid has {invalid_vertices} invalid vertex connections.")
    topology["nonmanifold_vertices"] = 0
    return (
        output_points,
        output_faces,
        {
            "original_faces_retained": len(sf),
            "added_faces": len(output_faces) - len(sf),
            "topology": topology,
            "full_volume": True,
            "may_rebuild_existing_surface": False,
        },
    )


def _nonmanifold_vertex_count(faces):
    vertices = faces.ravel()
    links = np.stack((faces[:, [1, 2]], faces[:, [2, 0]], faces[:, [0, 1]]), axis=1).reshape(-1, 2)
    order = np.argsort(vertices)
    _, starts, counts = np.unique(vertices[order], return_index=True, return_counts=True)
    invalid = 0
    for start, count in zip(starts, counts):
        adjacency = {}
        for a, b in links[order[start : start + count]]:
            adjacency.setdefault(int(a), []).append(int(b))
            adjacency.setdefault(int(b), []).append(int(a))
        seen = set()
        pending = [next(iter(adjacency))]
        while pending:
            vertex = pending.pop()
            if vertex in seen:
                continue
            seen.add(vertex)
            pending.extend(adjacency[vertex])
        invalid += len(seen) != len(adjacency) or any(len(link) != 2 for link in adjacency.values())
    return int(invalid)
