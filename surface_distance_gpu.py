"""Optional GPU angle-weighted pseudonormal distance queries.

Reference: https://doi.org/10.1109/TVCG.2005.49
Warp supplies closest-point traversal under Apache-2.0.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import warp as wp

from depth_adaptive_qem import select_cuda_device

wp.config.quiet = True
wp.config.kernel_cache_dir = str(Path(tempfile.gettempdir()) / "meshmill-volume-kernels")


@wp.kernel
def query_normals(
    mesh: wp.uint64,
    queries: wp.array(dtype=wp.vec3),
    ids: wp.array(dtype=int),
    fn: wp.array(dtype=wp.vec3),
    vn: wp.array(dtype=wp.vec3),
    en: wp.array(dtype=wp.vec3),
    limit: float,
    out: wp.array(dtype=float),
):
    i = wp.tid()
    q = queries[i]
    hit = wp.mesh_query_point_no_sign(mesh, q, limit)
    if hit.result:
        closest = wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)
        normal = fn[hit.face]
        u = hit.u
        v = hit.v
        w = 1.0 - u - v
        eps = 1.0e-6
        if u > 1.0 - eps:
            normal = vn[ids[hit.face * 3]]
        elif v > 1.0 - eps:
            normal = vn[ids[hit.face * 3 + 1]]
        elif w > 1.0 - eps:
            normal = vn[ids[hit.face * 3 + 2]]
        elif u < eps:
            normal = en[hit.face * 3]
        elif v < eps:
            normal = en[hit.face * 3 + 1]
        elif w < eps:
            normal = en[hit.face * 3 + 2]
        sign = 1.0
        if wp.dot(q - closest, normal) < 0.0:
            sign = -1.0
        out[i] = sign * wp.length(q - closest)
    else:
        out[i] = limit


class GPUSurfaceDistance:
    def __init__(self, p, f, preference="system"):
        self.center = p.mean(0)
        p = p - self.center
        self.limit = float(np.linalg.norm(np.ptp(p, axis=0)) * 4)
        self.device = "cuda:" + str(select_cuda_device(preference))
        wp.init()
        t = time.perf_counter()
        tri = p[f].astype(float)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-30)
        vn = np.zeros_like(p, dtype=float)
        for j in range(3):
            a = tri[:, (j + 1) % 3] - tri[:, j]
            b = tri[:, (j + 2) % 3] - tri[:, j]
            angle = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))
            np.add.at(vn, f[:, j], n * angle[:, None])
        edges = np.stack([f[:, [1, 2]], f[:, [2, 0]], f[:, [0, 1]]], axis=1).reshape(-1, 2)
        unique, inv = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
        normals = np.zeros((len(unique), 3))
        np.add.at(normals, inv, np.repeat(n, 3, axis=0))
        en = normals[inv]
        self.ids = wp.array(f.ravel().astype(np.int32), dtype=int, device=self.device)
        self.mesh = wp.Mesh(
            points=wp.array(p.astype(np.float32), dtype=wp.vec3, device=self.device),
            indices=self.ids,
        )
        self.fn = wp.array(n.astype(np.float32), dtype=wp.vec3, device=self.device)
        self.vn = wp.array(vn.astype(np.float32), dtype=wp.vec3, device=self.device)
        self.en = wp.array(en.astype(np.float32), dtype=wp.vec3, device=self.device)
        wp.synchronize_device(self.device)
        self.setup = time.perf_counter() - t
        self.query_seconds = 0.0
        self.query_count = 0

    def evaluate(self, xyz):
        t = time.perf_counter()
        queries = wp.array(
            (xyz - self.center).astype(np.float32), dtype=wp.vec3, device=self.device
        )
        out = wp.empty(len(xyz), dtype=float, device=self.device)
        wp.launch(
            query_normals,
            dim=len(xyz),
            inputs=[self.mesh.id, queries, self.ids, self.fn, self.vn, self.en, self.limit, out],
            device=self.device,
        )
        values = out.numpy()
        self.query_seconds += time.perf_counter() - t
        self.query_count += len(xyz)
        return values
