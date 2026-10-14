"""Measure small one-ring relaxations on a depth-density candidate mesh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_depth_density_adaptive import connected_arrays, load_binary_stl
from benchmark_depth_density_weighted_qem import (
    constrain_extrema,
    match_surface_area,
    remove_nonmanifold_surplus,
)

from meshmill import (
    arrays_polydata,
    mesh_quality_metrics,
    mesh_surface_area,
    save_stl,
)


def one_ring_target(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return the mean of incident face centroids for each vertex."""
    face_mean = points[faces].mean(axis=1)
    vertex_ids = faces.ravel()
    repeated = np.repeat(face_mean, 3, axis=0)
    neighbor_sum = np.stack(
        [
            np.bincount(vertex_ids, weights=repeated[:, axis], minlength=len(points))
            for axis in range(3)
        ],
        axis=1,
    )
    count = np.bincount(vertex_ids, minlength=len(points))
    target = np.array(points, copy=True, dtype=np.float64)
    valid = count > 0
    target[valid] = neighbor_sum[valid] / count[valid, None]
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-output", type=Path)
    parser.add_argument("--alphas", default="0,0.001,0.002,0.005,0.01,0.02")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()
    source_points, source_faces, _ = connected_arrays(load_binary_stl(args.source))
    points, faces, _ = connected_arrays(load_binary_stl(args.candidate))
    target = one_ring_target(points, faces)
    source_area = mesh_surface_area(source_points, source_faces)
    results = []
    if args.auto:
        faces, topology = remove_nonmanifold_surplus(points, faces)
        matched, details = match_surface_area(source_points, source_faces, points, faces)
        results.append(
            {
                "auto": details,
                "topology": topology,
                "quality": mesh_quality_metrics(
                    source_points, source_faces, matched, faces, 200_000
                ),
            }
        )
        if args.mesh_output:
            save_stl(arrays_polydata(matched, faces), args.mesh_output)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        return 0
    for alpha in (float(value) for value in args.alphas.split(",")):
        relaxed = points + alpha * (target - points)
        relaxed, maximum_change = constrain_extrema(source_points, relaxed)
        metrics = mesh_quality_metrics(source_points, source_faces, relaxed, faces, 200_000)
        results.append(
            {
                "alpha": alpha,
                "extrema_maximum_change": maximum_change,
                "signed_area_percent": 100.0
                * (mesh_surface_area(relaxed, faces) / source_area - 1.0),
                "quality": metrics,
            }
        )
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
