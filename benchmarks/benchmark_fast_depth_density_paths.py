"""Compare fast topology paths for the depth-density production candidate."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import fast_simplification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_depth_density_adaptive import connected_arrays, load_binary_stl
from depth_adaptive_qem import (
    _match_surface_area,
    _remove_nonmanifold_surplus,
    adaptive_probe_projection,
)
from meshmill import mesh_quality_metrics


def finish(source_points, source_faces, points, faces):
    faces, topology = _remove_nonmanifold_surplus(points, faces)
    points, surface = _match_surface_area(
        source_points, source_faces, points, faces
    )
    return points, faces, topology, surface


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    points, faces, connectivity_seconds = connected_arrays(
        load_binary_stl(root / "samples" / "original-scan.stl")
    )
    target = 250_000
    projected, depth_report = adaptive_probe_projection(points, faces, target, "system")
    variants = [
        ("direct-agg8", [(target, 8.0)]),
        ("direct-agg4.9", [(target, 4.9)]),
        ("staged-500k-agg8-then-250k-agg7", [(500_000, 8.0), (target, 7.0)]),
        ("staged-500k-agg5-then-250k-agg7", [(500_000, 5.0), (target, 7.0)]),
        ("staged-1m-agg8-then-250k-agg7", [(1_000_000, 8.0), (target, 7.0)]),
    ]
    results = {
        "connectivity_seconds": connectivity_seconds,
        "depth_seconds": depth_report["seconds"],
        "runs": [],
    }
    for name, passes in variants:
        current_points = projected
        current_faces = faces
        started = time.perf_counter()
        for pass_target, aggressiveness in passes:
            current_points, current_faces = fast_simplification.simplify(
                current_points,
                current_faces,
                target_count=pass_target,
                agg=aggressiveness,
                verbose=False,
                preserve_border=True,
            )
        current_points, current_faces, topology, surface = finish(
            points, faces, current_points, current_faces
        )
        seconds = depth_report["seconds"] + time.perf_counter() - started
        results["runs"].append(
            {
                "name": name,
                "seconds": seconds,
                "triangles": len(current_faces),
                "vertices": len(current_points),
                "quality": mesh_quality_metrics(
                    points, faces, current_points, current_faces, 200_000
                ),
                "topology": topology,
                "surface": surface,
            }
        )
        print(json.dumps(results["runs"][-1], indent=2), flush=True)
    output = root / "benchmarks" / "results" / "fast-depth-density-paths-250k.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
