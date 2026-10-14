"""Validate the depth-density candidate against production reducers at one target."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import fast_simplification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_depth_density_adaptive import connected_arrays, load_binary_stl

from depth_adaptive_qem import (
    adaptive_probe_projection,
    simplify_depth_density_adaptive_qem,
)
from meshmill import arrays_polydata, mesh_quality_metrics, save_stl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "samples" / "original-scan.stl",
    )
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=200_000)
    parser.add_argument("--gpu", default="system")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points, faces, connectivity_seconds = connected_arrays(load_binary_stl(args.input))
    report = {
        "source_triangles": len(faces),
        "source_vertices": len(points),
        "target_triangles": args.target,
        "sample_count": args.sample_count,
        "connectivity_seconds": connectivity_seconds,
        "runs": [],
    }

    def record(name, seconds, output_points, output_faces, details=None):
        mesh_path = args.output_dir / f"{name.lower().replace(' ', '-')}.stl"
        save_stl(arrays_polydata(output_points, output_faces), mesh_path)
        report["runs"].append(
            {
                "name": name,
                "seconds": seconds,
                "triangles": len(output_faces),
                "vertices": len(output_points),
                "quality": mesh_quality_metrics(
                    points,
                    faces,
                    output_points,
                    output_faces,
                    args.sample_count,
                ),
                "details": details or {},
                "mesh": str(mesh_path),
            }
        )
        (args.output_dir / "results.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    for name, aggressiveness in (("Fast QEM", 7.0), ("Density balanced", 4.9)):
        started = time.perf_counter()
        output_points, output_faces = fast_simplification.simplify(
            points,
            faces,
            target_count=args.target,
            agg=aggressiveness,
            verbose=False,
            preserve_border=True,
        )
        record(name, time.perf_counter() - started, output_points, output_faces)

    projected, depth_report = adaptive_probe_projection(points, faces, args.target, args.gpu)
    started = time.perf_counter()
    output_points, output_faces = fast_simplification.simplify(
        projected,
        faces,
        target_count=args.target,
        agg=8.0,
        verbose=False,
        preserve_border=True,
    )
    record(
        "Depth Adaptive QEM",
        depth_report["seconds"] + time.perf_counter() - started,
        output_points,
        output_faces,
        depth_report,
    )

    started = time.perf_counter()
    output_points, output_faces, hybrid_report = simplify_depth_density_adaptive_qem(
        points,
        faces,
        args.target,
        gpu_preference=args.gpu,
    )
    record(
        "Depth Density Adaptive QEM",
        time.perf_counter() - started,
        output_points,
        output_faces,
        hybrid_report,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
