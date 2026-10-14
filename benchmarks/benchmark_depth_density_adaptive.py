"""Search density-gated Depth Adaptive QEM parameters on the full sample mesh."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import fast_simplification
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth_adaptive_qem import (
    adaptive_probe_projection,
    simplify_depth_adaptive_qem,
)
from meshmill import mesh_quality_metrics

STL_HEADER_BYTES = 84
STL_RECORD_BYTES = 50
STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def load_binary_stl(path: Path) -> np.ndarray:
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(STL_HEADER_BYTES)
    count = struct.unpack_from("<I", header, 80)[0]
    if STL_HEADER_BYTES + count * STL_RECORD_BYTES != size:
        raise ValueError("Benchmark requires a valid binary STL.")
    records = np.memmap(
        path,
        dtype=STL_DTYPE,
        mode="r",
        offset=STL_HEADER_BYTES,
        shape=(count,),
    )
    return np.asarray(records["vertices"], dtype=np.float32)


def connected_arrays(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    points, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32, copy=False)
    return points, faces, time.perf_counter() - started


def measure(source_points, source_faces, output_points, output_faces, sample_count):
    return mesh_quality_metrics(
        source_points,
        source_faces,
        output_points,
        output_faces,
        sample_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "samples" / "original-scan.stl",
    )
    parser.add_argument("--target", type=int, default=250_000)
    parser.add_argument("--sample-count", type=int, default=200_000)
    parser.add_argument("--gpu", default="system")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    load_started = time.perf_counter()
    points, faces, connectivity_seconds = connected_arrays(load_binary_stl(args.input))
    report = {
        "source": {
            "triangles": len(faces),
            "vertices": len(points),
            "connectivity_seconds": connectivity_seconds,
            "load_and_connectivity_seconds": time.perf_counter() - load_started,
        },
        "target_triangles": args.target,
        "sample_count": args.sample_count,
        "runs": [],
    }

    def record(name, seconds, output_points, output_faces, details=None):
        report["runs"].append(
            {
                "name": name,
                "seconds": seconds,
                "triangles": len(output_faces),
                "vertices": len(output_points),
                "quality": measure(
                    points,
                    faces,
                    output_points,
                    output_faces,
                    args.sample_count,
                ),
                "details": details or {},
            }
        )
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    started = time.perf_counter()
    output_points, output_faces = fast_simplification.simplify(
        points,
        faces,
        target_count=args.target,
        agg=7.0,
        verbose=False,
        preserve_border=True,
    )
    record("Fast QEM", time.perf_counter() - started, output_points, output_faces)

    started = time.perf_counter()
    output_points, output_faces = fast_simplification.simplify(
        points,
        faces,
        target_count=args.target,
        agg=4.9,
        verbose=False,
        preserve_border=True,
    )
    record("Density balanced", time.perf_counter() - started, output_points, output_faces)

    started = time.perf_counter()
    output_points, output_faces, details = simplify_depth_adaptive_qem(
        points,
        faces,
        args.target,
        aggressiveness=8.0,
        gpu_preference=args.gpu,
    )
    record(
        "Depth Adaptive QEM",
        time.perf_counter() - started,
        output_points,
        output_faces,
        details,
    )

    parameter_grid = (
        (1.00, 1.25, -0.25, 1.0),
        (1.00, 1.50, 0.00, 1.5),
        (1.00, 1.75, 0.25, 1.5),
        (1.00, 2.00, 0.50, 2.0),
    )
    for floor, ceiling, center, slope in parameter_grid:
        projected, details = adaptive_probe_projection(
            points,
            faces,
            args.target,
            args.gpu,
            density_adaptive=True,
            density_floor=floor,
            density_ceiling=ceiling,
            density_center=center,
            density_slope=slope,
        )
        for aggressiveness in (6.0, 8.0):
            qem_started = time.perf_counter()
            output_points, output_faces = fast_simplification.simplify(
                projected,
                faces,
                target_count=args.target,
                agg=aggressiveness,
                verbose=False,
                preserve_border=True,
            )
            variant_details = dict(details)
            variant_details["qem_seconds"] = time.perf_counter() - qem_started
            record(
                (
                    "Depth Density Adaptive QEM "
                    f"floor={floor:.2f} ceiling={ceiling:.2f} "
                    f"center={center:.2f} slope={slope:.2f} "
                    f"agg={aggressiveness:.1f}"
                ),
                details["seconds"] + variant_details["qem_seconds"],
                output_points,
                output_faces,
                variant_details,
            )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
