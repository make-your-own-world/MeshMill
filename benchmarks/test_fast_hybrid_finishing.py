"""Measure bounded finishing on the existing Fast-QEM-based depth result."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_depth_density_adaptive import connected_arrays, load_binary_stl
from depth_adaptive_qem import _match_surface_area, _remove_nonmanifold_surplus
from meshmill import mesh_quality_metrics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_points, source_faces, _ = connected_arrays(
        load_binary_stl(root / "samples" / "original-scan.stl")
    )
    result_root = root / "benchmarks" / "results" / "depth-density-production-250k"
    output_points, output_faces, _ = connected_arrays(
        load_binary_stl(result_root / "depth-adaptive-qem.stl")
    )
    started = time.perf_counter()
    finished_faces, topology = _remove_nonmanifold_surplus(output_points, output_faces)
    finished_points, surface = _match_surface_area(
        source_points,
        source_faces,
        output_points,
        finished_faces,
    )
    finishing_seconds = time.perf_counter() - started
    quality = mesh_quality_metrics(
        source_points,
        source_faces,
        finished_points,
        finished_faces,
        200_000,
    )
    print(
        json.dumps(
            {
                "finishing_seconds": finishing_seconds,
                "triangles": len(finished_faces),
                "vertices": len(finished_points),
                "quality": quality,
                "topology": topology,
                "surface": surface,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
