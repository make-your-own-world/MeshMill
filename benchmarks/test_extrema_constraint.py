"""Measure a six-coordinate bounds constraint on an existing benchmark mesh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_depth_density_adaptive import connected_arrays, load_binary_stl

from meshmill import arrays_polydata, mesh_quality_metrics, save_stl


def constrain_extrema(source_points: np.ndarray, output_points: np.ndarray) -> np.ndarray:
    constrained = np.array(output_points, copy=True)
    source_low = source_points.min(axis=0)
    source_high = source_points.max(axis=0)
    for axis in range(3):
        constrained[np.argmin(constrained[:, axis]), axis] = source_low[axis]
        constrained[np.argmax(constrained[:, axis]), axis] = source_high[axis]
    return constrained


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source_points, source_faces, _ = connected_arrays(load_binary_stl(args.source))
    output_points, output_faces, _ = connected_arrays(load_binary_stl(args.candidate))
    before = mesh_quality_metrics(source_points, source_faces, output_points, output_faces, 200_000)
    constrained = constrain_extrema(source_points, output_points)
    after = mesh_quality_metrics(source_points, source_faces, constrained, output_faces, 200_000)
    save_stl(arrays_polydata(constrained, output_faces), args.output)
    result = {"before": before, "after": after}
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
