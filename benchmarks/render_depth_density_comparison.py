"""Render repeatable wireframe and density views for depth-density validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import render_algorithm_comparison as renderer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    renderer.MODELS.clear()
    renderer.MODELS.extend(
        [
            ("Original", renderer.ROOT / "samples" / "original-scan.stl"),
            ("Fast QEM", args.results / "fast-qem.stl"),
            ("Density balanced", args.results / "density-balanced.stl"),
            ("Depth Adaptive", args.results / "depth-adaptive-qem.stl"),
            ("Depth + Density", args.results / "depth-density-adaptive-qem.stl"),
        ]
    )
    renderer.OUTPUT = args.output
    renderer.OUTPUT.mkdir(parents=True, exist_ok=True)
    print(renderer.make_contact_sheet("wireframe"))
    print(renderer.make_contact_sheet("density"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
