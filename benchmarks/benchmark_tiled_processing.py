"""Reproducible consistency, performance, and assembly benchmark for tiled processing."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from out_of_core import (
    binary_stl_count,
    build_index,
    default_index_directory,
    optimize_indexed_mesh,
)


def streamed_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def run_case(index: Path, output: Path, target: int, workers: int) -> dict:
    started = time.perf_counter()
    report = optimize_indexed_mesh(index, output, target, workers=workers)
    elapsed = time.perf_counter() - started
    return {
        **report,
        "workers_requested": workers,
        "seconds": elapsed,
        "sha256": streamed_sha256(output),
        "file_triangles": binary_stl_count(output),
        "file_bytes": output.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    index = default_index_directory(source)
    if args.rebuild_index and index.exists():
        shutil.rmtree(index)
    if not index.exists():
        build_index(source, index)

    results = {}
    for name, workers in (("sequential", 1), ("parallel", args.workers)):
        results_directory = index / "results"
        if results_directory.exists():
            shutil.rmtree(results_directory)
        results[name] = run_case(
            index, output_root / f"{name}.stl", args.target, workers
        )

    results["cached"] = run_case(
        index, output_root / "cached.stl", args.target, args.workers
    )
    hashes = {case["sha256"] for case in results.values()}
    counts_match = all(
        case["output_triangles"] == case["file_triangles"] for case in results.values()
    )
    summary = {
        "source": {
            "triangles": binary_stl_count(source),
            "bytes": source.stat().st_size,
        },
        "requested_target": args.target,
        "runs": results,
        "validation": {
            "byte_identical": len(hashes) == 1,
            "assembled_counts_match": counts_match,
            "assembly_loss_triangles": sum(
                case["output_triangles"] - case["file_triangles"]
                for case in results.values()
            ),
        },
    }
    summary["speedup"] = (
        results["sequential"]["seconds"] / results["parallel"]["seconds"]
    )
    report_path = output_root / "results.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if len(hashes) == 1 and counts_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
