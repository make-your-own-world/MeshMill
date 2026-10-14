"""Compare serial and parallel GPU-guided tiled reduction without changing app defaults."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkStaticPointLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth_adaptive_qem import simplify_depth_adaptive_qem
from out_of_core import (
    TRIANGLE_RECORD_DTYPE,
    build_index,
    optimize_indexed_mesh,
)

DEPTH_ADAPTIVE = "Depth Adaptive QEM (GPU / Faster)"


def load_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        stream.seek(80)
        triangle_count = struct.unpack("<I", stream.read(4))[0]
    records = np.memmap(
        path,
        dtype=TRIANGLE_RECORD_DTYPE,
        mode="r",
        offset=84,
        shape=(triangle_count,),
    )
    flat = np.ascontiguousarray(records["vertices"].reshape(-1, 3), dtype=np.float64)
    points, inverse = np.unique(flat, axis=0, return_inverse=True)
    return np.ascontiguousarray(points), np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int32)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def arrays_polydata(points: np.ndarray, faces: np.ndarray) -> vtkPolyData:
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    packed = np.empty((len(faces), 4), dtype=np.int64)
    packed[:, 0] = 3
    packed[:, 1:] = faces
    cells = vtkCellArray()
    cells.SetCells(
        len(faces), numpy_to_vtkIdTypeArray(np.ascontiguousarray(packed).ravel(), deep=True)
    )
    polydata = vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(cells)
    return polydata


def quality_metrics(
    source_points: np.ndarray, output_points: np.ndarray, sample_count: int = 20_000
) -> dict[str, float]:
    """Use the same deterministic sampled nearest-point metric as MeshMill's Stats panel."""
    rng = np.random.default_rng(20261013)
    sample = source_points[
        rng.choice(len(source_points), min(sample_count, len(source_points)), replace=False)
    ]
    output_poly = arrays_polydata(output_points, np.empty((0, 3), dtype=np.int32))
    locator = vtkStaticPointLocator()
    locator.SetDataSet(output_poly)
    locator.BuildLocator()
    vtk_points = output_poly.GetPoints()
    distances = np.asarray(
        [
            np.linalg.norm(sample_point - vtk_points.GetPoint(locator.FindClosestPoint(sample_point)))
            for sample_point in sample
        ],
        dtype=np.float64,
    )
    diagonal = max(float(np.linalg.norm(source_points.max(0) - source_points.min(0))), 1e-12)
    return {
        "rms_percent": 100.0 * float(np.sqrt(np.mean(distances * distances))) / diagonal,
        "p95_percent": 100.0 * float(np.percentile(distances, 95)) / diagonal,
    }


def run_tiled(
    index: Path,
    output: Path,
    target: int,
    gpu: str,
    workers: int,
    parallel_gpu_workers: bool,
) -> dict:
    results = index / "results"
    if results.exists():
        shutil.rmtree(results)
    started = time.perf_counter()
    report = optimize_indexed_mesh(
        index,
        output,
        target,
        algorithm=DEPTH_ADAPTIVE,
        gpu_preference=gpu,
        workers=workers,
        parallel_gpu_workers=parallel_gpu_workers,
    )
    report["seconds"] = time.perf_counter() - started
    report["sha256"] = sha256_file(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tile-triangles", type=int, default=65_536)
    args = parser.parse_args()

    source = args.source.resolve()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    index = work / "source.meshmill-index"
    if not index.exists():
        build_index(source, index, target_triangles_per_tile=args.tile_triangles)
    source_points, source_faces = load_binary_stl(source)

    global_started = time.perf_counter()
    global_points, global_faces, _ = simplify_depth_adaptive_qem(
        source_points, source_faces, args.target, 7.0, args.gpu
    )
    global_seconds = time.perf_counter() - global_started
    serial_path = work / "tiled-serial.stl"
    parallel_path = work / "tiled-parallel.stl"
    serial = run_tiled(index, serial_path, args.target, args.gpu, 1, False)
    parallel = run_tiled(index, parallel_path, args.target, args.gpu, args.workers, True)
    serial_points, serial_faces = load_binary_stl(serial_path)
    parallel_points, parallel_faces = load_binary_stl(parallel_path)

    result = {
        "source_triangles": len(source_faces),
        "target_triangles": args.target,
        "gpu": args.gpu,
        "global": {
            "seconds": global_seconds,
            "triangles": len(global_faces),
            "quality": quality_metrics(source_points, global_points),
        },
        "tiled_serial": {
            **serial,
            "quality": quality_metrics(source_points, serial_points),
        },
        "tiled_parallel": {
            **parallel,
            "quality": quality_metrics(source_points, parallel_points),
        },
        "validation": {
            "serial_parallel_byte_identical": serial["sha256"] == parallel["sha256"],
            "serial_parallel_triangle_match": len(serial_faces) == len(parallel_faces),
        },
    }
    (work / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if all(result["validation"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
