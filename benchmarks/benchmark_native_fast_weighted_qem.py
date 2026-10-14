"""Benchmark an isolated density/normal-weighted Fast QEM prototype."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_depth_density_adaptive import connected_arrays, load_binary_stl
from depth_adaptive_qem import (
    _area_weighted_vertex_normals,
    _match_surface_area,
    _remove_nonmanifold_surplus,
    _robust_density,
    adaptive_probe_projection,
)
from meshmill import mesh_quality_metrics


def run_native(
    executable: Path,
    working: Path,
    points: np.ndarray,
    faces: np.ndarray,
    density: np.ndarray,
    normals: np.ndarray,
    target: int,
    density_weight: float,
    normal_weight: float,
    attribute_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    input_path = working / "weighted-input.bin"
    output_path = working / "weighted-output.bin"
    header = struct.pack(
        "<QQQdddd",
        len(points),
        len(faces),
        target,
        8.0,
        density_weight,
        normal_weight,
        attribute_scale,
    )
    with input_path.open("wb") as stream:
        stream.write(header)
        stream.write(np.ascontiguousarray(points, dtype=np.float64).tobytes())
        stream.write(np.ascontiguousarray(faces, dtype=np.int32).tobytes())
        stream.write(np.ascontiguousarray(density, dtype=np.float32).tobytes())
        stream.write(np.ascontiguousarray(normals, dtype=np.float32).tobytes())
    started = time.perf_counter()
    subprocess.run([executable, input_path, output_path], check=True)
    seconds = time.perf_counter() - started
    with output_path.open("rb") as stream:
        output_vertices, output_faces = struct.unpack("<QQ", stream.read(16))
        result_points = np.frombuffer(
            stream.read(output_vertices * 3 * 8), dtype=np.float64
        ).reshape(-1, 3).copy()
        result_faces = np.frombuffer(
            stream.read(output_faces * 3 * 4), dtype=np.int32
        ).reshape(-1, 3).copy()
    return result_points, result_faces, seconds


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    prototype = root / "tmp" / "fast-weighted-qem"
    points, faces, _ = connected_arrays(load_binary_stl(root / "samples" / "original-scan.stl"))
    target = 250_000
    projected, depth_report = adaptive_probe_projection(points, faces, target, "system")
    density = _robust_density(points, faces)
    normals = _area_weighted_vertex_normals(projected, faces)
    source_area = 0.5 * np.linalg.norm(
        np.cross(
            points[faces][:, 1] - points[faces][:, 0],
            points[faces][:, 2] - points[faces][:, 0],
        ),
        axis=1,
    ).sum()
    output_pitch_squared = 4.0 * source_area / (np.sqrt(3.0) * target)
    variants = [
        (0.05, 0.00, 1.0),
        (0.075, 0.00, 1.0),
        (0.15, 0.00, 1.0),
        (0.25, 0.00, 1.0),
        (0.50, 0.00, 1.0),
    ]
    results = []
    for density_weight, normal_weight, scale_multiplier in variants:
        result_points, result_faces, qem_seconds = run_native(
            prototype / "weighted_qem.exe",
            prototype,
            projected,
            faces,
            density,
            normals,
            target,
            density_weight,
            normal_weight,
            float(output_pitch_squared * scale_multiplier),
        )
        finish_started = time.perf_counter()
        result_faces, topology = _remove_nonmanifold_surplus(result_points, result_faces)
        result_points, surface = _match_surface_area(
            points, faces, result_points, result_faces
        )
        finishing_seconds = time.perf_counter() - finish_started
        record = {
            "density_weight": density_weight,
            "normal_weight": normal_weight,
            "scale_multiplier": scale_multiplier,
            "seconds": depth_report["seconds"] + qem_seconds + finishing_seconds,
            "qem_seconds": qem_seconds,
            "finishing_seconds": finishing_seconds,
            "triangles": len(result_faces),
            "vertices": len(result_points),
            "quality": mesh_quality_metrics(
                points, faces, result_points, result_faces, 200_000
            ),
            "topology": topology,
            "surface": surface,
        }
        results.append(record)
        print(json.dumps(record, indent=2), flush=True)
    output = root / "benchmarks" / "results" / "native-fast-weighted-qem-250k-density.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
