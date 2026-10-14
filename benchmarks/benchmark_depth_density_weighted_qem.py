"""Test density as a QEM attribute after GPU depth-probe surface analysis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkFiltersCore import vtkQuadricDecimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_depth_density_adaptive import connected_arrays, load_binary_stl

from depth_adaptive_qem import adaptive_probe_projection
from meshmill import arrays_polydata, mesh_quality_metrics, polydata_arrays


def robust_density(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return a bounded robust z-score for vertices per incident surface area."""
    incident_count = np.bincount(faces.ravel(), minlength=len(points)).astype(np.float32)
    triangles = points[faces]
    area = np.float32(0.5) * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    incident_area = np.bincount(
        faces.ravel(),
        weights=np.repeat(area, 3),
        minlength=len(points),
    ).astype(np.float32)
    valid = incident_area > np.float32(1e-12)
    density = np.zeros(len(points), dtype=np.float32)
    density[valid] = np.log(
        np.maximum(incident_count[valid] / incident_area[valid], np.float32(1e-30))
    )
    median = np.median(density[valid])
    mad = np.median(np.abs(density[valid] - median))
    sigma = max(float(np.float32(1.4826) * mad), 1e-6)
    density = np.clip((density - median) / sigma, -4.0, 4.0)
    density[~valid] = -4.0
    return np.ascontiguousarray((density + 4.0) / 8.0, dtype=np.float32)


def area_weighted_vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = points[faces]
    face_cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    ).astype(np.float32, copy=False)
    vertex_ids = faces.ravel()
    repeated = np.repeat(face_cross, 3, axis=0)
    normals = np.stack(
        [
            np.bincount(vertex_ids, weights=repeated[:, axis], minlength=len(points))
            for axis in range(3)
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    return np.ascontiguousarray(normals)


def reduce_weighted(
    points: np.ndarray,
    faces: np.ndarray,
    density: np.ndarray,
    normals: np.ndarray,
    target: int,
    scalar_weight: float,
    normal_weight: float,
    volume_preservation: bool,
    boundary_weight: float = 1.0,
    length_weighted_boundaries: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    source = arrays_polydata(points, faces)
    values = numpy_to_vtk(density, deep=True)
    values.SetName("sampling_density")
    source.GetPointData().SetScalars(values)
    normal_values = numpy_to_vtk(normals, deep=True)
    normal_values.SetName("surface_normals")
    source.GetPointData().SetNormals(normal_values)
    decimator = vtkQuadricDecimation()
    decimator.SetInputData(source)
    decimator.SetTargetReduction(1.0 - target / len(faces))
    decimator.AttributeErrorMetricOn()
    decimator.ScalarsAttributeOn()
    decimator.SetScalarsWeight(scalar_weight)
    decimator.SetBoundaryWeightFactor(boundary_weight)
    decimator.SetWeighBoundaryConstraintsByLength(length_weighted_boundaries)
    if normal_weight > 0.0:
        decimator.NormalsAttributeOn()
        decimator.SetNormalsWeight(normal_weight)
    else:
        decimator.NormalsAttributeOff()
    if volume_preservation:
        decimator.VolumePreservationOn()
    else:
        decimator.VolumePreservationOff()
    decimator.Update()
    return polydata_arrays(decimator.GetOutput())


def constrain_extrema(
    source_points: np.ndarray, output_points: np.ndarray
) -> tuple[np.ndarray, float]:
    """Restore six source bounds by changing only the corresponding output coordinates."""
    constrained = np.array(output_points, copy=True)
    source_low = source_points.min(axis=0)
    source_high = source_points.max(axis=0)
    maximum_change = 0.0
    for axis in range(3):
        low_index = int(np.argmin(constrained[:, axis]))
        high_index = int(np.argmax(constrained[:, axis]))
        maximum_change = max(
            maximum_change,
            abs(float(constrained[low_index, axis] - source_low[axis])),
            abs(float(constrained[high_index, axis] - source_high[axis])),
        )
        constrained[low_index, axis] = source_low[axis]
        constrained[high_index, axis] = source_high[axis]
    return constrained, maximum_change


def surface_area(points: np.ndarray, faces: np.ndarray) -> float:
    triangles = points[faces]
    return float(
        0.5
        * np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        ).sum()
    )


def match_surface_area(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    output_points: np.ndarray,
    output_faces: np.ndarray,
    *,
    maximum_alpha: float = 0.08,
    iterations: int = 12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the smallest bounded one-ring relaxation that matches source surface area."""
    source_area = surface_area(source_points, source_faces)
    initial_area = surface_area(output_points, output_faces)
    if initial_area <= source_area:
        constrained, extrema_change = constrain_extrema(source_points, output_points)
        return constrained, {
            "alpha": 0.0,
            "initial_area_ratio": initial_area / source_area,
            "final_area_ratio": surface_area(constrained, output_faces) / source_area,
            "extrema_maximum_change": extrema_change,
        }

    face_mean = output_points[output_faces].mean(axis=1)
    vertex_ids = output_faces.ravel()
    repeated = np.repeat(face_mean, 3, axis=0)
    neighbor_sum = np.stack(
        [
            np.bincount(
                vertex_ids,
                weights=repeated[:, axis],
                minlength=len(output_points),
            )
            for axis in range(3)
        ],
        axis=1,
    )
    count = np.bincount(vertex_ids, minlength=len(output_points))
    target = np.array(output_points, copy=True, dtype=np.float64)
    valid = count > 0
    target[valid] = neighbor_sum[valid] / count[valid, None]
    delta = target - output_points

    low = 0.0
    high = maximum_alpha
    high_points, _ = constrain_extrema(source_points, output_points + high * delta)
    if surface_area(high_points, output_faces) > source_area:
        constrained = high_points
        alpha = high
    else:
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            candidate, _ = constrain_extrema(source_points, output_points + middle * delta)
            if surface_area(candidate, output_faces) > source_area:
                low = middle
            else:
                high = middle
        alpha = 0.5 * (low + high)
        constrained, _ = constrain_extrema(source_points, output_points + alpha * delta)

    constrained, extrema_change = constrain_extrema(source_points, constrained)
    return constrained, {
        "alpha": alpha,
        "initial_area_ratio": initial_area / source_area,
        "final_area_ratio": surface_area(constrained, output_faces) / source_area,
        "extrema_maximum_change": extrema_change,
    }


def remove_nonmanifold_surplus(
    points: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove the smallest surplus faces from edges shared by more than two triangles."""
    face_ids = np.repeat(np.arange(len(faces), dtype=np.int64), 3)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    # Concatenation groups edges by edge slot, while face IDs above are grouped by face.
    face_ids = np.concatenate(
        (
            np.arange(len(faces), dtype=np.int64),
            np.arange(len(faces), dtype=np.int64),
            np.arange(len(faces), dtype=np.int64),
        )
    )
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    sorted_face_ids = face_ids[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_edges, axis=0), axis=1)) + 1]
    ends = np.r_[starts[1:], len(sorted_edges)]
    nonmanifold_groups = np.flatnonzero((ends - starts) > 2)
    if not len(nonmanifold_groups):
        return np.ascontiguousarray(faces), {
            "nonmanifold_edges_before": 0,
            "removed_faces": 0,
        }
    triangles = points[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    remove = np.zeros(len(faces), dtype=bool)
    for group in nonmanifold_groups:
        members = np.unique(sorted_face_ids[starts[group] : ends[group]])
        active = members[~remove[members]]
        surplus = max(0, len(active) - 2)
        if surplus:
            remove[active[np.argsort(areas[active])[:surplus]]] = True
    return np.ascontiguousarray(faces[~remove]), {
        "nonmanifold_edges_before": len(nonmanifold_groups),
        "removed_faces": int(np.count_nonzero(remove)),
    }


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
    parser.add_argument("--weights", default="0,0.01,0.1,1,10")
    parser.add_argument("--normal-weights", default="0")
    parser.add_argument("--volume-preservation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    points, faces, connectivity_seconds = connected_arrays(load_binary_stl(args.input))
    projected, depth_report = adaptive_probe_projection(points, faces, args.target, args.gpu)
    density_started = time.perf_counter()
    density = robust_density(points, faces)
    normals = area_weighted_vertex_normals(projected, faces)
    density_seconds = time.perf_counter() - density_started
    report = {
        "source_triangles": len(faces),
        "source_vertices": len(points),
        "target_triangles": args.target,
        "connectivity_seconds": connectivity_seconds,
        "depth_analysis": depth_report,
        "density_seconds": density_seconds,
        "runs": [],
    }
    weights = tuple(float(value) for value in args.weights.split(","))
    normal_weights = tuple(float(value) for value in args.normal_weights.split(","))
    for preserve_volume in (args.volume_preservation,):
        for weight in weights:
            for normal_weight in normal_weights:
                started = time.perf_counter()
                output_points, output_faces = reduce_weighted(
                    projected,
                    faces,
                    density,
                    normals,
                    args.target,
                    weight,
                    normal_weight,
                    preserve_volume,
                )
                seconds = time.perf_counter() - started
                report["runs"].append(
                    {
                        "name": (
                            f"density={weight:g} normals={normal_weight:g} volume={preserve_volume}"
                        ),
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
                    }
                )
                args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
