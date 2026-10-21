"""GPU-guided adaptive physical probes followed by boundary-preserving Fast QEM."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkQuadricDecimation

from qem_progress import simplify as simplify_with_progress

_cuda_dll_directories = []
if os.name == "nt":
    frozen_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    bundled_root = frozen_root / "cuda_runtime"
    legacy_component_root = (
        Path(os.environ.get("LOCALAPPDATA", Path.home())) / "MeshMill" / "components" / "cuda"
    )
    for runtime_root in (bundled_root, legacy_component_root):
        if not runtime_root.is_dir():
            continue
        sys.path.insert(0, str(runtime_root))
        runtime_dll_directories = [runtime_root]
        runtime_dll_directories.extend(
            path for path in (runtime_root / "nvidia").glob("*/bin") if path.is_dir()
        )
        for directory in runtime_dll_directories:
            _cuda_dll_directories.append(os.add_dll_directory(str(directory)))
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
    try:
        nvrtc_spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
    except ModuleNotFoundError:
        nvrtc_spec = None
    nvrtc_directories: list[Path] = []
    if nvrtc_spec is not None and nvrtc_spec.submodule_search_locations:
        package_root = next(iter(nvrtc_spec.submodule_search_locations), None)
        if package_root:
            nvrtc_directories.append(Path(package_root) / "bin")
    if any(frozen_root.glob("nvrtc64_*.dll")):
        nvrtc_directories.append(frozen_root)
    for nvrtc_directory in nvrtc_directories:
        if nvrtc_directory.is_dir():
            _cuda_dll_directories.append(os.add_dll_directory(str(nvrtc_directory)))
            os.environ["PATH"] = str(nvrtc_directory) + os.pathsep + os.environ.get("PATH", "")
            break

_cupy_import_error: str | None = None
try:
    import cupy as cp
except (ImportError, OSError) as exc:
    cp = None
    _cupy_import_error = repr(exc)

if os.environ.get("MESHMILL_GPU_DIAGNOSTICS") and _cupy_import_error:
    print(f"MeshMill GPU backend import failed: {_cupy_import_error}", file=sys.stderr)


def cuda_devices() -> list[dict[str, Any]]:
    """Return CUDA devices with stable PCI identifiers for application preferences."""
    if cp is None:
        return []
    devices: list[dict[str, Any]] = []
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001 - enumeration must not block application startup
        return []
    for device_id in range(count):
        try:
            properties = cp.cuda.runtime.getDeviceProperties(device_id)
            name = properties.get("name", f"CUDA GPU {device_id}")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            pci_bus_id = str(cp.cuda.runtime.deviceGetPCIBusId(device_id))
            devices.append(
                {
                    "id": device_id,
                    "name": str(name),
                    "pci_bus_id": pci_bus_id,
                    "total_memory": int(properties.get("totalGlobalMem", 0)),
                }
            )
        except Exception:  # noqa: BLE001,S112 - one inaccessible adapter should not hide the others
            continue
    return devices


def select_cuda_device(preference: str = "system") -> int | None:
    """Select a CUDA device by stable PCI ID, or choose the device with most free VRAM."""
    if cp is None:
        return None
    if not preference or preference == "system":
        candidates: list[tuple[float, int, int]] = []
        for device_id in range(cp.cuda.runtime.getDeviceCount()):
            try:
                with cp.cuda.Device(device_id):
                    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
                free_ratio = free_bytes / max(1, total_bytes)
                candidates.append((free_ratio, int(free_bytes), device_id))
            except Exception as error:  # noqa: BLE001 - CUDA may expose backend-specific errors
                if os.environ.get("MESHMILL_GPU_DIAGNOSTICS"):
                    print(f"MeshMill CUDA device {device_id} could not be queried: {error!r}", file=sys.stderr)
                continue
        selected_id = max(candidates)[2] if candidates else 0
        device = cp.cuda.Device(selected_id)
        device.use()
        return int(device.id)
    device = cp.cuda.Device.from_pci_bus_id(preference)
    device.use()
    return int(device.id)


def gpu_backend_available(preference: str = "system") -> bool:
    """Return whether a CUDA device can execute the adaptive-probe analysis."""
    if cp is None:
        return False
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        select_cuda_device(preference)
        probe = cp.arange(1, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        return float(probe.get()[0]) == 0.0
    except Exception:  # noqa: BLE001 - CuPy exposes several backend/compiler exception types
        return False


def gpu_backend_supported(preference: str = "system") -> bool:
    """Return whether the packaged backend and a CUDA device are present without compiling kernels."""
    if os.environ.get("MESHMILL_DISABLE_CUDA") == "1":
        return False
    if cp is None:
        return False
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        if preference != "system":
            select_cuda_device(preference)
        return not (os.name == "nt" and not _cuda_dll_directories)
    except Exception as exc:  # noqa: BLE001 - capability detection must not block application startup
        if os.environ.get("MESHMILL_GPU_DIAGNOSTICS"):
            print(f"MeshMill GPU capability check failed: {exc!r}", file=sys.stderr)
        return False


def _gpu_cache_marker(preference: str = "system") -> Path | None:
    """Return a versioned local marker for MeshMill's successfully compiled GPU kernels."""
    if cp is None:
        return None
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        device_id = select_cuda_device(preference)
        device = cp.cuda.runtime.getDeviceProperties(device_id or 0)
        device_name = device.get("name", b"gpu")
        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8", errors="replace")
        identity = re.sub(r"[^A-Za-z0-9._-]+", "-", str(device_name)).strip("-")
    except Exception:  # noqa: BLE001 - detection must not prevent the CPU fallback
        identity = "gpu"
    try:
        runtime = cp.cuda.runtime.runtimeGetVersion()
    except Exception:  # noqa: BLE001 - the marker can still identify the backend without a version
        runtime = "unknown"
    cupy_version = getattr(cp, "__version__", "unknown")
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "MeshMill" / "gpu-cache"
    return base / f"cupy-{cupy_version}-cuda-{runtime}-{identity}.ready"


def first_time_gpu_setup_needed(preference: str = "system") -> bool:
    """Return whether this GPU/backend combination lacks a completed MeshMill setup marker."""
    if not gpu_backend_supported(preference):
        return False
    marker = _gpu_cache_marker(preference)
    return marker is not None and not marker.exists()


def _mark_gpu_cache_ready(preference: str = "system") -> None:
    marker = _gpu_cache_marker(preference)
    if marker is None:
        return
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("Depth Adaptive QEM GPU kernels prepared.\n", encoding="utf-8")
    except OSError as exc:
        if os.environ.get("MESHMILL_GPU_DIAGNOSTICS"):
            print(f"MeshMill GPU cache marker failed: {exc!r}", file=sys.stderr)


def prepare_gpu_backend(preference: str = "system") -> dict[str, Any]:
    """Initialize the optimization worker and compile the adaptive-analysis kernels."""
    started = time.perf_counter()
    if not gpu_backend_supported(preference):
        return {
            "backend": "unavailable",
            "seconds": time.perf_counter() - started,
        }
    # A small closed mesh exercises every array, reduction, grouping, density, and projection
    # operation used by the production path. CuPy caches the compiled kernels, while running this
    # inside MeshMill's persistent optimization process also pays the CUDA-context startup cost.
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )
    _projected, report = adaptive_probe_projection(
        points,
        faces,
        target=8,
        gpu_preference=preference,
        density_adaptive=True,
    )
    return {
        "backend": report.get("backend", "CUDA"),
        "device_id": report.get("device_id"),
        "seconds": time.perf_counter() - started,
    }


def probe_source_statistics(points: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    """Read-only, target-independent inputs for physical probe dimensions."""
    sample_faces = faces[:: max(1, len(faces) // 500_000)]
    triangles = points[sample_faces]
    edges = np.concatenate(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        )
    )
    valid_edges = edges[edges > 1e-12]
    probe_diameter = float(np.median(valid_edges)) if len(valid_edges) else 0.0
    twice_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    area = float(0.5 * twice_area.sum() * (len(faces) / len(sample_faces)))
    return {"diameter": probe_diameter, "sampled_area": area}


def physical_probe_dimensions(
    points: np.ndarray, faces: np.ndarray, target: int, source_statistics=None
) -> dict[str, float]:
    """Derive probe dimensions, retaining the same sampling and arithmetic."""
    statistics = source_statistics if source_statistics is not None else probe_source_statistics(points, faces)
    probe_diameter = statistics["diameter"]
    area = statistics["sampled_area"]
    output_pitch = float(np.sqrt(4.0 * area / (np.sqrt(3.0) * target)))
    return {
        "diameter": probe_diameter,
        "tip_radius": probe_diameter * 0.5,
        "minimum_pitch": max(output_pitch, probe_diameter, 1e-12),
    }


def _bincount(keys: Any, weights: Any, count: int) -> Any:
    return cp.bincount(keys, weights=weights, minlength=count).astype(cp.float32)


def readonly_gpu_edge_counts(faces: np.ndarray, gpu_preference: str = "system"):
    """Count integer edge incidence for reporting only, with a CPU-fallback sentinel.

    Packed keys preserve the CPU lexicographic edge order exactly. No vertex
    coordinates, collapse decisions, or topology repairs use this result.
    """
    if cp is None or len(faces) < 262_144 or faces.dtype != np.int32:
        return None
    if faces.size and int(faces.min()) < 0:
        return None
    try:
        device_id = select_cuda_device(gpu_preference)
        with cp.cuda.Device(device_id):
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            if free_bytes < len(faces) * 192:
                return None
            device_faces = cp.asarray(faces)
            keys = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                low = cp.minimum(device_faces[:, a], device_faces[:, b]).astype(cp.uint64)
                high = cp.maximum(device_faces[:, a], device_faces[:, b]).astype(cp.uint64)
                keys.append((low << cp.uint64(32)) | high)
            _, counts = cp.unique(cp.concatenate(keys), return_counts=True)
            return cp.asnumpy(counts)
    except Exception:  # noqa: BLE001 - optional acceleration falls back on driver/OOM failures
        return None


def readonly_gpu_density_field(points, faces, gpu_preference="system", progress=None,
                               cancel_event=None):
    """Compute display-only density, leaving editable coordinates and QEM untouched.

    Float64 weighted accumulation follows the CPU chunk boundaries. GPU arithmetic
    can change final display channels by one level; this field is not a reduction input.
    """
    if cp is None or len(faces) < 262_144:
        return None
    try:
        with cp.cuda.Device(select_cuda_device(gpu_preference)):
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            if free_bytes < points.nbytes * 8 + faces.nbytes * 4 + len(points) * 96:
                return None
            p, f = cp.asarray(points), cp.asarray(faces)
            n = len(points)
            count = cp.bincount(f.ravel(), minlength=n).astype(cp.float32)
            area_sum = cp.zeros(n, cp.float32)
            neighbors = cp.zeros(n, cp.float32)

            def checkpoint(stage, done):
                cp.cuda.get_current_stream().synchronize()
                if cancel_event is not None and cancel_event.is_set():
                    return False
                if progress is not None:
                    progress(stage, done, len(faces))
                return True

            for start in range(0, len(faces), 250_000):
                if cancel_event is not None and cancel_event.is_set():
                    return None
                chunk = f[start:start + 250_000]
                a, b, c = (p[chunk[:, i]] for i in range(3))
                area = 0.5 * cp.linalg.norm(cp.cross(b - a, c - a), axis=1)
                area_sum += cp.bincount(chunk.ravel(), weights=cp.repeat(area, 3),
                                        minlength=n).astype(cp.float32)
                neighbors += cp.bincount(chunk.ravel(), minlength=n).astype(cp.float32)
                if not checkpoint("Measuring triangle density", start + len(chunk)):
                    return None
            valid = (count > 0) & (area_sum > 0)
            log = cp.zeros(n, cp.float32)
            log[valid] = cp.log(cp.maximum(count[valid] / area_sum[valid], cp.float32(1e-30)))
            neighbor_sum = cp.zeros(n, cp.float32)
            for start in range(0, len(faces), 250_000):
                if cancel_event is not None and cancel_event.is_set():
                    return None
                chunk = f[start:start + 250_000]
                mean = log[chunk].mean(axis=1)
                neighbor_sum += cp.bincount(chunk.ravel(), weights=cp.repeat(mean, 3),
                                            minlength=n).astype(cp.float32)
                if not checkpoint("Smoothing density regions", start + len(chunk)):
                    return None
            return cp.asnumpy((log + neighbor_sum) / (1.0 + neighbors)), cp.asnumpy(valid)
    except Exception:  # noqa: BLE001 - optional presentation acceleration has a CPU fallback
        # Optional presentation acceleration includes a CPU fallback for OOM/driver failures.
        return None


def adaptive_probe_projection(
    points: np.ndarray,
    faces: np.ndarray,
    target: int,
    gpu_preference: str = "system",
    *,
    density_adaptive: bool = False,
    density_floor: float = 1.0,
    density_ceiling: float = 1.5,
    density_center: float = 0.25,
    density_slope: float = 1.5,
    progress: Callable[[str, int, int], None] | None = None,
    protected_vertices: np.ndarray | None = None,
    source_statistics: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project supported scan-scale excursions onto locally fitted surface trajectories."""
    if not gpu_backend_available(gpu_preference):
        return np.ascontiguousarray(points), {
            "backend": "Fast QEM fallback",
            "seconds": 0.0,
            "reason": "CUDA adaptive-probe backend unavailable",
        }

    started = time.perf_counter()
    device_id = select_cuda_device(gpu_preference)
    stage = "Analyzing depth and density on GPU" if density_adaptive else "Analyzing surface on GPU"
    if progress is not None:
        progress(stage, 0, 6)
    physical = physical_probe_dimensions(points, faces, target, source_statistics)
    tip_radius = physical["tip_radius"]
    displacement_cap = tip_radius * 0.25
    pitches = [physical["minimum_pitch"] * factor for factor in (8.0, 4.0, 2.0, 1.0)]

    gp = cp.asarray(points)
    gf = cp.asarray(faces)
    triangles = gp[gf]
    cross = cp.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = cp.linalg.norm(cross, axis=1)
    face_normals = cross / cp.maximum(twice_area[:, None], cp.float32(1e-12))
    centers = cp.mean(triangles, axis=1)
    weights = twice_area * cp.float32(0.5)
    low = cp.min(gp, axis=0)
    high = cp.max(gp, axis=0)
    vertex_delta = cp.zeros_like(gp)
    vertex_weight = cp.zeros(len(points), dtype=cp.float32)
    unresolved = cp.ones(len(faces), dtype=cp.bool_)
    level_stats: list[dict[str, Any]] = []

    density_gain = None
    density_stats = None
    if density_adaptive:
        # Sampling density is measured per unit of incident surface area. Log space makes
        # multiplicative scan-density changes additive; median/MAD normalization prevents a
        # small oversampled region from compressing the useful range for the rest of the mesh.
        vertex_ids = gf.ravel()
        incident_count = cp.bincount(vertex_ids, minlength=len(points)).astype(cp.float32)
        incident_area = _bincount(
            vertex_ids,
            cp.repeat(weights, 3),
            len(points),
        )
        valid_density = incident_area > cp.float32(1e-12)
        log_density = cp.zeros(len(points), dtype=cp.float32)
        log_density[valid_density] = cp.log(
            cp.maximum(
                incident_count[valid_density] / incident_area[valid_density],
                cp.float32(1e-30),
            )
        )
        valid_values = log_density[valid_density]
        median = cp.median(valid_values)
        mad = cp.median(cp.abs(valid_values - median))
        robust_sigma = cp.maximum(cp.float32(1.4826) * mad, cp.float32(1e-6))
        robust_z = (log_density - median) / robust_sigma
        density_gain = cp.float32(density_floor) + cp.float32(density_ceiling - density_floor) / (
            cp.float32(1.0)
            + cp.exp(-cp.float32(density_slope) * (robust_z - cp.float32(density_center)))
        )
        density_gain[~valid_density] = cp.float32(density_floor)
        density_stats = {
            "median_log_density": float(median.get()),
            "robust_sigma": float(robust_sigma.get()),
            "gain_min": float(cp.min(density_gain).get()),
            "gain_mean": float(cp.mean(density_gain).get()),
            "gain_max": float(cp.max(density_gain).get()),
            "floor": float(density_floor),
            "ceiling": float(density_ceiling),
            "center_z": float(density_center),
            "slope": float(density_slope),
        }
    if progress is not None:
        progress(stage, 1, 6)

    for level_index, pitch in enumerate(pitches, start=2):
        dims = cp.maximum(cp.ceil((high - low) / pitch).astype(cp.int64) + 1, 1)
        cell = cp.floor((centers - low) / pitch).astype(cp.int64)
        raw_keys = cell[:, 0] + dims[0] * (cell[:, 1] + dims[1] * cell[:, 2])
        _occupied_keys, keys = cp.unique(raw_keys, return_inverse=True)
        cell_count = int(cp.max(keys).get()) + 1
        area_sum = _bincount(keys, weights, cell_count)
        weighted_normal = cp.stack(
            [_bincount(keys, weights * face_normals[:, axis], cell_count) for axis in range(3)],
            axis=1,
        )
        normal_length = cp.linalg.norm(weighted_normal, axis=1)
        mean_normal = weighted_normal / cp.maximum(normal_length[:, None], cp.float32(1e-12))
        coherence = normal_length / cp.maximum(area_sum, cp.float32(1e-12))
        mean_center = cp.stack(
            [_bincount(keys, weights * centers[:, axis], cell_count) for axis in range(3)],
            axis=1,
        ) / cp.maximum(area_sum[:, None], cp.float32(1e-12))
        local_normal = mean_normal[keys]
        local_center = mean_center[keys]
        residual = cp.abs(cp.sum(local_normal * (centers - local_center), axis=1))
        residual_mean = _bincount(keys, weights * residual, cell_count) / cp.maximum(
            area_sum, cp.float32(1e-12)
        )
        residual_sq = _bincount(keys, weights * residual * residual, cell_count) / cp.maximum(
            area_sum, cp.float32(1e-12)
        )
        residual_sigma = cp.sqrt(cp.maximum(residual_sq - residual_mean * residual_mean, 0.0))
        inlier = residual <= cp.maximum(
            residual_mean[keys] + cp.float32(2.5) * residual_sigma[keys],
            cp.float32(tip_radius),
        )
        inlier_weight = weights * inlier
        inlier_area = _bincount(keys, inlier_weight, cell_count)
        clipped_rms = cp.sqrt(
            _bincount(keys, inlier_weight * residual * residual, cell_count)
            / cp.maximum(inlier_area, cp.float32(1e-12))
        )
        maximum_angle = np.arctan2(tip_radius, pitch * 0.5)
        coherence_limit = float(np.cos(maximum_angle))
        valid_cell = (coherence >= cp.float32(coherence_limit)) & (
            clipped_rms <= cp.float32(tip_radius)
        )
        assigned = unresolved & valid_cell[keys]
        dust = (
            assigned
            & (residual > cp.float32(tip_radius * 0.5))
            & (residual <= cp.float32(tip_radius * 1.5))
        )
        flat_vertices = gf[dust].ravel()
        if int(cp.count_nonzero(dust).get()):
            selected_triangles = triangles[dust]
            selected_normal = local_normal[dust]
            selected_center = local_center[dust]
            selected_weight = weights[dust]
            signed = cp.sum(
                (selected_triangles - selected_center[:, None, :]) * selected_normal[:, None, :],
                axis=2,
            )
            repeated_weight = cp.repeat(selected_weight, 3)
            for axis in range(3):
                delta = -signed * selected_normal[:, None, axis]
                vertex_delta[:, axis] += _bincount(
                    flat_vertices, repeated_weight * delta.ravel(), len(points)
                )
            vertex_weight += _bincount(flat_vertices, repeated_weight, len(points))
        level_stats.append(
            {
                "pitch": float(pitch),
                "assigned_faces": int(cp.count_nonzero(assigned).get()),
                "probe_count": len(cp.unique(keys[assigned])),
                "projected_outlier_faces": int(cp.count_nonzero(dust).get()),
            }
        )
        unresolved &= ~assigned
        if progress is not None:
            progress(stage, level_index, 6)

    delta = vertex_delta / cp.maximum(vertex_weight[:, None], cp.float32(1e-12))
    if density_gain is not None:
        # Sparse regions retain their measured positions. Dense regions receive progressively
        # more of the depth-probe projection because redundant samples provide stronger local
        # evidence for the fitted surface trajectory.
        delta *= density_gain[:, None]
    length = cp.linalg.norm(delta, axis=1)
    delta *= cp.minimum(1.0, cp.float32(displacement_cap) / cp.maximum(length, cp.float32(1e-12)))[
        :, None
    ]
    projected = gp + delta
    if protected_vertices is not None and np.any(protected_vertices):
        protected = cp.asarray(protected_vertices, dtype=cp.bool_)
        projected[protected] = gp[protected]
    cp.cuda.Stream.null.synchronize()
    if progress is not None:
        progress(stage, 6, 6)
    report = {
        "backend": "CUDA",
        "device_id": device_id,
        "device_preference": gpu_preference,
        "seconds": time.perf_counter() - started,
        "probe_geometry": {
            **physical,
            "maximum_pitch": pitches[0],
            "displacement_cap": displacement_cap,
        },
        "levels": level_stats,
        "unresolved_faces": int(cp.count_nonzero(unresolved).get()),
        "moved_vertex_share": float(cp.mean(length > 1e-6).get()),
    }
    if density_stats is not None:
        report["density_adaptation"] = density_stats
    _mark_gpu_cache_ready(gpu_preference)
    return cp.asnumpy(projected).astype(np.float32, copy=False), report


def simplify_depth_adaptive_qem(
    points: np.ndarray,
    faces: np.ndarray,
    target: int,
    aggressiveness: float = 8.0,
    gpu_preference: str = "system",
    progress: Callable[[str, int, int], None] | None = None,
    protected_vertices: np.ndarray | None = None,
    source_statistics: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run Depth Adaptive QEM, or boundary-preserving Fast QEM without CUDA."""
    if progress is not None:
        progress("Analyzing surface on GPU", 0, 6)
    try:
        projected, report = adaptive_probe_projection(
            points,
            faces,
            target,
            gpu_preference,
            progress=progress,
            protected_vertices=protected_vertices,
            source_statistics=source_statistics,
        )
    except Exception as error:  # noqa: BLE001 - unavailable GPU backends fall back safely
        if os.environ.get("MESHMILL_GPU_DIAGNOSTICS"):
            print(f"MeshMill GPU analysis failed: {error!r}", file=sys.stderr)
        projected = np.ascontiguousarray(points)
        report = {
            "backend": "Fast QEM fallback",
            "seconds": 0.0,
            "reason": f"GPU adaptive-probe analysis failed: {type(error).__name__}",
        }
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
    if progress is not None:
        progress("Reducing mesh topology", 0, len(faces) - target)
    qem_started = time.perf_counter()
    output_points, output_faces = simplify_with_progress(
        projected,
        faces,
        target_count=target,
        agg=aggressiveness,
        verbose=False,
        preserve_border=True,
        progress=progress,
    )
    if progress is not None:
        progress("Reducing mesh topology", len(faces) - len(output_faces), len(faces) - target)
    report["qem_seconds"] = time.perf_counter() - qem_started
    return output_points, output_faces, report


def simplify_depth_density_adaptive_qem(
    points: np.ndarray,
    faces: np.ndarray,
    target: int,
    aggressiveness: float = 8.0,
    gpu_preference: str = "system",
    progress: Callable[[str, int, int], None] | None = None,
    protected_vertices: np.ndarray | None = None,
    source_statistics: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Combine density-adaptive depth probes, Fast QEM, and bounded finishing."""
    if progress is not None:
        progress("Analyzing depth and density on GPU", 0, 6)
    try:
        projected, report = adaptive_probe_projection(
            points,
            faces,
            target,
            gpu_preference,
            density_adaptive=True,
            progress=progress,
            protected_vertices=protected_vertices,
            source_statistics=source_statistics,
        )
        if report.get("backend") == "Fast QEM fallback":
            report["backend"] = "Fast QEM fallback"
    except Exception as error:  # noqa: BLE001 - unavailable GPU backends fall back safely
        if os.environ.get("MESHMILL_GPU_DIAGNOSTICS"):
            print(f"MeshMill GPU analysis failed: {error!r}", file=sys.stderr)
        projected = np.ascontiguousarray(points)
        report = {
            "backend": "Density balanced fallback",
            "seconds": 0.0,
            "reason": f"GPU adaptive-probe analysis failed: {type(error).__name__}",
        }
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
    if progress is not None:
        progress("Reducing mesh topology", 0, len(faces) - target)
    qem_started = time.perf_counter()
    output_points, output_faces = simplify_with_progress(
        projected,
        faces,
        target_count=target,
        agg=aggressiveness,
        verbose=False,
        preserve_border=True,
        progress=progress,
    )
    if progress is not None:
        progress("Reducing mesh topology", len(faces) - len(output_faces), len(faces) - target)
    report["qem_seconds"] = time.perf_counter() - qem_started
    if progress is not None:
        progress("Finishing topology and surface", 0, 2)
    finishing_started = time.perf_counter()
    output_faces, topology_report = _remove_nonmanifold_surplus(output_points, output_faces)
    if progress is not None:
        progress("Finishing topology and surface", 1, 2)
    output_points, surface_report = _match_surface_area(points, faces, output_points, output_faces)
    if progress is not None:
        progress("Finishing topology and surface", 2, 2)
    report["finishing_seconds"] = time.perf_counter() - finishing_started
    report["topology_finishing"] = topology_report
    report["surface_finishing"] = surface_report
    report["requested_aggressiveness"] = aggressiveness
    return output_points, output_faces, report


def _arrays_polydata(points: np.ndarray, faces: np.ndarray) -> vtkPolyData:
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    offsets = np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64)
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(faces, dtype=np.int64).ravel(), deep=True),
    )
    poly = vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(cells)
    return poly


def _polydata_arrays(poly: vtkPolyData) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float64)
    cells = poly.GetPolys()
    offsets = np.asarray(vtk_to_numpy(cells.GetOffsetsArray()))
    connectivity = np.asarray(vtk_to_numpy(cells.GetConnectivityArray()))
    if len(offsets) != cells.GetNumberOfCells() + 1 or np.any(np.diff(offsets) != 3):
        raise ValueError("Weighted QEM produced non-triangle cells.")
    return points, np.asarray(connectivity.reshape(-1, 3), dtype=np.int32)


def _robust_density(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    incident_count = np.bincount(faces.ravel(), minlength=len(points)).astype(np.float32)
    triangles = points[faces]
    area = np.float32(0.5) * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    incident_area = np.bincount(
        faces.ravel(), weights=np.repeat(area, 3), minlength=len(points)
    ).astype(np.float32)
    valid = incident_area > np.float32(1e-12)
    density = np.zeros(len(points), dtype=np.float32)
    density[valid] = np.log(
        np.maximum(incident_count[valid] / incident_area[valid], np.float32(1e-30))
    )
    values = density[valid]
    median = np.median(values) if len(values) else 0.0
    mad = np.median(np.abs(values - median)) if len(values) else 0.0
    sigma = max(float(np.float32(1.4826) * mad), 1e-6)
    density = np.clip((density - median) / sigma, -4.0, 4.0)
    density[~valid] = -4.0
    return np.ascontiguousarray((density + 4.0) / 8.0, dtype=np.float32)


def _area_weighted_vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = points[faces]
    face_cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
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


def _weighted_qem(
    points: np.ndarray,
    faces: np.ndarray,
    density: np.ndarray,
    normals: np.ndarray,
    target: int,
) -> tuple[np.ndarray, np.ndarray]:
    source = _arrays_polydata(points, faces)
    density_values = numpy_to_vtk(density, deep=True)
    density_values.SetName("sampling_density")
    source.GetPointData().SetScalars(density_values)
    normal_values = numpy_to_vtk(normals, deep=True)
    normal_values.SetName("surface_normals")
    source.GetPointData().SetNormals(normal_values)
    decimator = vtkQuadricDecimation()
    decimator.SetInputData(source)
    decimator.SetTargetReduction(1.0 - target / len(faces))
    decimator.AttributeErrorMetricOn()
    decimator.ScalarsAttributeOn()
    decimator.SetScalarsWeight(0.15)
    decimator.NormalsAttributeOn()
    decimator.SetNormalsWeight(0.1)
    decimator.VolumePreservationOff()
    decimator.Update()
    return _polydata_arrays(decimator.GetOutput())


def _surface_area(points: np.ndarray, faces: np.ndarray) -> float:
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


def _constrain_extrema(
    source_points: np.ndarray, output_points: np.ndarray
) -> tuple[np.ndarray, float]:
    constrained = np.array(output_points, copy=True)
    low = source_points.min(axis=0)
    high = source_points.max(axis=0)
    maximum_change = 0.0
    for axis in range(3):
        low_index = int(np.argmin(constrained[:, axis]))
        high_index = int(np.argmax(constrained[:, axis]))
        maximum_change = max(
            maximum_change,
            abs(float(constrained[low_index, axis] - low[axis])),
            abs(float(constrained[high_index, axis] - high[axis])),
        )
        constrained[low_index, axis] = low[axis]
        constrained[high_index, axis] = high[axis]
    return constrained, maximum_change


def _match_surface_area(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    output_points: np.ndarray,
    output_faces: np.ndarray,
    maximum_alpha: float = 0.08,
) -> tuple[np.ndarray, dict[str, float]]:
    source_area = _surface_area(source_points, source_faces)
    initial_area = _surface_area(output_points, output_faces)
    if initial_area <= source_area:
        constrained, extrema_change = _constrain_extrema(source_points, output_points)
        return constrained, {
            "alpha": 0.0,
            "initial_area_ratio": initial_area / source_area,
            "final_area_ratio": _surface_area(constrained, output_faces) / source_area,
            "extrema_maximum_change": extrema_change,
        }
    face_mean = output_points[output_faces].mean(axis=1)
    vertex_ids = output_faces.ravel()
    repeated = np.repeat(face_mean, 3, axis=0)
    neighbor_sum = np.stack(
        [
            np.bincount(vertex_ids, weights=repeated[:, axis], minlength=len(output_points))
            for axis in range(3)
        ],
        axis=1,
    )
    count = np.bincount(vertex_ids, minlength=len(output_points))
    target = np.array(output_points, copy=True, dtype=np.float64)
    valid = count > 0
    target[valid] = neighbor_sum[valid] / count[valid, None]
    delta = target - output_points
    low_alpha = 0.0
    high_alpha = maximum_alpha
    constrained, _ = _constrain_extrema(source_points, output_points + high_alpha * delta)
    if _surface_area(constrained, output_faces) > source_area:
        alpha = high_alpha
    else:
        for _ in range(12):
            middle = 0.5 * (low_alpha + high_alpha)
            candidate, _ = _constrain_extrema(source_points, output_points + middle * delta)
            if _surface_area(candidate, output_faces) > source_area:
                low_alpha = middle
            else:
                high_alpha = middle
        alpha = 0.5 * (low_alpha + high_alpha)
        constrained, _ = _constrain_extrema(source_points, output_points + alpha * delta)
    constrained, extrema_change = _constrain_extrema(source_points, constrained)
    return constrained, {
        "alpha": alpha,
        "initial_area_ratio": initial_area / source_area,
        "final_area_ratio": _surface_area(constrained, output_faces) / source_area,
        "extrema_maximum_change": extrema_change,
    }


def _remove_nonmanifold_surplus(
    points: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    face_range = np.arange(len(faces), dtype=np.int64)
    face_ids = np.concatenate((face_range, face_range, face_range))
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    sorted_face_ids = face_ids[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_edges, axis=0), axis=1)) + 1]
    ends = np.r_[starts[1:], len(sorted_edges)]
    groups = np.flatnonzero((ends - starts) > 2)
    if not len(groups):
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
    for group in groups:
        members = np.unique(sorted_face_ids[starts[group] : ends[group]])
        active = members[~remove[members]]
        surplus = max(0, len(active) - 2)
        if surplus:
            remove[active[np.argsort(areas[active])[:surplus]]] = True
    return np.ascontiguousarray(faces[~remove]), {
        "nonmanifold_edges_before": len(groups),
        "removed_faces": int(np.count_nonzero(remove)),
    }
