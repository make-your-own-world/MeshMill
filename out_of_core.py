"""Bounded-memory indexing primitives for binary STL meshes.

The module deliberately has no Qt, VTK, or GPU dependency. It provides the stable storage and
work-unit boundary that rendering and optional GPU compute backends can share.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import fast_simplification
import numpy as np

from worker_lifecycle import OwnedProcessPool, completed_futures

INDEX_VERSION = 1
STL_RECORD_BYTES = 50
STL_HEADER_BYTES = 84
MAX_OPEN_TILE_FILES = 64
ESTIMATED_OPTIMIZATION_BYTES_PER_TRIANGLE = 320
TRIANGLE_RECORD_DTYPE = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)


class IndexCancelled(Exception):
    """Raised when index creation is cancelled without publishing a partial index."""


@dataclass(frozen=True)
class TileRecord:
    level: int
    morton: int
    triangle_count: int
    relative_path: str


@dataclass(frozen=True)
class IndexManifest:
    index_version: int
    source_size: int
    source_mtime_ns: int
    source_fingerprint: str
    triangle_count: int
    bounds: tuple[float, float, float, float, float, float]
    max_depth: int
    target_triangles_per_tile: int
    tiles: tuple[TileRecord, ...]


def binary_stl_count(path: Path) -> int:
    size = path.stat().st_size
    if size < STL_HEADER_BYTES:
        raise ValueError("The file is too small to be a binary STL.")
    with path.open("rb") as stream:
        header = stream.read(STL_HEADER_BYTES)
    count = struct.unpack_from("<I", header, 80)[0]
    if STL_HEADER_BYTES + STL_RECORD_BYTES * count != size:
        raise ValueError("Out-of-core indexing currently requires a valid binary STL.")
    return count


def source_fingerprint(path: Path, sample_bytes: int = 1 << 20) -> str:
    """Hash stable source metadata and bounded samples without reading the complete file."""
    stat = path.stat()
    digest = sha256()
    digest.update(struct.pack("<QQ", stat.st_size, stat.st_mtime_ns))
    with path.open("rb") as stream:
        digest.update(stream.read(sample_bytes))
        if stat.st_size > sample_bytes:
            stream.seek(max(0, stat.st_size - sample_bytes))
            digest.update(stream.read(sample_bytes))
    return digest.hexdigest()


def file_sha256(path: Path, block_bytes: int = 8 * 1024**2) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def default_index_directory(source: Path) -> Path:
    """Return a private, stable cache path without exposing the source filename."""
    cache_root = (
        Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "MeshMill" / "indexes"
    )
    normalized_source = os.path.normcase(os.fspath(source.resolve())).encode("utf-8")
    source_key = sha256(normalized_source).hexdigest()[:24]
    return cache_root / f"{source_key}.meshmill-index"


def _require_disk_space(directory: Path, required_bytes: int, operation: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < required_bytes:
        raise OSError(
            f"{operation} needs about {required_bytes / 1024**3:.2f} GB of free space, "
            f"but only {free / 1024**3:.2f} GB is available."
        )


def _cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise IndexCancelled


def _progress(callback: Callable[[str, int, int], None] | None, stage: str, done: int, total: int):
    if callback is not None:
        callback(stage, done, total)


def _part1by2(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.uint64, copy=False) & np.uint64(0x1FFFFF)
    value = (value | (value << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    value = (value | (value << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    value = (value | (value << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    value = (value | (value << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    return (value | (value << np.uint64(2))) & np.uint64(0x1249249249249249)


def morton3(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return _part1by2(x) | (_part1by2(y) << np.uint64(1)) | (_part1by2(z) << np.uint64(2))


def _bounds(records: np.memmap, block_triangles: int, cancel_event, progress_callback):
    low = np.full(3, np.inf, dtype=np.float64)
    high = np.full(3, -np.inf, dtype=np.float64)
    total = len(records)
    for start in range(0, total, block_triangles):
        _cancelled(cancel_event)
        vertices = records[start : start + block_triangles]["vertices"]
        low = np.minimum(low, vertices.min(axis=(0, 1)))
        high = np.maximum(high, vertices.max(axis=(0, 1)))
        _progress(progress_callback, "bounds", min(total, start + len(vertices)), total)
    if not np.all(np.isfinite(low)) or np.any(high < low):
        raise ValueError("The STL contains invalid coordinate bounds.")
    return low, high


def _analysis_ranges(total: int, block_triangles: int) -> list[tuple[int, int]]:
    return [
        (start, min(total, start + block_triangles))
        for start in range(0, total, block_triangles)
    ]


def _worker_bounds(source: str, triangle_count: int, start: int, stop: int):
    records = np.memmap(
        source,
        dtype=TRIANGLE_RECORD_DTYPE,
        mode="r",
        offset=STL_HEADER_BYTES,
        shape=(triangle_count,),
    )
    vertices = records[start:stop]["vertices"]
    return start, stop, vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))


def _worker_tile_counts(
    source: str,
    triangle_count: int,
    start: int,
    stop: int,
    low: np.ndarray,
    extent: np.ndarray,
    grid: int,
):
    records = np.memmap(
        source,
        dtype=TRIANGLE_RECORD_DTYPE,
        mode="r",
        offset=STL_HEADER_BYTES,
        shape=(triangle_count,),
    )
    centroids = records[start:stop]["vertices"].mean(axis=1, dtype=np.float64)
    cells = np.floor((centroids - low) / extent * grid).astype(np.int64)
    np.clip(cells, 0, grid - 1, out=cells)
    codes, counts = np.unique(
        morton3(cells[:, 0], cells[:, 1], cells[:, 2]), return_counts=True
    )
    return start, stop, tuple(zip(codes.tolist(), counts.tolist()))


def _parallel_bounds(
    executor: ProcessPoolExecutor,
    source: Path,
    triangle_count: int,
    ranges: list[tuple[int, int]],
    cancel_event,
    progress_callback,
):
    futures = [
        executor.submit(_worker_bounds, os.fspath(source), triangle_count, start, stop)
        for start, stop in ranges
    ]
    low = np.full(3, np.inf, dtype=np.float64)
    high = np.full(3, -np.inf, dtype=np.float64)
    done = 0
    for future in completed_futures(futures, lambda: _cancelled(cancel_event)):
        _cancelled(cancel_event)
        start, stop, chunk_low, chunk_high = future.result()
        low = np.minimum(low, chunk_low)
        high = np.maximum(high, chunk_high)
        done += stop - start
        _progress(progress_callback, "bounds", done, triangle_count)
    if not np.all(np.isfinite(low)) or np.any(high < low):
        raise ValueError("The STL contains invalid coordinate bounds.")
    return low, high


def _parallel_tile_counts(
    executor: ProcessPoolExecutor,
    source: Path,
    triangle_count: int,
    ranges: list[tuple[int, int]],
    low: np.ndarray,
    extent: np.ndarray,
    grid: int,
    cancel_event,
    progress_callback,
) -> dict[int, int]:
    futures = [
        executor.submit(
            _worker_tile_counts,
            os.fspath(source),
            triangle_count,
            start,
            stop,
            low,
            extent,
            grid,
        )
        for start, stop in ranges
    ]
    merged: dict[int, int] = {}
    done = 0
    for future in completed_futures(futures, lambda: _cancelled(cancel_event)):
        _cancelled(cancel_event)
        start, stop, result = future.result()
        for code, count in result:
            merged[int(code)] = merged.get(int(code), 0) + int(count)
        done += stop - start
        _progress(progress_callback, "planning", done, triangle_count)
    return merged


def _parallel_plan_adaptive_tiles(
    executor: ProcessPoolExecutor,
    source: Path,
    triangle_count: int,
    ranges: list[tuple[int, int]],
    low: np.ndarray,
    extent: np.ndarray,
    target: int,
    cancel_event,
    progress_callback,
) -> tuple[int, dict[tuple[int, int], int]]:
    if triangle_count <= target:
        return 0, {(0, 0): triangle_count}
    leaves: dict[tuple[int, int], int] = {}
    split_parents = {0}
    for level in range(1, 22):
        counts = _parallel_tile_counts(
            executor,
            source,
            triangle_count,
            ranges,
            low,
            extent,
            1 << level,
            cancel_event,
            progress_callback,
        )
        next_split: set[int] = set()
        for code, count in counts.items():
            if code >> 3 not in split_parents:
                continue
            if count > target:
                next_split.add(code)
            else:
                leaves[(level, code)] = count
        if not next_split:
            return level, leaves
        split_parents = next_split
    raise ValueError("The mesh cannot be partitioned within the configured tile target.")


def _tile_counts(
    records: np.memmap,
    low: np.ndarray,
    extent: np.ndarray,
    grid: int,
    block_triangles: int,
    cancel_event,
    progress_callback,
) -> dict[int, int]:
    counts: dict[int, int] = {}
    total = len(records)
    for start in range(0, total, block_triangles):
        _cancelled(cancel_event)
        chunk = records[start : start + block_triangles]
        centroids = chunk["vertices"].mean(axis=1, dtype=np.float64)
        cells = np.floor((centroids - low) / extent * grid).astype(np.int64)
        np.clip(cells, 0, grid - 1, out=cells)
        codes, chunk_counts = np.unique(
            morton3(cells[:, 0], cells[:, 1], cells[:, 2]), return_counts=True
        )
        for code, count in zip(codes, chunk_counts, strict=True):
            key = int(code)
            counts[key] = counts.get(key, 0) + int(count)
        _progress(progress_callback, "planning", min(total, start + len(chunk)), total)
    return counts


def _plan_adaptive_tiles(
    records: np.memmap,
    low: np.ndarray,
    extent: np.ndarray,
    target: int,
    block_triangles: int,
    cancel_event,
    progress_callback,
) -> tuple[int, dict[tuple[int, int], int]]:
    """Subdivide only overflowing octree leaves and return their measured counts."""
    if len(records) <= target:
        return 0, {(0, 0): len(records)}
    leaves: dict[tuple[int, int], int] = {}
    split_parents = {0}
    for level in range(1, 22):
        grid = 1 << level
        counts = _tile_counts(
            records, low, extent, grid, block_triangles, cancel_event, progress_callback
        )
        next_split: set[int] = set()
        for code, count in counts.items():
            parent = code >> 3
            if parent not in split_parents:
                continue
            if count > target:
                next_split.add(code)
            else:
                leaves[(level, code)] = count
        if not next_split:
            return level, leaves
        split_parents = next_split
    raise ValueError("The mesh cannot be partitioned within the configured tile target.")


def _leaf_assignments(
    centroids: np.ndarray,
    low: np.ndarray,
    extent: np.ndarray,
    max_depth: int,
    leaves: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    assigned_levels = np.full(len(centroids), -1, dtype=np.int16)
    assigned_codes = np.zeros(len(centroids), dtype=np.uint64)
    for level in range(max_depth + 1):
        level_codes = np.fromiter(
            (code for leaf_level, code in leaves if leaf_level == level), dtype=np.uint64
        )
        if not len(level_codes):
            continue
        grid = 1 << level
        cells = np.floor((centroids - low) / extent * grid).astype(np.int64)
        np.clip(cells, 0, grid - 1, out=cells)
        codes = morton3(cells[:, 0], cells[:, 1], cells[:, 2])
        selected = (assigned_levels < 0) & np.isin(codes, level_codes)
        assigned_levels[selected] = level
        assigned_codes[selected] = codes[selected]
    if np.any(assigned_levels < 0):
        raise RuntimeError("Adaptive tile planning left triangles without a leaf owner.")
    return assigned_levels, assigned_codes


def build_index(
    source: Path,
    index_directory: Path,
    *,
    target_triangles_per_tile: int = 262_144,
    block_triangles: int = 262_144,
    analysis_workers: int | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> IndexManifest:
    """Create a deterministic spatial index without holding the full mesh in RAM.

    Triangle records are assigned by centroid. Boundary overlap belongs to later operation-specific
    work-unit construction, which prevents the persistent index from duplicating source geometry.
    """
    source = source.resolve()
    triangle_count = binary_stl_count(source)
    _require_disk_space(
        index_directory.parent,
        source.stat().st_size + max(64 * 1024**2, source.stat().st_size // 20),
        "Tiled indexing",
    )
    target_triangles_per_tile = max(65_536, int(target_triangles_per_tile))
    block_triangles = max(16_384, int(block_triangles))
    records = np.memmap(
        source, dtype=TRIANGLE_RECORD_DTYPE, mode="r", offset=STL_HEADER_BYTES,
        shape=(triangle_count,),
    )
    block_count = max(1, (triangle_count + block_triangles - 1) // block_triangles)
    if analysis_workers is None:
        analysis_workers = min(8, max(1, (os.cpu_count() or 2) // 2), block_count)
    analysis_workers = max(1, min(int(analysis_workers), block_count))
    if analysis_workers > 1 and block_count > 1:
        ranges = _analysis_ranges(triangle_count, block_triangles)
        with OwnedProcessPool(max_workers=analysis_workers) as executor:
            low, high = _parallel_bounds(
                executor,
                source,
                triangle_count,
                ranges,
                cancel_event,
                progress_callback,
            )
            extent = np.maximum(high - low, np.finfo(np.float64).eps)
            max_depth, planned_counts = _parallel_plan_adaptive_tiles(
                executor,
                source,
                triangle_count,
                ranges,
                low,
                extent,
                target_triangles_per_tile,
                cancel_event,
                progress_callback,
            )
    else:
        low, high = _bounds(records, block_triangles, cancel_event, progress_callback)
        extent = np.maximum(high - low, np.finfo(np.float64).eps)
        max_depth, planned_counts = _plan_adaptive_tiles(
            records,
            low,
            extent,
            target_triangles_per_tile,
            block_triangles,
            cancel_event,
            progress_callback,
        )

    staging = index_directory.with_name(index_directory.name + ".building")
    staging.mkdir(parents=True, exist_ok=True)
    for stale in staging.glob("tile-*.bin"):
        stale.unlink()

    counts: dict[tuple[int, int], int] = {}
    handles: OrderedDict[tuple[int, int], object] = OrderedDict()
    try:
        for start in range(0, triangle_count, block_triangles):
            _cancelled(cancel_event)
            chunk = records[start : start + block_triangles]
            centroids = chunk["vertices"].mean(axis=1, dtype=np.float64)
            levels, codes = _leaf_assignments(
                centroids, low, extent, max_depth, planned_counts
            )
            keys = (levels.astype(np.uint64) << np.uint64(56)) | codes
            order = np.argsort(keys, kind="stable")
            sorted_keys = keys[order]
            boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
            for group in np.split(order, boundaries):
                level = int(levels[group[0]])
                code = int(codes[group[0]])
                key = (level, code)
                handle = handles.get(key)
                if handle is None:
                    if len(handles) >= MAX_OPEN_TILE_FILES:
                        _old_key, old_handle = handles.popitem(last=False)
                        old_handle.flush()
                        old_handle.close()
                    handle = (staging / f"tile-L{level:02d}-{code:016x}.bin").open("ab")
                    handles[key] = handle
                else:
                    handles.move_to_end(key)
                payload = np.ascontiguousarray(chunk[group]).view(np.uint8)
                handle.write(payload.tobytes())
                counts[key] = counts.get(key, 0) + len(group)
            _progress(
                progress_callback,
                "partition",
                min(triangle_count, start + len(chunk)),
                triangle_count,
            )
    except Exception:
        for handle in handles.values():
            handle.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        del records
    for handle in handles.values():
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
    if counts != planned_counts:
        raise RuntimeError("Spatial index counts changed between planning and partitioning.")

    stat = source.stat()
    tiles = tuple(
        TileRecord(level, code, count, f"tile-L{level:02d}-{code:016x}.bin")
        for (level, code), count in sorted(counts.items())
    )
    manifest = IndexManifest(
        INDEX_VERSION,
        stat.st_size,
        stat.st_mtime_ns,
        source_fingerprint(source),
        triangle_count,
        (
            float(low[0]), float(high[0]),
            float(low[1]), float(high[1]),
            float(low[2]), float(high[2]),
        ),
        max_depth,
        target_triangles_per_tile,
        tiles,
    )
    manifest_payload = asdict(manifest)
    manifest_payload["tiles"] = [asdict(tile) for tile in tiles]
    temporary_manifest = staging / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, staging / "manifest.json")
    backup = index_directory.with_name(index_directory.name + ".previous")
    had_previous = index_directory.exists()
    if had_previous:
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(index_directory, backup)
    try:
        os.replace(staging, index_directory)
    except Exception:
        if had_previous and backup.exists() and not index_directory.exists():
            os.replace(backup, index_directory)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)
    _progress(progress_callback, "complete", triangle_count, triangle_count)
    return manifest


def load_manifest(index_directory: Path) -> IndexManifest:
    payload = json.loads((index_directory / "manifest.json").read_text(encoding="utf-8"))
    tiles = tuple(TileRecord(**tile) for tile in payload.pop("tiles"))
    payload["bounds"] = tuple(payload["bounds"])
    return IndexManifest(tiles=tiles, **payload)


def index_matches(source: Path, manifest: IndexManifest) -> bool:
    stat = source.stat()
    return (
        manifest.index_version == INDEX_VERSION
        and manifest.source_size == stat.st_size
        and manifest.source_mtime_ns == stat.st_mtime_ns
        and manifest.source_fingerprint == source_fingerprint(source)
    )


def load_tile_mesh(index_directory: Path, tile: TileRecord) -> tuple[np.ndarray, np.ndarray]:
    """Load one indexed STL tile and weld identical source coordinates within that tile."""
    records = np.memmap(
        index_directory / tile.relative_path,
        dtype=TRIANGLE_RECORD_DTYPE,
        mode="r",
        shape=(tile.triangle_count,),
    )
    triangle_vertices = np.ascontiguousarray(records["vertices"].reshape(-1, 3))
    del records
    points, inverse = np.unique(triangle_vertices, axis=0, return_inverse=True)
    faces = np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int32)
    return np.ascontiguousarray(points, dtype=np.float64), faces


def boundary_vertex_mask(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    """Return vertices on edges owned by only one triangle in the current tile."""
    edges = np.sort(
        np.vstack((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)])), axis=1
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    mask = np.zeros(vertex_count, dtype=bool)
    mask[np.unique(unique_edges[counts == 1])] = True
    return mask


def missing_protected_vertices(
    source_points: np.ndarray, protected: np.ndarray, output_points: np.ndarray
) -> int:
    """Count protected float32 coordinates absent from a tile result."""
    source = np.ascontiguousarray(source_points[protected], dtype=np.float32)
    output = np.ascontiguousarray(output_points, dtype=np.float32)
    if not len(source):
        return 0
    coordinate_key = np.dtype((np.void, source.dtype.itemsize * 3))
    source_keys = source.view(coordinate_key).ravel()
    output_keys = output.view(coordinate_key).ravel()
    return int(np.count_nonzero(~np.isin(source_keys, output_keys)))


def _write_binary_stl(
    path: Path,
    tile_results: list[tuple[Path, int]],
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> None:
    triangle_count = sum(count for _path, count in tile_results)
    temporary = path.with_name(path.name + ".partial")
    try:
        with temporary.open("wb") as stream:
            header = b"MeshMill tiled optimization"
            stream.write(header.ljust(80, b"\0"))
            stream.write(struct.pack("<I", triangle_count))
            for index, (result_path, expected_count) in enumerate(tile_results, start=1):
                _cancelled(cancel_event)
                with np.load(result_path, allow_pickle=False) as payload:
                    points = payload["points"]
                    faces = payload["faces"]
                if len(faces) != expected_count:
                    raise RuntimeError("A tiled result changed after validation.")
                triangles = points[faces].astype("<f4", copy=False)
                cross = np.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                )
                lengths = np.linalg.norm(cross, axis=1)
                normals = np.zeros_like(cross, dtype="<f4")
                valid = lengths > 1e-20
                normals[valid] = cross[valid] / lengths[valid, None]
                records = np.empty(len(faces), dtype=TRIANGLE_RECORD_DTYPE)
                records["normal"] = normals
                records["vertices"] = triangles
                records["attribute"] = 0
                stream.write(records.tobytes())
                _progress(progress_callback, "assembling", index, len(tile_results))
            stream.flush()
            os.fsync(stream.fileno())
        _cancelled(cancel_event)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def allocate_tile_targets(tiles: tuple[TileRecord, ...], target: int) -> list[int]:
    """Allocate an exact global target deterministically with a four-face tile floor."""
    minimums = np.asarray([min(4, tile.triangle_count) for tile in tiles], dtype=np.int64)
    minimum_total = int(minimums.sum())
    if target < minimum_total:
        raise ValueError(
            f"The requested target is below the tiled minimum of {minimum_total:,} triangles."
        )
    counts = np.asarray([tile.triangle_count for tile in tiles], dtype=np.int64)
    capacity = counts - minimums
    remaining = target - minimum_total
    if remaining <= 0 or int(capacity.sum()) == 0:
        return minimums.tolist()
    shares = remaining * capacity.astype(np.float64) / float(capacity.sum())
    additions = np.minimum(capacity, np.floor(shares).astype(np.int64))
    unresolved = remaining - int(additions.sum())
    fractions = shares - additions
    order = np.lexsort((np.arange(len(tiles)), -fractions))
    for index in order:
        if unresolved <= 0:
            break
        available = int(capacity[index] - additions[index])
        if available <= 0:
            continue
        grant = min(available, unresolved)
        additions[index] += grant
        unresolved -= grant
    allocated = minimums + additions
    if int(allocated.sum()) != target or np.any(allocated > counts):
        raise RuntimeError("Tiled target allocation failed its count contract.")
    return allocated.tolist()


def _tile_result_count(path: Path, source_count: int) -> int | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            points = payload["points"]
            faces = payload["faces"]
            checksum = str(payload["checksum"].item())
        digest = sha256(points.tobytes() + faces.tobytes()).hexdigest()
        valid = (
            1 <= len(faces) <= source_count
            and faces.ndim == 2
            and faces.shape[1] == 3
            and len(points) > 0
            and int(faces.max(initial=-1)) < len(points)
            and checksum == digest
        )
        return len(faces) if valid else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _tile_result_valid(path: Path, source_count: int) -> bool:
    return _tile_result_count(path, source_count) is not None


def _save_tile_result(path: Path, points: np.ndarray, faces: np.ndarray) -> None:
    points = np.ascontiguousarray(points, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    checksum = sha256(points.tobytes() + faces.tobytes()).hexdigest()
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.savez(stream, points=points, faces=faces, checksum=np.asarray(checksum))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_run_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _optimize_tile(
    index_directory: str,
    tile: TileRecord,
    tile_number: int,
    tile_target: int,
    run_directory: str,
    aggressiveness: float,
    algorithm: str,
    gpu_preference: str,
) -> tuple[int, int, int, str]:
    index_path = Path(index_directory)
    result_path = Path(run_directory) / f"tile-{tile_number:08d}.npz"
    points, faces = load_tile_mesh(index_path, tile)
    protected_vertices = boundary_vertex_mask(len(points), faces)
    if tile_target < len(faces):
        if algorithm.startswith("Depth + Density Adaptive QEM"):
            from depth_adaptive_qem import simplify_depth_density_adaptive_qem

            output_points, output_faces, _report = simplify_depth_density_adaptive_qem(
                points,
                faces,
                tile_target,
                aggressiveness,
                gpu_preference,
                protected_vertices=protected_vertices,
            )
        elif algorithm.startswith("Depth Adaptive QEM"):
            from depth_adaptive_qem import simplify_depth_adaptive_qem

            output_points, output_faces, _report = simplify_depth_adaptive_qem(
                points,
                faces,
                tile_target,
                aggressiveness,
                gpu_preference,
                protected_vertices=protected_vertices,
            )
        else:
            effective_aggressiveness = (
                min(5.0, max(0.0, aggressiveness * 0.70))
                if algorithm == "Density balanced"
                else aggressiveness
            )
            output_points, output_faces = fast_simplification.simplify(
                points,
                faces,
                target_count=tile_target,
                agg=effective_aggressiveness,
                verbose=False,
                preserve_border=True,
            )
    else:
        output_points, output_faces = points, faces
    if not 1 <= len(output_faces) <= tile.triangle_count:
        raise RuntimeError(
            f"Tile {tile_number} produced {len(output_faces):,} triangles from "
            f"{tile.triangle_count:,}."
        )
    missing_boundary_vertices = missing_protected_vertices(
        points, protected_vertices, output_points
    )
    if missing_boundary_vertices:
        raise RuntimeError(
            f"Tile {tile_number} changed {missing_boundary_vertices} protected boundary "
            "vertices. The tiled result was not assembled."
        )
    _save_tile_result(result_path, output_points, output_faces)
    if not _tile_result_valid(result_path, tile.triangle_count):
        raise RuntimeError(f"Tile {tile_number} failed result validation.")
    return tile_number, tile.triangle_count, len(output_faces), os.fspath(result_path)


def optimize_indexed_mesh(
    index_directory: Path,
    output_path: Path,
    target_triangles: int,
    *,
    aggressiveness: float = 7.0,
    algorithm: str = "Fast QEM",
    gpu_preference: str = "system",
    workers: int | None = None,
    parallel_gpu_workers: bool = False,
    memory_budget_bytes: int | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, int]:
    """Optimize indexed tiles within bounded memory and assemble a binary STL incrementally.

    Exact source coordinates on tile borders are protected by Fast QEM's boundary preservation.
    STL assembly remains triangle-soup based, so it does not require a global vertex array.
    """
    manifest = load_manifest(index_directory)
    target_triangles = max(4, min(int(target_triangles), manifest.triangle_count))
    estimated_result_bytes = target_triangles * (STL_RECORD_BYTES + 48)
    _require_disk_space(
        output_path.parent,
        estimated_result_bytes + max(64 * 1024**2, estimated_result_bytes // 10),
        "Tiled optimization",
    )
    tile_targets = allocate_tile_targets(manifest.tiles, target_triangles)
    work_directory = index_directory / "results"
    work_directory.mkdir(exist_ok=True)
    run_key = sha256(
        json.dumps(
            {
                "index": manifest.source_fingerprint,
                "target": target_triangles,
                "algorithm": algorithm,
                "aggressiveness": aggressiveness,
                "gpu": gpu_preference,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:24]
    run_directory = work_directory / run_key
    run_directory.mkdir(exist_ok=True)
    for partial in run_directory.glob("*.partial"):
        partial.unlink()
    results_by_tile: dict[int, tuple[Path, int]] = {}
    run_manifest_path = run_directory / "run.json"
    run_manifest = {
        "format": 1,
        "source_fingerprint": manifest.source_fingerprint,
        "source_triangles": manifest.triangle_count,
        "target_triangles": target_triangles,
        "algorithm": algorithm,
        "aggressiveness": aggressiveness,
        "gpu_preference": gpu_preference,
        "complete": False,
        "tiles": [],
    }
    processed = 0
    cached_tiles = 0
    pending: list[tuple[int, TileRecord, int]] = []
    for tile_number, (tile, tile_target) in enumerate(
        zip(manifest.tiles, tile_targets, strict=True), start=1
    ):
        _cancelled(cancel_event)
        result_path = run_directory / f"tile-{tile_number:08d}.npz"
        cached_count = _tile_result_count(result_path, tile.triangle_count)
        if cached_count is not None:
            results_by_tile[tile_number] = (result_path, cached_count)
            processed += tile.triangle_count
            cached_tiles += 1
            _progress(
                progress_callback,
                "optimizing tiles",
                processed,
                manifest.triangle_count,
            )
            continue
        pending.append((tile_number, tile, tile_target))

    def checkpoint() -> None:
        run_manifest["tiles"] = [
            {
                "number": number,
                "target_triangles": tile_targets[number - 1],
                "output_triangles": count,
                "result": result_path.name,
            }
            for number, (result_path, count) in sorted(results_by_tile.items())
        ]
        _write_run_manifest(run_manifest_path, run_manifest)

    checkpoint()

    gpu_algorithm = algorithm.startswith("Depth ")
    if workers is None:
        workers = min(8, max(1, (os.cpu_count() or 2) // 2), max(1, len(pending)))
    workers = (
        1
        if gpu_algorithm and not parallel_gpu_workers
        else max(1, min(int(workers), max(1, len(pending))))
    )
    if memory_budget_bytes is not None and pending:
        largest_tile = max(tile.triangle_count for _number, tile, _target in pending)
        per_worker = max(
            64 * 1024**2,
            largest_tile * ESTIMATED_OPTIMIZATION_BYTES_PER_TRIANGLE,
        )
        workers = min(workers, max(1, int(memory_budget_bytes) // per_worker))

    if workers == 1:
        completed_results = (
            _optimize_tile(
                os.fspath(index_directory),
                tile,
                tile_number,
                tile_target,
                os.fspath(run_directory),
                aggressiveness,
                algorithm,
                gpu_preference,
            )
            for tile_number, tile, tile_target in pending
        )
        for tile_number, source_count, output_count, result_path in completed_results:
            _cancelled(cancel_event)
            results_by_tile[tile_number] = (Path(result_path), output_count)
            processed += source_count
            checkpoint()
            _progress(progress_callback, "optimizing tiles", processed, manifest.triangle_count)
    elif pending:
        with OwnedProcessPool(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _optimize_tile,
                    os.fspath(index_directory),
                    tile,
                    tile_number,
                    tile_target,
                    os.fspath(run_directory),
                    aggressiveness,
                    algorithm,
                    gpu_preference,
                )
                for tile_number, tile, tile_target in pending
            ]
            for future in completed_futures(futures, lambda: _cancelled(cancel_event)):
                _cancelled(cancel_event)
                tile_number, source_count, output_count, result_path = future.result()
                results_by_tile[tile_number] = (Path(result_path), output_count)
                processed += source_count
                checkpoint()
                _progress(
                    progress_callback, "optimizing tiles", processed, manifest.triangle_count
                )
    _cancelled(cancel_event)
    results = [results_by_tile[index] for index in range(1, len(manifest.tiles) + 1)]
    output_triangle_count = sum(count for _path, count in results)
    _progress(progress_callback, "assembling", 0, len(results))
    _write_binary_stl(output_path, results, cancel_event, progress_callback)
    if binary_stl_count(output_path) != output_triangle_count:
        raise RuntimeError("Assembled STL count does not match the validated tile results.")
    run_manifest["complete"] = True
    run_manifest["output_triangles"] = output_triangle_count
    run_manifest["output_sha256"] = file_sha256(output_path)
    checkpoint()
    return {
        "source_triangles": manifest.triangle_count,
        "target_triangles": target_triangles,
        "output_triangles": output_triangle_count,
        "tile_count": len(results),
        "cached_tiles": cached_tiles,
        "workers": workers,
    }
