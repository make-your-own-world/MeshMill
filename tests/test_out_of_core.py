import shutil
import struct
import threading
from hashlib import sha256
from pathlib import Path

import numpy as np

from out_of_core import (
    IndexCancelled,
    TileRecord,
    allocate_tile_targets,
    binary_stl_count,
    build_index,
    index_matches,
    load_manifest,
    optimize_indexed_mesh,
)


def write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    with path.open("wb") as stream:
        stream.write(b"MeshMill test".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(triangles)))
        for triangle in triangles.astype("<f4"):
            stream.write(struct.pack("<3f", 0.0, 0.0, 1.0))
            stream.write(triangle.tobytes())
            stream.write(b"\0\0")


def read_binary_stl_triangles(path: Path) -> np.ndarray:
    count = binary_stl_count(path)
    record_dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    records = np.memmap(path, dtype=record_dtype, mode="r", offset=84, shape=(count,))
    triangles = np.asarray(records["vertices"]).copy()
    del records
    return triangles


def boundary_edge_count(triangles: np.ndarray) -> int:
    points, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    del points
    faces = inverse.reshape(-1, 3)
    edges = np.sort(
        np.vstack((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)])), axis=1
    )
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1))


def test_index_partitions_every_triangle_and_reopens(tmp_path):
    source = tmp_path / "mesh.stl"
    triangles = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[10, 0, 0], [11, 0, 0], [10, 1, 0]],
            [[0, 10, 0], [1, 10, 0], [0, 11, 0]],
            [[10, 10, 10], [11, 10, 10], [10, 11, 10]],
        ],
        dtype=np.float32,
    )
    write_binary_stl(source, triangles)
    destination = tmp_path / "mesh.meshmill-index"

    created = build_index(
        source,
        destination,
        target_triangles_per_tile=1,
        block_triangles=2,
    )
    loaded = load_manifest(destination)

    assert loaded == created
    assert sum(tile.triangle_count for tile in loaded.tiles) == len(triangles)
    assert max(tile.triangle_count for tile in loaded.tiles) <= 65_536
    assert all(
        (destination / tile.relative_path).stat().st_size == tile.triangle_count * 50
        for tile in loaded.tiles
    )
    assert index_matches(source, loaded)
    assert all(tile.level <= loaded.max_depth for tile in loaded.tiles)


def test_index_detects_source_change(tmp_path):
    source = tmp_path / "mesh.stl"
    triangle = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    write_binary_stl(source, triangle)
    destination = tmp_path / "mesh.meshmill-index"
    manifest = build_index(source, destination)

    with source.open("ab") as stream:
        stream.write(b"changed")

    assert not index_matches(source, manifest)


def test_cancel_does_not_publish_or_leave_staging(tmp_path):
    source = tmp_path / "mesh.stl"
    triangle = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    write_binary_stl(source, np.repeat(triangle, 32, axis=0))
    destination = tmp_path / "mesh.meshmill-index"
    cancel = threading.Event()
    cancel.set()

    try:
        build_index(source, destination, cancel_event=cancel)
    except IndexCancelled:
        pass
    else:
        raise AssertionError("cancelled indexing unexpectedly completed")

    assert not destination.exists()
    assert not destination.with_name(destination.name + ".building").exists()


def test_parallel_analysis_matches_sequential_index(tmp_path):
    source = tmp_path / "parallel.stl"
    triangle = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    triangles = np.repeat(triangle, 32_768, axis=0)
    triangles[:, :, 0] += np.arange(len(triangles), dtype=np.float32)[:, None]
    write_binary_stl(source, triangles)

    sequential = build_index(
        source,
        tmp_path / "sequential.meshmill-index",
        block_triangles=16_384,
        analysis_workers=1,
    )
    parallel = build_index(
        source,
        tmp_path / "parallel.meshmill-index",
        block_triangles=16_384,
        analysis_workers=2,
    )

    assert parallel.bounds == sequential.bounds
    assert parallel.max_depth == sequential.max_depth
    assert parallel.tiles == sequential.tiles


def test_indexed_mesh_assembles_without_global_geometry_array(tmp_path):
    source = tmp_path / "source.stl"
    triangles = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[10, 0, 0], [11, 0, 0], [10, 1, 0]],
            [[0, 10, 0], [1, 10, 0], [0, 11, 0]],
            [[10, 10, 10], [11, 10, 10], [10, 11, 10]],
        ],
        dtype=np.float32,
    )
    write_binary_stl(source, triangles)
    index_directory = tmp_path / "source.meshmill-index"
    build_index(source, index_directory)
    output = tmp_path / "result.stl"
    updates = []

    report = optimize_indexed_mesh(
        index_directory,
        output,
        len(triangles),
        progress_callback=lambda stage, done, total: updates.append((stage, done, total)),
    )

    assert output.exists()
    assert binary_stl_count(output) == len(triangles)
    assert report["output_triangles"] == len(triangles)
    assert updates[-1] == ("assembling", report["tile_count"], report["tile_count"])


def test_tile_target_allocation_is_exact_and_deterministic():
    tiles = tuple(
        TileRecord(1, index, count, f"tile-{index}.bin")
        for index, count in enumerate((10, 20, 30, 40))
    )
    first = allocate_tile_targets(tiles, 57)
    second = allocate_tile_targets(tiles, 57)

    assert first == second
    assert sum(first) == 57
    assert all(4 <= target <= tile.triangle_count for target, tile in zip(first, tiles))


def test_corrupt_checkpoint_is_rebuilt_and_output_is_deterministic(tmp_path):
    source = tmp_path / "source.stl"
    triangles = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[10, 0, 0], [11, 0, 0], [10, 1, 0]],
            [[0, 10, 0], [1, 10, 0], [0, 11, 0]],
            [[10, 10, 10], [11, 10, 10], [10, 11, 10]],
        ],
        dtype=np.float32,
    )
    write_binary_stl(source, triangles)
    index_directory = tmp_path / "source.meshmill-index"
    build_index(source, index_directory)
    first_output = tmp_path / "first.stl"
    first = optimize_indexed_mesh(index_directory, first_output, len(triangles))
    first_hash = sha256(first_output.read_bytes()).hexdigest()
    run_directory = next((index_directory / "results").iterdir())
    assembled_tiles = []
    for result_path in sorted(run_directory.glob("tile-*.npz")):
        with np.load(result_path, allow_pickle=False) as payload:
            assembled_tiles.append(payload["points"][payload["faces"]])
    assert np.array_equal(read_binary_stl_triangles(first_output), np.vstack(assembled_tiles))

    checkpoint = next((index_directory / "results").glob("*/tile-*.npz"))
    checkpoint.write_bytes(b"corrupt")
    second_output = tmp_path / "second.stl"
    second = optimize_indexed_mesh(index_directory, second_output, len(triangles))

    assert sha256(second_output.read_bytes()).hexdigest() == first_hash
    assert second["output_triangles"] == first["output_triangles"]


def test_connected_surface_reassembles_without_artificial_boundary_edges(tmp_path):
    width, height = 260, 130
    x, y = np.meshgrid(
        np.arange(width + 1, dtype=np.float32),
        np.arange(height + 1, dtype=np.float32),
    )
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size, dtype=np.float32)))
    row = width + 1
    lower_left = (
        np.arange(height, dtype=np.int64)[:, None] * row
        + np.arange(width, dtype=np.int64)[None, :]
    ).ravel()
    faces = np.vstack(
        (
            np.column_stack((lower_left, lower_left + 1, lower_left + row)),
            np.column_stack((lower_left + 1, lower_left + row + 1, lower_left + row)),
        )
    )
    triangles = points[faces]
    source = tmp_path / "grid.stl"
    write_binary_stl(source, triangles)
    index_directory = tmp_path / "grid.meshmill-index"
    manifest = build_index(
        source,
        index_directory,
        target_triangles_per_tile=65_536,
        block_triangles=16_384,
    )
    assert len(manifest.tiles) > 1
    output = tmp_path / "grid-reduced.stl"

    report = optimize_indexed_mesh(index_directory, output, 50_000)
    output_triangles = read_binary_stl_triangles(output)

    assert len(output_triangles) == report["output_triangles"]
    assert boundary_edge_count(output_triangles) == boundary_edge_count(triangles)

    sequential_hash = sha256(output.read_bytes()).hexdigest()
    shutil.rmtree(index_directory / "results")
    parallel_output = tmp_path / "grid-parallel.stl"
    parallel_report = optimize_indexed_mesh(
        index_directory, parallel_output, 50_000, workers=2
    )
    assert parallel_report["output_triangles"] == report["output_triangles"]
    assert sha256(parallel_output.read_bytes()).hexdigest() == sequential_hash

    shutil.rmtree(index_directory / "results")
    cancel = threading.Event()
    cancelled_output = tmp_path / "grid-cancelled.stl"

    def cancel_after_first_tile(stage, done, total):
        if stage == "optimizing tiles" and 0 < done < total:
            cancel.set()

    try:
        optimize_indexed_mesh(
            index_directory,
            cancelled_output,
            50_000,
            workers=1,
            cancel_event=cancel,
            progress_callback=cancel_after_first_tile,
        )
    except IndexCancelled:
        pass
    else:
        raise AssertionError("cancelled tiled optimization unexpectedly completed")
    assert not cancelled_output.exists()

    cancel.clear()
    resumed_output = tmp_path / "grid-resumed.stl"
    resumed = optimize_indexed_mesh(
        index_directory, resumed_output, 50_000, workers=1, cancel_event=cancel
    )
    assert resumed["cached_tiles"] >= 1
    assert sha256(resumed_output.read_bytes()).hexdigest() == sequential_hash

    assembly_cancel = threading.Event()
    assembly_cancelled_output = tmp_path / "grid-assembly-cancelled.stl"

    def cancel_during_assembly(stage, done, total):
        if stage == "assembling" and 0 < done < total:
            assembly_cancel.set()

    try:
        optimize_indexed_mesh(
            index_directory,
            assembly_cancelled_output,
            50_000,
            workers=1,
            cancel_event=assembly_cancel,
            progress_callback=cancel_during_assembly,
        )
    except IndexCancelled:
        pass
    else:
        raise AssertionError("cancelled assembly unexpectedly completed")
    assert not assembly_cancelled_output.exists()
    assert not assembly_cancelled_output.with_name(
        assembly_cancelled_output.name + ".partial"
    ).exists()
