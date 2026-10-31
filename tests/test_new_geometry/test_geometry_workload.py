from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from geometry_workload import difference, merge, record, snapshot


def test_repeated_passes_and_devices():
    points = np.zeros((4, 3), dtype=np.float32)
    faces = np.zeros((2, 3), dtype=np.int32)
    before = snapshot()
    record("CPU", points, faces)
    record("CPU", points, faces)
    record("GPU", points, faces)
    delta = difference(snapshot(), before)
    assert delta == {"CPU": [8, 4, 144], "GPU": [4, 2, 72]}


def test_concurrent_receipts_and_process_merge():
    points = np.zeros((4, 3), dtype=np.float32)
    faces = np.zeros((2, 3), dtype=np.int32)
    before = snapshot()
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda _: record("CPU", points, faces), range(100)))
    merge({"GPU": [4, 2, 72]})
    assert difference(snapshot(), before) == {"CPU": [400, 200, 7200], "GPU": [4, 2, 72]}


def test_empty_mesh_and_invalid_receipt():
    before = snapshot()
    record("CPU", np.empty((0, 3)), np.empty((0, 3), dtype=int))
    assert snapshot() == before
    with pytest.raises(ValueError):
        merge({"GPU": [-1, 0, 0]})
