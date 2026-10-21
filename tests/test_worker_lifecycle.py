import ctypes
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from worker_lifecycle_cases import slow_task

from worker_lifecycle import OwnedProcessPool, completed_futures


def wait_until(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Worker lifecycle condition did not complete")
        time.sleep(0.025)


@pytest.mark.parametrize("active", [False, True])
def test_shutdown_joins_idle_and_active_workers(tmp_path, active):
    pool = OwnedProcessPool(max_workers=2)
    try:
        if active:
            futures = [pool.submit(slow_task, str(tmp_path), f"worker-{n}") for n in range(2)]
            wait_until(lambda: len(list(tmp_path.glob("worker-*"))) == 2)
        else:
            futures = [pool.submit(abs, -n) for n in range(4)]
            assert [future.result(20) for future in futures] == list(range(4))
        processes = list(pool._processes.values())
        started = time.monotonic()
        pool.terminate_and_join()
        assert time.monotonic() - started < 8
        assert all(not process.is_alive() and process.exitcode is not None for process in processes)
        assert all(future.done() for future in futures)
        pool.terminate_and_join()
    finally:
        pool.terminate_and_join()


def test_shutdown_before_first_submission():
    pool = OwnedProcessPool()
    pool.terminate_and_join()
    with pytest.raises(RuntimeError):
        pool.submit(abs, -1)


def test_context_exception_stops_running_tasks(tmp_path):
    with pytest.raises(ValueError, match="test failure"), OwnedProcessPool() as pool:
        pool.submit(slow_task, str(tmp_path), "worker")
        wait_until(lambda: (tmp_path / "worker").exists())
        processes = list(pool._processes.values())
        raise ValueError("test failure")
    assert all(not process.is_alive() for process in processes)


def test_cancellation_is_observed_while_all_tasks_are_busy(tmp_path):
    event = threading.Event()
    with pytest.raises(InterruptedError), OwnedProcessPool() as pool:
        future = pool.submit(slow_task, str(tmp_path), "worker")
        wait_until(lambda: (tmp_path / "worker").exists())
        processes = list(pool._processes.values())
        event.set()

        def check():
            if event.is_set():
                raise InterruptedError()

        list(completed_futures([future], check))
    assert all(not process.is_alive() for process in processes)


@pytest.mark.parametrize("mode", ["active", "idle", "native", "nested", "watchdog", "bootstrap"])
def test_workers_exit_after_owner_is_force_closed(tmp_path, mode):
    if mode == "native" and os.name != "nt":
        pytest.skip("Windows job containment under a GIL-holding native call")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    owner = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("worker_lifecycle_cases.py")),
         mode, str(tmp_path)], env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    handles = []
    try:
        expected = 3 if mode == "nested" else 1
        wait_until(lambda: len(list(tmp_path.iterdir())) == expected)
        pids = [int(path.read_text()) for path in tmp_path.iterdir()]
        assert len(set(pids)) == expected
        if os.name == "nt":
            from ctypes import wintypes
            api = ctypes.WinDLL("kernel32", use_last_error=True)
            api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            api.OpenProcess.restype = wintypes.HANDLE
            api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            api.CloseHandle.argtypes = [wintypes.HANDLE]
            handles = [api.OpenProcess(0x100000, False, pid) for pid in pids]
            assert all(handles)
        owner.kill()
        owner.communicate(timeout=10)
        if os.name == "nt":
            assert all(api.WaitForSingleObject(handle, 8000) == 0 for handle in handles)
        else:
            def stopped(pid):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return True
                status = Path(f"/proc/{pid}/stat")
                return status.exists() and status.read_text().split()[2] == "Z"
            wait_until(lambda: all(stopped(pid) for pid in pids), timeout=10)
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout=10)
        if os.name == "nt":
            for handle in handles:
                if handle:
                    api.CloseHandle(handle)
