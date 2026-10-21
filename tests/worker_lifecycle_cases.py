"""Importable tasks and isolated owners for worker lifecycle regression tests."""

import ctypes
import multiprocessing
import os
import sys
import time
from pathlib import Path

from worker_lifecycle import OwnedProcessPool


def slow_task(directory, label, hold_gil=False):
    Path(directory, label).write_text(str(os.getpid()))
    if hold_gil and os.name == "nt":
        ctypes.PyDLL("kernel32").Sleep(60_000)
    else:
        time.sleep(60)


def nested_task(directory):
    Path(directory, "worker").write_text(str(os.getpid()))
    with OwnedProcessPool(max_workers=2) as pool:
        futures = [pool.submit(slow_task, directory, f"nested-{n}") for n in range(2)]
        for future in futures:
            future.result()


def owner(mode, directory):
    if mode == "watchdog":
        import worker_lifecycle
        worker_lifecycle._WindowsJob = lambda: None
    with OwnedProcessPool() as pool:
        if mode == "idle":
            Path(directory, "worker").write_text(str(pool.submit(os.getpid).result(20)))
            time.sleep(60)
            return
        if mode == "nested":
            future = pool.submit(nested_task, directory)
        else:
            future = pool.submit(slow_task, directory, "worker", mode == "native")
        if mode == "bootstrap":
            Path(directory, "worker").write_text(str(next(iter(pool._processes))))
        future.result()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    owner(sys.argv[1], sys.argv[2])
