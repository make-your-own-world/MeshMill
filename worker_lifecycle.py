"""Owned spawned pools with explicit cancellation and parent-death containment.

Python 3.12 has no public terminate_workers API. Keep its two private process-pool
integration points here, covered by real subprocess lifecycle tests.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing.connection import wait as wait_handles


def _watch_parent() -> None:
    parent = multiprocessing.parent_process()
    if parent is not None:
        # The sentinel identifies this process instance, unlike polling a reusable PID.
        wait_handles([parent.sentinel])
        os._exit(1)


def _initialize_worker() -> None:
    threading.Thread(target=_watch_parent, name="owner-watchdog", daemon=True).start()


class _WindowsJob:
    """A non-inheritable owner handle kills worker descendants when the owner exits."""

    def __init__(self) -> None:
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD), ("minimum", ctypes.c_size_t),
                ("maximum", ctypes.c_size_t), ("active", wintypes.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits), ("io", ctypes.c_uint64 * 6),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t),
            ]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        self.api.SetInformationJobObject.restype = wintypes.BOOL
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.AssignProcessToJobObject.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process) -> None:
        # Windows multiprocessing exposes its actual process handle as the sentinel.
        if not self.api.AssignProcessToJobObject(self.handle, process.sentinel):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


class OwnedProcessPool(ProcessPoolExecutor):
    """Spawned workers belong to one executor and cannot outlive their owner on Windows."""

    def __init__(self, max_workers: int = 1):
        self._worker_job = _WindowsJob() if os.name == "nt" else None
        try:
            super().__init__(
                max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
            )
        except BaseException:
            if self._worker_job is not None:
                self._worker_job.close()
            raise

    def _spawn_process(self):
        previous = set(self._processes)
        super()._spawn_process()
        if self._worker_job is not None:
            try:
                for pid in self._processes.keys() - previous:
                    self._worker_job.assign(self._processes[pid])
            except BaseException:
                # Do not leave an uncontained child running after a failed assignment.
                self._worker_job.close()
                for process in self._processes.values():
                    if process.is_alive():
                        process.terminate()
                    process.join()
                self._broken = "Worker process containment failed"
                raise

    def terminate_and_join(self, timeout: float = 2.0) -> None:
        with self._shutdown_lock:
            self._shutdown_thread = True
            self._cancel_pending_futures = True
            processes = list((self._processes or {}).values())
        if self._worker_job is not None:
            self._worker_job.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + timeout
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()
        for process in processes:
            process.join(timeout)
        if any(process.is_alive() for process in processes):
            raise RuntimeError("An optimization worker did not stop")
        super().shutdown(wait=True, cancel_futures=True)

    def shutdown(self, wait=True, *, cancel_futures=False):
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
        if wait and self._worker_job is not None:
            self._worker_job.close()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.terminate_and_join()
        else:
            self.shutdown(wait=True)
        return False


def completed_futures(futures, check_cancelled):
    """Observe cancellation even when every worker is inside a long native call."""
    pending = set(futures)
    while pending:
        check_cancelled()
        done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
        yield from done
