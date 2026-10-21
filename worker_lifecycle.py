"""Owned spawned pools with explicit cancellation and parent-death containment.

Python 3.12 has no public terminate_workers API. Keep its two private process-pool
integration points here, covered by real subprocess lifecycle tests.
"""

from __future__ import annotations

import ast
import ctypes
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing.connection import wait as wait_handles
from pathlib import Path


def _spawn_owner(command: str) -> int | None:
    """Read multiprocessing's parent argument without executing its command."""
    if "--multiprocessing-fork" not in command:
        return None
    start = command.find("spawn_main(")
    if start >= 0:
        end = command.find(")", start)
        try:
            call = ast.parse(command[start:end + 1], mode="eval").body
            for keyword in call.keywords:
                if keyword.arg == "parent_pid" and isinstance(keyword.value, ast.Constant):
                    value = keyword.value.value
                    return value if type(value) is int and value > 0 else None
        except (SyntaxError, AttributeError):
            return None
    for argument in command.split():
        if argument.startswith("parent_pid="):
            value = argument.partition("=")[2].strip('"')
            return int(value) if value.isdecimal() and int(value) > 0 else None
    return None


def _owned_orphan_candidate(record, roots) -> bool:
    owner = _spawn_owner(record.get("command") or "")
    if owner is None or owner != record.get("parent"):
        return False
    roots = [Path(root).resolve() for root in roots]
    image = Path(record.get("image") or "").resolve()
    if image.name.lower() in ("meshmill.exe", "meshmillcli.exe") and image.parent in roots:
        return True
    # A shared python.exe is insufficient evidence. Require MeshMill's native
    # adapter loaded from this application's trusted directory.
    return any(Path(module).name.startswith("_meshmill_qem")
               and Path(module).resolve().parent in roots for module in record.get("modules", []))


def _stop_dead_owner_worker(record) -> bool:
    """Hold the worker handle while checking identity and parent lifetime."""
    from ctypes import wintypes

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    api.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]

    def created(handle):
        values = [wintypes.FILETIME() for _ in range(4)]
        if not api.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
            return None
        return (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime

    child = api.OpenProcess(0x101001, False, record["pid"])
    if not child:
        return False
    try:
        child_time = created(child)
        if child_time is None or child_time != record.get("created"):
            return False  # PID was reused after discovery.
        parent = api.OpenProcess(0x101000, False, record["parent"])
        if parent:
            try:
                parent_time = created(parent)
                if parent_time is None:
                    return False
                if parent_time <= child_time and api.WaitForSingleObject(parent, 0) != 0:
                    return False
            finally:
                api.CloseHandle(parent)
        elif ctypes.get_last_error() != 87:
            return False  # Access denied or unknown status is not proof of death.
        return bool(api.TerminateProcess(child, 1)) and api.WaitForSingleObject(child, 5000) == 0
    finally:
        api.CloseHandle(child)


def recover_orphan_workers(*, excluded_pids=(), trusted_roots=None):
    """Recover verified older workers; leave live owners and unknown processes alone."""
    if os.name != "nt":
        return []
    roots = trusted_roots or [Path(__file__).resolve().parent, Path(sys.executable).resolve().parent]
    if trusted_roots is None:
        # A development worktree can use the original checkout's virtualenv.
        # Trust that checkout only when this running interpreter identifies it.
        environment = Path(sys.prefix).resolve()
        checkout = environment.parent
        if ((environment / "pyvenv.cfg").is_file()
                and (checkout / "meshmill.py").is_file()
                and (checkout / "worker_lifecycle.py").is_file()):
            roots.append(checkout)
    script = r'''
$rows = foreach ($p in (Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('python.exe','pythonw.exe','MeshMill.exe','MeshMillCLI.exe') -and
    $_.CommandLine -like '*--multiprocessing-fork*'
})) {
    try {
        $g = Get-Process -Id $p.ProcessId -ErrorAction Stop
        $mods = @($g.Modules | ForEach-Object { $_.FileName })
        [pscustomobject]@{pid=[int]$p.ProcessId; parent=[int]$p.ParentProcessId;
            command=$p.CommandLine; image=$g.Path; modules=$mods;
            created=$g.StartTime.ToUniversalTime().ToFileTimeUtc()}
    } catch {}
}
ConvertTo-Json -InputObject @($rows) -Depth 4 -Compress
'''
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=20, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        records = json.loads(result.stdout or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    stopped = []
    for record in records:
        if record["pid"] in (*excluded_pids, os.getpid()):
            continue
        if _owned_orphan_candidate(record, roots) and _stop_dead_owner_worker(record):
            stopped.append(record["pid"])
    return stopped


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
