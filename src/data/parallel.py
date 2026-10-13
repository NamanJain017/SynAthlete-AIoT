"""Run independent per-file jobs on all CPU cores, within the laptop's free memory.

Used by the audit now and by the loaders later. Everything here is CPU-only on purpose:
teammates have no GPU, and file parsing and hashing do not get faster on a GPU anyway.
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

MEMORY_HEADROOM = 0.8  # never plan to use more than 80 % of the RAM that is free right now


def available_memory_bytes() -> int | None:
    """Free physical memory right now, or None if the OS does not tell us."""
    if sys.platform == "win32":

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None


def pick_workers(n_tasks: int, bytes_per_task: int = 0, max_workers: int | None = None) -> int:
    """As many processes as there are cores and tasks, but never more than free RAM allows."""
    workers = min(n_tasks, max_workers or os.cpu_count() or 1)
    free = available_memory_bytes()
    if free and bytes_per_task > 0:
        workers = min(workers, int(free * MEMORY_HEADROOM // bytes_per_task))
    return max(1, workers)


def _init_worker(threads: int) -> None:
    # Each process gets its share of the cores, so pyarrow's own threads x processes
    # do not oversubscribe the CPU.
    try:
        import pyarrow

        pyarrow.set_cpu_count(threads)
    except ImportError:
        pass


def parallel_map(
    func: Callable[[T], R], items: Iterable[T], workers: int, label: str = ""
) -> list[R]:
    """func over items in separate processes; results come back in input order.

    workers=1 runs everything in this process (simple to debug, used by most tests).
    func must be defined at module level so Windows can start it in a fresh process.
    A failure in any job is raised here, never skipped.
    """
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [func(x) for x in items]
    threads = max(1, (os.cpu_count() or 1) // workers)
    results: list = [None] * len(items)
    step = max(1, len(items) // 10)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(threads,)
    ) as pool:
        futures = {pool.submit(func, x): i for i, x in enumerate(items)}
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if label and (done % step == 0 or done == len(items)):
                print(f"    {label}: {done}/{len(items)} done", flush=True)
    return results
