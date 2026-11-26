"""CPU and memory of this process and every child (the workers and their MAD-NG processes)."""

from __future__ import annotations

import os
import time

import psutil

#: Seconds between two samples; a redraw in between reuses the last one.
SAMPLE_INTERVAL = 1.0
GIB = 1024**3


class ResourceMonitor:
    """Cores busy and memory of the process tree, sampled at most every ``SAMPLE_INTERVAL``."""

    def __init__(self) -> None:
        self._root = psutil.Process(os.getpid())
        self._known: dict[int, psutil.Process] = {}
        self._last = float("-inf")
        self.cores = 0.0
        self.memory = 0
        self.peak_memory = 0
        self.processes = 1
        self.machine_cores = psutil.cpu_count() or 1

    def sample(self) -> None:
        """Refresh ``cores`` (busy cores), ``memory`` (bytes, shared pages counted once)."""
        now = time.monotonic()
        if now - self._last < SAMPLE_INTERVAL:
            return
        self._last = now
        alive: dict[int, psutil.Process] = {}
        percent, memory = 0.0, 0
        for found in [self._root, *self._root.children(recursive=True)]:
            proc = self._known.get(found.pid, found)  # keeps the cpu_percent baseline
            try:
                percent += proc.cpu_percent(None)
                info = proc.memory_full_info()
                memory += getattr(info, "pss", info.uss)
            except psutil.Error:
                continue
            alive[found.pid] = proc
        self._known = alive
        self.cores, self.memory, self.processes = percent / 100, memory, len(alive)
        self.peak_memory = max(self.peak_memory, memory)
