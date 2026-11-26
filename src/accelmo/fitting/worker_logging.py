"""Worker-process logging: a file per worker, with WARNING and above relayed to the parent."""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path

from accelmo.logs import DATE_FORMAT, FILE_FORMAT

#: Where a run's workers log by default, relative to its run log directory.
WORKER_LOG_PATH = Path("workers") / "python.log"
#: Records at this level and above reach the parent's loggers (and so its dashboard).
RELAY_LEVEL = logging.WARNING

#: Set in the parent before workers fork, so each worker inherits it.
_queue: mp.Queue | None = None


class _Relay(logging.Handler):
    """Hands a worker's record to the parent's logger of the same name."""

    def emit(self, record: logging.LogRecord) -> None:
        record.msg = f"{record.processName}: {record.msg}"
        logging.getLogger(record.name).handle(record)


def start_relay() -> None:
    """Start relaying worker records into this process's loggers; call before forking workers."""
    global _queue
    if _queue is not None:
        return
    _queue = mp.Queue()
    listener = QueueListener(_queue, _Relay())
    listener.start()
    atexit.register(listener.stop)


def attach_in_worker(logfile: Path | None, level: int) -> None:
    """Replace this worker's inherited handlers by a relay and, with ``logfile``, a file at ``level``.

    Does nothing outside a child process, so a worker run in-process leaves the logging alone.
    """
    if mp.parent_process() is None:
        return
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(RELAY_LEVEL)
    if _queue is not None:
        relay = QueueHandler(_queue)
        relay.setLevel(RELAY_LEVEL)
        root.addHandler(relay)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logfile, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(file_handler)
        root.setLevel(min(level, RELAY_LEVEL))
