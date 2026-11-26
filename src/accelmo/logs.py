"""One call to set up logging in the main process of a script."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

CONSOLE_FORMAT = "%(levelname)s %(name)s: %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def configure_logging(
    file: Path | None = None,
    *,
    console_level: int = logging.WARNING,
    file_level: int = logging.INFO,
) -> None:
    """Print ``console_level`` and above to the terminal and, with ``file``, write ``file_level`` to it.

    The live dashboard hides the console handler while it is on screen and shows warnings
    in its own strip. Workers write their own files (see ``WorkerRuntime.logging_level``).
    Replaces any handlers already on the root logger.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root.addHandler(console)
    root.setLevel(min(console_level, file_level) if file else console_level)
    if file is not None:
        file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(file, mode="a")
        handler.setLevel(file_level)
        handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(handler)
