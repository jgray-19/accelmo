"""Model backends: MAD-NG (``madng``) and xtrack (``xtrack``). See :mod:`accelmo.backend.protocol`."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from accelmo.backend.protocol import Backend

BACKENDS = ("madng", "xtrack")


def get_backend(name: str) -> Backend:
    """The backend module ``accelmo.backend.<name>.engines``."""
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; valid: {BACKENDS}")
    try:
        return import_module(f"accelmo.backend.{name}.engines")
    except ModuleNotFoundError as err:
        if name == "xtrack" and err.name in {"xtrack", "xtrack_tools"}:
            raise ImportError(
                "The xtrack backend needs xtrack (with tpsa) and xtrack_tools, which are not "
                "installed by accelmo."
            ) from err
        raise
