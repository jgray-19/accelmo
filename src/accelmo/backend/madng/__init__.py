"""MAD-NG backend: everything that imports ``pymadng`` or ``pymadng_utils.mad`` or sends Lua."""

from .interface import MadInterface

__all__ = ["MadInterface"]
