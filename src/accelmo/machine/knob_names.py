"""Names of settable magnet strengths (``MQ.1.dk1l``, ``BR.QFO11.dx``), shared by every backend."""

from __future__ import annotations

from pymadng_utils.mad.accelerator_mad_interface import MAGNET_STRENGTH_SUFFIXES


def is_magnet_strength_name(name: str) -> bool:
    """True if ``name`` is a settable magnet-strength name (e.g. ``MQ.1.dk1l``).

    Callers use it to tell a genuine magnet strength from a stray or mistyped knob name
    before routing values to the model.
    """
    return any(name.endswith(suffix) for suffix in MAGNET_STRENGTH_SUFFIXES)
