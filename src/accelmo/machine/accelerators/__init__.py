"""Accelerator definitions for the supported machines (LHC, PSB, SPS, FCC).

Each class carries the machine-specific knob specifications, BPM pattern and
tune configuration the MAD-NG interface needs.
"""

from accelmo.machine.accelerators.base import (
    MISALIGNMENT_ATTRS,
    Accelerator,
    KnobSpec,
    MagnetFamily,
)
from accelmo.machine.accelerators.fcc import FCC
from accelmo.machine.accelerators.lhc import LHC
from accelmo.machine.accelerators.psb import PSB
from accelmo.machine.accelerators.sps import SPS

__all__ = [
    "FCC",
    "LHC",
    "MISALIGNMENT_ATTRS",
    "PSB",
    "SPS",
    "Accelerator",
    "KnobSpec",
    "MagnetFamily",
]
