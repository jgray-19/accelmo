"""AC-kicker (a, b) knobs on the PSB AC dipole."""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING

import pytest

from accelmo.backend.madng.interface import MadInterface
from accelmo.machine.accelerators import PSB

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.serial

DRIVE_TUNES = (0.16, 0.24)
KNOBS = {
    "hackicker_a.volt": 3e-3,
    "hackicker_b.volt": 1.1e-3,
    "vackicker_a.volt": -2e-3,
    "vackicker_b.volt": 0.7e-3,
}


def test_acd_kicker_knobs_kick_and_derivatives(seq_psb: Path) -> None:
    """First pass after cycling to the before-marker is turn 1: kick and d/dknob are exact."""
    accelerator = PSB(
        ring=3,
        sequence_file=seq_psb,
        errors={"acd": {"volt"}},
        acd_drive_tunes=(DRIVE_TUNES,),
    )
    before, after = accelerator.acd_marker_name("before"), accelerator.acd_marker_name("after")
    iface = MadInterface(
        accelerator=accelerator,
        bpm_range=f"{after}/{before}",
        tracking_anchor_mode="acd",
        initial_model_values=KNOBS,
        discard_mad_output=True,
        create_knobs=True,
    )
    try:
        assert sorted(iface.knob_names) == sorted(KNOBS)
        iface.cycle_to_start(before)
        py = iface.py_name
        iface.mad.send(f"""
local X = MAD.damap{{nv=6, np=#knob_names, mo=1, po=1, pn=knob_names}}
for _, k in ipairs(knob_names) do loaded_sequence[k] = loaded_sequence[k] + X[k] end
local _, flw = MAD.track{{sequence=loaded_sequence, range="{before}/{after}", X0=X, nturn=1}}
local out = {{flw[1].px:get0(), flw[1].py:get0(), loaded_sequence.beam.pc}}
for j = 1, #knob_names do
    out[#out + 1] = flw[1].px:get(7 + j); out[#out + 1] = flw[1].py:get(7 + j)
end
{py}:send(out, true)
""")
        px, py_, pc, *grads = iface.mad.recv()
        dpx = dict(zip(iface.knob_names, grads[0::2], strict=True))
        dpy = dict(zip(iface.knob_names, grads[1::2], strict=True))
        for plane, kick, deriv, freq in (
            ("hackicker", px, dpx, DRIVE_TUNES[0]),
            ("vackicker", py_, dpy, DRIVE_TUNES[1]),
        ):
            a, b = KNOBS[f"{plane}_a.volt"], KNOBS[f"{plane}_b.volt"]
            sin, cos = math.sin(2 * math.pi * freq), math.cos(2 * math.pi * freq)
            scale = 0.3 / pc
            assert kick == pytest.approx(scale * (a * sin + b * cos), rel=1e-13, abs=0)
            assert deriv[f"{plane}_a.volt"] == pytest.approx(scale * sin, rel=1e-13)
            assert deriv[f"{plane}_b.volt"] == pytest.approx(scale * cos, rel=1e-13)
    finally:
        with contextlib.suppress(Exception):
            del iface
