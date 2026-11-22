"""MAD-NG closed-orbit engine: the loss-only (plain-knob) orbit equals the parametric one."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from accelmo.backend import get_backend
from accelmo.fitting.worker import WorkerConfig
from accelmo.machine.accelerators import PSB

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.serial

CORRECTOR_STATE = {"kbr3dvt3l4": 1e-4, "kbr3dhz2l4": 1e-4}
QUAD_SHIFT = 2e-4


def _engine(seq_psb: Path):
    accelerator = PSB(ring=3, sequence_file=seq_psb, errors={"quad": {"k1"}})
    config = WorkerConfig(
        accelerator=accelerator,
        tracking_start_bpm="$start",
        tracking_end_bpm="$end",
        magnet_range="$start/$end",
        cycle_sequence=False,
        interface_options={"machine_state": CORRECTOR_STATE},
    )
    model = get_backend("madng").open_model(accelerator, magnet_range="$start/$end")
    knobs = dict.fromkeys(model.knob_names, 0.0)
    model.close()
    engine = get_backend("madng").ClosedOrbitEngine(
        config, knobs, optics_names=[], orbit_coords=["x", "y"], worker_id=0
    )
    engine.set_pt(0.0)
    return engine, knobs


def test_plain_orbit_matches_parametric_orbit_and_restores_knobs(seq_psb: Path) -> None:
    engine, knobs = _engine(seq_psb)
    keys = iter(range(4))

    def solve(*, jacobian: bool):
        # A fresh key cold-starts cofind, so both solves take the same iterations.
        engine.set_warm_start(f"cold{next(keys)}")
        return engine.solve(["y", "x"], jacobian=jacobian)

    shifted = {**knobs, engine.knob_names[0]: QUAD_SHIFT}
    engine.set_knobs(shifted)
    full = solve(jacobian=True)
    plain = solve(jacobian=False)
    engine.set_knobs(knobs)
    plain_base = solve(jacobian=False)
    full_base = solve(jacobian=True)
    engine.close()

    assert np.abs(full.values).max() > 1e-4
    assert plain.jacobian is None
    assert plain.names == full.names
    np.testing.assert_allclose(plain.values, full.values, rtol=0, atol=1e-13)
    np.testing.assert_allclose(plain_base.values, full_base.values, rtol=0, atol=1e-13)
    # The knob shift moves the orbit, so set_knobs in plain mode took effect.
    assert np.abs(plain.values - plain_base.values).max() > 5e-7
    # Back in parametric mode the Jacobian is the one of the unshifted knobs (measured 7.9e-10, second order).
    predicted = full_base.values + full_base.jacobian[:, :, 0] * QUAD_SHIFT
    np.testing.assert_allclose(predicted, full.values, rtol=0, atol=2e-9)
