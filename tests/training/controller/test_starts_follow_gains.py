"""Starts that follow the BPM gains (starts_follow_gains) on xtrack truth with known gains."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from accelmo.tracking.ic_iteration import Step
from accelmo.tracking.lm_problem import StartGainMap
from tests.training.controller.test_controller_acd_sinusoid import _truth_acd_knobs
from tests.training.controller.test_ic_iteration import GAINS, IC_ERRORS, _setup
from tests.training.controller_test_utils import UNIT_LOSS

if TYPE_CHECKING:
    from pathlib import Path

    from accelmo.backend.madng.interface import MadInterface

pytest.importorskip("xtrack_tools")
pytestmark = pytest.mark.serial

GAIN_TOLERANCE = 1e-6
LOSS_FLOOR = 1e-17  # x UNIT_LOSS; measured 2026-11-23: gain errors <= 2e-7


@pytest.mark.parametrize("slope_scale", [0.0, 1e-3])
def test_follow_gains_recovers_gains_when_the_map_passes_through_truth(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: MadInterface, slope_scale: float
) -> None:
    """Noise-free data, knobs at truth, a start map with s(g_true) = truth start: gains come back.

    The map's slope is arbitrary and the measured start is offset, so the fit only
    reaches loss 0 at the true gains if the start follows them through the callback.
    """
    make_fitter, strengths, truth_start = _setup(
        tmp_path, seq_psb, loaded_psb_interface, gains=GAINS, gain_sigma=10.0
    )
    knobs = {**strengths, **_truth_acd_knobs()}
    fitter = make_fitter(Step(IC_ERRORS, fit_start_coords=False, knobs=knobs, starts=None))
    fitter.parameters = replace(fitter.parameters, starts_follow_gains=True)
    true_gains = np.array([GAINS.get(name.split(".", 2)[2], 0.0) for name in fitter.gain_names])
    slope = slope_scale * np.random.default_rng(0).standard_normal((1, 4, len(true_gains)))
    start_map = StartGainMap(truth_start[None, :], slope, true_gains)
    fitter.options = replace(fitter.options, initial_conditions_callback=lambda *_: start_map)

    result = fitter.run()

    for plane in ("x", "y"):
        for bpm, gain in GAINS.items():
            fitted = result.extra[f"bpmgain.{plane}.{bpm}"]
            print(f"{plane} {bpm}: fitted {fitted:.6e} truth {gain:.6e}")
            assert fitted == pytest.approx(gain, abs=GAIN_TOLERANCE)
    assert fitter.diagnostics.best_loss / fitter.session.n_training_points < LOSS_FLOOR * UNIT_LOSS
