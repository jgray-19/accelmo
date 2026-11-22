"""ACD sinusoid mode: forward multi-turn tracking through the model AC kickers vs xtrack truth."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from accelmo.config import OptimiserConfig, SimulationConfig
from accelmo.fitting.config import OutputConfig, SequenceConfig
from accelmo.machine.accelerators import PSB
from accelmo.machine.machine_state import merge_machine_states
from accelmo.tracking.config.helpers import create_arc_measurement_config
from accelmo.tracking.fitter import ACDSinusoidFitter, FitterOptions
from tests.training.controller.test_controller_acd import (
    PSB_HORIZONTAL_ACD_VOLTAGE,
    PSB_VERTICAL_ACD_VOLTAGE,
    _generate_acd_track,
)
from tests.training.controller_test_utils import (
    evaluate_controller_worker_losses,
)

if TYPE_CHECKING:
    from pathlib import Path

    from accelmo.backend.madng.interface import MadInterface

pytest.importorskip("xtrack_tools")
pytestmark = pytest.mark.serial

DRIVE_TUNES = (0.16, 0.24)  # as in _generate_acd_track
RAMP_TURNS = 3
FLATTOP_TURNS = 30
ACD_BEFORE = "HACMAP_before"
START_OFFSET = {"x": 1e-4, "px": 1e-5, "y": -1e-4, "py": -1e-5}


def kick_knobs(volts, lags, suffix: str = "") -> dict[str, float]:
    """(a, b) kicker knobs of volt sin(2 pi (turn freq + lag)) per plane (h, v)."""
    knobs = {}
    for plane, volt, lag in zip(("hackicker", "vackicker"), volts, lags, strict=True):
        knobs[f"{plane}_a.volt{suffix}"] = volt * np.cos(2 * np.pi * lag)
        knobs[f"{plane}_b.volt{suffix}"] = volt * np.sin(2 * np.pi * lag)
    return knobs


def kick_error(knobs: dict[str, float], truth: dict[str, float] | None = None) -> float:
    """Max over planes of max |(a, b) - truth| / max |truth (a, b)|."""
    truth = truth or _truth_acd_knobs()
    errors = []
    for plane in ("hackicker", "vackicker"):
        names = [n for n in truth if n.startswith(plane)]
        ref = np.array([truth[n] for n in names])
        fit = np.array([knobs[n] for n in names])
        errors.append(np.max(np.abs(fit - ref)) / np.max(np.abs(ref)))
    return max(errors)


def _truth_acd_knobs(ramp_turns: int = RAMP_TURNS) -> dict[str, float]:
    """xtrack lag 0 at at_turn = ramp_turns (first flattop pass) is MAD turn 1."""
    return kick_knobs(
        (PSB_HORIZONTAL_ACD_VOLTAGE, PSB_VERTICAL_ACD_VOLTAGE),
        [(ramp_turns - 1) * freq for freq in DRIVE_TUNES],
    )


def _build(
    tmp_path: Path,
    seq_psb: Path,
    iface: MadInterface,
    optimiser: str = "adam",
    start_offset: dict[str, float] | None = None,
):
    """Fitter on xtrack truth; ``start_offset`` shifts the measured start (before-marker, turn 1)."""
    data = tmp_path / "acd_sinusoid.parquet"
    correctors, strengths, tune_knobs = _generate_acd_track(
        iface,
        FLATTOP_TURNS,
        data,
        dpp_value=0.0,
        ramp_up_turns=RAMP_TURNS,
    )
    frame = pd.read_parquet(data)
    start_row = (frame["name"] == ACD_BEFORE) & (frame["turn"] == 1)
    truth_start = frame.loc[start_row, list(START_OFFSET)].to_numpy()[0]
    for column, offset in (start_offset or {}).items():
        frame.loc[start_row, column] += offset
    frame.to_parquet(data, index=False)
    accelerator = PSB(
        ring=3,
        kinetic_energy=iface.accelerator.kinetic_energy,
        sequence_file=seq_psb,
        errors={"acd": {"volt"}},
        acd_drive_tunes=(DRIVE_TUNES,),
    )
    start = {name: 0.9 * value for name, value in _truth_acd_knobs().items()}
    ctrl = ACDSinusoidFitter(
        accelerator,
        OptimiserConfig(
            max_epochs=200,
            warmup_epochs=0,
            warmup_lr_start=1e-3,
            max_lr=1e-3,
            min_lr=1e-3,
            optimiser_type=optimiser,
            gradient_converged_value=1e-14,
        ),
        SimulationConfig(
            num_workers=1,
            num_batches=1,
            n_run_turns=FLATTOP_TURNS - 1,
            fit_start_coords=start_offset is not None,
        ),
        SequenceConfig(magnet_range="$start/$end"),
        create_arc_measurement_config(
            data, machine_state=merge_machine_states(correctors, tune_knobs)
        ),
        options=FitterOptions(
            initial_knob_strengths={**strengths, **start},
            output_config=OutputConfig(mad_logfile=tmp_path / "mad.log", write_run_log=False),
        ),
    )
    return ctrl, truth_start


def test_acd_sinusoid_loss_minimal_at_truth(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: MadInterface
) -> None:
    """Truth magnets + truth kick reproduce the xtrack data to the integrator floor."""
    ctrl, _ = _build(tmp_path, seq_psb, loaded_psb_interface)
    truth = _truth_acd_knobs()
    shifted = _truth_acd_knobs(RAMP_TURNS + 1e-3 / DRIVE_TUNES[0])
    shifted = {**truth, **{k: v for k, v in shifted.items() if k.startswith("hackicker")}}
    scaled = {k: v * 1.001 if k.startswith("vackicker") else v for k, v in truth.items()}
    loss_truth, loss_lag, loss_volt = evaluate_controller_worker_losses(
        ctrl, [truth, shifted, scaled]
    )
    # measured 2026-11-12: truth 1.2e-19, lag + 1e-3 8.0e-12, volt x 1.001 1.8e-13
    assert loss_truth < 1e-18
    assert loss_lag > 1e6 * loss_truth
    assert loss_volt > 1e5 * loss_truth


def test_acd_sinusoid_fit_recovers_kick(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: MadInterface
) -> None:
    """From 0.9 x truth, fitting the (a, b) kick (magnets at truth) recovers the xtrack kick."""
    ctrl, _ = _build(tmp_path, seq_psb, loaded_psb_interface, optimiser="levenberg_marquardt")
    fitted = ctrl.run().knobs
    rel_error = kick_error(fitted)
    rel_improvement = rel_error / kick_error(ctrl.machine.initial_knobs)
    # measured 2026-11-13 (LM, loss 2.9e-20): kick error 6.3e-7, improvement 6.3e-6
    assert rel_error < 1e-6
    assert rel_improvement < 1e-5


def test_acd_sinusoid_fit_recovers_start_and_kick(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: MadInterface
) -> None:
    """A wrong measured start is refitted with the kick: both return to the xtrack truth."""
    ctrl, truth_start = _build(
        tmp_path,
        seq_psb,
        loaded_psb_interface,
        optimiser="levenberg_marquardt",
        start_offset=START_OFFSET,
    )
    fitted = ctrl.run().knobs
    rel_error = kick_error(fitted)
    rel_improvement = rel_error / kick_error(ctrl.machine.initial_knobs)
    start_error = np.abs(ctrl.fitted_starts[0] - truth_start)
    start_rel_error = np.max(start_error / np.abs(truth_start))
    start_rel_improvement = np.max(start_error / np.abs(list(START_OFFSET.values())))
    # measured 2026-11-12 (loss 1.7e-20): kick 4.7e-7 / 4.7e-6, start 1.9e-5 / 8.2e-6
    assert rel_error < 1e-6
    assert rel_improvement < 1e-5
    assert start_rel_error < 5e-5
    assert start_rel_improvement < 2e-5
