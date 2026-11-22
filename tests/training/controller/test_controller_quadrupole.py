"""
Quadrupole-focused integration tests for controller logic.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import pytest

from accelmo.fitting.config import OutputConfig, SequenceConfig
from accelmo.machine.accelerators import LHC
from accelmo.machine.machine_state import merge_machine_states
from accelmo.tracking.config.helpers import create_arc_measurement_config
from accelmo.tracking.fitter import ArcByArcFitter, FitterOptions
from tests.training.controller_test_utils import (
    _generate_nonoise_track,
    _make_optimiser_config_quad,
    _make_simulation_config_quad,
    evaluate_controller_worker_loss,
    make_lm_config,
    relative_truth_errors,
)

if TYPE_CHECKING:
    from pathlib import Path

    from accelmo.config import OptimiserConfig
    from accelmo.backend.madng.aba_mad_interface import AbaMadInterface


logger = logging.getLogger(__name__)
pytestmark = pytest.mark.serial
# lr = 1 is the quasi-Newton step; the first step (H0 = I) is capped at the knob scale.
LBFGS_SETTINGS = {
    "lbfgs": {
        "warmup_epochs": 0,
        "warmup_lr_start": 1.0,
        "max_lr": 1.0,
        "min_lr": 1.0,
        "lbfgs_max_step_norm": 1e-5,
    }
}


def _build_lhc_quad_controller(
    *,
    tmp_path: Path,
    seq_b1: Path,
    loaded_interface: AbaMadInterface,
    start_marker: str,
    optimiser_config: OptimiserConfig | None = None,
) -> tuple[ArcByArcFitter, dict[str, float]]:
    magnet_range = "BPM.13R1.B1/BPM.13L2.B1"
    bpm_start_points = [f"BPM.{i}R1.B1" for i in range(13, 14)]
    bpm_end_points = [f"BPM.{i}L2.B1" for i in range(13, 14)]

    flattop_turns = 100
    off_magnet_path = tmp_path / "track_off_magnet.parquet"

    corrector_file, magnet_strengths, tune_knobs = _generate_nonoise_track(
        loaded_interface,
        flattop_turns,
        off_magnet_path,
        0.0,
        start_marker=start_marker,
        perturb_quads=True,
    )

    ctrl = ArcByArcFitter(
        LHC(
            beam=1,
            kinetic_energy=6800,
            sequence_file=seq_b1,
            errors={"quad": {"k1"}},
        ),
        optimiser_config or _make_optimiser_config_quad(),
        _make_simulation_config_quad(),
        SequenceConfig(magnet_range=magnet_range),
        create_arc_measurement_config(
            off_magnet_path, machine_state=merge_machine_states(corrector_file, tune_knobs)
        ),
        bpm_start_points,
        bpm_end_points,
        options=FitterOptions(
            output_config=OutputConfig(
                mad_logfile=tmp_path / "mad_logfile.log",
                write_run_log=False,
            ),
            true_strengths=magnet_strengths.copy(),
        ),
    )
    return ctrl, magnet_strengths.copy()


@pytest.mark.parametrize("start_marker", ["MSIA.EXIT.B1", "E.CELL.12.B1"])
def test_controller_quad_opt_simple(
    tmp_path: Path,
    seq_b1: Path,
    start_marker: str,
    loaded_interface: AbaMadInterface,
    controller_test_mode: str,
) -> None:
    ctrl, true_values = _build_lhc_quad_controller(
        tmp_path=tmp_path,
        seq_b1=seq_b1,
        loaded_interface=loaded_interface,
        start_marker=start_marker,
        optimiser_config=make_lm_config(),
    )
    logger.info("Starting controller with logfile at %s", tmp_path / "mad_logfile.log")
    if controller_test_mode == "loss_regression":
        initial_loss = evaluate_controller_worker_loss(ctrl, ctrl.machine.initial_knobs)
        true_loss = evaluate_controller_worker_loss(ctrl, true_values)
        assert true_loss < initial_loss * 1e-6
        return
    estimate = ctrl.run().knobs
    rel_error, rel_improvement = relative_truth_errors(
        estimate, true_values, ctrl.machine.initial_knobs
    )
    # measured 2026-11-12: rel error = rel improvement = 2.8e-6 (both start markers)
    assert rel_error < 1e-5
    assert rel_improvement < 1e-5


@pytest.mark.parametrize("optimiser_type", ["adam", "lbfgs"])
def test_controller_quad_opt_simple_without_early_stopping_reaches_truth(
    tmp_path: Path,
    seq_b1: Path,
    loaded_interface: AbaMadInterface,
    controller_test_mode: str,
    optimiser_type: str,
) -> None:
    """The SGD optimisers end to end (the other controller fits use Levenberg-Marquardt)."""
    ctrl, true_values = _build_lhc_quad_controller(
        tmp_path=tmp_path,
        seq_b1=seq_b1,
        loaded_interface=loaded_interface,
        start_marker="MSIA.EXIT.B1",
        # Never stop early: run every epoch.
        optimiser_config=dataclasses.replace(
            _make_optimiser_config_quad(),
            loss_change_tolerance=0.0,
            gradient_converged_value=-1.0,
            optimiser_type=optimiser_type,
            **LBFGS_SETTINGS.get(optimiser_type, {}),
        ),
    )
    if controller_test_mode == "loss_regression":
        initial_loss = evaluate_controller_worker_loss(ctrl, ctrl.machine.initial_knobs)
        true_loss = evaluate_controller_worker_loss(ctrl, true_values)
        assert true_loss < initial_loss * 1e-6
        return

    estimate = ctrl.run().knobs
    rel_error, rel_improvement = relative_truth_errors(
        estimate, true_values, ctrl.machine.initial_knobs
    )
    # measured 2026-11-12, 300 epochs: adam 8.2e-3, lbfgs 4.9e-5 (rel error = rel improvement)
    limit = {"adam": 2e-2, "lbfgs": 2e-4}[optimiser_type]
    assert rel_error < limit
    assert rel_improvement < limit
