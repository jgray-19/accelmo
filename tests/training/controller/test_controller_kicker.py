"""Kicker-focused integration tests for controller logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from accelmo.fitting.config import OutputConfig, SequenceConfig
from accelmo.machine.accelerators import PSB
from accelmo.machine.machine_state import merge_machine_states
from accelmo.tracking.config.helpers import create_arc_measurement_config
from accelmo.tracking.config.models import KickerConfig
from accelmo.tracking.fitter import FitterOptions, KickerFitter
from tests.training.controller_test_utils import (
    _generate_kicker_track,
    _make_simulation_config_quad,
    evaluate_controller_worker_loss,
    make_lm_config,
    relative_truth_errors,
)

if TYPE_CHECKING:
    from pathlib import Path

    from accelmo.backend.madng.interface import MadInterface

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.serial


def _build_kicker_controller(
    *,
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: MadInterface,
    flattop_turns: int = 10,
) -> tuple[KickerFitter, dict[str, float]]:
    magnet_range = "$start/$end"
    off_magnet_path = tmp_path / "track_kicker_off_magnet.parquet"

    corrector_file, magnet_strengths, tune_knobs, kicker_name = _generate_kicker_track(
        loaded_psb_interface,
        flattop_turns,
        off_magnet_path,
        dpp_value=0.0,
        bpm_pattern=r"(?i)br3\.bpm.*",
    )
    optimiser_config = make_lm_config()

    ctrl = KickerFitter(
        PSB(
            ring=3,
            kinetic_energy=loaded_psb_interface.accelerator.kinetic_energy,
            sequence_file=seq_psb,
            errors={"quad": {"k1"}},
        ),
        optimiser_config,
        _make_simulation_config_quad(),
        SequenceConfig(magnet_range=magnet_range),
        create_arc_measurement_config(
            off_magnet_path, machine_state=merge_machine_states(corrector_file, tune_knobs)
        ),
        KickerConfig(
            kicker_name=kicker_name,
            turns_after_kicker=flattop_turns,
        ),
        options=FitterOptions(
            output_config=OutputConfig(
                mad_logfile=tmp_path / "mad_logfile_kicker.log",
                write_run_log=False,
            ),
            true_strengths=magnet_strengths.copy(),
        ),
    )
    return ctrl, magnet_strengths.copy()


def test_controller_kicker_has_no_held_out_validation(
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: MadInterface,
) -> None:
    """The kicker plan disables validation, so no turns are held out.

    KickerTrackingPlan sets enable_validation=False: DataManager reserves no
    validation turns and TrackingSession.validation_loss returns None (the caller then
    falls back to training loss). This asserts the real behaviour of a Kicker-style
    run, complementing the arc-by-arc case where validation returns a real number.
    """
    ctrl, _magnet_strengths = _build_kicker_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
        flattop_turns=4,
    )
    assert not ctrl.tracking_plan.enable_validation
    assert ctrl.data_manager.validation_turn_batches == []

    ctrl.start_session(ctrl.machine.initial_knobs)
    try:
        assert ctrl.session.validation_loss(ctrl.machine.initial_knobs) is None
    finally:
        ctrl.session.terminate()


def test_check_degeneracy_finds_the_quad_behind_the_last_bpm(
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: MadInterface,
) -> None:
    """One turn from the kicker never reaches a BPM after BR.QDE1 (between BR3.BPM1L3 and the
    kicker), so its knob is the only flat direction; four turns constrain every quad."""
    one_turn, _ = _build_kicker_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
        flattop_turns=1,
    )
    report = one_turn.check_degeneracy()
    assert len(report.knob_names) == 48
    assert report.n_degenerate == 1
    assert report.zero_sensitivity_knobs == ["BR.QDE1.dk1l"]
    (flat,) = report.degenerate_directions
    assert abs(flat.relative_eigenvalue) < 1e-14
    qde1 = report.knob_names.index("BR.QDE1.dk1l")
    assert abs(abs(flat.components[qde1]) - 1.0) < 1e-12

    four_turns, _ = _build_kicker_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
        flattop_turns=4,
    )
    report = four_turns.check_degeneracy()
    assert not report.is_degenerate
    assert report.zero_sensitivity_knobs == []
    assert report.condition_number < 1e6


def test_controller_quad_opt_with_kicker(
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: MadInterface,
    controller_test_mode: str,
) -> None:
    loss_regression = controller_test_mode == "loss_regression"
    ctrl, magnet_strengths = _build_kicker_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
        flattop_turns=4 if loss_regression else 10,
    )
    logger.info(
        "Starting kicker controller with logfile at %s",
        tmp_path / "mad_logfile_kicker.log",
    )

    initial_loss = evaluate_controller_worker_loss(ctrl, ctrl.machine.initial_knobs)

    if loss_regression:
        true_loss = evaluate_controller_worker_loss(ctrl, magnet_strengths)
        # Round-off only: a run at the true strengths lands around 1e-18.
        assert true_loss < 1e-16, (
            f"True-strength kicker loss should be numerically tiny, got {true_loss:.3e}"
        )
        assert true_loss < initial_loss * 1e-3, (
            "Kicker loss should strongly prefer the true quadrupole strengths "
            f"(initial={initial_loss:.3e}, true={true_loss:.3e})"
        )
        return

    estimate = ctrl.run().knobs
    rel_error, rel_improvement = relative_truth_errors(
        estimate, magnet_strengths, ctrl.machine.initial_knobs
    )
    # measured 2026-11-12: rel error = rel improvement = 5.2e-3 (one weak knob, 10 turns)
    assert rel_error < 1e-2
    assert rel_improvement < 1e-2
