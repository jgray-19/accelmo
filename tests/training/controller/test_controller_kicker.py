"""Kicker-focused integration tests for controller logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

from accelmo.config import TrackingParameters
from accelmo.fitting.config import OutputConfig, SequenceConfig
from accelmo.machine.accelerators import PSB
from accelmo.machine.machine_state import merge_machine_states
from accelmo.tracking.config.helpers import single_measurement_config
from accelmo.tracking.fitter import FitterOptions, KickerFitter, tracking_lm_config
from tests.training.controller_test_utils import (
    LM_TEST_ITERATIONS,
    QUAD_RUNTIME,
    QUAD_TRACKING_DATA,
    UNIT_LOSS,
    _generate_kicker_track,
    evaluate_controller_worker_loss,
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
    **parameters: object,
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
    optimiser_config = tracking_lm_config(LM_TEST_ITERATIONS)

    ctrl = KickerFitter(
        PSB(
            ring=3,
            kinetic_energy=loaded_psb_interface.accelerator.kinetic_energy,
            sequence_file=seq_psb,
            errors={"quad": {"k1"}},
        ),
        optimiser_config,
        QUAD_RUNTIME,
        QUAD_TRACKING_DATA,
        SequenceConfig(magnet_range=magnet_range),
        single_measurement_config(
            off_magnet_path, machine_state=merge_machine_states(corrector_file, tune_knobs)
        ),
        kicker_name,
        flattop_turns,
        options=FitterOptions(
            output_config=OutputConfig(
                mad_logfile=tmp_path / "mad_logfile_kicker.log",
                write_run_log=False,
            ),
            true_strengths=magnet_strengths.copy(),
            parameters=TrackingParameters(**parameters),
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


def test_fitted_start_uncertainty_is_the_marginal_of_the_joint_fit(
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: MadInterface,
) -> None:
    """With ``fit_start_coords`` the start is a fit parameter, not a measured input.

    One noise-free kicker track: every reading is used once, so the knob covariance is the knob
    block of the inverse joint (knobs + start) normal matrix. The measured start variances
    must not enter. Gauss-Newton is pinned: ``lm_hessian`` is the Gauss-Newton Hessian, which
    the true Hessian differs from by the residual term.
    """
    ctrl, _truth = _build_kicker_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
        flattop_turns=4,
        fit_start_coords=True,
        uncertainty_hessian="gauss_newton",
    )
    result = ctrl.run()
    assert result.diagnostics.converged
    names = ctrl.knob_names
    n_knobs = len(names)
    joint = ctrl.lm_hessian  # chi² Hessian, knobs then the 4 start coordinates
    assert joint.shape == (n_knobs + 4, n_knobs + 4)

    # The uncertainty normal matrix is the joint one with the start projected out (Schur).
    k, s = slice(0, n_knobs), slice(n_knobs, None)
    schur = joint[k, k] - joint[k, s] @ np.linalg.solve(joint[s, s], joint[s, k])
    normal, _ = ctrl.build_initial_normal_matrix(start_knobs=result.knobs)
    scale = np.trace(normal) / np.trace(schur)
    np.testing.assert_allclose(scale * schur, normal, rtol=1e-9, atol=0)

    marginal = np.sqrt(np.diag(np.linalg.inv(scale * joint))[:n_knobs])
    reported = np.array([result.uncertainties[name] for name in names])
    np.testing.assert_allclose(reported, marginal, rtol=1e-9)


def test_knob_prior_bounds_the_reported_uncertainty(
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: MadInterface,
) -> None:
    """A Gaussian knob prior is information: the posterior sigma cannot exceed its width.

    Fitted starts: every reading is used once and none carries start noise, so the
    covariance is ``(A + P)⁻¹`` with ``P = 1/width²``, ``A`` the Gauss-Newton normal matrix.
    """
    free, _ = _build_kicker_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
        flattop_turns=4,
        fit_start_coords=True,
        uncertainty_hessian="gauss_newton",
    )
    free_result = free.run()
    data_sigma = np.array([free_result.uncertainties[name] for name in free.knob_names])
    normal, _ = free.build_initial_normal_matrix(start_knobs=free_result.knobs)
    np.testing.assert_allclose(np.sqrt(np.diag(np.linalg.inv(normal))), data_sigma, rtol=1e-9)
    width = 0.01 * data_sigma.min()

    ctrl, _ = _build_kicker_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
        flattop_turns=4,
        fit_start_coords=True,
        uncertainty_hessian="gauss_newton",
        knob_sigmas={"dk1l": width},
    )
    result = ctrl.run()
    assert result.diagnostics.converged
    reported = np.array([result.uncertainties[name] for name in ctrl.knob_names])
    # A + P is half the LM chi-square Hessian at the best point, starts projected out
    joint = ctrl.lm_hessian
    n_knobs = len(ctrl.knob_names)
    k, s = slice(0, n_knobs), slice(n_knobs, None)
    schur = joint[k, k] - joint[k, s] @ np.linalg.solve(joint[s, s], joint[s, k])
    posterior = np.sqrt(np.diag(np.linalg.inv(0.5 * schur)))
    assert np.all(reported < width)
    np.testing.assert_allclose(reported, posterior, rtol=1e-10)


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
        assert true_loss < 1e-16 * UNIT_LOSS, (
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
