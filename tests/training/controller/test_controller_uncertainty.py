from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from accelmo.config import TrackingDataConfig, WorkerRuntime
from accelmo.fitting.config import OutputConfig
from accelmo.fitting.protocol import Evaluate
from accelmo.fitting.uncertainty import HESSIAN_MIN_EIGENVALUE, sandwich_uncertainties
from accelmo.machine.machine_state import merge_machine_states
from accelmo.tracking.fitter import ArcByArcFitter, FitterOptions, TrackingFitter
from accelmo.tracking.uncertainty import (
    UncertaintyPart,
    merge_uncertainty_parts,
    noise_matrix,
)


def test_sandwich_uncertainties_are_finite_and_non_negative_for_an_indefinite_normal_matrix() -> (
    None
):
    uncertainties = sandwich_uncertainties(
        np.array([[4.0, 0.0], [0.0, -1e-12]], dtype=np.float64),
        np.diag([4.0, HESSIAN_MIN_EIGENVALUE]),
    )

    assert np.all(np.isfinite(uncertainties))
    assert np.all(uncertainties >= 0.0)
    assert np.isclose(uncertainties[0], 0.5)
    assert np.isclose(uncertainties[1], 1.0 / np.sqrt(HESSIAN_MIN_EIGENVALUE))


def test_shared_reading_noise_is_summed_before_squaring() -> None:
    # Two workers observe reading 7; worker b also starts from reading 9.
    g_a, g_b, g_start = np.array([1.0, 2.0]), np.array([3.0, -1.0]), np.array([0.5, 0.5])
    part_a = UncertaintyPart(np.eye(2), np.array([7]), g_a[None, :], np.array([0.1]))
    part_b = UncertaintyPart(
        2.0 * np.eye(2),
        np.array([7, 9, 11]),
        np.stack([g_b, g_start, g_start]),
        np.array([0.1, 0.2, np.inf]),
    )

    merged = merge_uncertainty_parts([part_a, part_b], n_knobs=2)

    shared = g_a + g_b
    np.testing.assert_allclose(merged.normal, 3.0 * np.eye(2))
    # The unobserved reading 11 (infinite variance) carries no noise.
    np.testing.assert_allclose(
        noise_matrix(merged), 0.1 * np.outer(shared, shared) + 0.2 * np.outer(g_start, g_start)
    )


def test_incremental_merge_matches_one_shot_merge() -> None:
    # The manager folds workers into a running merge chunk by chunk.
    rng = np.random.default_rng(1)
    parts = [
        UncertaintyPart(
            np.eye(3) * (i + 1),
            rng.integers(0, 30, size=50),
            rng.normal(size=(50, 3)),
            np.full(50, 0.3),
        )
        for i in range(5)
    ]

    one_shot = merge_uncertainty_parts(parts, n_knobs=3)
    running = merge_uncertainty_parts(parts[:2], n_knobs=3)
    running = merge_uncertainty_parts([*parts[2:4], running], n_knobs=3)
    running = merge_uncertainty_parts([running, parts[4]], n_knobs=3)

    np.testing.assert_array_equal(running.reading_ids, one_shot.reading_ids)
    np.testing.assert_allclose(running.normal, one_shot.normal, rtol=1e-14)
    np.testing.assert_allclose(noise_matrix(running), noise_matrix(one_shot), rtol=1e-12)


def test_sandwich_reduces_to_inverse_normal_matrix_for_independent_readings() -> None:
    rng = np.random.default_rng(0)
    jacobian = rng.normal(size=(40, 3))
    variances = rng.uniform(0.5, 2.0, size=40)
    # Each reading is used once with w = 1/σ²: B = A, so Cov = A⁻¹.
    part = UncertaintyPart(
        jacobian.T @ (jacobian / variances[:, None]),
        np.arange(40),
        -jacobian / variances[:, None],
        variances,
    )

    merged = merge_uncertainty_parts([part], n_knobs=3)
    sigmas = sandwich_uncertainties(merged.normal, noise_matrix(merged))

    np.testing.assert_allclose(sigmas, np.sqrt(np.diag(np.linalg.inv(part.normal))))


def _collect_epoch_gradient(ctrl: TrackingFitter, knob_updates: dict[str, float]) -> np.ndarray:
    """The training chi² gradient summed over all workers and batches."""
    gradient = np.zeros(len(ctrl.machine.knob_names), dtype=np.float64)
    channels = ctrl.session.training.channels
    for batch in range(ctrl.tracking_data.num_batches):
        channels.send_all(Evaluate(knob_updates, batch))
        for result in channels.recv_all():
            gradient += np.asarray(result.grad, dtype=np.float64).reshape(-1)
    return gradient


#: Norm-relative Hessian error budgets at truth: the measured central-difference error
#: (LHC 4.8e-7, PSB 1.7e-9) with a factor of about 10 of margin.
LHC_HESSIAN_REL_TOL = 5e-6
PSB_HESSIAN_REL_TOL = 2e-8
#: Measured miss of the Gauss-Newton Hessian 5% off the truth: 1.06e-5, above the true Hessian's tolerance.
GAUSS_NEWTON_MISS_REL = 5e-6
#: Single-direction runs, measured: A vs finite differences 5.3e-9, B vs A 6.9e-16,
#: sigma vs sqrt(diag(inv(A))) 6.1e-8.
DIRECTION_HESSIAN_REL_TOL = 5e-8
DIRECTION_NOISE_REL_TOL = 1e-12
DIRECTION_SIGMA_REL_TOL = 1e-6


def _finite_difference_hessian(
    ctrl: TrackingFitter, base_knobs: dict[str, float], indices: np.ndarray
) -> np.ndarray:
    """Central differences of the summed training gradient in the knobs ``indices``."""
    names = ctrl.machine.knob_names
    matrix = np.zeros((indices.size, indices.size), dtype=np.float64)
    for col, knob_idx in enumerate(indices):
        name = names[knob_idx]
        step = max(1e-6, 1e-2 * max(abs(base_knobs[name]), 1e-4))
        plus_knobs = base_knobs.copy()
        minus_knobs = base_knobs.copy()
        plus_knobs[name] += step
        minus_knobs[name] -= step
        grad_plus = _collect_epoch_gradient(ctrl, plus_knobs)
        grad_minus = _collect_epoch_gradient(ctrl, minus_knobs)
        matrix[:, col] = (grad_plus[indices] - grad_minus[indices]) / (2.0 * step)
    return matrix


def _assert_matrices_close(measured: np.ndarray, predicted: np.ndarray, rel_tol: float) -> None:
    """The Frobenius norm of the difference is below ``rel_tol`` of the predicted norm."""
    scale = np.linalg.norm(predicted)
    assert scale > 0.0
    assert np.linalg.norm(measured - predicted) < rel_tol * scale


#: Relative offset of the subset knobs from the truth in the off-truth Hessian test.
OFF_TRUTH_OFFSET = 0.05
#: Absolute offset of the fitted energy ``pt`` from the truth in the energy Hessian test.
PT_OFFSET = 1e-4


def _lhc_subset_hessians(
    tmp_path,
    seq_b1,
    loaded_interface,
    hessian: str,
    offset: float,
    *,
    optimise_energy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """``(2 A, finite-difference d(gradient)/d(knob))`` of three knobs, ``offset`` off the truth.

    The three knobs are two central ones, moved by ``offset`` of their value, and with
    ``optimise_energy`` the energy ``pt``, moved by ``PT_OFFSET``; the residuals are not zero there.
    """
    from accelmo.config import TrackingParameters
    from accelmo.fitting.config import SequenceConfig
    from accelmo.machine.accelerators import LHC
    from accelmo.tracking.config.helpers import single_measurement_config
    from tests.training.controller_test_utils import (
        _generate_nonoise_track,
        _make_optimiser_config_quad,
    )

    magnet_range = "BPM.13R1.B1/BPM.13L2.B1"
    bpm_start_points = ["BPM.13R1.B1"]
    bpm_end_points = ["BPM.13L2.B1"]
    flattop_turns = 64
    start_marker = "MSIA.EXIT.B1"
    measurement_file = tmp_path / "track_off_magnet.parquet"

    corrector_file, magnet_strengths, tune_knobs = _generate_nonoise_track(
        loaded_interface,
        flattop_turns,
        measurement_file,
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
            optimise_energy=optimise_energy,
        ),
        _make_optimiser_config_quad(),
        WorkerRuntime(num_workers=4),
        TrackingDataConfig(num_batches=2),
        SequenceConfig(magnet_range=magnet_range),
        single_measurement_config(
            measurement_file, machine_state=merge_machine_states(corrector_file, tune_knobs)
        ),
        bpm_start_points,
        bpm_end_points,
        options=FitterOptions(
            output_config=OutputConfig(
                mad_logfile=tmp_path / "controller_hessian.log",
                write_run_log=False,
            ),
            true_strengths=magnet_strengths.copy(),
            parameters=TrackingParameters(uncertainty_hessian=hessian),
        ),
    )

    terminated = False
    try:
        ctrl.start_session(ctrl.machine.initial_knobs)
        base_knobs = ctrl.machine.true_strengths.copy()
        n_knobs = len(ctrl.machine.knob_names)
        centre = n_knobs // 2
        if optimise_energy:
            assert ctrl.machine.knob_names[-1] == "pt"
            subset = np.array([centre - 1, centre, n_knobs - 1], dtype=int)
            base_knobs["pt"] = base_knobs.get("pt", 0.0) + PT_OFFSET
        else:
            subset = np.array([centre - 1, centre, centre + 1], dtype=int)
        for knob_idx in subset:
            name = ctrl.machine.knob_names[knob_idx]
            if name != "pt":
                base_knobs[name] += offset * abs(base_knobs[name])

        fd_matrix = _finite_difference_hessian(ctrl, base_knobs, subset)

        ctrl.session.set_training_knobs(base_knobs)
        total_hessian, _ = ctrl.session.stop_and_collect_uncertainty(n_knobs).primary
        terminated = True
    finally:
        if not terminated:
            ctrl.session.terminate()

    assert total_hessian.shape == (n_knobs, n_knobs)
    assert np.all(np.isfinite(total_hessian))
    sym_hessian = 0.5 * (total_hessian + total_hessian.T)
    return 2.0 * sym_hessian[np.ix_(subset, subset)], fd_matrix


@pytest.mark.slow
@pytest.mark.serial
def test_controller_worker_hessian_matches_finite_difference_on_reduced_knob_subset(
    tmp_path,
    seq_b1,
    loaded_interface,
) -> None:
    predicted, fd_matrix = _lhc_subset_hessians(
        tmp_path, seq_b1, loaded_interface, "true", offset=0.0
    )

    _assert_matrices_close(fd_matrix, predicted, LHC_HESSIAN_REL_TOL)


@pytest.mark.slow
@pytest.mark.serial
def test_true_hessian_matches_finite_difference_away_from_the_truth(
    tmp_path,
    seq_b1,
    loaded_interface,
) -> None:
    predicted, fd_matrix = _lhc_subset_hessians(
        tmp_path, seq_b1, loaded_interface, "true", offset=OFF_TRUTH_OFFSET
    )

    _assert_matrices_close(fd_matrix, predicted, LHC_HESSIAN_REL_TOL)


@pytest.mark.slow
@pytest.mark.serial
def test_true_hessian_with_the_fitted_energy_matches_finite_difference(
    tmp_path,
    seq_b1,
    loaded_interface,
) -> None:
    predicted, fd_matrix = _lhc_subset_hessians(
        tmp_path, seq_b1, loaded_interface, "true", offset=OFF_TRUTH_OFFSET, optimise_energy=True
    )

    _assert_matrices_close(fd_matrix, predicted, LHC_HESSIAN_REL_TOL)


@pytest.mark.slow
@pytest.mark.serial
def test_gauss_newton_hessian_misses_the_residual_term_away_from_the_truth(
    tmp_path,
    seq_b1,
    loaded_interface,
) -> None:
    predicted, fd_matrix = _lhc_subset_hessians(
        tmp_path, seq_b1, loaded_interface, "gauss_newton", offset=OFF_TRUTH_OFFSET
    )

    assert np.linalg.norm(fd_matrix - predicted) > GAUSS_NEWTON_MISS_REL * np.linalg.norm(fd_matrix)


PSB_READING_VARIANCE = (1e-4) ** 2


def _restrict_to_directions(monkeypatch, sdirs: tuple[int, ...]) -> None:
    """Plan only the workers tracking in the given directions (``+1`` forward, ``-1`` backward)."""
    from accelmo.tracking.config.tracking import TrackingPlan

    all_specs = TrackingPlan.build_range_specs
    monkeypatch.setattr(
        TrackingPlan,
        "build_range_specs",
        lambda self, ctx: [spec for spec in all_specs(self, ctx) if spec.sdir in sdirs],
    )


def _make_psb_fitter(
    tmp_path,
    seq_psb,
    loaded_psb_interface,
    *,
    bpm_start_points: list[str],
    bpm_end_points: list[str],
    num_workers: int,
    num_batches: int,
    exact_start_bpms: tuple[str, ...] = (),
    exact_start_columns: tuple[str, ...] = ("x", "y", "px", "py"),
    flattop_turns: int = 256,
    validation_fraction: float = 0.1,
    hessian: str = "true",
):
    """A PSB ring-3 arc-by-arc quadrupole fitter on noise-free data, and its measurement file.

    ``exact_start_bpms`` are never observed: their declared variance is infinite.
    """
    from accelmo.config import OptimiserConfig, TrackingParameters
    from accelmo.fitting.config import SequenceConfig
    from accelmo.machine.accelerators import PSB
    from accelmo.tracking.config.helpers import single_measurement_config
    from tests.training.controller_test_utils import (
        _generate_nonoise_track,
    )

    measurement_file = tmp_path / "track_off_magnet_psb.parquet"
    corrector_file, magnet_strengths, tune_knobs = _generate_nonoise_track(
        loaded_psb_interface,
        flattop_turns,
        measurement_file,
        0.0,
        perturb_quads=True,
        bpm_pattern=r"br3\.bpm.*",
        apply_orbit_correction=False,
        target_qx=0.17,
        target_qy=0.225,
    )
    if exact_start_bpms:
        track = pd.read_parquet(measurement_file)
        exact = track["name"].str.upper().isin([name.upper() for name in exact_start_bpms])
        track.loc[exact, [f"var_{column}" for column in exact_start_columns]] = np.inf
        track.to_parquet(measurement_file, index=False)
    optimiser_config = OptimiserConfig(
        max_epochs=1,
        warmup_epochs=0,
        warmup_lr_start=1e-6,
        max_lr=3e-4,
        min_lr=3e-4,
        optimiser_type="adam",
    )
    ctrl = ArcByArcFitter(
        PSB(
            ring=3,
            kinetic_energy=loaded_psb_interface.accelerator.kinetic_energy,
            sequence_file=seq_psb,
            errors={"quad": {"k1"}},
        ),
        optimiser_config,
        WorkerRuntime(num_workers=num_workers),
        TrackingDataConfig(num_batches=num_batches, validation_fraction=validation_fraction),
        SequenceConfig("$start/$end"),
        single_measurement_config(
            measurement_file, machine_state=merge_machine_states(corrector_file, tune_knobs)
        ),
        bpm_start_points,
        bpm_end_points,
        options=FitterOptions(
            output_config=OutputConfig(
                mad_logfile=tmp_path / "controller_psb_hessian.log", write_run_log=False
            ),
            true_strengths=magnet_strengths.copy(),
            parameters=TrackingParameters(uncertainty_hessian=hessian),
        ),
    )
    return ctrl, measurement_file


@pytest.mark.slow
@pytest.mark.serial
def test_controller_worker_hessian_matches_finite_difference_for_psb_100um_variances(
    tmp_path,
    seq_psb,
    loaded_psb_interface,
) -> None:
    ctrl, measurement_file = _make_psb_fitter(
        tmp_path,
        seq_psb,
        loaded_psb_interface,
        bpm_start_points=["BR3.BPM1L3", "BR3.BPM5L3", "BR3.BPM9L3"],
        bpm_end_points=["BR3.BPM13L3", "BR3.BPM15L3", "BR3.BPM16L3"],
        num_workers=4,
        num_batches=4,
    )
    measurement_df = pd.read_parquet(measurement_file)
    assert np.allclose(measurement_df["var_x"].dropna().to_numpy(), PSB_READING_VARIANCE)
    assert np.allclose(measurement_df["var_y"].dropna().to_numpy(), PSB_READING_VARIANCE)

    terminated = False
    try:
        ctrl.start_session(ctrl.machine.initial_knobs)
        base_knobs = ctrl.machine.true_strengths.copy()
        base_vec = np.array(
            [base_knobs[name] for name in ctrl.machine.knob_names],
            dtype=np.float64,
        )
        n_knobs = len(base_vec)
        centre = n_knobs // 2
        subset = np.array([centre - 1, centre, centre + 1], dtype=int)

        fd_matrix = _finite_difference_hessian(ctrl, base_knobs, subset)

        ctrl.session.set_training_knobs(base_knobs)
        total_hessian, _ = ctrl.session.stop_and_collect_uncertainty(n_knobs).primary
        terminated = True
    finally:
        if not terminated:
            ctrl.session.terminate()

    assert total_hessian.shape == (n_knobs, n_knobs)
    assert np.all(np.isfinite(total_hessian))
    sym_hessian = 0.5 * (total_hessian + total_hessian.T)
    eigenvalues = np.linalg.eigvalsh(sym_hessian)
    assert np.min(eigenvalues) >= -1e-9

    # No range crosses s = 0, so the quadrupoles outside BPM1L3..BPM16L3 (QFO11 before
    # the first BPM, QDE16 and QFO162 after the last) are never observed.
    row_norms = np.linalg.norm(sym_hessian, axis=1)
    zero_row_knobs = {ctrl.machine.knob_names[idx] for idx in np.where(row_norms == 0.0)[0]}
    assert zero_row_knobs == {
        "BR.QFO11.dk1l",
        "BR.QDE16.dk1l",
        "BR.QFO162.dk1l",
    }
    positive_modes = eigenvalues[eigenvalues > 1e-9]
    assert positive_modes.size == n_knobs - len(zero_row_knobs)

    predicted = 2.0 * sym_hessian[np.ix_(subset, subset)]

    _assert_matrices_close(fd_matrix, predicted, PSB_HESSIAN_REL_TOL)


@pytest.mark.slow
@pytest.mark.serial
@pytest.mark.parametrize("sdir", [1, -1], ids=["forward", "backward"])
def test_single_direction_sigma_is_the_inverse_finite_difference_normal_matrix(
    tmp_path, seq_psb, loaded_psb_interface, monkeypatch, sdir
) -> None:
    """One direction with an exact start reads every point once: noise ``B`` equals the Gauss-Newton ``A``.

    The reported sigma is then ``sqrt(diag(inv(A)))`` with ``A = JᵀWJ`` from finite differences.
    Forward and backward workers sharing readings are pinned by the merge unit tests above.
    """
    _restrict_to_directions(monkeypatch, (sdir,))
    start_bpm, end_bpm = "BR3.BPM1L3", "BR3.BPM16L3"
    ctrl, _ = _make_psb_fitter(
        tmp_path,
        seq_psb,
        loaded_psb_interface,
        bpm_start_points=[start_bpm],
        bpm_end_points=[end_bpm],
        num_workers=1,
        num_batches=2,
        exact_start_bpms=(start_bpm if sdir > 0 else end_bpm,),
        hessian="gauss_newton",
    )
    n_knobs = len(ctrl.machine.knob_names)
    terminated = False
    try:
        ctrl.start_session(ctrl.machine.initial_knobs)
        assert [config.geometry.sdir for config in ctrl.session.training.metadata] == [sdir]
        base_knobs = ctrl.machine.true_strengths.copy()
        fd_normal = 0.5 * _finite_difference_hessian(ctrl, base_knobs, np.arange(n_knobs))
        ctrl.session.set_training_knobs(base_knobs)
        normal, noise = ctrl.session.stop_and_collect_uncertainty(n_knobs).primary
        terminated = True
    finally:
        if not terminated:
            ctrl.session.terminate()

    observed = np.flatnonzero(np.linalg.norm(normal, axis=1) > 0.0)
    fd_normal = fd_normal[np.ix_(observed, observed)]
    normal = normal[np.ix_(observed, observed)]
    noise = noise[np.ix_(observed, observed)]
    assert observed.size == n_knobs - 3
    _assert_matrices_close(normal, fd_normal, DIRECTION_HESSIAN_REL_TOL)
    _assert_matrices_close(noise, normal, DIRECTION_NOISE_REL_TOL)
    expected = np.sqrt(np.diag(np.linalg.inv(0.5 * (fd_normal + fd_normal.T))))
    np.testing.assert_allclose(
        sandwich_uncertainties(normal, noise), expected, rtol=DIRECTION_SIGMA_REL_TOL
    )


PROBE_TURNS = 4
#: Probe shift as a fraction of the reading's sigma. The one-sided probe truncates as the
#: shift and float noise takes over below about 1e-6, so 1e-5 sits at the minimum.
PROBE_SHIFT_SIGMA = 1e-5
#: Measured at that shift: exact starts 5e-10, noisy starts 4e-6 to 8e-6 (probe truncation).
PROBE_EXACT_REL_TOL = 1e-8
PROBE_NOISY_REL_TOL = 1e-4


def _truth_gradient(ctrl, track_file, knobs: dict[str, float]) -> np.ndarray:
    """The raw training gradient at ``knobs`` for the data in ``track_file``."""
    from accelmo.tracking.config.models import MeasurementConfig

    ctrl.measurement_config = MeasurementConfig({track_file: ctrl.measurement_config.details[0]})
    ctrl.data_manager = None
    ctrl.start_session(ctrl.machine.initial_knobs)
    try:
        return _collect_epoch_gradient(ctrl, knobs)
    finally:
        ctrl.session.terminate()


@pytest.mark.slow
@pytest.mark.serial
@pytest.mark.parametrize(
    ("sdirs", "exact_start"),
    [
        pytest.param((1,), True, id="forward-exact_start"),
        pytest.param((1,), False, id="forward-noisy_start"),
        pytest.param((-1,), False, id="backward-noisy_start"),
        pytest.param((1, -1), True, id="both-exact_start"),
        pytest.param((1, -1), False, id="both-noisy_start"),
    ],
)
def test_reported_noise_matrix_matches_single_reading_probes(
    tmp_path, seq_psb, loaded_psb_interface, monkeypatch, sdirs, exact_start
) -> None:
    """``B = Σ σ² G Gᵀ`` rebuilt by shifting one reading at a time.

    At truth the gradient responds linearly, ``G_r = Δg / δ``, through the real data path.
    With both directions the workers share readings, and with noisy starts the start
    coordinates (x, px, y, py) move every downstream residual; ``B`` differs from the normal
    matrix ``A`` in both cases, and the probes must reproduce it.
    """
    _restrict_to_directions(monkeypatch, sdirs)
    ctrl, measurement_file = _make_psb_fitter(
        tmp_path,
        seq_psb,
        loaded_psb_interface,
        bpm_start_points=["BR3.BPM1L3"],
        bpm_end_points=["BR3.BPM16L3"],
        exact_start_bpms=("BR3.BPM1L3", "BR3.BPM16L3") if exact_start else (),
        num_workers=len(sdirs),
        num_batches=1,
        flattop_turns=PROBE_TURNS,
        validation_fraction=0.0,
    )
    n_knobs = len(ctrl.machine.knob_names)
    base_knobs = ctrl.machine.true_strengths.copy()
    terminated = False
    try:
        ctrl.start_session(ctrl.machine.initial_knobs)
        base_gradient = _collect_epoch_gradient(ctrl, base_knobs)
        ctrl.session.set_training_knobs(base_knobs)
        _, noise = ctrl.session.stop_and_collect_uncertainty(n_knobs).primary
        terminated = True
    finally:
        if not terminated:
            ctrl.session.terminate()

    track = pd.read_parquet(measurement_file)
    probe_file = tmp_path / "probe.parquet"
    expected = np.zeros((n_knobs, n_knobs))
    for column in ("x", "px", "y", "py"):
        variances = track[f"var_{column}"].to_numpy()
        for row in np.flatnonzero(np.isfinite(variances) & (variances > 0.0)):
            shift = PROBE_SHIFT_SIGMA * np.sqrt(variances[row])
            probe = track.copy()
            probe.iloc[row, probe.columns.get_loc(column)] += shift
            probe.to_parquet(probe_file, index=False)
            gradient = _truth_gradient(ctrl, probe_file, base_knobs)
            sensitivity = 0.5 * (gradient - base_gradient) / shift
            expected += variances[row] * np.outer(sensitivity, sensitivity)

    _assert_matrices_close(
        noise, expected, PROBE_EXACT_REL_TOL if exact_start else PROBE_NOISY_REL_TOL
    )


@pytest.mark.slow
@pytest.mark.serial
def test_recomputing_both_hessians_in_one_session_leaves_gauss_newton_unchanged(
    tmp_path, seq_psb, loaded_psb_interface
) -> None:
    """Both Hessians come from one session, and the Gauss-Newton one equals its single-mode run."""
    ctrl, _ = _make_psb_fitter(
        tmp_path,
        seq_psb,
        loaded_psb_interface,
        bpm_start_points=["BR3.BPM1L3"],
        bpm_end_points=["BR3.BPM16L3"],
        num_workers=1,
        num_batches=2,
        exact_start_bpms=("BR3.BPM1L3",),
    )
    knobs = ctrl.machine.true_strengths.copy()

    both = ctrl.recompute_uncertainties(knobs)
    single = ctrl.recompute_uncertainties(knobs, ("gauss_newton",))

    assert list(both) == ["gauss_newton", "true"]
    assert list(single) == ["gauss_newton"]
    observed = [name for name, sigma in both["gauss_newton"].items() if np.isfinite(sigma)]
    assert observed
    for name in observed:
        assert both["gauss_newton"][name] == single["gauss_newton"][name]
        # noise-free data at the truth: the residual term is numerical noise only
        np.testing.assert_allclose(both["true"][name], both["gauss_newton"][name], rtol=1e-6)
