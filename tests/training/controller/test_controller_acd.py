"""ACD-focused integration tests for controller logic."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import xtrack as xt

from accelmo.config import SimulationConfig
from accelmo.fitting.config import OutputConfig, SequenceConfig
from accelmo.machine.accelerators import PSB
from accelmo.machine.machine_state import merge_machine_states
from accelmo.tracking.config.helpers import create_arc_measurement_config
from accelmo.tracking.fitter import ACDMarkerFitter, FitterOptions
from tests.training.controller_test_utils import (
    _load_mad_twiss_for_tracking,
    _make_simulation_config_quad,
    evaluate_controller_worker_loss,
    make_lm_config,
    relative_truth_errors,
    strip_inline_flags,
)

if TYPE_CHECKING:
    from pathlib import Path

    from accelmo.backend.madng.aba_mad_interface import AbaMadInterface

from pymadng_utils.io.utils import save_knobs
from xtrack_tools.monitors import get_monitor_names_at_pattern, process_tracking_data
from xtrack_tools.tracking import run_tracking

from tests.training.helpers import TRACK_COLUMNS, generate_xsuite_env_with_errors

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.serial

PSB_HORIZONTAL_ACD_VOLTAGE = 7.65803463522e-05
PSB_VERTICAL_ACD_VOLTAGE = 4.86557716491e-05


def _generate_acd_track(
    interface_with_beam: AbaMadInterface,
    flattop_turns: int,
    destination: Path,
    dpp_value: float,
    bpm_pattern: str = r"(?i)br3\.bpm.*",
    apply_orbit_correction: bool = True,
    target_qx: float = 0.17,
    target_qy: float = 0.23,
    rel_error: float = 0.02,
    ramp_up_turns: int = 3,
) -> tuple[Path | None, dict[str, float], Path | None]:
    """Generate a parquet file with ACD-style tracking data (pre/post-kick markers)."""
    corrector_file: Path | None = None
    tune_knobs: Path | None = None
    if apply_orbit_correction:
        corrector_file = destination.parent / f"corrector_{destination.stem}.tfs"
    tune_knobs = destination.parent / f"tune_knobs_{destination.stem}.txt"

    env, magnet_strengths, matched_tunes, _ = generate_xsuite_env_with_errors(
        interface_with_beam,
        dpp_value=dpp_value,
        corrector_file=corrector_file,
        perturb_quads=True,
        apply_orbit_correction=apply_orbit_correction,
        target_qx=target_qx,
        target_qy=target_qy,
        rel_error=rel_error,
    )
    save_knobs(matched_tunes, tune_knobs)

    accelerator = interface_with_beam.accelerator
    acd_elem = accelerator.ac_dipole_name
    acd_after = accelerator.acd_marker_name("after")
    acd_before = accelerator.acd_marker_name("before")

    seq_name = accelerator.seq_name.lower()
    line: xt.Line = env[seq_name]

    mad_twiss = _load_mad_twiss_for_tracking(interface_with_beam, dpp_value)
    tws = line.twiss(method="4d")
    tt = line.get_table()
    s_acd = float(tt["s_center", acd_elem.lower()])
    s_before = s_acd - 1e-9
    s_after = s_acd + 1e-9

    acd_line = line.copy()

    elem_name_lower = acd_elem.lower()
    # The PSB AC dipole loads as a zero-length xsuite Marker, which carries no
    # `length` attribute at all; only the LHC element is a real thin kicker.
    elem_length = float(getattr(acd_line.element_dict[elem_name_lower], "length", 0.0))
    acd_line.env.elements[elem_name_lower + "_drift"] = xt.Drift(length=elem_length)
    acd_line.replace(elem_name_lower, elem_name_lower + "_drift")

    # MAD-X monitor elements load as zero-length Drifts in xsuite; merge_consecutive_drifts
    # would absorb them. Convert BPMs to Markers so they survive the merge.
    bpm_pattern_clean = strip_inline_flags(bpm_pattern)
    for name in list(acd_line.element_names):
        if re.match(rf"(?i:{bpm_pattern_clean})", name):
            acd_line.env.element_dict[name] = xt.Marker()

    acd_line.merge_consecutive_drifts()

    # Insert thin marker at _before (pre-kick observation point)
    acd_line.env.elements[acd_before] = xt.Marker()
    acd_line.insert(acd_line.env.place(acd_before, at=s_before))

    driven_tunes = (
        0.16,
        0.24,
    )  # Choose driven tunes to be close to but distinct from the natural tunes
    qxd_qx = driven_tunes[0] - tws["qx"] % 1
    qyd_qx = driven_tunes[1] - tws["qy"] % 1
    logger.info(f"Qxd/Qx: {qxd_qx}, Qyd/Qx: {qyd_qx}")

    # Ramp the exciter up over a few turns before the flattop. The first physical
    # turn always sees zero kick (the beam starts on the closed orbit) and the pre-kick
    # 'before' marker lags a further turn behind, so these warm-up turns are discarded
    # below; only the fully-driven flattop turns are kept.
    ramp_profile = [
        0,
        ramp_up_turns,
        ramp_up_turns + flattop_turns,
        ramp_up_turns + flattop_turns + 1,
    ]
    acd_line.env.elements[f"{elem_name_lower}h"] = xt.ACDipole(
        plane="x",
        volt=PSB_HORIZONTAL_ACD_VOLTAGE,
        freq=driven_tunes[0],
        lag=0,
        ramp=ramp_profile,
    )
    acd_line.env.elements[f"{elem_name_lower}v"] = xt.ACDipole(
        plane="y",
        volt=PSB_VERTICAL_ACD_VOLTAGE,
        freq=driven_tunes[1],
        lag=0,
        ramp=ramp_profile,
    )
    acd_line.insert(f"{elem_name_lower}v", at=s_acd)
    acd_line.insert(f"{elem_name_lower}h", at=s_acd)

    # Insert thin marker at _after (post-kick observation point, just after the exciter)
    acd_line.env.elements[acd_after] = xt.Marker()
    acd_line.insert(acd_line.env.place(acd_after, at=s_after))

    monitor_pattern = rf"(?i:{bpm_pattern_clean}|{re.escape(acd_before)}|{re.escape(acd_after)})"
    monitor_names = get_monitor_names_at_pattern(acd_line, monitor_pattern)

    start_elem = acd_line.element_names[0].upper()
    co_row = mad_twiss.loc[start_elem] if start_elem in mad_twiss.index else mad_twiss.iloc[0]
    particles: xt.Particles = acd_line.build_particles(
        x=float(co_row["x"]),
        px=float(co_row["px"]),
        y=float(co_row["y"]),
        py=float(co_row["py"]),
        delta=dpp_value,
    )

    monitored_line = run_tracking(
        line=acd_line,
        particles=particles,
        nturns=ramp_up_turns + flattop_turns,
        monitor_names=monitor_names,
    )
    tracking_df = process_tracking_data(
        monitored_line,
        ramp_turns=ramp_up_turns,
        flattop_turns=flattop_turns,
        add_variance_columns=True,
    )
    tracking_df["bunch_number"] = 0
    tracking_df = tracking_df.loc[:, TRACK_COLUMNS].copy()
    tracking_df["name"] = tracking_df["name"].astype(str)

    # process_tracking_data uppercases names; rename ACD markers back to lowercase
    # to match MAD-NG's naming convention (acd_marker_name returns lowercase suffix)
    tracking_df["name"] = tracking_df["name"].replace(
        {
            acd_before.upper(): acd_before,
            acd_after.upper(): acd_after,
        }
    )

    if acd_after not in set(tracking_df["name"]):
        raise ValueError(f"ACD 'after' marker {acd_after!r} missing from tracking output")
    if acd_before not in set(tracking_df["name"]):
        raise ValueError(f"ACD 'before' marker {acd_before!r} missing from tracking output")

    print(f"All BPMs in tracking data: {set(tracking_df['name'])}")

    tracking_df.to_parquet(destination, index=False)
    return corrector_file, magnet_strengths, tune_knobs


def _build_acd_controller(
    *,
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: AbaMadInterface,
    flattop_turns: int = 10,
) -> tuple[ACDMarkerFitter, dict[str, float]]:
    magnet_range = "$start/$end"
    off_magnet_path = tmp_path / "track_acd_off_magnet.parquet"

    corrector_file, magnet_strengths, tune_knobs = _generate_acd_track(
        loaded_psb_interface,
        flattop_turns,
        off_magnet_path,
        dpp_value=0.0,
        bpm_pattern=r"(?i)br3\.bpm.*",
    )

    optimiser_config = make_lm_config()

    ctrl = ACDMarkerFitter(
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
        options=FitterOptions(
            output_config=OutputConfig(
                mad_logfile=tmp_path / "mad_logfile_acd.log",
                write_run_log=False,
            ),
            true_strengths=magnet_strengths.copy(),
        ),
    )
    return ctrl, magnet_strengths.copy()


def test_controller_quad_opt_with_acd(
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: AbaMadInterface,
    controller_test_mode: str,
) -> None:
    loss_regression = controller_test_mode == "loss_regression"
    ctrl, magnet_strengths = _build_acd_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
        flattop_turns=4 if loss_regression else 10,
    )
    logger.info(
        "Starting ACD controller with logfile at %s",
        tmp_path / "mad_logfile_acd.log",
    )

    initial_loss = evaluate_controller_worker_loss(ctrl, ctrl.machine.initial_knobs)

    if loss_regression:
        true_loss = evaluate_controller_worker_loss(ctrl, magnet_strengths)
        assert true_loss < 5e-15, (
            "True-strength ACD loss should be near the xsuite/MAD-NG integrator floor (~5e-15), "
            f"got {true_loss:.3e} (initial={initial_loss:.3e})"
        )
        assert true_loss < initial_loss * 1e-3, (
            "ACD loss should strongly prefer the true quadrupole strengths "
            f"(initial={initial_loss:.3e}, true={true_loss:.3e})"
        )
        return

    estimate = ctrl.run().knobs
    rel_error, rel_improvement = relative_truth_errors(
        estimate, magnet_strengths, ctrl.machine.initial_knobs
    )
    # measured 2026-11-12: rel error = rel improvement = 2.5e-6 (knobs start at 0)
    assert rel_error < 1e-5
    assert rel_improvement < 1e-5


GAINS = {"BR3.BPM1L3": 0.03, "BR3.BPM4L3": -0.02}


def test_lm_gain_prior_does_not_depend_on_particles_per_worker(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: AbaMadInterface
) -> None:
    """Same noise-free ACD data and gain prior on 2 workers (40 training turns each) or 8 workers
    (10 each, an even split): the chi-square is the same, so the fitted gains must be too."""
    data = tmp_path / "track_acd_gains.parquet"
    corrector_file, strengths, tune_knobs = _generate_acd_track(
        loaded_psb_interface, 42, data, dpp_value=0.0, bpm_pattern=r"(?i)br3\.bpm.*"
    )
    frame = pd.read_parquet(data)
    for bpm, gain in GAINS.items():  # the BPMs report (1 + g) x position
        frame.loc[frame["name"] == bpm, ["x", "y"]] *= 1.0 + gain
    frame.to_parquet(data, index=False)

    def fitted_gains(num_workers: int) -> dict[str, float]:
        ctrl = ACDMarkerFitter(
            PSB(
                ring=3,
                kinetic_energy=loaded_psb_interface.accelerator.kinetic_energy,
                sequence_file=seq_psb,
                errors={"quad": {"k1"}},
            ),
            make_lm_config(),
            SimulationConfig(
                num_workers=num_workers,
                num_batches=1,
                validation_fraction=0.0,
                bpm_gain_sigma=0.01,
            ),
            SequenceConfig(magnet_range="$start/$end"),
            create_arc_measurement_config(
                data, machine_state=merge_machine_states(corrector_file, tune_knobs)
            ),
            options=FitterOptions(
                initial_knob_strengths=strengths,
                output_config=OutputConfig(mad_logfile=tmp_path / "mad.log", write_run_log=False),
            ),
        )
        return ctrl.run().extra

    few, many = fitted_gains(2), fitted_gains(8)
    for plane in ("x", "y"):
        for bpm, gain in GAINS.items():
            name = f"bpmgain.{plane}.{bpm}"
            print(
                f"{name}: truth {gain:+.4e} 2 workers {few[name]:+.6e} 8 workers {many[name]:+.6e}"
            )
            assert few[name] == pytest.approx(many[name], rel=1e-6)


def test_lm_knob_prior_shifts_a_knob_by_the_gaussian_posterior_mean(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: AbaMadInterface
) -> None:
    """Gaussian posterior: a zero-centred prior of width s on a knob fitted at v0 with data width
    h = sqrt((A^-1)_kk) (A = J^T W J in chi-square units) shifts it by v0 h^2 / (s^2 + h^2).
    s = 30 h keeps the shift (0.4 h) linear. Checks the prior is in the data's chi-square units,
    on 2 and 8 workers."""
    data = tmp_path / "track_acd_prior.parquet"
    corrector_file, strengths, tune_knobs = _generate_acd_track(
        loaded_psb_interface, 42, data, dpp_value=0.0, bpm_pattern=r"(?i)br3\.bpm.*"
    )

    def fitter(num_workers: int, prior: dict[str, float] | None) -> ACDMarkerFitter:
        return ACDMarkerFitter(
            PSB(
                ring=3,
                kinetic_energy=loaded_psb_interface.accelerator.kinetic_energy,
                sequence_file=seq_psb,
                errors={"quad": {"k1"}},
            ),
            make_lm_config(),
            SimulationConfig(
                num_workers=num_workers,
                num_batches=1,
                validation_fraction=0.0,
                knob_sigmas=prior,
            ),
            SequenceConfig(magnet_range="$start/$end"),
            create_arc_measurement_config(
                data, machine_state=merge_machine_states(corrector_file, tune_knobs)
            ),
            options=FitterOptions(
                initial_knob_strengths=strengths,
                output_config=OutputConfig(mad_logfile=tmp_path / "mad.log", write_run_log=False),
            ),
        )

    knobs = fitter(2, None).run().knobs
    normal, names = fitter(2, None).build_initial_normal_matrix(knobs)  # A at the fit
    widths = np.sqrt(np.diag(np.linalg.inv(normal)))
    k = max(range(len(names)), key=lambda i: abs(knobs[names[i]]) / widths[i])
    name, v0, h = names[k], knobs[names[k]], float(widths[k])
    sigma = 30.0 * h
    expected = v0 * h**2 / (sigma**2 + h**2)
    for workers in (2, 8):
        shift = v0 - fitter(workers, {name: sigma}).run().knobs[name]
        print(
            f"{name}: v0 {v0:+.6e} h {h:.3e}, {workers} workers shift {shift:+.6e} expected {expected:+.6e}"
        )
        assert shift == pytest.approx(expected, rel=1e-3)
