"""ACD sinusoid fit modes (ic_iteration) on xtrack truth."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from accelmo.config import OptimiserConfig, SimulationConfig
from accelmo.fitting.config import OutputConfig, SequenceConfig
from accelmo.machine.accelerators import PSB
from accelmo.machine.machine_state import merge_machine_states
from accelmo.tracking.config import MeasurementConfig
from accelmo.tracking.config.helpers import create_arc_measurement_config
from accelmo.tracking.fitter import ACDSinusoidFitter, FitterOptions
from accelmo.tracking.ic_iteration import State, Step, cold_joint, fit_step, joint
from tests.training.controller.test_controller_acd import (
    PSB_HORIZONTAL_ACD_VOLTAGE,
    PSB_VERTICAL_ACD_VOLTAGE,
    _generate_acd_track,
)
from tests.training.controller.test_controller_acd_sinusoid import (
    ACD_BEFORE,
    DRIVE_TUNES,
    FLATTOP_TURNS,
    RAMP_TURNS,
    START_OFFSET,
    _truth_acd_knobs,
    kick_error,
    kick_knobs,
)

if TYPE_CHECKING:
    from pathlib import Path

    from accelmo.backend.madng.interface import MadInterface

pytest.importorskip("xtrack_tools")
pytestmark = pytest.mark.serial

IC_ERRORS = {"acd": {"volt"}}
K_ERRORS = {"quad": {"k1"}}
WINDOW = 10
DROP_TURNS = 7


def _setup(
    tmp_path: Path,
    seq_psb: Path,
    iface: MadInterface,
    drop_turns: int = 0,
    gains: dict[str, float] | None = None,
    gain_sigma: float | None = None,
):
    """xtrack truth with offset measured start; returns factory, truth strengths, truth start.

    ``drop_turns`` > 0 adds a second file: the first without its first ``drop_turns`` turns.
    """
    data = tmp_path / "acd_sinusoid.parquet"
    correctors, strengths, tune_knobs = _generate_acd_track(
        iface, FLATTOP_TURNS, data, dpp_value=0.0, ramp_up_turns=RAMP_TURNS
    )
    frame = pd.read_parquet(data)
    start_row = (frame["name"] == ACD_BEFORE) & (frame["turn"] == 1)
    truth_start = frame.loc[start_row, list(START_OFFSET)].to_numpy()[0]
    for column, offset in START_OFFSET.items():
        frame.loc[start_row, column] += offset
    for bpm, gain in (gains or {}).items():  # the BPMs report (1 + g) x position
        frame.loc[frame["name"] == bpm, ["x", "y"]] *= 1.0 + gain
    frame.to_parquet(data, index=False)
    measurement = create_arc_measurement_config(
        data, machine_state=merge_machine_states(correctors, tune_knobs)
    )
    if drop_turns:
        later = tmp_path / "acd_sinusoid_later.parquet"
        frame[frame["turn"] > drop_turns].to_parquet(later, index=False)
        details = next(iter(measurement.measurements.values()))
        measurement = MeasurementConfig({data: details, later: details})

    def make_fitter(step: Step) -> ACDSinusoidFitter:
        accelerator = PSB(
            ring=3,
            kinetic_energy=iface.accelerator.kinetic_energy,
            sequence_file=seq_psb,
            errors=step.errors,
            acd_drive_tunes=(DRIVE_TUNES,) * (2 if drop_turns else 1),
        )
        return ACDSinusoidFitter(
            accelerator,
            OptimiserConfig(
                max_epochs=200,
                warmup_epochs=0,
                warmup_lr_start=1e-3,
                max_lr=1e-3,
                min_lr=1e-3,
                optimiser_type="levenberg_marquardt",
                gradient_converged_value=1e-14,
            ),
            SimulationConfig(
                num_workers=1,
                num_batches=1,
                n_run_turns=min(step.n_turns or FLATTOP_TURNS, FLATTOP_TURNS - 1),
                fit_start_coords=step.fit_start_coords,
                bpm_gain_sigma=gain_sigma,
            ),
            SequenceConfig(magnet_range="$start/$end"),
            measurement,
            options=FitterOptions(
                initial_knob_strengths=step.knobs,
                initial_starts=step.starts,
                output_config=OutputConfig(mad_logfile=tmp_path / "mad.log", write_run_log=False),
            ),
        )

    return make_fitter, strengths, truth_start


def _kick_start(scale: float = 0.9) -> dict[str, float]:
    return {name: scale * value for name, value in _truth_acd_knobs().items()}


def test_ic_step_recovers_start_and_kick_at_truth_k(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: MadInterface
) -> None:
    """IC step with truth k: start and kick back to xtrack truth."""
    make_fitter, strengths, truth_start = _setup(tmp_path, seq_psb, loaded_psb_interface)
    state = fit_step(
        make_fitter,
        State({**strengths, **_kick_start()}, None, np.inf),
        IC_ERRORS,
        fit_start_coords=True,
    )
    kick_err = kick_error(state.knobs)
    start_error = np.max(np.abs(state.starts[0] - truth_start) / np.abs(truth_start))
    print(f"loss {state.loss:.3e} kick {kick_err:.3e} start {start_error:.3e}")
    assert state.loss < 1e-18
    assert kick_err < 1e-6
    assert start_error < 5e-5


def test_windowed_ic_step_recovers_kick(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: MadInterface
) -> None:
    """Windows of 10 turns (starts at turns 1 and 11, one worker each) share one kick."""
    make_fitter, strengths, truth_start = _setup(tmp_path, seq_psb, loaded_psb_interface)
    state = fit_step(
        make_fitter,
        State({**strengths, **_kick_start()}, None, np.inf),
        IC_ERRORS,
        fit_start_coords=True,
        n_turns=WINDOW,
    )
    kick_err = kick_error(state.knobs)
    # session order is worker order, not turn order: the turn-1 window is the closest row
    start_error = np.min(np.max(np.abs(state.starts - truth_start) / np.abs(truth_start), axis=1))
    print(f"windows {len(state.starts)} loss {state.loss:.3e} kick {kick_err:.3e}")
    assert len(state.starts) == (FLATTOP_TURNS - 1) // WINDOW
    assert state.loss < 1e-18
    assert kick_err < 1e-6
    assert start_error < 5e-5


def test_kick_per_file(tmp_path: Path, seq_psb: Path, loaded_psb_interface: MadInterface) -> None:
    """Two files (the second starts DROP_TURNS later): one (a, b) kick per file."""
    make_fitter, strengths, _ = _setup(
        tmp_path, seq_psb, loaded_psb_interface, drop_turns=DROP_TURNS
    )
    volts = (PSB_HORIZONTAL_ACD_VOLTAGE, PSB_VERTICAL_ACD_VOLTAGE)
    truth_per_file = [
        kick_knobs(volts, [(RAMP_TURNS - 1 + drop) * f for f in DRIVE_TUNES], f"_f{i}")
        for i, drop in enumerate((0, DROP_TURNS))
    ]
    start = {n: 0.9 * v for truth in truth_per_file for n, v in truth.items()}
    state = fit_step(
        make_fitter,
        State({**strengths, **start}, None, np.inf),
        IC_ERRORS,
        fit_start_coords=True,
        n_turns=WINDOW,
    )
    errors = [kick_error(state.knobs, truth) for truth in truth_per_file]
    print(f"starts {len(state.starts)} loss {state.loss:.3e} kick errors {errors}")
    assert len(state.starts) == 4
    # measured 2026-11-12: loss 2.0e-20, kick 5.5e-7 (file 0), 1.1e-6 (file 1)
    assert state.loss < 1e-18
    assert max(errors) < 2e-6


def test_k_step_keeps_fixed_kick(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: MadInterface
) -> None:
    """Quads fitted with the kick fixed at truth (not a knob): truth loss is kept."""
    make_fitter, strengths, truth_start = _setup(tmp_path, seq_psb, loaded_psb_interface)
    knobs = {**strengths, **_truth_acd_knobs()}
    state = fit_step(
        make_fitter, State(knobs, truth_start[None, :], np.inf), K_ERRORS, fit_start_coords=False
    )
    print(f"loss {state.loss:.3e}")
    assert state.loss < 1e-18


def _loss_at(make_fitter, state: State, errors, *, fit_start_coords: bool) -> float:
    """Worker loss of ``state`` in a fitter with free set ``errors`` (no fitting)."""
    fitter = make_fitter(Step(errors, fit_start_coords, state.knobs, state.starts))
    session = fitter.start_session(fitter.machine.worker_start_knobs)
    try:
        session.send_init_condition_updates(state.starts)
        return sum(session.training_losses(fitter.machine.initial_knobs))
    finally:
        session.terminate()


def test_handoff_preserves_loss(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: MadInterface
) -> None:
    """The state after an IC step has the same loss whichever free set evaluates it."""
    make_fitter, _, _ = _setup(tmp_path, seq_psb, loaded_psb_interface)
    state = fit_step(
        make_fitter, State(_kick_start(), None, np.inf), IC_ERRORS, fit_start_coords=True
    )
    loss_ic = _loss_at(make_fitter, state, IC_ERRORS, fit_start_coords=True)
    loss_k = _loss_at(make_fitter, state, K_ERRORS, fit_start_coords=False)
    print(f"fit {state.loss:.9e} ic {loss_ic:.9e} k {loss_k:.9e}")
    assert loss_k == pytest.approx(loss_ic, rel=1e-9)


def test_joint_recovers_quads(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: MadInterface
) -> None:
    """IC + kick at design quads, then one joint fit (quads, kick, start), returns to truth.

    A joint fit straight from design quads stalls (loss 4.6e-8): Qy moves 0.12 from the
    drive tune and the vertical kick grows 10x (tune / kick-amplitude degeneracy).
    """
    make_fitter, strengths, truth_start = _setup(tmp_path, seq_psb, loaded_psb_interface)
    ic = fit_step(make_fitter, State(_kick_start(), None, np.inf), IC_ERRORS, fit_start_coords=True)
    final = joint(make_fitter, ic, IC_ERRORS, K_ERRORS)[-1]
    quads = [n for n in strengths if n in final.knobs]
    k_error = max(abs(final.knobs[n] - strengths[n]) / abs(strengths[n]) for n in quads)
    kick_err = kick_error(final.knobs)
    start_error = np.max(np.abs(final.starts[0] - truth_start) / np.abs(truth_start))
    print(
        f"joint loss {final.loss:.3e} k {k_error:.3e} kick {kick_err:.3e} start {start_error:.3e}"
    )
    # measured 2026-11-13: loss 2.35e-21, k 1.33e-5, kick 7.7e-7, start 3.0e-6
    assert final.loss < 1e-20
    assert k_error < 2e-5
    assert kick_err < 2e-6
    assert start_error < 1e-5


def test_cold_joint_recovers_quads(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: MadInterface
) -> None:
    """Turn continuation (10 turns, then all 29) from design quads returns to xtrack truth."""
    make_fitter, strengths, truth_start = _setup(tmp_path, seq_psb, loaded_psb_interface)
    history = cold_joint(make_fitter, State(_kick_start(), None, np.inf), IC_ERRORS, K_ERRORS)
    final = history[-1]
    quads = [n for n in strengths if n in final.knobs]
    k_error = max(abs(final.knobs[n] - strengths[n]) / abs(strengths[n]) for n in quads)
    kick_err = kick_error(final.knobs)
    start_error = np.max(np.abs(final.starts[0] - truth_start) / np.abs(truth_start))
    print(
        f"cold losses {[f'{s.loss:.2e}' for s in history[1:]]} "
        f"k {k_error:.3e} kick {kick_err:.3e} start {start_error:.3e}"
    )
    # measured 2026-11-12: loss 2.22e-21, k 8.4e-5, kick 7.6e-7, start 3.3e-6; k is set by
    # the floor along the weakest direction (normal-matrix cond 3e7 at 29 turns, BR.QFO dk1l)
    assert final.loss < 1e-20
    assert k_error < 1e-4
    assert kick_err < 2e-6
    assert start_error < 1e-5


GAINS = {"BR3.BPM1L3": 0.03, "BR3.BPM4L3": -0.02, "BR3.BPM7L3": 0.05}


def test_lm_recovers_bpm_gains(
    tmp_path: Path, seq_psb: Path, loaded_psb_interface: MadInterface
) -> None:
    """Known gains on three BPMs, noise-free data, quads and kick at truth: LM returns them.

    Measured 2026-11-13: errors <= 2e-7 with a loose prior (sigma 10; sigma 0.1 shrinks them 2%).
    """
    make_fitter, strengths, _ = _setup(
        tmp_path, seq_psb, loaded_psb_interface, gains=GAINS, gain_sigma=10.0
    )
    knobs = {**strengths, **_truth_acd_knobs()}
    fitter = make_fitter(Step(IC_ERRORS, fit_start_coords=True, knobs=knobs, starts=None))
    result = fitter.run()
    for plane in ("x", "y"):
        for bpm, gain in GAINS.items():
            fitted = result.extra[f"bpmgain.{plane}.{bpm}"]
            print(f"{plane} {bpm}: fitted {fitted:.6e} truth {gain:.6e}")
            assert fitted == pytest.approx(gain, abs=1e-6)
