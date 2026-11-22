"""The xtrack backend against MAD-NG and against plain xsuite, on the real PSB ring 3.

Asserted:
* the same knobs and observed BPMs as MAD-NG;
* corrector machine states give MAD-NG's closed orbit (same sign convention);
* the TPSA knob derivatives equal central differences of the tracked orbit;
* the horizontal one-turn tracking and its knob Jacobian match MAD-NG.
The vertical PSB plane differs between the two codes' lattice models (see
``docs/backends.rst``) and is not compared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from accelmo.backend import get_backend
from accelmo.backend.protocol import TrackRequest
from accelmo.fitting.worker import WorkerConfig
from accelmo.machine.accelerators import PSB

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.serial

ERRORS = {"quad": {"k1"}}


def _accelerator(seq_psb: Path, backend: str) -> PSB:
    return PSB(ring=3, sequence_file=seq_psb, errors=ERRORS, backend=backend)


def _knob_names(seq_psb: Path, backend: str) -> tuple[list[str], list[str]]:
    model = get_backend(backend).open_model(
        _accelerator(seq_psb, backend), magnet_range="$start/$end"
    )
    names, bpms = list(model.knob_names), list(model.all_bpms)
    model.close()
    return names, bpms


def test_same_knobs_and_bpms_as_madng(seq_psb: Path) -> None:
    madng, xtrack = _knob_names(seq_psb, "madng"), _knob_names(seq_psb, "xtrack")
    assert len(madng[0]) == 48
    assert set(xtrack[0]) == set(madng[0])
    assert xtrack[1] == madng[1]


def _orbit(seq_psb: Path, backend: str, state: dict[str, float], knobs: dict[str, float]):
    config = WorkerConfig(
        accelerator=_accelerator(seq_psb, backend),
        tracking_start_bpm="$start",
        tracking_end_bpm="$end",
        magnet_range="$start/$end",
        cycle_sequence=False,
        interface_options={"machine_state": state},
    )
    engine = get_backend(backend).ClosedOrbitEngine(
        config, knobs, optics_names=[], orbit_coords=["x", "y"], worker_id=0
    )
    engine.set_pt(0.0)
    solution = engine.solve(["x", "y"])
    engine.close()
    return solution


@pytest.mark.parametrize(("variable", "plane"), [("kbr3dvt3l4", 1), ("kbr3dhz2l4", 0)])
def test_corrector_state_gives_madng_closed_orbit(seq_psb: Path, variable: str, plane: int) -> None:
    """Measured 5e-12 m (vertical) and 3e-13 m (horizontal) on a ~1 mm orbit."""
    knobs = dict.fromkeys(_knob_names(seq_psb, "madng")[0], 0.0)
    madng = _orbit(seq_psb, "madng", {variable: 1e-4}, knobs)
    xtrack = _orbit(seq_psb, "xtrack", {variable: 1e-4}, knobs)
    assert xtrack.names == madng.names
    assert np.abs(madng.values[plane]).max() > 5e-4
    np.testing.assert_allclose(xtrack.values[plane], madng.values[plane], rtol=0, atol=1e-10)


def _tracking(seq_psb: Path, backend: str, knobs: dict[str, float]):
    _, bpms = _knob_names(seq_psb, "madng")
    config = WorkerConfig(
        accelerator=_accelerator(seq_psb, backend),
        tracking_start_bpm=bpms[0],
        tracking_end_bpm=bpms[-1],
        magnet_range="$start/$end",
    )
    plan = TrackRequest(
        observables=("x", "px", "y", "py"),
        batch_size=2,
        num_batches=1,
        tracking_range=config.tracking_range,
        n_run_turns=1,
        start_on_first_turn=False,
        validation=False,
        file_index=0,
        turn_offset=0,
    )
    engine = get_backend(backend).TrackingEngine(config, plan, knobs, worker_id=0)
    engine.load_particles(
        [[[1e-3, 1e-4, -1e-3, 5e-5, 0, 0], [-5e-4, 0, 2e-4, -1e-4, 0, 0]]], [[0.0, 1e-3]]
    )
    engine.set_knobs(knobs)
    engine.set_pts(0.0, 0, 0)
    result = engine.track(0, start_derivatives=True)
    order = [engine.knob_names.index(name) for name in sorted(knobs)]
    engine.close()
    return result, order


def test_horizontal_tracking_matches_madng(seq_psb: Path) -> None:
    """Measured: x 6e-10 m on 5 mm, d(x)/dk 6e-9 on 4e-2 (knobs at 0 and at +-2e-3)."""
    rng = np.random.default_rng(1)
    names = _knob_names(seq_psb, "madng")[0]
    knobs = {name: float(rng.uniform(-2e-3, 2e-3)) for name in names}
    (madng, m_order), (xtrack, x_order) = (
        _tracking(seq_psb, b, knobs) for b in ("madng", "xtrack")
    )
    for o in ("x", "px"):
        np.testing.assert_allclose(xtrack.obs[o], madng.obs[o], rtol=0, atol=2e-9)
        np.testing.assert_allclose(
            xtrack.d_dk[o][:, x_order], madng.d_dk[o][:, m_order], rtol=0, atol=2e-8
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Bundled MAD-NG mad_Linux 1.1.15: off-momentum quadrupole map differs from xsuite "
        "vertically (one turn, pt=1e-3: 3.2e-7 m in y, d(y)/d(y0) 3e-4 rel); the MAD-NG dev "
        "build agrees to 4e-10 m"
    ),
)
def test_vertical_tracking_matches_madng(seq_psb: Path) -> None:
    """The same bounds as the horizontal plane; the second particle is at pt = 1e-3."""
    rng = np.random.default_rng(1)
    names = _knob_names(seq_psb, "madng")[0]
    knobs = {name: float(rng.uniform(-2e-3, 2e-3)) for name in names}
    (madng, m_order), (xtrack, x_order) = (
        _tracking(seq_psb, b, knobs) for b in ("madng", "xtrack")
    )
    for o in ("y", "py"):
        np.testing.assert_allclose(xtrack.obs[o], madng.obs[o], rtol=0, atol=2e-9)
        np.testing.assert_allclose(
            xtrack.d_dk[o][:, x_order], madng.d_dk[o][:, m_order], rtol=0, atol=2e-8
        )


def test_tpsa_derivatives_equal_plain_xsuite_differences(seq_psb: Path) -> None:
    """d(x)/d(knob) from the TPSA map vs central differences of the tracked orbit (its constant part)."""
    names = _knob_names(seq_psb, "xtrack")[0]
    knobs = dict.fromkeys(names, 0.0)
    result, order = _tracking(seq_psb, "xtrack", knobs)
    knob = sorted(knobs)[3]
    step = 1e-7
    shifted = [
        _tracking(seq_psb, "xtrack", {**knobs, knob: sign * step})[0].obs["x"] for sign in (1, -1)
    ]
    fd = (shifted[0] - shifted[1]) / (2 * step)
    np.testing.assert_allclose(result.d_dk["x"][:, order[3]], fd, rtol=0, atol=1e-8)
    assert np.abs(fd).max() > 1e-3


def test_kicker_start_matches_madng(seq_psb: Path) -> None:
    """Multi-turn from the thick kicker BE3.KFA14L1: both codes start at its centre.

    Measured 2.4e-11 m after 5 turns; starting xsuite at the kicker entrance gave 8.8e-5 m.
    """
    knobs = dict.fromkeys(_knob_names(seq_psb, "madng")[0], 0.0)
    results = {}
    for backend in ("madng", "xtrack"):
        config = WorkerConfig(
            accelerator=_accelerator(seq_psb, backend),
            tracking_start_bpm="$start",
            tracking_end_bpm="$end",
            magnet_range="$start/$end",
            initial_condition_marker="BE3.KFA14L1",
        )
        plan = TrackRequest(
            observables=("x", "y"),
            batch_size=1,
            num_batches=1,
            tracking_range=config.tracking_range,
            n_run_turns=5,
            start_on_first_turn=True,
            validation=True,
            file_index=0,
            turn_offset=0,
        )
        engine = get_backend(backend).TrackingEngine(config, plan, knobs, worker_id=0)
        engine.load_particles([[[5e-4, 5e-5, -5e-4, 2e-5, 0, 0]]], [[0.0]])
        engine.set_pts(0.0, 0, 0)
        results[backend] = engine.track(0, start_derivatives=False)
        engine.close()
    for o in ("x", "y"):
        assert np.abs(results["madng"].obs[o]).max() > 1e-4
        np.testing.assert_allclose(
            results["xtrack"].obs[o], results["madng"].obs[o], rtol=0, atol=1e-9
        )
