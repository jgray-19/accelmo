"""The xtrack closed-orbit solver (Newton from a predicted start) on the real PSB ring 3.

Every result of a warm solve (a state that was solved before, at other knobs) is compared with
a cold solve of the same knobs and state, which has no predicted start and so uses xtrack's own
closed-orbit search. The Jacobian is also checked against central differences of cold orbits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from accelmo.backend import get_backend
from accelmo.fitting.worker import WorkerConfig
from accelmo.machine.accelerators import PSB

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

pytestmark = pytest.mark.serial

KNOB_SPREAD = 2e-3  # integrated dk1l [1/m], uniform +-
#: Corrector states (MAD-X globals) that give each series its own closed orbit.
STATES = {"horizontal": {"kbr3dhz2l4": 1e-4}, "vertical": {"kbr3dvt3l4": -1e-4}}
#: Knob scale per step: a large first move, then Levenberg-Marquardt-like small ones.
STEPS = (0.0, 1.0, 1.0 + 1e-3, 1.0 + 1e-5, 1.0 + 1e-6)
OBSERVABLES = ["x", "y"]
FD_STEP = 1e-6
FD_KNOBS = 4
SEED = 1
TOL_ORBIT = 1e-9  # relative to the largest orbit value
TOL_JACOBIAN = 1e-6  # relative to the largest Jacobian entry
TOL_FD = 1e-4  # central differences are limited by the closed-orbit tolerance


def _engine(seq_psb: Path):
    accelerator = PSB(ring=3, sequence_file=seq_psb, errors={"quad": {"k1"}}, backend="xtrack")
    backend = get_backend("xtrack")
    model = backend.open_model(accelerator, magnet_range="$start/$end")
    names = list(model.knob_names)
    model.close()
    config = WorkerConfig(accelerator=accelerator, magnet_range="$start/$end")
    engine = backend.ClosedOrbitEngine(
        config, dict.fromkeys(names, 0.0), optics_names=[], orbit_coords=OBSERVABLES, worker_id=0
    )
    engine.set_pt(0.0)
    return engine, names


def _knobs(names: list[str]) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    return {name: float(rng.uniform(-KNOB_SPREAD, KNOB_SPREAD)) for name in names}


def _scaled(knobs: Mapping[str, float], scale: float) -> dict[str, float]:
    return {name: value * scale for name, value in knobs.items()}


def _solve(engine, key: str, state: Mapping[str, float], knobs, *, jacobian: bool = True):
    engine.set_warm_start(key)
    engine.set_vars({"kbr3dhz2l4": 0.0, "kbr3dvt3l4": 0.0, **state})
    engine.set_knobs(knobs)
    return engine.solve(OBSERVABLES, jacobian=jacobian)


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_warm_solves_match_cold_search(seq_psb: Path) -> None:
    """Jacobian and loss-only solves along a Levenberg-Marquardt-like path, two states interleaved."""
    engine, names = _engine(seq_psb)
    base = _knobs(names)
    try:
        for step, scale in enumerate(STEPS):
            knobs = _scaled(base, scale)
            for label, state in STATES.items():
                cold = _solve(engine, f"cold-{label}-{step}", state, knobs)
                warm = _solve(engine, label, state, knobs)
                trial = _solve(engine, label, state, _scaled(base, scale + 1e-7), jacobian=False)
                cold_trial = _solve(
                    engine, f"cold-trial-{label}-{step}", state, _scaled(base, scale + 1e-7)
                )
                assert _relative(warm.values, cold.values) < TOL_ORBIT, (label, step)
                assert _relative(warm.jacobian, cold.jacobian) < TOL_JACOBIAN, (label, step)
                assert _relative(trial.values, cold_trial.values) < TOL_ORBIT, (label, step)
                assert warm.names == cold.names
    finally:
        engine.close()


def test_jacobian_matches_central_differences_of_cold_orbits(seq_psb: Path) -> None:
    engine, names = _engine(seq_psb)
    knobs = _knobs(names)
    try:
        for label, state in STATES.items():
            _solve(engine, label, state, _scaled(knobs, 0.9))
            warm = _solve(engine, label, state, knobs)
            for index, name in enumerate(names[:FD_KNOBS]):
                orbits = []
                for sign in (+1, -1):
                    shifted = {**knobs, name: knobs[name] + sign * FD_STEP}
                    orbits.append(_solve(engine, f"fd-{label}-{name}-{sign}", state, shifted))
                difference = (orbits[0].values - orbits[1].values) / (2 * FD_STEP)
                assert _relative(warm.jacobian[:, :, index], difference) < TOL_FD, (label, name)
    finally:
        engine.close()


def test_loss_only_solve_before_any_jacobian_solve(seq_psb: Path) -> None:
    """A state with no stored sensitivity searches, and agrees with the Jacobian solve's orbit."""
    engine, names = _engine(seq_psb)
    knobs = _scaled(_knobs(names), 0.5)
    try:
        plain = _solve(engine, "plain-first", STATES["horizontal"], knobs, jacobian=False)
        full = _solve(engine, "full", STATES["horizontal"], knobs)
        assert _relative(plain.values, full.values) < TOL_ORBIT
        assert plain.jacobian is None
    finally:
        engine.close()


def test_momentum_change_discards_the_stored_orbit(seq_psb: Path) -> None:
    engine, names = _engine(seq_psb)
    knobs = _knobs(names)
    state = STATES["horizontal"]
    try:
        _solve(engine, "momentum", state, knobs)
        engine.set_pt(5e-4)
        warm = _solve(engine, "momentum", state, knobs)
        cold = _solve(engine, "momentum-cold", state, knobs)
        engine.set_pt(0.0)
        zero = _solve(engine, "momentum-zero", state, knobs)
        assert _relative(warm.values, cold.values) < TOL_ORBIT
        assert _relative(warm.jacobian, cold.jacobian) < TOL_JACOBIAN
        assert np.abs(warm.values - zero.values).max() > 1e-5  # the momentum offset moved the orbit
    finally:
        engine.close()


def test_a_lost_closed_orbit_returns_none_and_the_next_solve_recovers(seq_psb: Path) -> None:
    engine, names = _engine(seq_psb)
    knobs = _knobs(names)
    state = STATES["horizontal"]
    try:
        before = _solve(engine, "recover", state, knobs)
        assert _solve(engine, "recover", state, dict.fromkeys(names, 0.5)) is None
        after = _solve(engine, "recover", state, knobs)
        assert _relative(after.values, before.values) < TOL_ORBIT
        assert _relative(after.jacobian, before.jacobian) < TOL_JACOBIAN
    finally:
        engine.close()
