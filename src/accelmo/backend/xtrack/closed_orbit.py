"""Closed orbit of the xtrack backend: chord-Newton from a predicted start.

``Line.find_closed_orbit`` is a finite-difference Newton on six variables. It needs 13 to 22
one-turn tracks and only skips them when the guess is within 1e-11 of the orbit. Here the start
is predicted from the last solve of the same state as ``x + S (k - K)``, with ``S = (1 - R)^-1 P``
from its parametric track, and refined by ``x <- x + (1 - R)^-1 (M(x) - x)`` with that solve's
one-turn matrix ``R``: one plain turn per iteration, and ``R`` changes so little between solves
that two or three iterations reach ``REFINE_TOL``. The parametric track then runs at the
converged orbit, so the Jacobian is exact. A state without a stored solve, or a refinement that
does not converge, uses xtrack's own search.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xtrack as xt
import xtrack.tpsa as xtpsa

from accelmo.backend.protocol import ClosedSolution

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)

COORDS = ("x", "px", "y", "py")
COORD_INDEX = {name: i for i, name in enumerate(COORDS)}
#: Largest Newton step [m] of the refinement, as xtrack's own search tolerance.
REFINE_TOL = 1e-11
MAX_REFINE = 8


@dataclass
class _Solved:
    """The last closed orbit of one state, with what predicts the next one."""

    start: np.ndarray  # (4,) transverse start of the closed orbit at ``knobs``
    orbit: xt.Particles  # the closed-orbit particle (its longitudinal part is kept)
    pt: float
    knobs: np.ndarray
    sensitivity: np.ndarray | None = None  # (4, n_knobs) d start / d knob
    one_turn: np.ndarray | None = None  # (4, 4)


class ClosedOrbitSolver:
    """Closed orbit at the BPMs, with its knob Jacobian, for one state (key) at a time."""

    def __init__(self, model, params, monomials: np.ndarray) -> None:
        self.model = model
        self.params = params
        self.monomials = monomials
        self.pt = 0.0
        self.key = "default"
        self._solved: dict[str, _Solved] = {}

    def solve(self, observables: Sequence[str], *, jacobian: bool) -> ClosedSolution | None:
        """Orbit values (and Jacobian) of ``observables`` at the BPMs; ``None`` if the orbit is lost."""
        state = self._solved.get(self.key)
        if state is not None and (state.pt != self.pt or state.one_turn is None):
            state = None
        names = list(self.model.all_bpms)
        observed = [name.lower() for name in names]
        orbit = self._closed_orbit(state, observed)
        if orbit is None:
            return None
        if jacobian:
            return self._solve_jacobian(observables, names, observed, orbit)
        data = self._track_plain(orbit, observed)
        if data is None:
            return None
        values = np.array([data[0][0, 0, COORD_INDEX[name]] for name in observables])
        self._remember(
            orbit, state.sensitivity if state else None, state.one_turn if state else None
        )
        return ClosedSolution(names, values, None)

    # ------------------------------------------------------------------ closed orbit

    def _closed_orbit(self, state: _Solved | None, observed: list[str]) -> xt.Particles | None:
        """The closed-orbit particle at the current knobs and momentum."""
        if state is None:
            return self._search(None)
        guess = self._predicted(state)
        self.params.plain()
        orbit = self._refine(guess, state.one_turn, observed)
        return orbit if orbit is not None else self._search(guess)

    def _predicted(self, state: _Solved) -> xt.Particles:
        """The closed orbit of the last solve moved to the current knobs to first order."""
        start = state.start.copy()
        if state.sensitivity is not None:
            start += state.sensitivity @ (np.asarray(self.params.values) - state.knobs)
        return _with_start(state.orbit, start)

    def _refine(self, orbit: xt.Particles, one_turn: np.ndarray, observed: list[str]):
        """Chord-Newton with a fixed one-turn matrix; ``None`` if it has not converged."""
        inverse = np.eye(4) - one_turn
        for _ in range(MAX_REFINE):
            data = self._track_plain(orbit, observed)
            if data is None:
                return None
            step = np.linalg.solve(inverse, data[1] - _start(orbit))
            orbit = _with_start(orbit, _start(orbit) + step)
            if np.abs(step).max() <= REFINE_TOL:
                return orbit
        return None

    def _search(self, guess: xt.Particles | None) -> xt.Particles | None:
        """xtrack's 4D closed-orbit search at ``ptau = pt``; ``None`` when none is found."""
        self.params.plain()
        ref = self.model.particle_ref
        delta0 = float(
            xt.Particles(p0c=ref.p0c[0], mass0=ref.mass0, q0=ref.q0, ptau=self.pt).delta[0]
        )
        for co_guess in (guess, None) if guess is not None else (None,):
            try:
                return self.model.line.find_closed_orbit(
                    method="4d", delta0=delta0, co_guess=co_guess
                )
            except Exception as exc:  # noqa: BLE001  xsuite raises several types when no orbit exists
                LOGGER.warning("xtrack backend: closed orbit not found (%s)", exc)
        return None

    def _track_plain(self, orbit: xt.Particles, observed: list[str]):
        """BPM coordinates of one plain turn from ``orbit`` and the start coordinates after it."""
        particle = orbit.copy()
        self.model.line.track(particle, multi_element_monitor_at=observed)
        if particle.state[0] <= 0:
            return None
        data = np.asarray(self.model.line.tracker.record_multi_element_last_track.data)
        return data, _start(particle)

    def _remember(self, orbit: xt.Particles, sensitivity, one_turn) -> None:
        self._solved[self.key] = _Solved(
            _start(orbit), orbit, self.pt, np.array(self.params.values), sensitivity, one_turn
        )

    # ------------------------------------------------------------------ with Jacobian

    def _solve_jacobian(
        self,
        observables: Sequence[str],
        names: list[str],
        observed: list[str],
        orbit: xt.Particles,
    ) -> ClosedSolution | None:
        recorded = list(dict.fromkeys(observables))
        rows = [recorded.index(name) for name in observables]
        self.params.apply()
        ref = self.model.particle_ref
        particle = xtpsa.ParticlesTpsa(
            order=1,
            descriptor=self.params.descriptor,
            x=orbit.x[0],
            px=orbit.px[0],
            y=orbit.y[0],
            py=orbit.py[0],
            zeta=orbit.zeta[0],
            delta=orbit.delta[0],
            p0c=ref.p0c[0],
            mass0=ref.mass0,
            q0=ref.q0,
        )
        line = self.model.line
        try:
            line.track(
                particle,
                multi_element_monitor_at=observed,
                monitor_monomials=dict.fromkeys(recorded, self.monomials),
            )
        except RuntimeError as exc:
            LOGGER.warning("xtrack backend: parametric turn lost (%s)", exc)
            return None
        record = np.asarray(line.tracker.record_multi_element_last_track.coefficients)[0]
        record = record.reshape(len(observed), len(recorded), -1)
        one_turn = particle.jacobian()[:4, :4]
        inverse = np.eye(4) - one_turn
        sensitivity = np.linalg.solve(inverse, particle.param_jacobian()[:4])
        step = np.linalg.solve(inverse, particle.const_part[:4] - _start(orbit))
        values = (record[:, rows, 0] + record[:, rows, 1:5] @ step).T
        jacobian = (record[:, rows, 1:5] @ sensitivity + record[:, rows, 5:]).transpose(1, 0, 2)
        self._remember(_with_start(orbit, _start(orbit) + step), sensitivity, one_turn)
        return ClosedSolution(names, values, jacobian)


def _start(particle: xt.Particles) -> np.ndarray:
    return np.array([float(getattr(particle, name)[0]) for name in COORDS])


def _with_start(particle: xt.Particles, start: np.ndarray) -> xt.Particles:
    moved = particle.copy()
    for name, value in zip(COORDS, start, strict=True):
        setattr(moved, name, float(value))
    return moved
