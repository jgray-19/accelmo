"""xtrack TPSA tracking and closed-orbit engines (see :mod:`accelmo.backend.protocol`).

The knobs are parameters of a ``madng_tpsa`` descriptor (``KnobParameters``). Tracking runs
one ``ParticlesTpsa`` per particle; the closed orbit is ``twiss(method="4d")`` plus one
parametric turn and the fixed-point shift ``d(orbit_s)/dk = R_s (I - R)^-1 P + P_s``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import madng_tpsa
import numpy as np
import xtrack.tpsa as xtpsa
from xtrack.tpsa._knobs import KnobParameters

from accelmo.backend.protocol import ClosedSolution, TrackResult
from accelmo.backend.xtrack.closed_orbit import COORD_INDEX, COORDS, ClosedOrbitSolver
from accelmo.backend.xtrack.model import XtrackModel, _range_indices

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from accelmo.backend.protocol import TrackGeometry, TrackRequest
    from accelmo.fitting.worker import WorkerConfig
    from accelmo.machine.accelerators import Accelerator
    from accelmo.machine.machine_state import MachineState

LOGGER = logging.getLogger(__name__)
#: One Yoshida-6 slice; LHC orbit responses match 128 kicks to 2e-11 relative.
CLOSED_ORBIT_MULTIPOLE_KICKS = 7


def _monomials(n_knobs: int) -> np.ndarray:
    """Recorded per coordinate: the constant, the four start-coordinate terms, one per knob."""
    monomials = np.zeros((5 + n_knobs, 6 + n_knobs), dtype=np.uint8)
    monomials[1 + np.arange(4), np.arange(4)] = 1
    monomials[5 + np.arange(n_knobs), 6 + np.arange(n_knobs)] = 1
    return monomials


def _model(
    config: WorkerConfig, init_knobs: Mapping[str, float], geometry: TrackGeometry | None = None
) -> XtrackModel:
    """The worker's model: ``geometry``'s segment, cycled; the whole ring, uncycled, without it."""
    model = XtrackModel(
        config.accelerator,
        magnet_range=config.magnet_range,
        bpm_range="$start/$end" if geometry is None else geometry.bpm_range,
        bad_bpms=config.bad_bpms,
        machine_state=config.machine_state,
        initial_model_values=init_knobs,
    )
    if config.b2_errors is not None or (geometry is not None and geometry.anchor_mode is not None):
        raise NotImplementedError("xtrack backend: no b2 errors or marker-anchored tracking")
    if geometry is not None:
        model.cycle_to_start(geometry.cycle_target)
        model.recount_bpms()
    return model


class _Parametric:
    """Knob parameters on a model's line; ``values`` holds the constant parts."""

    def __init__(self, model: XtrackModel, names: list[str]) -> None:
        self.model = model
        self.names = names
        self.descriptor = madng_tpsa.Descriptor(6, 1, num_params=len(names), param_order=1)
        self.values = model.knob_values()
        self.knobs = KnobParameters(model.line, names, self.descriptor) if names else None
        self.applied = False

    def set(self, knobs: Mapping[str, float]) -> None:
        for i, name in enumerate(self.names):
            if name in knobs:
                self.values[i] = float(knobs[name])
        if self.applied and self.knobs is not None:
            self.knobs.apply(list(self.values))

    def apply(self) -> None:
        if self.knobs is not None:
            self.knobs.apply(list(self.values))
            self.applied = True

    def plain(self) -> None:
        if self.knobs is not None and self.applied:
            self.knobs.apply_doubles(list(self.values))
        else:
            self.model.set_knob_values(dict(zip(self.names, self.values)))
        self.applied = False


class TrackingEngine:
    """One parametric ``ParticlesTpsa`` track per particle."""

    def __init__(
        self,
        config: WorkerConfig,
        plan: TrackRequest,
        init_knobs: Mapping[str, float],
        *,
        worker_id: int,
        logfile: Path | None = None,
    ) -> None:
        del worker_id, logfile
        if plan.geometry.sdir != 1 or config.accelerator.has_ac_kickers:
            raise NotImplementedError(
                "xtrack backend: backward tracking and AC kickers are not supported"
            )
        self.plan = plan
        self.model = _model(config, init_knobs, plan.geometry)
        self.knob_names = list(self.model.knob_names)
        line = self.model.line
        names = line.element_names
        if plan.geometry.tracking_range is None:
            self.ele_start, self.ele_stop, observed = 0, None, self.model.all_bpms
            self.nbpms = len(observed)
        else:
            if plan.geometry.n_turns != 1:
                raise NotImplementedError("xtrack backend: a named range is tracked for one turn")
            start, stop = _range_indices(names, plan.geometry.tracking_range)
            self.ele_start, self.ele_stop = start, stop + 1
            inside = {name.upper() for name in names[start : stop + 1]}
            observed = [name for name in self.model.all_bpms if name in inside]
            self.nbpms = self.model.nbpms
        self.observed = [name.lower() for name in observed]
        self.params = _Parametric(self.model, [] if plan.validation else self.knob_names)
        self.params.apply()
        self.monomials = _monomials(len(self.params.names))

    def set_knobs(self, knobs: Mapping[str, float]) -> None:
        """See :meth:`accelmo.backend.protocol.TrackingEngine.set_knobs`."""
        self.params.set(knobs)

    def load_particles(
        self, init_coords: list[list[list[float]]], init_pts: list[list[float]]
    ) -> None:
        """See :meth:`accelmo.backend.protocol.TrackingEngine.load_particles`."""
        self.coords = [np.asarray(batch, dtype=float) for batch in init_coords]
        self.offsets = [np.asarray(batch, dtype=float) for batch in init_pts]
        self.pts = [np.array(batch) for batch in self.offsets]

    def set_pts(self, pt: float, first_batch: int, last_batch: int) -> None:
        """See :meth:`accelmo.backend.protocol.TrackingEngine.set_pts`."""
        for batch in range(first_batch, last_batch + 1):
            self.pts[batch] = pt + self.offsets[batch]

    def set_start_coords(
        self, x: np.ndarray, px: np.ndarray, y: np.ndarray, py: np.ndarray
    ) -> None:
        """See :meth:`accelmo.backend.protocol.TrackingEngine.set_start_coords`."""
        flat = np.column_stack([np.ravel(v) for v in (x, px, y, py)])
        first = 0
        for batch in self.coords:
            batch[:, :4] = flat[first : first + len(batch)]
            first += len(batch)

    def _track_one(self, coords: np.ndarray, pt: float) -> np.ndarray | None:
        """``(turns * points, coords, 5 + K)`` coefficients, or ``None`` when lost."""
        line, ref = self.model.line, self.model.particle_ref
        particle = xtpsa.ParticlesTpsa(
            order=1,
            descriptor=self.params.descriptor,
            x=coords[0],
            px=coords[1],
            y=coords[2],
            py=coords[3],
            zeta=0.0,
            ptau=pt,
            p0c=ref.p0c[0],
            mass0=ref.mass0,
            q0=ref.q0,
        )
        try:
            line.track(
                particle,
                ele_start=self.ele_start,
                ele_stop=self.ele_stop,
                num_turns=self.plan.geometry.n_turns,
                multi_element_monitor_at=self.observed,
                monitor_monomials=dict.fromkeys(COORDS, self.monomials),
            )
        except RuntimeError as exc:
            LOGGER.debug("xtrack backend: particle lost (%s)", exc)
            return None
        record = np.asarray(line.tracker.record_multi_element_last_track.coefficients)
        n = len(self.monomials)
        turns = record[: self.plan.geometry.n_turns]
        return turns.reshape(-1, len(COORDS), n)

    def preflight(self) -> dict[str, int | bool]:
        """See :meth:`accelmo.backend.protocol.TrackingEngine.preflight`."""
        record = self._track_one(self.coords[0][0], float(self.pts[0][0]))
        expected = self.nbpms * self.plan.geometry.n_turns
        if record is None:
            return {"observed": 0, "expected": expected, "lost": True}
        return {"observed": len(record), "expected": expected, "lost": False}

    def track(self, batch: int, *, start_derivatives: bool) -> TrackResult:
        """See :meth:`accelmo.backend.protocol.TrackingEngine.track`."""
        coords, pts = self.coords[batch], self.pts[batch]
        n_points = self.nbpms * self.plan.geometry.n_turns
        n_knobs = len(self.params.names)
        records = np.full((len(coords), n_points, len(COORDS), 5 + n_knobs), np.nan)
        n_lost = 0
        for i, (start, pt) in enumerate(zip(coords, pts)):
            record = self._track_one(start, float(pt))
            if record is None:
                n_lost += 1
            else:
                records[i] = record
        observables = self.plan.observables
        obs = {o: records[:, :, COORD_INDEX[o], 0] for o in observables}
        if self.plan.validation:
            return TrackResult(obs, n_total=len(coords), n_lost=n_lost)
        d_dk = {o: records[:, :, COORD_INDEX[o], 5:].transpose(0, 2, 1) for o in observables}
        d_dc0 = (
            {o: records[:, :, COORD_INDEX[o], 1:5].transpose(0, 2, 1) for o in observables}
            if start_derivatives
            else None
        )
        return TrackResult(obs, d_dk, d_dc0, n_lost, len(coords))

    def close(self) -> None:
        """Release the line."""
        self.model.close()


class ClosedOrbitEngine:
    """Closed orbit at the BPMs with its knob Jacobian (orbit observables only)."""

    def __init__(
        self,
        config: WorkerConfig,
        init_knobs: Mapping[str, float],
        *,
        optics_names: Sequence[str],
        orbit_coords: Sequence[str],
        worker_id: int,
        logfile: Path | None = None,
        extra_knobs: Sequence[str] | None = None,
    ) -> None:
        del worker_id, logfile
        if extra_knobs is not None:
            raise NotImplementedError("xtrack backend: no corrector calibration (extra_knobs)")
        if optics_names:
            raise NotImplementedError("xtrack backend: closed orbit only (no optics)")
        self.model = _model(config, init_knobs)
        LOGGER.info(
            "xtrack backend: %d zero-field magnets removed", self.model.remove_zero_field_magnets()
        )
        for configure in (
            self.model.line.configure_sextupole_model,
            self.model.line.configure_octupole_model,
        ):
            configure(
                model="drift-kick-drift-exact",
                integrator="yoshida-6",
                num_multipole_kicks=CLOSED_ORBIT_MULTIPOLE_KICKS,
            )
        self.orbit_coords = list(orbit_coords)
        self.knob_names = list(self.model.knob_names)
        self.fit_pt = False
        self.params = _Parametric(self.model, self.knob_names)
        self.solver = ClosedOrbitSolver(self.model, self.params, _monomials(len(self.knob_names)))

    def set_knobs(self, knobs: Mapping[str, float]) -> None:
        """See :meth:`accelmo.backend.protocol.ClosedOrbitEngine.set_knobs`."""
        self.params.set(knobs)

    def get_vars(self, names: Sequence[str]) -> dict[str, float]:
        """See :meth:`accelmo.backend.protocol.ClosedOrbitEngine.get_vars`."""
        return self.model.get_vars(names)

    def set_vars(self, values: Mapping[str, float]) -> None:
        """See :meth:`accelmo.backend.protocol.ClosedOrbitEngine.set_vars`."""
        self.model.set_vars(values)

    def set_pt(self, pt: float) -> None:
        """See :meth:`accelmo.backend.protocol.ClosedOrbitEngine.set_pt`."""
        self.solver.pt = float(pt)

    def set_warm_start(self, key: str) -> None:
        """See :meth:`accelmo.backend.protocol.ClosedOrbitEngine.set_warm_start`."""
        self.solver.key = key

    def solve(self, observables: Sequence[str], *, jacobian: bool = True) -> ClosedSolution | None:
        """See :meth:`accelmo.backend.protocol.ClosedOrbitEngine.solve`."""
        return self.solver.solve(observables, jacobian=jacobian)

    def close(self) -> None:
        """Release the line."""
        self.model.close()


def open_model(
    accelerator: Accelerator,
    *,
    magnet_range: str,
    machine_state: MachineState | None = None,
    b2_errors: Path | None = None,
    bad_bpms: list[str] | None = None,
    debug: bool = False,
    mad_logfile: Path | None = None,
    anchor_mode: str | None = None,
    anchor_markers: Sequence[str] = (),
) -> XtrackModel:
    """See :meth:`accelmo.backend.protocol.Backend.open_model`."""
    del debug, mad_logfile
    if b2_errors is not None or anchor_mode is not None or anchor_markers:
        raise NotImplementedError(
            "xtrack backend: b2 errors and tracking anchors are not supported"
        )
    return XtrackModel(
        accelerator, magnet_range=magnet_range, bad_bpms=bad_bpms, machine_state=machine_state
    )


def closed_orbit_table(accelerator, magnet_strengths, pt, machine_state, columns):
    """Not implemented for xtrack."""
    raise NotImplementedError("xtrack backend: the momentum reference table is not implemented")
