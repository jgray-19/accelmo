"""Particle tracking worker for multi-turn beam dynamics simulations.

A :class:`TrackingWorker` tracks its particles through its BPM range and returns
the weighted least-squares loss and its knob gradient. Constructed with
``validation=True`` it instead scores held-out turns: no knob parameters, no
gradients, only the loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from accelmo.backend.protocol import TrackRequest
from accelmo.fitting.bpm_gains import PLANES, bpm_gain_names, bpm_reading, without_bpm_gains
from accelmo.fitting.protocol import (
    Ack,
    GradReply,
    LossReply,
    SetKnobs,
    SetUncertaintyMode,
    TotalChi2,
    UncertaintyProgress,
    UpdateInitCoords,
)
from accelmo.fitting.worker import AbstractWorker
from accelmo.tracking.dispatch.setup import TrackingWorkerConfig
from accelmo.tracking.uncertainty import UncertaintyPart

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from accelmo.backend.protocol import SecondOrderBlocks, TrackingEngine, TrackResult
    from accelmo.config import TrackingDataConfig, TrackingParameters, WorkerRuntime
    from accelmo.fitting.protocol import Command, Evaluate

LOGGER = logging.getLogger(__name__)


@dataclass
class PrecomputedTrackingWeights:
    """Per-observable inverse-variance weights ``1/σ²`` (0 drops a point)."""

    x: np.ndarray
    y: np.ndarray
    px: np.ndarray
    py: np.ndarray


@dataclass
class TrackingData:
    """Reference data for a tracking-loss evaluation.

    Position and momentum comparison arrays use shape
    ``(n_particles, n_data_points, 2)``, with the last axis storing the two
    transverse components for each observable family.

    Reading ids identify one measured grid cell (file, turn, marker) across all
    workers, so the uncertainty propagation can add up every use of the same noisy
    reading -- as an observation or as a start coordinate.
    """

    position_comparisons: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    momentum_comparisons: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    position_variances: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    momentum_variances: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    init_coords: np.ndarray  # Shape: (n_particles, 6)
    init_pts: np.ndarray  # Shape: (n_particles,)
    reading_ids: np.ndarray  # Shape: (n_particles, n_data_points)
    init_reading_ids: np.ndarray  # Shape: (n_particles,)
    init_variances: np.ndarray  # Shape: (n_particles, 4), var of the start x, px, y, py
    #: Start turn minus the file's first turn (AC-kicker phase). Shape: (n_particles,)
    start_turn_offsets: np.ndarray
    precomputed_weights: PrecomputedTrackingWeights | None
    #: BPMs with a fitted gain, in gain-knob order; empty when no gains are fitted.
    gain_bpms: tuple[str, ...] = ()
    #: Index of each point's BPM in ``gain_bpms``, -1 for none. Shape: (n_data_points,)
    point_gain_index: np.ndarray | None = None


class ParticleLostError(Exception):
    """Raised when the backend reports lost particles."""


OBSERVABLE_SPECS: dict[str, tuple[str, int]] = {
    "x": ("position_comparisons", 0),
    "y": ("position_comparisons", 1),
    "px": ("momentum_comparisons", 0),
    "py": ("momentum_comparisons", 1),
}

#: Last digit of a reading id: which number in a (file, turn, marker) grid cell was
#: read. The start coordinates x0, px0, y0, py0 are the readings of their cell.
READING_CODES: dict[str, int] = {"x": 0, "y": 1, "px": 2, "py": 3}
#: Reading codes in the order of the start coordinates, the rows of ``d{o}_dc0``.
START_CODES = np.array([READING_CODES[name] for name in ("x", "px", "y", "py")])


def active_observables(kick_plane: str, include_momentum: bool) -> tuple[str, ...]:
    """Observables a worker of this kick plane compares."""
    if kick_plane == "xy":
        return ("x", "y", "px", "py") if include_momentum else ("x", "y")
    if kick_plane == "x":
        return ("x", "px") if include_momentum else ("x",)
    if kick_plane == "y":
        return ("y", "py") if include_momentum else ("y",)
    raise ValueError(f"Unsupported kick plane {kick_plane!r}")


def _largest_divisor_at_most(value: int, limit: int) -> int:
    """Return the largest positive divisor of ``value`` not exceeding ``limit``."""
    if value <= 0:
        return 0
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


class TrackingWorker(AbstractWorker[TrackingData, TrackingWorkerConfig]):
    """Worker for particle tracking simulations.

    Tracks every particle of a batch through the worker's range, compares the
    observed positions (and, with ``TrackingDataConfig.optimise_momenta``, momenta)
    with the measurement, and returns the loss and its knob gradient from the
    differential-algebra map.
    """

    engine: TrackingEngine

    def __init__(
        self,
        conn: Connection,
        worker_id: int,
        data: TrackingData,
        config: TrackingWorkerConfig,
        runtime: WorkerRuntime,
        tracking_data: TrackingDataConfig,
        parameters: TrackingParameters,
        validation: bool = False,
    ) -> None:
        self.validation = validation
        self.tracking_data = tracking_data
        self.parameters = parameters
        super().__init__(conn, worker_id, data, config, runtime)

    def prepare_data(self, data: TrackingData) -> None:
        """Split the measurement and start coordinates into batches."""
        self.observables = active_observables(
            self.config.kick_plane, self.tracking_data.optimise_momenta
        )
        n_init = len(data.init_coords)
        if self.validation:
            # Every batch has one fixed ``batch_size``, so a divisor batch count keeps
            # every held-out turn.
            num_batches = _largest_divisor_at_most(n_init, self.tracking_data.num_batches)
        else:
            num_batches = min(self.tracking_data.num_batches, n_init)
        if num_batches <= 0:
            raise ValueError(f"Worker {self.worker_id}: No initial coordinates available")
        LOGGER.debug(
            f"Worker {self.worker_id}: Processing {n_init} particles in {num_batches} batches"
        )

        init_coords = data.init_coords
        if np.isnan(init_coords).any():
            raise ValueError(f"Worker {self.worker_id}: NaNs found in initial coordinates")
        if data.precomputed_weights is None:
            raise ValueError("Precomputed weights must be provided for TrackingWorker")

        self.comparisons: dict[str, list[np.ndarray]] = {}
        self.weights: dict[str, list[np.ndarray]] = {}
        for observable in self.observables:
            source_attr, plane_idx = OBSERVABLE_SPECS[observable]
            values = getattr(data, source_attr)[:n_init][:, :, plane_idx]
            weights = getattr(data.precomputed_weights, observable)[:n_init]
            self.comparisons[observable] = np.array_split(values, num_batches)
            self.weights[observable] = np.array_split(weights, num_batches)
        self._prepare_gains(data)

        # Flat copy kept in sync with the engine by per-epoch start-coordinate updates.
        self._init_coords_np = np.ascontiguousarray(init_coords, dtype=np.float64)
        self.init_coords = [batch.tolist() for batch in np.array_split(init_coords, num_batches)]
        self.init_pts = [batch.tolist() for batch in np.array_split(data.init_pts, num_batches)]
        self.batch_size = len(self.init_coords[0])
        self.num_batches = num_batches
        self.turn_offsets = np.unique(data.start_turn_offsets[:n_init])
        # Reading ids carry the file index in their high 32 bits; a worker holds one file.
        self.file_index = int(data.init_reading_ids[0] >> 32)

        self.uncertainty_hessians = (self.parameters.uncertainty_hessian,)

        if not self.validation:
            self._prepare_uncertainty_batches(data, n_init, num_batches)

    def _prepare_gains(self, data: TrackingData) -> None:
        """Gain bookkeeping: names in reply order, values, and each point's gain index."""
        self.fit_gains = bool(data.gain_bpms)
        self.gain_names = bpm_gain_names(data.gain_bpms)
        self.n_gain_bpms = len(data.gain_bpms)
        self.gain_values = np.zeros(len(self.gain_names))
        if self.fit_gains:
            self.point_gain_index = np.asarray(data.point_gain_index)
            self.has_gain = self.point_gain_index >= 0

    def _update_gains(self, knobs: dict[str, float]) -> None:
        for i, name in enumerate(self.gain_names):
            if name in knobs:
                self.gain_values[i] = knobs[name]

    def _gain_factor(self, plane_idx: int) -> np.ndarray:
        """``1 + g`` of each observation point for one plane."""
        gains = self.gain_values[plane_idx * self.n_gain_bpms : (plane_idx + 1) * self.n_gain_bpms]
        return bpm_reading(
            1.0, np.where(self.has_gain, gains[np.maximum(self.point_gain_index, 0)], 0.0)
        )

    def _reading_factor(self, observable: str) -> np.ndarray | float:
        """``1 + g`` of each observation point of ``observable``; momenta carry no gain."""
        if not self.fit_gains or observable not in PLANES:
            return 1.0
        return self._gain_factor(PLANES.index(observable))

    def _apply_gains(
        self, results: dict[str, np.ndarray], *, gain_columns: bool = False
    ) -> dict[str, np.ndarray]:
        """Turn tracked positions into BPM readings.

        Positions become ``(1 + g) m`` (``m`` is kept as ``<o>_tracked`` for
        :meth:`_gain_sums`); their knob and start derivatives scale by ``1 + g``.
        ``gain_columns`` also appends the gain derivatives, ``m`` at each BPM's points,
        to the Jacobian (knobs first, then gains), for the uncertainty and the LM fit.
        """
        if not self.fit_gains:
            return results
        for observable in self.observables:
            plane_idx = PLANES.index(observable) if observable in PLANES else None
            tracked = results[observable]
            factor = self._reading_factor(observable)
            results[observable] = tracked * factor
            results[f"{observable}_tracked"] = tracked
            key = self._gradient_key(observable)
            if key not in results:
                continue
            if not gain_columns:
                results[key] = results[key] * factor
                continue
            gain_jac = np.zeros((tracked.shape[0], len(self.gain_names), tracked.shape[1]))
            if plane_idx is not None:
                points = np.flatnonzero(self.has_gain)
                rows = plane_idx * self.n_gain_bpms + self.point_gain_index[points]
                gain_jac[:, rows, points] = tracked[:, points]
            results[key] = np.concatenate([results[key] * factor, gain_jac], axis=1)
            start_key = f"d{observable}_dc0"
            if start_key in results:
                results[start_key] = results[start_key] * factor
        return results

    def _gain_sums(self, results: dict[str, np.ndarray], batch: int) -> np.ndarray:
        """Per-gain ``Σ w m (y - m)`` and ``Σ w m²`` over this batch, with ``m`` the tracked position.

        The loss is quadratic in each gain, ``∂L/∂g = 2 (g Σ w m² - Σ w m (y - m))``,
        so these sums, added over every worker, give the best gains in closed form.
        """
        weights = self._batch_weights(batch)
        sums = np.zeros((2, len(self.gain_names)))
        points = np.flatnonzero(self.has_gain)
        for plane_idx, observable in enumerate(PLANES):
            if observable not in self.observables:
                continue
            tracked = results[f"{observable}_tracked"][:, points]
            weight = weights[observable][:, points]
            measured = self.comparisons[observable][batch][:, points]
            columns = plane_idx * self.n_gain_bpms + self.point_gain_index[points]
            n = len(self.gain_names)
            sums[0] += np.bincount(
                columns, (weight * tracked * (measured - tracked)).sum(axis=0), n
            )
            sums[1] += np.bincount(columns, (weight * tracked**2).sum(axis=0), n)
        return sums

    def _prepare_uncertainty_batches(
        self, data: TrackingData, n_init: int, num_batches: int
    ) -> None:
        """Split reading ids and variances like the comparisons, for the uncertainty part."""
        self.variances = {}
        for observable in self.observables:
            source_attr, plane_idx = OBSERVABLE_SPECS[observable]
            variances = getattr(data, source_attr.replace("comparisons", "variances"))
            self.variances[observable] = np.array_split(
                variances[:n_init, :, plane_idx], num_batches
            )
        self.reading_ids = np.array_split(data.reading_ids[:n_init], num_batches)
        self.init_reading_ids = np.array_split(data.init_reading_ids[:n_init], num_batches)
        self.init_variances = np.array_split(data.init_variances[:n_init], num_batches)

    def on_start(self, knobs: dict[str, float]) -> None:
        """Build the tracking engine, load the particles and check the observation geometry."""
        self._update_gains(knobs)
        init_knobs = without_bpm_gains(knobs)
        turn_offset = 0
        if self.config.accelerator.has_ac_kickers:
            if len(self.turn_offsets) > 1:
                raise ValueError(
                    f"Worker {self.worker_id}: particles start {self.turn_offsets.tolist()} turns "
                    "after their file's first turn; one track shares one AC-kicker phase"
                )
            turn_offset = int(self.turn_offsets[0])
        plan = TrackRequest(
            geometry=self.config.geometry,
            observables=self.observables,
            batch_size=self.batch_size,
            num_batches=self.num_batches,
            validation=self.validation,
            file_index=self.file_index,
            turn_offset=turn_offset,
        )
        self.engine = self.backend.TrackingEngine(
            self.config, plan, init_knobs, worker_id=self.worker_id, logfile=self.engine_logfile
        )
        knob_names = set(self.engine.knob_names)
        # Non-optimised pt is not an element strength: a fixed tracking scalar.
        self.fixed_pt = float(init_knobs.get("pt", 0.0)) if "pt" not in knob_names else 0.0
        # Replies carry one gradient entry per fitted knob (not the fixed model values).
        self.n_reply_knobs = len(knob_names)
        self.engine.load_particles(self.init_coords, self.init_pts)
        self.run_preflight_check()
        LOGGER.debug(
            f"Worker {self.worker_id}: Ready for computation with {self.engine.nbpms} BPMs"
        )

    def run_preflight_check(self) -> None:
        """Validate the observation geometry once with a single-particle dry run.

        Tracks one particle through the configured range and turns and confirms the
        engine observes exactly ``nbpms * n_turns`` points, the size every real
        run fills. A mis-scoped observe/range (e.g. an off-plane BPM left observed
        outside the range) then fails here with a clear message instead of deep in
        the optimisation loop.
        """
        report = self.engine.preflight()
        if report.get("lost"):
            # A lost particle is handled the same way at runtime (the epoch is
            # rejected); a truncated track gives a misleading observation count.
            LOGGER.warning(
                "Worker %d: preflight particle lost; skipping observation-count check "
                "(range=%s sdir=%d)",
                self.worker_id,
                self.config.geometry.tracking_range,
                self.config.geometry.sdir,
            )
            return
        observed = int(report["observed"])
        expected = int(report["expected"])
        if observed != expected:
            raise ValueError(
                f"Worker {self.worker_id}: preflight observed {observed} BPM points per run "
                f"but the result vectors hold {expected} (nbpms={self.engine.nbpms} x n_turns). "
                f"An observe/range mismatch would overflow tracking; check the observe "
                f"pattern and bad_bpms for plane {self.config.kick_plane!r} "
                f"(range={self.config.geometry.tracking_range!r}, sdir={self.config.geometry.sdir})."
            )
        LOGGER.debug(
            "Worker %d: preflight OK (%d observed == %d allocated)",
            self.worker_id,
            observed,
            expected,
        )

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _run_tracking_batch(
        self, knob_updates: dict[str, float], batch: int, *, gain_columns: bool = False
    ) -> dict[str, np.ndarray]:
        """Track one batch at ``knob_updates`` and return all outputs.

        ``gain_columns`` appends the gain derivatives to the knob Jacobian (see :meth:`_apply_gains`).
        """
        machine_pt = knob_updates.get("pt", self.fixed_pt)
        self._update_gains(knob_updates)
        self.engine.set_knobs(knob_updates)
        self.engine.set_pts(machine_pt, batch, batch)
        start_derivatives = self.parameters.start_derivatives and not self.validation
        return self._results(
            self.engine.track(batch, start_derivatives=start_derivatives), gain_columns=gain_columns
        )

    def _results(self, result: TrackResult, *, gain_columns: bool = False) -> dict[str, np.ndarray]:
        """Flatten a :class:`TrackResult` into ``{o, d{o}_dk, d{o}_dc0}`` and apply the gains.

        A validation worker gets the observables alone. Lost particles raise
        :class:`ParticleLostError`.
        """
        results: dict[str, np.ndarray] = dict(result.obs)
        if self.validation:
            return self._apply_gains(results)
        if result.d_dk is None:
            raise ValueError(f"Worker {self.worker_id}: the engine returned no knob derivatives")
        for observable in self.observables:
            results[self._gradient_key(observable)] = result.d_dk[observable]
        if result.d_dc0 is not None:
            for observable in self.observables:
                results[f"d{observable}_dc0"] = result.d_dc0[observable]
        if result.n_lost > 0:
            pct = 100.0 * result.n_lost / result.n_total
            raise ParticleLostError(
                f"Worker {self.worker_id}: {result.n_lost}/{result.n_total} particles lost "
                f"({pct:.1f}%) during tracking"
            )
        return self._apply_gains(results, gain_columns=gain_columns)

    @staticmethod
    def _gradient_key(observable: str) -> str:
        """Return the gradient result key for an observable."""
        return f"d{observable}_dk"

    def _batch_weights(self, batch: int) -> dict[str, np.ndarray]:
        """Return the weights of one batch for each observable."""
        return {observable: self.weights[observable][batch] for observable in self.observables}

    def _residuals(self, results: dict[str, np.ndarray], batch: int) -> dict[str, np.ndarray]:
        """Return residual arrays for the current batch."""
        return {
            observable: results[observable] - self.comparisons[observable][batch]
            for observable in self.observables
        }

    def _compute_loss(self, results: dict[str, np.ndarray], batch: int) -> float:
        """Weighted least-squares loss of one batch."""
        weights = self._batch_weights(batch)
        residuals = self._residuals(results, batch)
        return float(
            sum(
                np.sum(weights[observable] * residuals[observable] ** 2)
                for observable in self.observables
            )
        )

    def _compute_loss_and_gradients(
        self, results: dict[str, np.ndarray], batch: int
    ) -> tuple[np.ndarray, float]:
        """Weighted least-squares loss and its knob gradient for one batch."""
        weights = self._batch_weights(batch)
        residuals = self._residuals(results, batch)
        gradient_shape = results[self._gradient_key(self.observables[0])].shape[1]
        grad = np.zeros(gradient_shape, dtype=np.float64)
        for observable in self.observables:
            grad += np.einsum(
                "pkm,pm->k",
                results[self._gradient_key(observable)],
                weights[observable] * residuals[observable],
            )
        return 2.0 * grad, self._compute_loss(results, batch)

    def _gauss_newton(self, results: dict[str, np.ndarray], batch: int) -> np.ndarray:
        """The normal matrix ``JᵀWJ`` of one batch."""
        weights = self._batch_weights(batch)
        terms = []
        for observable in self.observables:
            jacobian = results[self._gradient_key(observable)]  # (particles, knobs, points)
            terms.append(
                np.einsum("pkm,pjm->kj", jacobian * weights[observable][:, None, :], jacobian)
            )
        return np.sum(terms, axis=0)

    def _start_blocks(self, results: dict[str, np.ndarray], batch: int) -> dict[str, np.ndarray]:
        """Gradient and Gauss-Newton blocks of the batch's start coordinates (x, px, y, py).

        ``grad`` (particles, 4), ``knob_start`` (knobs, particles, 4) and ``start_start``
        (particles, 4, 4): one particle's start only moves its own track. ``start`` holds
        the current start coordinates.
        """
        weights = self._batch_weights(batch)
        residuals = self._residuals(results, batch)
        n_particles, n_knobs = residuals[self.observables[0]].shape[0], self.n_uncertainty_params
        grad = np.zeros((n_particles, 4))
        knob_start = np.zeros((n_knobs, n_particles, 4))
        start_start = np.zeros((n_particles, 4, 4))
        for observable in self.observables:
            dstart = results[f"d{observable}_dc0"]  # (particles, 4, points)
            weighted = dstart * weights[observable][:, None, :]
            grad += np.einsum("pcm,pm->pc", weighted, residuals[observable])
            knob_start += np.einsum(
                "pkm,pcm->kpc", results[self._gradient_key(observable)], weighted
            )
            start_start += np.einsum("pcm,pdm->pcd", weighted, dstart)
        first = sum(len(b) for b in self.init_coords[:batch])
        return {
            "grad": 2.0 * grad,
            "knob_start": 2.0 * knob_start,
            "start_start": 2.0 * start_start,
            "start": self._init_coords_np[first : first + n_particles, :4].copy(),
        }

    def evaluate(self, message: Evaluate) -> GradReply:
        """Chi-square of one batch and its gradient (and Gauss-Newton blocks with ``curvature``)."""
        if self.validation:
            raise ValueError(
                f"Worker {self.worker_id}: validation workers do not evaluate gradients"
            )
        n_cols = self.n_uncertainty_params if message.curvature else self.n_reply_knobs
        zeros = np.zeros(n_cols)
        zero_curvature = np.zeros((n_cols, n_cols)) if message.curvature else None
        batch = int(message.batch)
        try:
            results = self._run_tracking_batch(
                message.knobs, batch, gain_columns=bool(message.curvature)
            )
        except ParticleLostError as exc:
            LOGGER.warning(
                "Worker %d: %s — sending zero contribution; epoch update will be rejected",
                self.worker_id,
                exc,
            )
            # NaN loss signals the aggregator to reject this epoch's knob updates.
            return GradReply(self.worker_id, float("nan"), zeros, zero_curvature)
        grad, loss = self._compute_loss_and_gradients(results, batch)
        if message.curvature:
            normal = self._gauss_newton(results, batch)
            extra = (
                self._start_blocks(results, batch) if self.parameters.start_derivatives else None
            )
            return GradReply(self.worker_id, loss, grad, normal, extra)
        return GradReply(
            self.worker_id,
            loss,
            grad,
            extra=self._gain_sums(results, batch) if self.fit_gains else None,
        )

    def total_chi2(self, knobs: dict[str, float]) -> float:
        """Chi-square of every batch at ``knobs``."""
        return sum(
            self._compute_loss(self._run_tracking_batch(knobs, batch), batch)
            for batch in range(self.num_batches)
        )

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def handle_command(self, command: Command) -> Ack | LossReply:
        """Answer a validation, diagnose, knob or start-coordinate command."""
        match command:
            case TotalChi2(knobs):
                return LossReply(self.worker_id, self.total_chi2(knobs))
            case SetKnobs(knobs):
                self._update_gains(knobs)
                self.engine.set_knobs(knobs)
                self.engine.set_pts(knobs.get("pt", self.fixed_pt), 0, self.num_batches - 1)
            case UpdateInitCoords(coords):
                self._send_init_condition_update(
                    *(np.asarray(coords[:, [i]], dtype=np.float64) for i in range(4))
                )
            case SetUncertaintyMode(hessians):
                self.uncertainty_hessians = hessians
            case _:
                return super().handle_command(command)
        return Ack(self.worker_id)

    def _send_init_condition_update(
        self,
        new_x: np.ndarray,
        new_px: np.ndarray,
        new_y: np.ndarray,
        new_py: np.ndarray,
    ) -> None:
        """Move every particle's start x, px, y, py.

        The positions are updated alongside the momenta because a launch point that
        sits on a closed orbit moves when the lattice does: freeing the magnets that
        shape that orbit while holding the start x/y fixed would launch the particle
        off the orbit the fitted knobs define. A caller that only re-derives momenta
        passes the current positions back unchanged.
        """
        self.engine.set_start_coords(new_x, new_px, new_y, new_py)
        self._init_coords_np[:, 0] = new_x.ravel()
        self._init_coords_np[:, 1] = new_px.ravel()
        self._init_coords_np[:, 2] = new_y.ravel()
        self._init_coords_np[:, 3] = new_py.ravel()

    # ------------------------------------------------------------------
    # Uncertainty propagation
    # ------------------------------------------------------------------

    def on_stop(self, failed: bool) -> None:
        """A training worker answers :class:`Stop` with its uncertainty part.

        A failed worker has already sent its :class:`ErrorReply` and sends nothing more.
        """
        if self.validation or failed:
            return
        LOGGER.debug(
            f"Worker {self.worker_id}: Propagating uncertainty {self.uncertainty_hessians}"
        )
        try:
            self.send_reply(self._compute_uncertainty_parts(self.n_uncertainty_params))
        except Exception as exc:  # noqa: BLE001
            self.send_error_payload(exc, phase="uncertainty")

    @property
    def n_uncertainty_params(self) -> int:
        """The magnet knobs, then every gain: the uncertainty covers both."""
        return self.n_reply_knobs + len(self.gain_names)

    def _second_order_blocks(self) -> list[SecondOrderBlocks]:
        """Per batch ``Σ w r ∂²m`` at the loaded knobs, from MAD-NG's second-order TPSA.

        ``r`` comes from a first-order track; the engine then switches to second order for good,
        and its ``track`` keeps returning the first derivatives.
        """
        weighted = []
        for batch in range(self.num_batches):
            results = self._results(self.engine.track(batch, start_derivatives=False))
            weights, residuals = self._batch_weights(batch), self._residuals(results, batch)
            weighted.append(
                {o: weights[o] * residuals[o] * self._reading_factor(o) for o in self.observables}
            )
        self.engine.begin_second_order()
        blocks = []
        for batch in range(self.num_batches):
            blocks.append(self.engine.second_order(batch, weighted[batch]))
            self.send_reply(UncertaintyProgress(self.worker_id))
        return blocks

    def _residual_blocks(
        self, blocks: SecondOrderBlocks, terms: list[tuple], n_knobs: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``Σ w r ∂²m_obs`` of one batch over (knobs, gains): ``(kk, start, start_start)``.

        ``m_obs = (1 + g) m``: the knob and start blocks are MAD-NG's (it was sent ``w r (1 + g)``);
        a gain only multiplies, so its mixed derivatives are ``∂m/∂k`` and ``∂m/∂c0`` of the tracked
        position, and the gain-gain block is zero.
        """
        k = self.n_reply_knobs
        kk = np.zeros((n_knobs, n_knobs))
        kk[:k, :k] = blocks.knob_knob
        start = np.zeros((len(blocks.start_start), 4, n_knobs))
        start[:, :, :k] = blocks.knob_start.transpose(0, 2, 1)
        if self.fit_gains:
            points = np.flatnonzero(self.has_gain)
            for observable, jacobian, d_dc0, weight, residual in terms:
                if observable not in PLANES:
                    continue
                plane_idx = PLANES.index(observable)
                columns = k + plane_idx * self.n_gain_bpms + self.point_gain_index[points]
                tracked_wr = (weight * residual)[:, points] / self._gain_factor(plane_idx)[points]
                cross = np.einsum("pkn,pn->kn", jacobian[:, :k][:, :, points], tracked_wr)
                knobs = np.arange(k)
                np.add.at(kk, (knobs[:, None], columns[None, :]), cross)
                np.add.at(kk, (columns[:, None], knobs[None, :]), cross.T)
                np.add.at(
                    start,
                    (slice(None), slice(None), columns),
                    d_dc0[:, :, points] * tracked_wr[:, None, :],
                )
        return kk, start, blocks.start_start

    def _compute_uncertainty_parts(self, n_knobs: int) -> dict[str, UncertaintyPart]:
        """This worker's uncertainty part for each requested Hessian; one second-order pass serves all."""
        hessians = self.uncertainty_hessians
        second_order = self._second_order_blocks() if "true" in hessians else None
        return {
            hessian: self._compute_uncertainty_part(
                n_knobs, second_order if hessian == "true" else None
            )
            for hessian in hessians
        }

    def _compute_uncertainty_part(
        self, n_knobs: int, second_order: list[SecondOrderBlocks] | None
    ) -> UncertaintyPart:
        """Track every batch at the loaded knobs and return this worker's uncertainty part.

        ``second_order`` (the true Hessian's residual term) is ``None`` for Gauss-Newton.

        With ``r = m − y`` and loss weights ``w = 1/σ²`` (masked):
        ``A = Σ w J Jᵀ``; an observed reading moves the gradient by ``−w J``; a start
        coordinate by ``Σ_n w J ∂m/∂c0``. :func:`merge_uncertainty_parts` adds these
        over every worker using the same reading. Only rows that can carry noise are
        built: observations with a non-zero weight and starts with a finite variance.

        With ``fit_start_coords`` each particle's start is a fit parameter: its direction is
        projected out (``J → J − A_ks A_ss⁻¹ ∂m/∂c0`` per particle, ``A`` its Schur complement)
        and the measured start variances do not enter.
        """
        fit_starts = self.parameters.fit_start_coords
        normal = np.zeros((n_knobs, n_knobs))
        ids, sensitivities, variances = [], [], []
        for batch in range(self.num_batches):
            results = self._results(
                self.engine.track(batch, start_derivatives=True), gain_columns=True
            )
            weights = self._batch_weights(batch)
            residuals = self._residuals(results, batch)
            n_particles = len(self.init_reading_ids[batch])
            start = np.zeros((n_particles, 4, n_knobs))
            start_start = np.zeros((n_particles, 4, 4))
            terms = []
            for observable in self.observables:
                jacobian = results[self._gradient_key(observable)]  # (particles, knobs, points)
                d_dc0 = results[f"d{observable}_dc0"]  # (particles, 4, points)
                weight = weights[observable]  # (particles, points)
                start += np.einsum("pkn,pcn->pck", jacobian * weight[:, None, :], d_dc0)
                start_start += np.einsum("pcn,pdn->pcd", d_dc0 * weight[:, None, :], d_dc0)
                terms.append((observable, jacobian, d_dc0, weight, residuals[observable]))
            if second_order is not None:
                extra_knobs, extra_start, extra_start_start = self._residual_blocks(
                    second_order[batch], terms, n_knobs
                )
                normal += extra_knobs
                start += extra_start
                start_start += extra_start_start
            if fit_starts:
                # (particles, knobs, 4): A_ks A_ss⁻¹ of each particle
                projection = np.linalg.solve(start_start, start).transpose(0, 2, 1)
            if fit_starts and second_order is not None:
                normal -= np.einsum("pkc,pcj->kj", projection, extra_start)
            for observable, jacobian, d_dc0, weight, _ in terms:
                if fit_starts:
                    jacobian = jacobian - np.einsum("pkc,pcn->pkn", projection, d_dc0)
                weighted = jacobian * weight[:, None, :]
                normal += np.einsum("pkn,pjn->kj", weighted, jacobian)
                noisy = weight != 0.0
                ids.append(self.reading_ids[batch][noisy] * 4 + READING_CODES[observable])
                sensitivities.append(-weighted.transpose(0, 2, 1)[noisy])
                variances.append(self.variances[observable][batch][noisy])
            if fit_starts:
                continue
            noisy_start = np.isfinite(self.init_variances[batch])  # (particles, 4)
            ids.append((self.init_reading_ids[batch][:, None] * 4 + START_CODES)[noisy_start])
            sensitivities.append(start[noisy_start])
            variances.append(self.init_variances[batch][noisy_start])

        return UncertaintyPart(
            normal=normal,
            reading_ids=np.concatenate(ids),
            sensitivities=np.concatenate(sensitivities),
            variances=np.concatenate(variances),
        )
