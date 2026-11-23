"""Closed-orbit worker with BPM gain and corrector calibration parameters (see :mod:`accelmo.poco.calibration`)."""

from __future__ import annotations

import logging
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import numpy as np

from accelmo.fitting.bpm_gains import PLANES, bpm_gains, bpm_reading
from accelmo.fitting.protocol import GradReply
from accelmo.poco.calibration import (
    CalibrationBlocks,
    add_series,
    corrector_gain_knob_name,
    corrector_gain_name,
)
from accelmo.poco.workers.closed_orbit import ClosedOrbitWorker

if TYPE_CHECKING:
    from accelmo.fitting.protocol import Evaluate
    from accelmo.poco.calibration import CalibrationSpec
    from accelmo.poco.workers.closed_orbit import SeriesState

LOGGER = logging.getLogger(__name__)


def _spec(series: SeriesState) -> CalibrationSpec:
    """The calibration layout of a series."""
    if series.data.calibration is None:
        raise ValueError("A calibrated worker needs a CalibrationSpec on every series")
    return series.data.calibration


class CalibratedClosedOrbitWorker(ClosedOrbitWorker):
    """Worker that fits ``(1 + b_bpm) * model(q, kicks * (1 + g_corrector))`` to the measured orbit changes.

    Every calibrated corrector that is on (a non-zero ``<kick_prefix><corrector>`` global) in either the series' state or its
    reference state has its whole kick scaled by its gain, in both states.

    Assumes the baseline lattice has no enabled correctors: a kick that is non-zero only in the baseline (not set by
    either state) gets no gain knob and is never scaled.

    The knob dict it receives carries the gains next to the magnet knobs (names from :mod:`accelmo.poco.calibration`);
    the engine never sees them. Its reply carries the blocks' header (sizes and chi²) in :attr:`~accelmo.fitting.protocol.GradReply.extra`, followed on the pipe by the blocks' arrays.
    """

    def _make_engine(self, knobs: dict[str, float]):
        return self.backend.ClosedOrbitEngine(
            self.config,
            knobs,
            extra_knobs=self._gain_knobs(),
            optics_names=self.optics_names,
            orbit_coords=self.orbit_coords,
            worker_id=self.worker_id,
            logfile=self.engine_logfile,
        )

    def _gain_knobs(self) -> list[str]:
        """One more knob per corrector that is on in this worker's series: its gain offset, after the magnet knobs."""
        correctors = list(
            dict.fromkeys(
                corrector
                for series in self.series
                for name, value in (
                    *series.data.machine_state.items(),
                    *series.data.reference_state.items(),
                )
                if (corrector := _spec(series).corrector_of(name)) in _spec(series).correctors
                and value != 0.0
            )
        )
        #: The correctors with a gain knob, in Jacobian column order (columns ``n_q`` onwards).
        self._correctors = correctors
        return [corrector_gain_knob_name(corrector) for corrector in correctors]

    def on_start(self, knobs: dict[str, float]) -> None:
        """Start the engine and count the magnet knobs."""
        super().on_start(knobs)
        #: Number of magnet knobs; ``n_knobs`` also counts the gain knobs.
        self.n_q = self.n_knobs - len(self._correctors)

    def _enter_state(self, role: str, knob_updates: dict[str, float]) -> None:
        """Enter *role*'s state with every corrector kick ``k`` set to ``k * (1 + g + gain knob)``.

        The gain knob is 0, so the value is ``k * (1 + g)``, and its Jacobian column is d(orbit)/d(g).
        """
        state = self._states[role]
        self.engine.set_vars(state)
        spec = _spec(self.series[0])
        scaled = []
        for name, kick in state.items():
            corrector = spec.corrector_of(name)
            if corrector in self._correctors and kick != 0.0:
                gain = 1.0 + float(knob_updates.get(corrector_gain_name(corrector), 0.0))
                scaled.append((name, kick, gain, corrector_gain_knob_name(corrector)))
        self.engine.scale_globals(scaled)

    def evaluate(self, message: Evaluate) -> GradReply:
        """Loss, gradient and calibration blocks at ``message.knobs``."""
        if message.loss_only:
            return self._loss_only(message.knobs)
        knob_updates = message.knobs
        self.engine.set_knobs(knob_updates)
        spec = _spec(self.series[0])
        n_g, n_b = len(spec.correctors), len(PLANES) * len(spec.bpms)
        blocks = CalibrationBlocks(
            self.n_q, n_g, n_b, self._shared_buffer(CalibrationBlocks.nbytes(self.n_q, n_g, n_b))
        )
        for series in self.series:
            if not self._evaluate_calibrated(series, knob_updates, spec, blocks):
                return GradReply(self.worker_id, float("nan"), np.zeros(1), extra={})
        blocks.symmetrise()
        payload = blocks.to_payload()
        payload["shared_memory"] = self._shared_block().name
        # The fitter reads the arrays from the shared block and everything else from the payload.
        return GradReply(self.worker_id, blocks.loss, np.zeros(1), extra=payload)

    _shared: shared_memory.SharedMemory | None = None

    def _shared_block(self, size: int = 0) -> shared_memory.SharedMemory:
        """The worker's block of shared memory the blocks are accumulated in, created on first use and reused every iteration."""
        if self._shared is None:
            self._shared = shared_memory.SharedMemory(create=True, size=size)
        return self._shared

    def _shared_buffer(self, size: int):
        """The buffer of :meth:`_shared_block`."""
        return self._shared_block(size).buf

    def close(self) -> None:
        """Close the engine."""
        super().close()
        if self._shared is not None:
            self._shared.close()
            self._shared.unlink()
            self._shared = None

    def _loss_only(self, knob_updates: dict[str, float]) -> GradReply:
        """Data loss alone, from plain closed-orbit solves (no knob parameters, no Jacobian, no Hessian).

        The same chi² as the blocks' ``loss`` of a full evaluation.
        """
        self.engine.set_knobs(knob_updates)
        total = 0.0
        for series in self.series:
            part = self._evaluate_calibrated_loss(series, knob_updates)
            if part is None:
                return GradReply(self.worker_id, float("nan"), np.zeros(1), extra={})
            total += part
        return GradReply(self.worker_id, total, np.zeros(1))

    def _evaluate_calibrated_loss(
        self, series: SeriesState, knob_updates: dict[str, float]
    ) -> float | None:
        """One series' ``sum w ((1 + b) * model - target)^2`` as in :func:`add_series`, from orbit values only."""
        loss = 0.0
        names, alignment = series.aligned()
        with self._in_series(series):
            for index, model in self._models(series, knob_updates, self._orbit_plain):
                if model is None:
                    return None
                observables, targets, weights = alignment[index]
                for o, observable in enumerate(observables):
                    gains = bpm_gains(knob_updates, observable.name, names)
                    residual = bpm_reading(model[o], gains) - np.asarray(targets[o], dtype=float)
                    w = np.asarray(weights[o], dtype=float)
                    loss += float(np.sum(w * np.where(w > 0.0, residual, 0.0) ** 2))
        return loss

    def _evaluate_calibrated(
        self,
        series: SeriesState,
        knob_updates: dict[str, float],
        spec: CalibrationSpec,
        blocks: CalibrationBlocks,
    ) -> bool:
        """Add one series' blocks to ``blocks``; ``False`` if its closed orbit is lost."""
        names, alignment = series.aligned()
        bpm_index = {name: i for i, name in enumerate(spec.bpms)}
        corrector_indices = [spec.correctors.index(corrector) for corrector in self._correctors]
        with self._in_series(series):
            for index, model in self._models(series, knob_updates, self._model_and_jacobian):
                if model is None:
                    return False
                values, jacobian = model
                observables, targets, weights = alignment[index]
                cols = np.empty((len(observables), len(names)), dtype=int)
                gains = np.empty((len(observables), len(names)))
                for o, observable in enumerate(observables):
                    plane = PLANES.index(observable.name)
                    cols[o] = [plane * len(spec.bpms) + bpm_index[name] for name in names]
                    gains[o] = bpm_gains(knob_updates, observable.name, names)
                q_jac, d_gain = jacobian[..., : self.n_q], jacobian[..., self.n_q :]
                add_series(
                    blocks, values, q_jac, d_gain, targets, weights, gains, cols, corrector_indices
                )
        return True
