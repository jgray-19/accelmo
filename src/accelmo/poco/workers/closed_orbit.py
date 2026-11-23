"""Closed-orbit worker: absolute or reference-subtracted orbit series, and absolute closed-twiss series."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from accelmo.fitting.protocol import GradReply
from accelmo.fitting.shared_reference import ReferencePublisher, ReferenceReader
from accelmo.fitting.worker import AbstractWorker, WorkerConfig
from accelmo.poco.workers.closed_twiss import (
    OBSERVABLE_KINDS,
    ORBIT_COORDS,
    Observable,
    align_observables,
    weighted_least_squares,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from accelmo.fitting.protocol import Evaluate
    from accelmo.poco.calibration import CalibrationSpec

LOGGER = logging.getLogger(__name__)
CLOSED_ORBIT_OBSERVABLES = ("x", "y")
_NO_MESSAGE = object()  # no reference message received yet in this iteration
SUPPORTED_OBSERVABLES = tuple(OBSERVABLE_KINDS)


@dataclass
class ClosedOrbitMeasurementData:
    """Internal prepared measurement and the model states it compares."""

    observables: list[Observable]
    pt: float = 0.0
    reference_pt: float = 0.0


@dataclass
class ClosedOrbitSeriesData:
    """Internal worker payload for measurements sharing a machine state."""

    bpm_names: list[str]
    measurements: list[ClosedOrbitMeasurementData]
    #: Model globals (quadrupole strengths, corrector kicks, ...) this series is measured at.
    machine_state: dict[str, float] = field(default_factory=dict)
    #: The state the reference orbit is solved at; a relative plane is the change from it to ``machine_state``.
    reference_state: dict[str, float] = field(default_factory=dict)
    absolute_planes: tuple[str, ...] = ()
    #: Gain layout for :class:`~accelmo.poco.workers.calibrated.CalibratedClosedOrbitWorker`.
    calibration: CalibrationSpec | None = None
    #: Take the reference orbit from the reference worker (see :mod:`accelmo.fitting.shared_reference`)
    #: instead of solving it here. The worker then expects one reference message per iteration.
    shared_reference: bool = False
    #: This worker only solves and publishes the reference orbits, once per iteration, for every series.
    reference_only: bool = False

    @property
    def needs_reference(self) -> bool:
        """Whether any orbit plane of this series is fitted relative to a reference state."""
        observables = self.measurements[0].observables
        return any(
            o.name in CLOSED_ORBIT_OBSERVABLES and o.name not in self.absolute_planes
            for o in observables
        )


@dataclass
class SeriesState:
    """What a worker keeps for one of its series: the prepared data and its alignment to the model BPMs."""

    #: Position of the series in its worker; part of the closed-orbit warm-start key.
    index: int
    data: ClosedOrbitSeriesData
    #: Observables of every measurement of the series, in the order the rows are evaluated.
    observables: list[Observable]
    measured_index: dict[str, int]
    #: 1 for an orbit plane fitted relative to the reference state, 0 for an absolute one.
    subtract: np.ndarray
    #: Model BPM order the alignments below were made for.
    twiss_bpm_order: list[str] | None = None
    #: Per measurement: ``(observables, targets, weights)`` in model BPM order.
    alignment: list[tuple[list[Observable], Any, Any]] | None = None

    def aligned(self) -> tuple[list[str], list[tuple[list[Observable], Any, Any]]]:
        """The model BPM order and per-measurement ``(observables, targets, weights)``."""
        if self.twiss_bpm_order is None or self.alignment is None:
            raise RuntimeError(f"Series {self.index} is not aligned to the model BPMs yet")
        return self.twiss_bpm_order, self.alignment

    def align(self, twiss_names: list[str], worker_id: int) -> None:
        """Align every measurement's targets and weights to the model BPM order."""
        alignment = []
        for measurement in self.data.measurements:
            targets, weights = align_observables(
                measurement.observables, self.measured_index, twiss_names, worker_id=worker_id
            )
            alignment.append((measurement.observables, targets, weights))
        self.alignment = alignment
        self.twiss_bpm_order = twiss_names


def validate_series(data: ClosedOrbitSeriesData) -> None:
    """Check a series' measurements, observables and planes."""
    if not data.measurements:
        raise ValueError("ClosedOrbitSeriesData needs at least one measurement")
    first = data.measurements[0]
    if not first.observables:
        raise ValueError("A closed-orbit measurement needs at least one observable")

    names = tuple(observable.name for observable in first.observables)
    unsupported = [name for name in names if name not in SUPPORTED_OBSERVABLES]
    if unsupported:
        raise ValueError(f"ClosedOrbitWorker supports {SUPPORTED_OBSERVABLES}, got {unsupported}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate closed-orbit observables: {names}")
    for measurement in data.measurements[1:]:
        other = tuple(observable.name for observable in measurement.observables)
        if other != names:
            raise ValueError(
                "Every measurement in a closed-orbit series must use the same "
                f"observable order, got {other} after {names}"
            )

    # Optics are fitted as measured in one twiss, never as a delta against a reference
    # state: such a series takes one measurement and only absolute orbit planes.
    if any(name not in CLOSED_ORBIT_OBSERVABLES for name in names):
        if len(data.measurements) != 1:
            raise ValueError("A closed-twiss series takes exactly one measurement")
        relative = [
            n for n in names if n in CLOSED_ORBIT_OBSERVABLES and n not in data.absolute_planes
        ]
        if relative:
            raise ValueError(f"A closed-twiss series fits orbit planes {relative} only as absolute")
        return

    absolute = tuple(dict.fromkeys(data.absolute_planes))
    unknown = set(absolute) - set(names)
    if unknown:
        raise ValueError(f"Unknown absolute plane(s) {sorted(unknown)}")
    has_relative = any(name not in absolute for name in names)
    has_momentum_signal = any(
        measurement.pt != measurement.reference_pt for measurement in data.measurements
    )
    if (
        data.machine_state == data.reference_state
        and not has_momentum_signal
        and has_relative
        and not absolute
    ):
        raise ValueError(
            "A closed-orbit series needs a machine state that differs from the reference, "
            "a momentum change, or an absolute plane"
        )


class ClosedOrbitWorker(AbstractWorker[list[ClosedOrbitSeriesData], WorkerConfig]):
    """Fit one or more closed-orbit series in one engine.

    Each series is a :class:`ClosedOrbitSeriesData`: its machine state and all its
    measurements. Every measurement keeps its own target, signal momentum and
    reference momentum. The losses, gradients and Hessians of the series are
    summed, so the fit is the one a process per series would give. Repeated model
    states are cached within one series evaluation only.
    """

    #: Step of the forward difference giving the orbit's ``pt`` derivative when ``pt`` is fitted.
    PT_STEP = 1e-6

    def _set_observables(self, observables: list[Observable], pt: float) -> None:
        """The observables the engine evaluates and the momentum it starts at."""
        self.observables = list(observables)
        self.orbit_coords = [obs.name for obs in self.observables if obs.name in ORBIT_COORDS]
        self.optics_names = [obs.name for obs in self.observables if obs.name not in ORBIT_COORDS]
        self.pt = float(pt)

    def _make_engine(self, knobs: dict[str, float]):
        return self.backend.ClosedOrbitEngine(
            self.config,
            knobs,
            optics_names=self.optics_names,
            orbit_coords=self.orbit_coords,
            worker_id=self.worker_id,
            logfile=self.engine_logfile,
        )

    def on_start(self, knobs: dict[str, float]) -> None:
        """Build the engine and note whether ``pt`` is a fitted knob (always the last one)."""
        self.engine = self._make_engine(knobs)
        self.n_knobs = len(self.engine.knob_names)
        self.engine.set_pt(self.pt)
        self.fit_pt = self.engine.fit_pt
        #: Length of the knob vector the fitter sees: the engine knobs, then ``pt`` if fitted.
        self.n_params = self.n_knobs + int(self.fit_pt)

    def _solve(self, observables: list[Observable], *, context: str = ""):
        """``(bpm_names, values, jacobian)`` at the loaded knobs, rows in ``observables`` order.

        ``values`` is ``(n_observables, n_bpms)`` and ``jacobian`` ``(n_observables, n_bpms,
        n_knobs)``; ``None`` when the closed orbit is lost, so the optimiser backtracks.
        """
        solution = self.engine.solve([obs.name for obs in observables])
        if solution is None:
            LOGGER.warning(
                "Worker %s: closed orbit not found%s; flagging step for backtrack",
                self.worker_id,
                context,
            )
            return None
        return solution.names, solution.values, solution.jacobian

    def _reply(
        self, grad: np.ndarray, loss: float, normal_matrix: np.ndarray, extra=None
    ) -> GradReply:
        """Chi² reply with the normal matrix ``JᵀWJ``."""
        return GradReply(self.worker_id, loss, grad, normal_matrix, extra)

    def _failure(self):
        """Reply for a lost closed orbit: NaN loss, so the optimiser backtracks."""
        n = self.n_params
        return self._reply(np.zeros(n), float("nan"), np.zeros((n, n)))

    def prepare_data(self, data: list[ClosedOrbitSeriesData]) -> None:
        """Store the measured series of this worker."""
        if not data:
            raise ValueError("A closed-orbit worker needs at least one series")
        self.series: list[SeriesState] = []
        for index, series in enumerate(data):
            first = series.measurements[0]
            if index == 0:
                self._set_observables(first.observables, first.pt)
            names = [o.name for o in first.observables]
            if [n for n in names if n in ORBIT_COORDS] + [
                n for n in names if n not in ORBIT_COORDS
            ] != (self.orbit_coords + self.optics_names):
                raise ValueError(
                    f"Every series in a worker needs the same observables, got {names}"
                )
            self.series.append(
                SeriesState(
                    index=index,
                    data=series,
                    observables=list(first.observables),
                    measured_index={name: i for i, name in enumerate(series.bpm_names)},
                    subtract=np.array(
                        [float(name not in series.absolute_planes) for name in self.orbit_coords],
                        dtype=float,
                    ),
                )
            )
        self.shared_reference = data[0].shared_reference
        self.reference_only = data[0].reference_only
        #: Model values of every global a series state sets, read on first use.
        self._baseline: dict[str, float] = {}
        self._states: dict[str, dict[str, float]] = {}
        self._reference_message: Any = _NO_MESSAGE
        self._publisher: ReferencePublisher | None = None
        self._reference_reader: ReferenceReader | None = None

    # ------------------------------------------------------------------
    # Machine states
    # ------------------------------------------------------------------

    @contextmanager
    def _in_series(self, series: SeriesState):
        """Make the series' signal and reference states available; on exit restore every global touched.

        The signal state is the model's own values under ``machine_state``, the
        reference state the same under ``reference_state``. The restore is what
        stops one series of a worker leaking its settings into the next.
        """
        machine_state, reference_state = series.data.machine_state, series.data.reference_state
        unread = [name for name in (*machine_state, *reference_state) if name not in self._baseline]
        if unread:
            self._baseline.update(self.engine.get_vars(unread))
        self._states = {
            "signal": {**self._baseline, **machine_state},
            "reference": {**self._baseline, **reference_state},
        }
        try:
            yield
        finally:
            self.engine.set_vars(self._baseline)

    def _enter_state(self, role: str, knob_updates: dict[str, float]) -> None:
        """Put the machine in the series' ``"signal"`` state or in the ``"reference"`` state."""
        del knob_updates
        self.engine.set_vars(self._states[role])

    def _set_co_key(self, series: SeriesState, role: str, pt: float) -> None:
        """Warm-start the next closed-orbit solve from this machine state's own last orbit.

        The state is (series index in this worker, signal/reference role, momentum).
        """
        self.engine.set_warm_start(f"{series.index}:{role}:{pt:.12e}")

    # ------------------------------------------------------------------
    # Solves
    # ------------------------------------------------------------------

    def _model_and_jacobian(self, series: SeriesState, *, context: str = ""):
        """``(values, jacobian)`` of the series' observables at the loaded state, or ``None`` if the orbit is lost."""
        solved = self._solve(series.observables, context=context)
        if solved is None:
            return None
        twiss_names, values, jacobian = solved
        if series.twiss_bpm_order != twiss_names:
            series.align(twiss_names, self.worker_id)
        return values, jacobian

    def _orbit_plain(self, series: SeriesState, *, context: str = ""):
        """Closed-orbit values ``(n_observables, n_bpms)`` without knob derivatives; ``None`` if the orbit is lost.

        The series must be orbit-only and already aligned by a full evaluation.
        """
        solution = self.engine.solve([obs.name for obs in series.observables], jacobian=False)
        if solution is None:
            LOGGER.warning(
                "Worker %s: closed orbit not found%s; flagging step for backtrack",
                self.worker_id,
                context,
            )
            return None
        if solution.names != series.twiss_bpm_order:
            raise RuntimeError("loss-only orbit BPM order differs from the aligned series")
        return solution.values

    def _models(
        self,
        series: SeriesState,
        knob_updates: dict[str, float],
        solve: Callable,
    ) -> Iterator[tuple[int, Any]]:
        """Yield ``(measurement index, model)`` for every measurement of the series; the model is ``None`` once an orbit is lost.

        ``solve(series, context=...)`` evaluates the loaded state. A relative
        plane compares the signal state with the reference state; each ``(role,
        pt)`` state is solved once. Call inside :meth:`_in_series`.
        """
        cache = {}

        def evaluate(role: str, pt: float, key_pt: float | None = None):
            key = (role, pt)
            if key not in cache:
                self._enter_state(role, knob_updates)
                self.engine.set_pt(pt)
                self._set_co_key(series, role, pt if key_pt is None else key_pt)
                cache[key] = solve(series, context=f" ({role}, pt={pt:+.9g})")
            return cache[key]

        for index, measurement in enumerate(series.data.measurements):
            if self.fit_pt:
                signal = self._signal_with_pt(evaluate, measurement, knob_updates["pt"])
            else:
                signal = evaluate("signal", float(measurement.pt))
            if signal is None or not np.any(series.subtract):
                yield index, signal
                if signal is None:
                    return
                continue
            reference = self._reference(evaluate, measurement)
            if reference is None:
                yield index, None
                return
            yield index, self._compare_to_reference(series, signal, reference)

    def _signal_with_pt(self, evaluate, measurement, pt: float):
        """Signal ``(orbit, jacobian)`` at the fitted ``pt``, the Jacobian gaining a last column d(orbit)/d(pt).

        The column is a forward difference of two solves; ``measurement.pt`` only keys the warm start.
        """
        step = self.PT_STEP
        key_pt = float(measurement.pt)
        base = evaluate("signal", pt, key_pt)
        if base is None:
            return None
        shifted = evaluate("signal", pt + step, key_pt + step)
        if shifted is None:
            return None
        column = ((shifted[0] - base[0]) / step)[:, :, None]
        return base[0], np.concatenate([base[1], column], axis=2)

    @staticmethod
    def _compare_to_reference(series: SeriesState, signal, reference):
        """Subtract the reference only in the relative observable planes."""
        if isinstance(signal, np.ndarray):  # values alone (loss-only)
            return signal - reference * series.subtract[:, None]
        model = signal[0] - reference[0] * series.subtract[:, None]
        ref_jacobian = reference[1]
        if (
            ref_jacobian.shape[-1] < signal[1].shape[-1]
        ):  # fitted pt: the reference does not depend on it
            pad = np.zeros((*ref_jacobian.shape[:-1], signal[1].shape[-1] - ref_jacobian.shape[-1]))
            ref_jacobian = np.concatenate([ref_jacobian, pad], axis=2)
        jacobian = signal[1] - ref_jacobian * series.subtract[:, None, None]
        return model, jacobian

    # ------------------------------------------------------------------
    # Shared reference
    # ------------------------------------------------------------------

    @contextmanager
    def _reference_round(self):
        """One iteration of a signal worker: exactly one reference message is consumed, however it ends."""
        self._reference_message = _NO_MESSAGE
        try:
            yield
        finally:
            if self.shared_reference and self._reference_message is _NO_MESSAGE:
                self._reference_message = self.conn.recv()

    def _reference(self, evaluate, measurement):
        """Reference ``(orbit, jacobian)`` of one measurement: solved here, or taken from the reference worker."""
        reference_pt = float(measurement.reference_pt)
        if not self.shared_reference:
            return evaluate("reference", reference_pt)
        if self._reference_message is _NO_MESSAGE:
            self._reference_message = self.conn.recv()
        message = self._reference_message
        if message is None:  # the reference worker lost the closed orbit
            return None
        if self._reference_reader is None:
            self._reference_reader = ReferenceReader()
        block = self._reference_reader.view(message)[message.reference_pts.index(reference_pt)]
        rows = message.rows(self.orbit_coords)
        return block[rows, :, 0], block[rows, :, 1:]

    def _publish_reference(self, knob_updates: dict[str, float]) -> GradReply:
        """Reference worker: solve the reference state of every distinct reference momentum and publish them."""
        if self._publisher is None:
            self._publisher = ReferencePublisher()
        series = self.series[0]
        reference_pts = tuple(
            dict.fromkeys(float(m.reference_pt) for m in series.data.measurements)
        )
        states = []
        n = self.n_params
        with self._in_series(series):
            for reference_pt in reference_pts:
                self._enter_state("reference", knob_updates)
                self.engine.set_pt(reference_pt)
                self._set_co_key(series, "reference", reference_pt)
                state = self._model_and_jacobian(
                    series, context=f" (shared reference, pt={reference_pt:+.9g})"
                )
                if state is None:
                    return self._failure()
                states.append(state)
        published = self._publisher.publish(states, reference_pts, tuple(self.orbit_coords))
        return self._reply(np.zeros(n), 0.0, np.zeros((n, n)), extra=published)

    def close(self) -> None:
        """Close the engine."""
        if self._publisher is not None:
            self._publisher.close()
        if self._reference_reader is not None:
            self._reference_reader.close()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, message: Evaluate) -> GradReply:
        """Sum the loss, gradient and Hessians of every series at ``message.knobs``."""
        self.engine.set_knobs(message.knobs)
        if self.reference_only:
            return self._publish_reference(message.knobs)
        n = self.n_params
        gradient = np.zeros(n)
        loss = 0.0
        normal_matrix = np.zeros((n, n))
        with self._reference_round():
            for series in self.series:
                part = self._evaluate_series(series, message.knobs)
                if part is None:
                    return self._failure()
                gradient += part[0]
                loss += part[1]
                normal_matrix += part[2]
        return self._reply(gradient, loss, normal_matrix)

    def _evaluate_series(self, series: SeriesState, knob_updates: dict[str, float]):
        """Gradient, chi² and normal matrix of one series; ``None`` if a closed orbit is lost."""
        n = self.n_params
        gradient = np.zeros(n)
        loss = 0.0
        normal_matrix = np.zeros((n, n))
        with self._in_series(series):
            for index, model in self._models(series, knob_updates, self._model_and_jacobian):
                if model is None:
                    return None
                part = weighted_least_squares(*model, *series.aligned()[1][index])
                gradient += part[0]
                loss += part[1]
                normal_matrix += part[2]
        return gradient, loss, normal_matrix
