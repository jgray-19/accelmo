"""Closed-twiss worker: fit knobs to a measured periodic optics solution.

One closed-orbit engine solve per iteration (:class:`~accelmo.backend.protocol.ClosedOrbitEngine`)
gives the *periodic* closed orbit, beta, alpha, phase and dispersion at every BPM and
their knob derivatives.

Nothing is seeded from the measurement. There is no starting point to propagate
from, so no measurement noise enters as an initial condition and no error can be
absorbed by an assumed anchor; every residual is attributable to the magnets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from accelmo.fitting.protocol import GradReply
from accelmo.fitting.weights import variance_to_weight
from accelmo.fitting.worker import AbstractWorker

if TYPE_CHECKING:
    from accelmo.fitting.protocol import Evaluate

LOGGER = logging.getLogger(__name__)

#: Observables that are phase-space coordinates rather than optical functions.
ORBIT_COORDS = ("x", "px", "y", "py")


class ObservableKind(StrEnum):
    """How a per-BPM observable is turned into a residual.

    ``POINTWISE``
        The model value at each BPM is compared directly against the measurement:
        closed orbit, beta, alpha, dispersion.
    ``ADVANCE``
        Consecutive BPM values are differenced first, so a cumulative model
        quantity is compared against a measured *advance*. Used for phase, where
        only the BPM-to-BPM advance is measurable and the absolute value carries
        an arbitrary origin.
    """

    POINTWISE = "pointwise"
    ADVANCE = "advance"


#: Observables the MAD-NG side knows how to evaluate on the closed twiss, mapped
#: to how their residual is formed. Names are ``gphys.optfun`` function names,
#: except the closed-orbit coordinates which are read off the map's constant part.
#:
#: ``betx``/``bety``/``alfx``/``alfy`` are the coupled Edwards-Teng/physical-plane
#: projections, requested via ``coupling=true`` on the ``twiss{}`` call in
#: ``run_closed_twiss_init.mad``. They are computed live as part of that twiss
#: (the beam is attached throughout), not read back off a bare saved map
#: afterward -- only the orbit coordinates (``x``/``y``/``px``/``py``, via
#: ``ORBIT_COORDS``) take that route, for their knob Jacobian. Never use the
#: uncoupled Ripken ``beta11``/``beta22``/``alfa11``/``alfa22`` columns here:
#: under real coupling their mode/plane identity is not guaranteed to track x/y,
#: so they are not comparable to an omc3 ``BETX``/``ALFX`` measurement by name
#: coincidence alone.
#:
#: ``mu1``/``mu2`` stay mode-indexed: MAD-NG has no ``mux``/``muy`` optical
#: function at all (confirmed against its own name list -- only ``mu``/``dmu``
#: with a mode-index suffix exist), so there is no physical-plane phase-advance
#: column to request in the first place.
#:
#: ``dx``/``dy`` are ``d(x)/d(pt)`` and ``d(y)/d(pt)`` - the MAD-X ``DX`` convention.
#: ``dpx``/``dpy`` are available but should normally be left out when fitting
#: omc3 output: omc3 derives its ``DPY`` from ``DY`` through the *model* transfer
#: matrix, so including both double-counts one measurement.
OBSERVABLE_KINDS: dict[str, ObservableKind] = {
    "x": ObservableKind.POINTWISE,
    "y": ObservableKind.POINTWISE,
    "px": ObservableKind.POINTWISE,
    "py": ObservableKind.POINTWISE,
    "betx": ObservableKind.POINTWISE,
    "bety": ObservableKind.POINTWISE,
    "alfx": ObservableKind.POINTWISE,
    "alfy": ObservableKind.POINTWISE,
    "dx": ObservableKind.POINTWISE,
    "dy": ObservableKind.POINTWISE,
    "dpx": ObservableKind.POINTWISE,
    "dpy": ObservableKind.POINTWISE,
    "mu1": ObservableKind.ADVANCE,
    "mu2": ObservableKind.ADVANCE,
}


@dataclass
class Observable:
    """One measured observable family, aligned to ``ClosedTwissData.bpm_names``.

    ``targets`` and ``variances`` have one entry per BPM for a ``POINTWISE``
    observable and one per *interval* (``n_bpms - 1``) for an ``ADVANCE`` one.
    A non-finite or non-positive variance drops that point from the fit, which is
    how partially-measured planes are handled without special-casing.
    """

    name: str
    targets: np.ndarray
    variances: np.ndarray

    @property
    def kind(self) -> ObservableKind:
        """Residual form for this observable."""
        try:
            return OBSERVABLE_KINDS[self.name]
        except KeyError:
            raise ValueError(
                f"Unknown observable '{self.name}'. Known: {sorted(OBSERVABLE_KINDS)}"
            ) from None


@dataclass
class ClosedTwissData:
    """Reference closed-twiss measurements for one momentum, for one worker.

    Arrays are ordered to match the model BPM ordering (the order the sequence's
    monitors are observed by twiss). ``bpm_names`` records that order so the
    worker can align the twiss output to these comparisons by name.

    ``pt`` is the known MAD-NG momentum coordinate of this measurement.
    It is a fixed input to twiss (``x0map.pt``), never an optimisation knob, so
    both the off-momentum bend response and the dispersive orbit come from the
    physics. Fitting several ``pt`` values jointly makes the per-magnet
    Jacobians independent and lifts the single-measurement rank deficiency.

    ``weight_scale`` and ``total_points`` are the *global* loss normalisation and
    must be identical across every worker in a fit. Each worker divides its
    inverse-variance weights by ``weight_scale`` and its loss/gradient/Hessian by
    ``total_points``. If derived per worker (from its own largest weight and point count), the
    optimiser would minimise ``sum_w (1/(max_w * N_w)) * chi2_w`` instead of the
    pooled ``sum_w chi2_w``. The most precisely measured momentum would then be
    down-weighted, and the reported 1-sigma (built from the un-normalised
    ``JᵀWJ``) would not describe the estimator that was minimised. They are
    computed once, over every worker's observables, by
    :func:`create_worker_payloads`.
    """

    bpm_names: list[str]
    observables: list[Observable]
    pt: float = 0.0
    weight_scale: float = 1.0
    total_points: int = 1


class ClosedTwissWorker(AbstractWorker[ClosedTwissData]):
    """Worker that fits knobs to a measured closed-twiss solution."""

    def prepare_data(self, data: ClosedTwissData) -> None:
        """Store the measured observables."""
        if not data.observables:
            raise ValueError("ClosedTwissData carries no observables to fit")

        LOGGER.debug(
            "Worker %s: closed-twiss data for %d BPMs, observables %s",
            self.worker_id,
            len(data.bpm_names),
            [obs.name for obs in data.observables],
        )

        self.observables = list(data.observables)
        self.orbit_coords = [obs.name for obs in self.observables if obs.name in ORBIT_COORDS]
        self.optics_names = [obs.name for obs in self.observables if obs.name not in ORBIT_COORDS]
        self._measured_index = {name: i for i, name in enumerate(data.bpm_names)}
        # Known momentum offset of this measurement; pinned on the closed orbit (not a knob).
        self.pt = float(data.pt)
        # Filled on the first compute once the twiss BPM ordering is known.
        self._twiss_bpm_order: list[str] | None = None
        # Global loss normalisation, identical for every worker in the fit. See
        # ``ClosedTwissData`` for why it must not be derived per worker.
        self.weight_scale = float(data.weight_scale)
        if not np.isfinite(self.weight_scale) or self.weight_scale <= 0.0:
            raise ValueError(f"Worker {self.worker_id}: weight_scale must be finite and positive")
        self.normalisation_points = max(1, int(data.total_points))

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
        """Build the engine; ``pt`` is never a knob here, it is pinned on the closed orbit."""
        self.engine = self._make_engine(knobs)
        self.n_knobs = len(self.engine.knob_names)
        self.engine.set_pt(self.pt)

    def _solve(self, observables: list[Observable], *, context: str = ""):
        """Solve the closed orbit (and twiss) at the loaded knobs.

        Returns ``(bpm_names, values, jacobian)`` with ``values`` of shape
        ``(n_observables, n_bpms)`` and ``jacobian`` of shape ``(n_observables,
        n_bpms, n_knobs)``, rows in the order of ``observables``; ``None`` when the
        closed orbit is lost (an unstable trial point), so the optimiser backtracks.
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
        self,
        grad: np.ndarray,
        loss: float,
        hessian: np.ndarray,
        normal_matrix: np.ndarray,
        extra=None,
    ) -> GradReply:
        """Normalise by the global point count; the physical ``normal`` stays un-normalised.

        Summed across workers, the inverse of ``normal`` is the covariance in real units.
        """
        points = self.normalisation_points
        return GradReply(
            self.worker_id, loss / points, grad / points, hessian / points, normal_matrix, extra
        )

    def _failure(self) -> GradReply:
        """Reply for a lost closed orbit: NaN loss, so the optimiser backtracks."""
        n = self.n_knobs
        return self._reply(np.zeros(n), float("nan"), np.zeros((n, n)), np.zeros((n, n)))

    def evaluate(self, message: Evaluate) -> GradReply:
        """Closed twiss at ``message.knobs`` and its weighted least-squares derivatives."""
        self.engine.set_knobs(message.knobs)
        solved = self._solve(self.observables)
        if solved is None:
            return self._failure()
        twiss_names, model, jacobian = solved
        if self._twiss_bpm_order != twiss_names:
            self.targets, self.raw_weights, self.weights = align_observables(
                self.observables,
                self._measured_index,
                twiss_names,
                self.weight_scale,
                worker_id=self.worker_id,
            )
            self._twiss_bpm_order = twiss_names
        # Every observable family contributes an independent block to the same
        # ``2 JᵀWJ`` normal equations; the inverse-variance weights make families in
        # different units commensurable, so no per-family scaling is applied.
        return self._reply(
            *weighted_loss_gradient_hessian(
                model,
                jacobian,
                self.observables,
                self.targets,
                self.weights,
                self.raw_weights,
                self.weight_scale,
            )
        )


def align_observables(
    observables,
    measured_index: dict[str, int],
    twiss_names: list[str],
    weight_scale: float,
    *,
    worker_id: int,
):
    """Align observable targets and weights to the model BPM ordering."""
    missing = [name for name in twiss_names if name not in measured_index]
    if missing:
        raise RuntimeError(
            f"Worker {worker_id}: {len(missing)} observed BPMs have no measurement, "
            f"e.g. {missing[:5]}"
        )
    order = np.array([measured_index[name] for name in twiss_names])
    if not np.all(np.diff(order) == 1):
        raise RuntimeError(
            f"Worker {worker_id}: the observed BPMs are not a contiguous run of "
            "the measured ordering; phase advances cannot be aligned by interval."
        )

    targets = []
    raw_weights = []
    for observable in observables:
        indices = order if observable.kind is ObservableKind.POINTWISE else order[:-1]
        targets.append(np.asarray(observable.targets, dtype=float)[indices])
        raw_weights.append(
            variance_to_weight(np.asarray(observable.variances, dtype=float)[indices])
        )
    return targets, raw_weights, [weight / weight_scale for weight in raw_weights]


def weighted_loss_gradient_hessian(
    model: np.ndarray,
    jacobian: np.ndarray,
    observables,
    targets,
    weights,
    raw_weights,
    weight_scale: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Evaluate weighted least squares for explicitly supplied observable blocks.

    ``weights`` are ``raw_weights / weight_scale`` (see :func:`align_observables`), so
    the Gauss-Newton Hessian ``2 JᵀWJ`` is the physical normal matrix ``Jᵀ W_raw J``
    times ``2 / weight_scale``. Only the latter is formed, once per observable, as a
    symmetric ``Aᵀ A`` product of the sqrt-weighted Jacobian.
    """
    n_knobs = jacobian.shape[-1]
    grad = np.zeros(n_knobs)
    normal_matrix = np.zeros((n_knobs, n_knobs))
    loss = 0.0

    for index, observable in enumerate(observables):
        values, jac = model[index], jacobian[index]
        if observable.kind is ObservableKind.ADVANCE:
            values, jac = _to_advance(values, jac)

        weight, raw_weight = weights[index], raw_weights[index]
        if not np.allclose(weight * weight_scale, raw_weight, rtol=1e-9, atol=0.0):
            raise ValueError("weights must equal raw_weights / weight_scale")
        residual = np.where(weight > 0.0, values - targets[index], 0.0)
        grad += 2.0 * (weight * residual) @ jac
        weighted_jac = jac * np.sqrt(raw_weight)[:, None]
        normal_matrix += weighted_jac.T @ weighted_jac  # numpy uses a symmetric rank-k update
        loss += float(np.sum(weight * residual**2))

    return grad, loss, (2.0 / weight_scale) * normal_matrix, normal_matrix


def _to_advance(values: np.ndarray, jacobian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a cumulative per-BPM quantity into BPM-to-BPM advances.

    ``gphys`` returns the phase from ``atan2``, so it is wrapped into a unit
    interval rather than accumulated around the ring. Reducing the consecutive
    difference modulo 1 undoes that, which is unambiguous as long as no BPM pair
    is separated by a full unit of phase - true of every real BPM layout, and the
    same assumption omc3 makes when it reports an advance in ``[0, 1)``.

    The modulo is locally the identity, so the Jacobian of the advance is the
    difference of consecutive Jacobian rows.
    """
    return np.mod(np.diff(values), 1.0), np.diff(jacobian, axis=0)
