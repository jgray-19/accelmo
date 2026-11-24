"""Closed-twiss observables and their weighted least-squares blocks.

One closed-orbit engine solve (:class:`~accelmo.backend.protocol.ClosedOrbitEngine`) gives the
*periodic* closed orbit, beta, alpha, phase and dispersion at every BPM and their knob
derivatives; :class:`~accelmo.poco.workers.closed_orbit.ClosedOrbitWorker` fits them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from accelmo.fitting.weights import variance_to_weight

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
    """One measured observable family, aligned to the series' ``bpm_names``.

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


def align_observables(
    observables,
    measured_index: dict[str, int],
    twiss_names: list[str],
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
    weights = []
    for observable in observables:
        indices = order if observable.kind is ObservableKind.POINTWISE else order[:-1]
        targets.append(np.asarray(observable.targets, dtype=float)[indices])
        weights.append(variance_to_weight(np.asarray(observable.variances, dtype=float)[indices]))
    return targets, weights


def _masked_residual(values: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``values - target`` where the weight is positive, else 0 (unmeasured BPMs may be NaN)."""
    return np.where(weight > 0.0, values - target, 0.0)


def weighted_chi2(model: np.ndarray, targets, weights) -> float:
    """Chi² of the inverse-variance weighted residuals of every orbit observable block."""
    return float(
        sum(
            np.sum(weight * _masked_residual(values, target, weight) ** 2)
            for values, target, weight in zip(model, targets, weights, strict=True)
        )
    )


def weighted_least_squares(
    model: np.ndarray, jacobian: np.ndarray, observables, targets, weights
) -> tuple[np.ndarray, float, np.ndarray]:
    """``(gradient, chi², JᵀWJ)`` of the inverse-variance weighted residuals of every observable block."""
    n_knobs = jacobian.shape[-1]
    grad = np.zeros(n_knobs)
    normal_matrix = np.zeros((n_knobs, n_knobs))
    loss = 0.0

    for index, observable in enumerate(observables):
        values, jac = model[index], jacobian[index]
        if observable.kind is ObservableKind.ADVANCE:
            values, jac = _to_advance(values, jac)

        weight = weights[index]
        residual = _masked_residual(values, targets[index], weight)
        grad += 2.0 * (weight * residual) @ jac
        weighted_jac = jac * np.sqrt(weight)[:, None]
        normal_matrix += weighted_jac.T @ weighted_jac  # numpy uses a symmetric rank-k update
        loss += float(np.sum(weight * residual**2))

    return grad, loss, normal_matrix


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
