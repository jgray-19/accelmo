"""The closed-twiss chi², gradient and normal matrix match the direct weighted formulas."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from accelmo.poco.workers.closed_twiss import ObservableKind, weighted_least_squares


def _reference(model, jacobian, targets, weights):
    n_knobs = jacobian.shape[-1]
    grad, normal, loss = np.zeros(n_knobs), np.zeros((n_knobs,) * 2), 0.0
    for i in range(len(model)):
        residual = np.where(weights[i] > 0.0, model[i] - targets[i], 0.0)
        grad += 2.0 * (weights[i] * residual) @ jacobian[i]
        normal += (jacobian[i].T * weights[i]) @ jacobian[i]
        loss += float(np.sum(weights[i] * residual**2))
    return grad, loss, normal


def test_matches_direct_formulas() -> None:
    rng = np.random.default_rng(1)
    n_obs, n_bpms, n_knobs = 2, 60, 25
    observables = [SimpleNamespace(kind=ObservableKind.POINTWISE) for _ in range(n_obs)]
    model, jacobian = rng.normal(size=(n_obs, n_bpms)), rng.normal(size=(n_obs, n_bpms, n_knobs))
    targets = list(rng.normal(size=(n_obs, n_bpms)))
    weights = rng.uniform(1e12, 1e16, size=(n_obs, n_bpms))
    weights[:, ::7] = 0.0  # invalid BPMs carry zero weight
    got = weighted_least_squares(model, jacobian, observables, targets, list(weights))
    for g, w in zip(got, _reference(model, jacobian, targets, weights), strict=True):
        np.testing.assert_allclose(g, w, rtol=1e-10)
