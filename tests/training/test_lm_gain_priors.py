"""LM gain priors and uncertainties on a linear problem whose data leave a common gain scale free."""

from __future__ import annotations

import numpy as np

from accelmo.fitting.priors import quadratic_prior
from accelmo.fitting.uncertainty import sandwich_uncertainties

N_GAINS, N_POINTS, SIGMA = 8, 40, 0.05


def _null_common_mode_problem(rng):
    """Residuals see only gain differences: every row of J sums to zero."""
    j = rng.normal(size=(N_POINTS, N_GAINS))
    j -= j.mean(axis=1, keepdims=True)
    w = np.diag(rng.uniform(0.5, 2.0, N_POINTS)) * 1e8
    truth = rng.normal(0.0, 0.03, N_GAINS) + 0.04
    return j, w, j @ truth, truth


def test_zero_mean_group_fixes_the_free_common_gain_at_zero() -> None:
    rng = np.random.default_rng(3)
    j, w, d, truth = _null_common_mode_problem(rng)
    group = np.arange(N_GAINS)
    prior = quadratic_prior(np.full(N_GAINS, 1.0 / SIGMA**2), [group], 1e10)
    solved = np.linalg.solve(j.T @ w @ j + prior, j.T @ w @ d)
    assert abs(solved.mean()) < 1e-10
    np.testing.assert_allclose(solved - solved.mean(), truth - truth.mean(), atol=1e-5)


def test_chained_sandwich_is_the_correlated_noise_covariance() -> None:
    """Knobs + gains; starts cancel the common gain; correlated noise."""
    rng = np.random.default_rng(4)
    n_knobs = 3
    j_gain, w, _, _ = _null_common_mode_problem(rng)
    j_knob = rng.normal(size=(N_POINTS, n_knobs))
    j_chained = np.hstack([j_knob, j_gain])  # starts follow the gains: it is not
    root = rng.normal(size=(N_POINTS, N_POINTS)) * 1e-5
    sigma = root @ root.T + np.diag(np.full(N_POINTS, 1e-9))
    ridge = np.r_[np.zeros(n_knobs), np.full(N_GAINS, 1.0 / SIGMA**2)]
    prior = quadratic_prior(ridge, [np.arange(n_knobs, n_knobs + N_GAINS)], 1e10)
    hessian = 2.0 * (j_chained.T @ w @ j_chained + prior)  # the LM's chi-square Hessian
    noise = j_chained.T @ w @ sigma @ w @ j_chained
    got = sandwich_uncertainties(0.5 * hessian, noise)
    inverse = np.linalg.inv(j_chained.T @ w @ j_chained + prior)
    want = inverse @ (j_chained.T @ w @ sigma @ w @ j_chained) @ inverse
    np.testing.assert_allclose(got, np.sqrt(np.diag(want)), rtol=1e-8)
    white = np.sqrt(np.diag(inverse))  # Hessian answer: right only for sigma = W⁻¹
    assert np.max(np.abs(got / white - 1.0)) > 0.1
