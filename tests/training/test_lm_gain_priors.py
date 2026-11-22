"""LM gain priors and uncertainties on a linear problem whose data leave a common gain scale free."""

from __future__ import annotations

import numpy as np

from accelmo.tracking.fitter import chained_sandwich_uncertainties, lm_prior_matrix

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
    prior = lm_prior_matrix(np.full(N_GAINS, 1.0 / SIGMA**2), [group], 1e10)
    solved = np.linalg.solve(j.T @ w @ j + prior, j.T @ w @ d)
    assert abs(solved.mean()) < 1e-10
    np.testing.assert_allclose(solved - solved.mean(), truth - truth.mean(), atol=1e-5)


def test_chained_sandwich_is_the_correlated_noise_covariance() -> None:
    """Knobs + gains; starts cancel the common gain; correlated noise; A, B in other weight units."""
    rng = np.random.default_rng(4)
    n_knobs = 3
    j_gain, w, _, _ = _null_common_mode_problem(rng)
    j_knob = rng.normal(size=(N_POINTS, n_knobs))
    j_fixed = np.hstack([j_knob, j_gain + 0.3])  # starts held fixed: the common gain is seen
    j_chained = np.hstack([j_knob, j_gain])  # starts follow the gains: it is not
    root = rng.normal(size=(N_POINTS, N_POINTS)) * 1e-5
    sigma = root @ root.T + np.diag(np.full(N_POINTS, 1e-9))
    ridge = np.r_[np.zeros(n_knobs), np.full(N_GAINS, 1.0 / SIGMA**2)]
    prior = lm_prior_matrix(ridge, [np.arange(n_knobs, n_knobs + N_GAINS)], 1e10)
    loss_per_chi2, weight_scale = 3.7e-6, 42.0
    hessian = 2.0 * loss_per_chi2 * (j_chained.T @ w @ j_chained + prior)
    normal = weight_scale * j_fixed.T @ w @ j_fixed
    noise = weight_scale**2 * j_chained.T @ w @ sigma @ w @ j_chained
    got = chained_sandwich_uncertainties(
        hessian, 2.0 * loss_per_chi2 * prior, normal, noise, np.arange(n_knobs)
    )
    inverse = np.linalg.inv(j_chained.T @ w @ j_chained + prior)
    want = inverse @ (j_chained.T @ w @ sigma @ w @ j_chained) @ inverse
    np.testing.assert_allclose(got, np.sqrt(np.diag(want)), rtol=1e-8)
    white = np.sqrt(np.diag(inverse))  # Hessian answer: right only for sigma = W⁻¹
    assert np.max(np.abs(got / white - 1.0)) > 0.1
