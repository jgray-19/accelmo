"""Gain uncertainties are the marginal sigmas of the full (u, b) normal matrix."""

from __future__ import annotations

import numpy as np

from accelmo.poco.calibration import CalibrationBlocks, gain_covariance


def test_marginal_sigmas_match_the_dense_inverse() -> None:
    n_q, n_g, n_b = 4, 2, 6
    n_u = n_q + n_g
    rng = np.random.default_rng(0)
    d = rng.uniform(1.0, 3.0, n_b)
    h_ub = rng.normal(size=(n_u, n_b))
    root = rng.normal(size=(n_u + 3, n_u))
    h_uu = h_ub @ np.diag(1.0 / d) @ h_ub.T + root.T @ root  # positive definite, diagonal b block
    physical = np.block([[h_uu, h_ub], [h_ub.T, np.diag(d)]])

    blocks = CalibrationBlocks(n_q, n_g, n_b)  # chi² Hessian: twice the normal matrix
    blocks.hess_uu[:], blocks.hess_ub[:], blocks.hess_bb[:] = 2 * h_uu, 2 * h_ub, 2 * d

    normal_u, sigma_u, sigma_b = gain_covariance(blocks)

    dense = np.sqrt(np.diag(np.linalg.inv(physical)))
    np.testing.assert_allclose(sigma_u, dense[:n_u], rtol=1e-9)
    np.testing.assert_allclose(sigma_b, dense[n_u:], rtol=1e-9)
    assert normal_u.shape == (n_u, n_u)
