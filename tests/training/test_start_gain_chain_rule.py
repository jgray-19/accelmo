"""chain_starts_through_gains against the exact gradient and Gauss-Newton Hessian of L(k, g, s(g))."""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag

from accelmo.tracking.fitter import chain_starts_through_gains

N_KNOBS, N_GAINS, N_PARTICLES, N_POINTS = 3, 4, 5, 6  # points per particle


def test_chain_rule_matches_the_least_squares_problem_with_starts_as_functions_of_gains() -> None:
    """Linear residuals r = J_t t + J_s s - d (t = knobs, gains), each particle's points depending
    only on its own start, and s(g) = s0 + S g: the chained blocks equal 2 J~^T W r and
    2 J~^T W J~ with J~ = J_t + J_s S E (E puts S in the gain columns)."""
    rng = np.random.default_rng(7)
    n = N_KNOBS + N_GAINS
    gains = slice(N_KNOBS, n)
    m = N_PARTICLES * N_POINTS
    j_t = rng.normal(size=(m, n))
    j_s = block_diag(*[rng.normal(size=(N_POINTS, 4)) for _ in range(N_PARTICLES)])
    w = np.diag(rng.uniform(0.5, 2.0, m))
    d = rng.normal(size=m)
    t = rng.normal(size=n)
    s_jac = rng.normal(size=(N_PARTICLES, 4, N_GAINS))
    s = rng.normal(size=4 * N_PARTICLES) + s_jac.reshape(-1, N_GAINS) @ t[gains]
    r = j_t @ t + j_s @ s - d

    blocks = np.array(
        [2 * (j_s.T @ w @ j_s)[4 * p : 4 * p + 4, 4 * p : 4 * p + 4] for p in range(N_PARTICLES)]
    )
    grad, hessian = chain_starts_through_gains(
        2 * j_t.T @ w @ r,
        2 * j_t.T @ w @ j_t,
        2 * j_s.T @ w @ r,
        2 * j_t.T @ w @ j_s,
        blocks,
        s_jac,
        gains,
    )

    j_total = j_t.copy()
    j_total[:, gains] += j_s @ s_jac.reshape(-1, N_GAINS)
    np.testing.assert_allclose(grad, 2 * j_total.T @ w @ r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hessian, 2 * j_total.T @ w @ j_total, rtol=1e-12, atol=1e-12)
