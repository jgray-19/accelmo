import numpy as np
import pytest

from accelmo.optimisers.lbfgs import LBFGSOptimiser


class TestLBFGSOptimiser:
    def test_initialization(self):
        optim = LBFGSOptimiser(
            history_size=10,
            eps=1e-12,
            weight_decay=0.0,
        )
        assert optim.history_size == 10
        assert optim.eps == 1e-12
        assert optim.weight_decay == 0.0
        assert optim.grad_scale is None
        assert optim.t == 0
        assert optim.prev_params is None
        assert optim.prev_grads is None
        assert len(optim.S) == 0
        assert len(optim.Y) == 0
        assert len(optim.RHO) == 0

    def test_step_first_step(self):
        optim = LBFGSOptimiser(max_step_norm=None)
        params = np.array([1.0, 2.0])
        grads = np.array([2.0, 4.0])
        lr = 0.01

        new_params = optim.step(params, grads, lr)

        # First step, no history: direction -g, normalised by the first gradient's norm
        expected_update = -lr * grads / np.linalg.norm(grads)
        expected_new_params = params + expected_update
        assert np.allclose(new_params, expected_new_params)
        assert optim.t == 1
        assert np.array_equal(optim.prev_params, params)
        assert np.allclose(optim.prev_grads, grads / np.linalg.norm(grads))

    def test_step_with_history(self):
        optim = LBFGSOptimiser(history_size=5, max_step_norm=None)
        params = np.array([1.0])
        grads = np.array([2.0])
        lr = 0.1

        # First step
        params = optim.step(params, grads, lr)

        # Second step
        grads2 = np.array([1.0])  # new grads
        new_params = optim.step(params, grads2, lr)

        assert new_params is not None

        assert optim.t == 2
        assert len(optim.S) == 1
        assert len(optim.Y) == 1
        assert len(optim.RHO) == 1

    def test_weight_decay(self):
        optim = LBFGSOptimiser(weight_decay=0.1, max_step_norm=None)
        params = np.array([1.0])
        grads = np.array([2.0])
        lr = 0.01

        new_params = optim.step(params, grads, lr)

        # Effective grads = grads + weight_decay * params
        effective_grads = grads + 0.1 * params
        expected_update = -lr * effective_grads / np.linalg.norm(effective_grads)
        expected_new_params = params + expected_update
        assert np.allclose(new_params, expected_new_params)

    def test_convergence_quadratic(self):
        optim = LBFGSOptimiser(history_size=10, max_step_norm=None)
        params = np.array([10.0])
        lr = 1.0  # Use lr=1.0 to see pure quasi-Newton effect

        # f(x) = 0.5 * x^2, so grad = x, Hessian = 1, inverse Hessian = 1
        # Newton step: x_new = x - H^-1 * grad = x - 1 * x = 0

        grads = params.copy()  # grad = x for f=0.5*x^2
        params = optim.step(params, grads, lr)

        # First step still behaves like gradient descent before any curvature history exists.
        assert abs(params[0]) < 10.0

        # Second step can exploit the stored curvature pair and should jump close to the optimum.
        grads = params.copy()
        params = optim.step(params, grads, lr)
        assert abs(params[0]) < 0.1

    def test_history_limit(self):
        optim = LBFGSOptimiser(history_size=2, max_step_norm=None)
        params = np.array([-1.0])  # start negative

        for i in range(5):
            grads = 2 * params  # gradient
            params = optim.step(params, grads, 0.01)

        # History should be limited to 2
        assert len(optim.S) == 2
        assert len(optim.Y) == 2
        assert len(optim.RHO) == 2

    def test_zero_gradients(self):
        optim = LBFGSOptimiser()
        params = np.array([1.0, 2.0])
        grads = np.array([0.0, 0.0])
        lr = 0.01

        new_params = optim.step(params, grads, lr)

        # With zero grads, no change
        assert np.allclose(new_params, params)

    def test_default_clipping_limits_first_step_norm(self):
        optim = LBFGSOptimiser()
        params = np.array([1.0, 2.0])
        grads = np.array([2.0, 4.0])
        lr = 0.01

        new_params = optim.step(params, grads, lr)
        step = new_params - params

        assert np.isclose(np.linalg.norm(step), lr)

    def test_state_roundtrip_restores_update_trajectory(self):
        params = np.array([1.5, -0.5, 2.0])
        lr = 0.05

        optim = LBFGSOptimiser(history_size=4, eps=1e-12, weight_decay=0.03)
        grads1 = np.array([0.2, -0.4, 0.1])
        params_after_first = optim.step(params, grads1, lr)

        grads2 = np.array([-0.1, 0.3, -0.2])
        params_after_second = optim.step(params_after_first, grads2, lr)

        state = optim.state_dict()
        restored = LBFGSOptimiser()
        restored.load_state_dict(state)

        grads3 = np.array([0.15, -0.2, 0.05])
        expected_next = optim.step(params_after_second, grads3, lr)
        restored_next = restored.step(params_after_second, grads3, lr)

        assert np.allclose(restored_next, expected_next)


LOSS_SCALES = (1.0, 1e-9)  # 1e-9: the PSB kicker tracking loss at the nominal model


def _quadratic_run(scale: float, steps: int = 300, lr: float = 0.5) -> float:
    """Minimise scale * 0.5 (x - x*)^T A (x - x*) with production settings; return |x - x*|."""
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    hess = scale * q @ np.diag(np.geomspace(1.0, 1e2, 8)) @ q.T
    target = rng.normal(scale=0.05, size=8)
    optim = LBFGSOptimiser(history_size=10, eps=1e-12, max_step_norm=1.0, powell_damping=0.2)
    x = np.zeros(8)
    for _ in range(steps):
        x = optim.step(x, hess @ (x - target), lr)
    return float(np.linalg.norm(x - target) / np.linalg.norm(target))


@pytest.mark.parametrize("scale", LOSS_SCALES)
def test_converges_independent_of_loss_scale(scale):
    """A quasi-Newton step is invariant to a constant factor on the loss."""
    assert _quadratic_run(scale) < 1e-8
