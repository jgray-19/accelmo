import numpy as np
import pytest

from accelmo.optimisers.levenberg_marquardt import (
    LevenbergMarquardtConfig,
    LevenbergMarquardtOptimiser,
)


def test_lm_solves_quadratic_newton_step() -> None:
    optim = LevenbergMarquardtOptimiser(
        LevenbergMarquardtConfig(),
        initial_params=np.array([0.0]),
    )

    update = optim.update(
        params=np.array([0.0]),
        loss=4.0,
        grad=np.array([-4.0]),
        hessian=np.array([[2.0]]),
    )

    assert update.accepted
    assert not update.converged
    assert np.allclose(update.next_params, [2.0])
    assert np.allclose(optim.best.value, [0.0])


def test_lm_rejects_worse_loss_and_increases_damping() -> None:
    optim = LevenbergMarquardtOptimiser(
        LevenbergMarquardtConfig(lambda_floor=1e-4),
        initial_params=np.array([0.0]),
    )
    hessian = np.array([[1.0]])

    first = optim.update(np.array([0.0]), 1.0, np.array([1.0]), hessian)
    rejected = optim.update(np.array([10.0]), 2.0, np.array([1.0]), hessian)

    assert first.accepted
    assert not rejected.accepted
    assert rejected.reason == "rejected"
    assert rejected.damping == pytest.approx(1e-4)
    # The retry is measured from the best point, not from the rejected one.
    assert np.allclose(optim.best.value, [0.0])
    assert not np.allclose(rejected.next_params, optim.best.value)


def test_lm_retries_from_best_instead_of_re_evaluating_it() -> None:
    """A rejected step must come back as a *new* point to evaluate.

    Returning ``best_params`` unchanged deadlocks the solve: the caller
    re-evaluates a point already stored as best, gets exactly ``best_loss``,
    and is rejected again for every remaining iteration.
    """
    optim = LevenbergMarquardtOptimiser(
        LevenbergMarquardtConfig(lambda_floor=1.0, lambda_increase=10.0),
        initial_params=np.array([0.0]),
    )
    hessian = np.array([[1.0]])
    grad = np.array([2.0])

    optim.update(np.array([0.0]), 1.0, grad, hessian)
    first = optim.update(np.array([100.0]), 5.0, grad, hessian)
    second = optim.update(first.next_params, 4.0, grad, hessian)

    # Undamped the step would be -grad/H = -2. Each rejection damps it further.
    assert first.next_params[0] == pytest.approx(-2.0 / (1.0 + 1.0))
    assert second.next_params[0] == pytest.approx(-2.0 / (1.0 + 10.0))
    assert abs(second.next_params[0]) < abs(first.next_params[0])


def test_lm_recovers_from_an_overshooting_gauss_newton_step() -> None:
    """End-to-end: an overshoot must be backtracked, not fatal.

    ``f(x) = (x^2 - 1)^2`` with residual ``r = x^2 - 1``. From x=0.1 the
    Gauss-Newton step is ~+5.0, which lands far uphill and is rejected. Only a
    working retry path recovers the root at |x| = 1.
    """
    optim = LevenbergMarquardtOptimiser(
        LevenbergMarquardtConfig(max_iterations=200, lambda_floor=1e-6),
        initial_params=np.array([0.1]),
    )
    params = np.array([0.1])
    rejections = 0
    for _ in range(200):
        residual = params[0] ** 2 - 1.0
        jac = 2.0 * params[0]
        update = optim.update(
            params, residual**2, np.array([2.0 * jac * residual]), np.array([[2.0 * jac**2]])
        )
        rejections += not update.accepted
        params = update.next_params
        if update.converged:
            break

    assert rejections > 0, "the overshoot should have been rejected at least once"
    assert abs(optim.best.value[0]) == pytest.approx(1.0, abs=1e-6)


def test_lm_stops_when_damping_is_exhausted() -> None:
    optim = LevenbergMarquardtOptimiser(
        LevenbergMarquardtConfig(lambda_floor=1e-4, lambda_max=1e2),
        initial_params=np.array([0.0]),
    )
    hessian = np.array([[1.0]])
    optim.update(np.array([0.0]), 1.0, np.array([1.0]), hessian)

    for _ in range(20):
        update = optim.update(np.array([10.0]), 2.0, np.array([1.0]), hessian)
        if update.converged:
            break

    assert update.converged
    assert update.reason == "damping_exhausted"
    assert np.allclose(update.next_params, optim.best.value)


def test_lm_reports_no_progress_when_nothing_was_ever_accepted() -> None:
    """A first evaluation that fails leaves no curvature to retry from."""
    optim = LevenbergMarquardtOptimiser(initial_params=np.array([0.0]))

    update = optim.update(
        np.array([0.0]), float("nan"), np.array([np.nan]), np.array([[np.nan]]), failed=True
    )

    assert not update.accepted
    assert update.converged, "the caller must stop rather than spin on the same point"
    assert update.reason == "no_progress"


def test_lm_state_roundtrip() -> None:
    optim = LevenbergMarquardtOptimiser(initial_params=np.array([1.0, 2.0]))
    hessian = np.eye(2)
    optim.update(np.array([1.0, 2.0]), 3.0, np.array([0.5, -0.5]), hessian)

    restored = LevenbergMarquardtOptimiser()
    restored.load_state_dict(optim.state_dict())

    assert restored.damping == optim.damping
    assert restored.best.loss == optim.best.loss
    assert np.allclose(restored.best.value, optim.best.value)
    assert np.allclose(restored.best_hessian, optim.best_hessian)
    # Without the gradient at the best point a resumed run cannot backtrack.
    assert np.allclose(restored.best_grad, optim.best_grad)


def test_refresh_after_every_accepted_step_reaches_the_fixed_point() -> None:
    """Loss (x - m)^2 with m rebuilt from the accepted x as m = x / 2 + 1: x -> m -> 2."""
    from accelmo.fitting.lm_loop import LMPoint, LMProblem, run_levenberg_marquardt

    class MovingMarker(LMProblem):
        def __init__(self) -> None:
            self.marker = 1.0
            self.refreshed_at: list[float] = []

        def evaluate(self, params: np.ndarray, iteration: int) -> LMPoint:
            residual = params[0] - self.marker
            return LMPoint(residual**2, np.array([2.0 * residual]), np.array([[2.0]]))

        def refresh(self, params: np.ndarray) -> bool:
            self.refreshed_at.append(float(params[0]))
            new = params[0] / 2.0 + 1.0
            moved = abs(new - self.marker) > 1e-12
            self.marker = new
            return moved

    problem = MovingMarker()
    config = LevenbergMarquardtConfig(max_iterations=200)
    optimiser, diagnostics = run_levenberg_marquardt(np.array([0.0]), problem, config)
    assert optimiser.best.value[0] == pytest.approx(2.0, abs=1e-11)
    # one refresh per accepted point, never twice at the same point
    assert len(problem.refreshed_at) == len(set(problem.refreshed_at))
    assert diagnostics.iterations < 200


def test_lm_stops_after_patience_stalled_steps_on_a_tiny_loss() -> None:
    """Accepted steps that each lower a 1e-7 loss by 1e-8 of itself: stop on the 5th, not before."""
    optim = LevenbergMarquardtOptimiser(
        LevenbergMarquardtConfig(
            loss_relative_tolerance=1e-6,
            loss_relative_floor=0.0,
            loss_stall_patience=5,
        ),
        initial_params=np.array([0.0]),
    )
    converged = []
    for k in range(7):
        update = optim.update(
            np.array([float(k)]), 1e-7 * (1.0 - 1e-8 * k), np.array([1e-3]), np.array([[2.0]])
        )
        converged.append(update.converged)
    assert (
        converged == [False] * 5 + [True] * 2
    )  # 1st update has no previous loss; stalls 1..5 end at k=5


def test_lm_stall_count_resets_on_a_real_improvement() -> None:
    optim = LevenbergMarquardtOptimiser(
        LevenbergMarquardtConfig(
            loss_relative_tolerance=1e-6,
            loss_relative_floor=0.0,
            loss_stall_patience=3,
        ),
        initial_params=np.array([0.0]),
    )
    losses = [
        1e-7,
        1e-7 * (1 - 1e-9),
        1e-7 * (1 - 2e-9),
        0.5e-7,
        0.5e-7 * (1 - 1e-9),
        0.5e-7 * (1 - 2e-9),
    ]
    flags = [
        optim.update(np.array([float(k)]), x, np.array([1e-3]), np.array([[2.0]])).converged
        for k, x in enumerate(losses)
    ]
    assert flags == [False] * 6  # the drop to 0.5e-7 restarts the count: two stalls < 3


STALL_CONFIG = LevenbergMarquardtConfig(
    loss_relative_tolerance=1e-6,
    loss_relative_floor=0.0,
    loss_stall_patience=3,
)
STALL_LOSSES = [1e-7 * (1.0 - 1e-9 * k) for k in range(5)]


def _stall_flags(optim: LevenbergMarquardtOptimiser, losses: list[float], first: int) -> list[bool]:
    return [
        optim.update(np.array([float(first + i)]), x, np.array([1e-3]), np.array([[2.0]])).converged
        for i, x in enumerate(losses)
    ]


def test_lm_resumed_run_keeps_its_stall_count() -> None:
    """Checkpoint after two stalled steps: the resumed run converges at the same step."""
    uninterrupted = _stall_flags(
        LevenbergMarquardtOptimiser(STALL_CONFIG, initial_params=np.array([0.0])), STALL_LOSSES, 0
    )
    first = LevenbergMarquardtOptimiser(STALL_CONFIG, initial_params=np.array([0.0]))
    head = _stall_flags(first, STALL_LOSSES[:3], 0)
    resumed = LevenbergMarquardtOptimiser(STALL_CONFIG, initial_params=np.array([0.0]))
    resumed.load_state_dict(first.state_dict())
    assert head + _stall_flags(resumed, STALL_LOSSES[3:], 3) == uninterrupted
    assert uninterrupted == [False] * 3 + [True] * 2


@pytest.mark.parametrize(
    "bad", [{"loss_stall_patience": 0}, {"loss_relative_floor": -1e-3}], ids=["patience", "floor"]
)
def test_lm_config_rejects_invalid_stall_settings(bad: dict) -> None:
    with pytest.raises(ValueError, match="LevenbergMarquardtConfig"):
        LevenbergMarquardtConfig(**bad)


def test_loss_only_trials_reach_the_same_solution_with_fewer_full_evaluations() -> None:
    """Loss x^4 with a constant Hessian 2 overshoots, so many steps are rejected."""
    from accelmo.fitting.lm_loop import LMPoint, LMProblem, run_levenberg_marquardt

    class Quartic(LMProblem):
        def __init__(self, loss_only_trials: bool) -> None:
            self.loss_only_trials = loss_only_trials
            self.full, self.trials = 0, 0

        def evaluate(self, params: np.ndarray, iteration: int) -> LMPoint:
            self.full += 1
            return LMPoint(params[0] ** 4, np.array([4.0 * params[0] ** 3]), np.array([[2.0]]))

        def trial_loss(self, params: np.ndarray) -> float | None:
            self.trials += 1
            return float(params[0] ** 4)

    config = LevenbergMarquardtConfig(max_iterations=200)
    plain, trial = Quartic(loss_only_trials=False), Quartic(loss_only_trials=True)
    best_plain = run_levenberg_marquardt(np.array([3.0]), plain, config)[0].best
    best_trial = run_levenberg_marquardt(np.array([3.0]), trial, config)[0].best
    assert best_trial.value[0] == best_plain.value[0]
    assert trial.trials > 0
    assert trial.full < plain.full
