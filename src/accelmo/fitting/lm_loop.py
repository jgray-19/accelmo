"""The Levenberg-Marquardt iteration shared by every closed-orbit / closed-twiss fitter."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from accelmo.fitting.results import FitDiagnostics
from accelmo.optimisers.levenberg_marquardt import (
    LevenbergMarquardtOptimiser,
    LevenbergMarquardtUpdate,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from accelmo.dashboard import FitDashboard
    from accelmo.optimisers.levenberg_marquardt import LevenbergMarquardtConfig
    from accelmo.runlog import RunLog

logger = logging.getLogger(__name__)


@dataclass
class LMPoint:
    """The objective at one point: loss, gradient and Gauss-Newton Hessian.

    ``failed`` marks a point where the closed orbit was lost. ``extra`` is whatever
    the evaluation wants handed back to ``on_accept``.
    """

    loss: float
    grad: np.ndarray
    hessian: np.ndarray
    failed: bool = False
    extra: Any = None


class LMProblem:
    """What :func:`run_levenberg_marquardt` asks of a fit; subclasses override the hooks they use."""

    #: Score each trial point by :meth:`trial_loss` first and evaluate it in full only if it
    #: improves on the best loss, so a rejected step costs a cheap loss evaluation.
    loss_only_trials = False

    def evaluate(self, params: np.ndarray, iteration: int) -> LMPoint:
        """The full objective at ``params``."""
        raise NotImplementedError

    def trial_loss(self, params: np.ndarray) -> float | None:
        """The loss alone at ``params``, ``None`` when the orbit is lost."""
        raise NotImplementedError

    def on_accept(self, params: np.ndarray, point: LMPoint) -> None:
        """Sees every accepted point."""

    def after_step(self, params: np.ndarray, optimiser: LevenbergMarquardtOptimiser) -> None:
        """Runs once the next point is chosen."""

    def refresh(self, params: np.ndarray) -> bool:
        """Runs at every newly accepted point; True if it changed the objective.

        E.g. the tracking start conditions were rebuilt from the fitted model. The old losses
        then no longer compare: the optimiser forgets its best point and the same point is
        evaluated again before the next step is taken.
        """
        return False


def run_levenberg_marquardt(
    params: np.ndarray,
    problem: LMProblem,
    config: LevenbergMarquardtConfig,
    *,
    run_log: RunLog | None = None,
    param_names: Sequence[str] | None = None,
    dashboard: FitDashboard | None = None,
) -> tuple[LevenbergMarquardtOptimiser, FitDiagnostics]:
    """Minimise with Levenberg-Marquardt steps ``(H + lam·diag H) delta = -g``.

    A rejected step is not a no-op: the optimiser retries from its best point with
    more damping, so the next point still needs evaluating. Only the terminal
    reasons (no curvature to retry from, or damping driven to the ceiling) end the
    solve early.

    With ``run_log``, every accepted iteration's loss is logged, and its parameters
    too when ``param_names`` names them. ``dashboard`` (already in its fit stage) shows each
    accepted iteration.

    Returns the optimiser, whose ``best_params`` are the solution, and how the fit ended.
    """
    optimiser = LevenbergMarquardtOptimiser(config, initial_params=params)
    last_time = time.time()
    last_update = None
    completed_iterations = 0
    accepted_evaluations = 0
    just_refreshed = False
    record_params = run_log is not None and param_names is not None
    if record_params:
        run_log.begin(_named(param_names, params))

    for iteration in range(config.max_iterations):
        completed_iterations = iteration + 1
        update, point = _next_update(problem, optimiser, params, iteration)
        last_update = update
        accepted_params = params
        if update.accepted:
            assert point is not None  # loss-only trials are never accepted
            accepted_evaluations += 1
            problem.on_accept(params, point)
        params = update.next_params

        if update.accepted:
            if not just_refreshed and problem.refresh(accepted_params):
                optimiser.reset_objective(accepted_params)
                params, just_refreshed = accepted_params, True
                continue
            just_refreshed = False

        if update.converged and not update.accepted:
            logger.warning(
                "Levenberg-Marquardt stopped at iter %d (%s, lam=%.1e)",
                iteration,
                update.reason,
                update.damping,
            )
            break
        problem.after_step(params, optimiser)
        now = time.time()
        if not update.accepted:
            _log_unaccepted(iteration, update, optimiser.best.loss)
            if dashboard is not None:
                dashboard.rejected(
                    epoch=iteration,
                    loss=update.loss,
                    grad_norm=update.grad_norm,
                    lr=update.damping,
                    epoch_time=now - last_time,
                )
            last_time = now
            continue

        if dashboard is not None:
            dashboard.update(
                epoch=iteration,
                loss=update.loss,
                validation_loss=None,
                grad_norm=update.grad_norm,
                lr=update.damping,
                epoch_time=now - last_time,
                true_diff=None,
                knobs=_named(param_names, accepted_params) if param_names else {},
                new_best=True,
            )
        last_time = now
        logger.debug(
            "LM iter %d: loss=%.3e, |g|=%.3e, lam=%.1e",
            iteration,
            update.loss,
            update.grad_norm,
            update.damping,
        )
        if run_log is not None:
            run_log.log(
                iteration, loss=update.loss, grad_norm=update.grad_norm, lm_lambda=update.damping
            )
            if record_params:
                run_log.record_knobs(iteration, _named(param_names, accepted_params))
        if update.converged:
            logger.info("Levenberg-Marquardt converged (%s) at iter %d", update.reason, iteration)
            break

    if dashboard is not None:
        dashboard.finish(
            last_update.reason
            if last_update is not None and last_update.converged
            else "max_iterations"
        )
    diagnostics = FitDiagnostics(
        converged=bool(last_update is not None and last_update.converged),
        reason=None if last_update is None else last_update.reason,
        iterations=completed_iterations,
        best_loss=float(optimiser.best.loss),
        gradient_norm=None if last_update is None else float(last_update.grad_norm),
        damping=None if last_update is None else float(last_update.damping),
        accepted_evaluations=accepted_evaluations,
    )
    if run_log is not None:
        run_log.finish(
            reason=diagnostics.reason,
            iterations=diagnostics.iterations,
            best_loss=diagnostics.best_loss,
        )
    return optimiser, diagnostics


def _next_update(
    problem: LMProblem, optimiser: LevenbergMarquardtOptimiser, params: np.ndarray, iteration: int
) -> tuple[LevenbergMarquardtUpdate, LMPoint | None]:
    """The optimiser's verdict on ``params`` and the full point it was given, if one was evaluated."""
    if problem.loss_only_trials and optimiser.best_hessian is not None:
        loss = problem.trial_loss(params)
        n = len(params)
        if loss is None:
            update = optimiser.update(
                params, float("nan"), np.zeros(n), np.zeros((n, n)), failed=True
            )
            return update, None
        if not loss < optimiser.best.loss:
            update = optimiser.update(params, loss, np.zeros(n), np.zeros((n, n)), failed=False)
            return update, None
    point = problem.evaluate(params, iteration)
    return optimiser.update(params, point.loss, point.grad, point.hessian, point.failed), point


def _log_unaccepted(iteration: int, update: LevenbergMarquardtUpdate, best_loss: float) -> None:
    if update.reason == "failed":
        logger.warning(
            "Iter %d: closed orbit lost; retrying from best (lam=%.1e)", iteration, update.damping
        )
    else:
        logger.debug(
            "Iter %d: step rejected (loss %.6e >= best %.6e); retrying from best (lam=%.1e)",
            iteration,
            update.loss,
            best_loss,
            update.damping,
        )


def _named(names: Sequence[str], params: np.ndarray) -> dict[str, float]:
    return dict(zip(names, (float(v) for v in params), strict=True))
