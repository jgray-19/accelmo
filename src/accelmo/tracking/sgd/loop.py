"""The mini-batch gradient-descent loop of the tracking fit."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from accelmo.fitting.protocol import Evaluate
from accelmo.fitting.reduction import reduce_replies
from accelmo.fitting.results import FitDiagnostics
from accelmo.optimisers.adam import AdamOptimiser
from accelmo.optimisers.base import BestTracker
from accelmo.optimisers.lbfgs import LBFGSOptimiser
from accelmo.tracking.sgd.checkpointing import OptimisationCheckpointer
from accelmo.tracking.sgd.scheduler import LRScheduler

if TYPE_CHECKING:
    from collections.abc import Callable

    from accelmo.config import OptimiserConfig
    from accelmo.dashboard import FitDashboard
    from accelmo.fitting.protocol import WorkerChannels
    from accelmo.runlog import RunLog
    from accelmo.tracking.config.models import CheckpointConfig

LOGGER = logging.getLogger(__name__)
#: Starting value for a ``pt`` knob that starts at exactly zero.
PT_SEED = 1e-6
OPTIMISERS = {cls.OPTIMISER_NAME: cls for cls in (AdamOptimiser, LBFGSOptimiser)}


@dataclass
class EpochState:
    """What one epoch produced: the knobs after it and the numbers logged for it."""

    epoch: int
    knobs: dict[str, float]
    #: Training chi-square per weighted point.
    loss: float
    grad_norm: float
    lr: float
    started: float
    #: Distance of ``knobs`` from the true strengths (see :meth:`SGDLoop._true_diff`).
    true_diff: float = 0.0
    validation_loss: float | None = None
    new_best: bool = False
    #: Whatever the epoch-end hook returned, appended to the log line.
    note: str | None = None

    @property
    def selection_loss(self) -> float:
        """The loss best-selection and early stopping use: validation when available."""
        return self.validation_loss if self.validation_loss is not None else self.loss


class SGDLoop:
    """Runs epochs of mini-batch gradient descent over the training workers.

    Each epoch sends every batch to the workers, steps the optimiser after each
    one, then validates, picks the best knobs so far and checks the stopping
    rules. :attr:`best` holds the result; :attr:`diagnostics` how the run ended.
    """

    def __init__(
        self,
        knob_names: list[str],
        true_strengths: dict[str, float],
        config: OptimiserConfig,
        num_batches: int,
    ):
        self.knob_names = knob_names
        self.true_strengths = true_strengths
        self.config = config
        self.num_batches = num_batches
        self.gain_names: list[str] = []
        self.gain_ridge = 0.0
        self.smoothed_loss_change = 0.0
        #: Best knobs so far and their selection loss; empty until the first epoch.
        self.best: BestTracker[dict[str, float]] = BestTracker(value={})
        #: How the last :meth:`run` ended.
        self.diagnostics: FitDiagnostics | None = None
        optimiser_cls = OPTIMISERS[config.optimiser_type]
        self.optimiser = optimiser_cls(**optimiser_cls.config_kwargs(config, len(knob_names)))
        self.scheduler = LRScheduler(
            warmup_epochs=config.warmup_epochs,
            decay_epochs=config.decay_epochs,
            start_lr=config.warmup_lr_start,
            max_lr=config.max_lr,
            min_lr=config.min_lr,
        )

    def run(
        self,
        knobs: dict[str, float],
        channels: WorkerChannels,
        *,
        n_points: int,
        run_log: RunLog | None = None,
        dashboard: FitDashboard | None = None,
        checkpoint_config: CheckpointConfig | None = None,
        validation_loss_fn: Callable[[dict[str, float]], float | None] | None = None,
        loss_callback: Callable[[int, float, float | None, float, float], None] | None = None,
        epoch_end_hook: Callable[[dict[str, float], dict[str, float]], str | None] | None = None,
        gain_names: list[str] | None = None,
        gain_ridge: float = 0.0,
    ) -> dict[str, float]:
        """Optimise from ``knobs`` and return the best knobs found.

        The loss is the training chi-square per weighted point (``n_points``). ``validation_loss_fn``
        scores the knobs after each epoch on held-out data; ``epoch_end_hook``
        runs after best-selection and returns a note for the log line.
        ``gain_names`` are BPM gains carried in the knobs but not stepped by the
        optimiser: after every batch each is set to its least-squares value given the
        current magnets, ``Σ w m (y - m) / (Σ w m² + gain_ridge / num_batches)``, from
        the sums the workers return; ``gain_ridge`` is the prior's ``1/σ²`` in the
        same units. ``dashboard`` is redrawn after every epoch.
        """
        self.gain_names = list(gain_names or [])
        self.gain_ridge = gain_ridge
        if self.gain_names:
            knobs = {**knobs, **{k: knobs.get(k, 0.0) for k in self.gain_names}}
        checkpointer = OptimisationCheckpointer(self, checkpoint_config)

        if knobs.get("pt") == 0.0:
            knobs = {**knobs, "pt": PT_SEED}

        prev_loss = None
        start_epoch = 0
        if checkpointer.restore:
            checkpoint_state = checkpointer.load(base_current_knobs=knobs)
            knobs = checkpoint_state["current_knobs"]
            prev_loss = checkpoint_state["prev_loss"]
            start_epoch = checkpoint_state["next_epoch"]
            stage_epochs = self.config.stage_epochs
            if stage_epochs and start_epoch % stage_epochs == 0:
                # A new stage: its data, and so its loss, differ from the last stage's.
                prev_loss = None
                self.best = BestTracker(value={})
                self.smoothed_loss_change = 0.0

        if run_log is not None:
            run_log.begin(knobs, resume_step=start_epoch if checkpointer.restore else None)

        last_completed_epoch = start_epoch - 1
        stop_reason = "max_epochs"
        epochs_run = start_epoch
        for epoch in range(start_epoch, self.config.max_epochs):
            epochs_run = epoch + 1
            state = self._run_epoch(epoch, knobs, channels, n_points)
            knobs = state.knobs
            state.true_diff = self._true_diff(knobs)
            if validation_loss_fn is not None:
                state.validation_loss = validation_loss_fn(knobs)
            state.new_best = self._select_best(state)
            if epoch_end_hook is not None:
                state.note = epoch_end_hook(knobs, self.best.value)

            stop = self._should_stop(state, prev_loss)
            prev_loss = state.selection_loss
            last_completed_epoch = epoch
            if stop is None and checkpointer.should_save_periodic(epoch):
                checkpointer.save(epoch, knobs, prev_loss)

            self._log_epoch(state, run_log, dashboard)
            if loss_callback is not None:
                loss_callback(
                    epoch, state.loss, state.validation_loss, state.grad_norm, state.true_diff
                )
            if stop is not None:
                stop_reason = stop
                break

        if checkpointer.should_save_final(last_completed_epoch):
            checkpointer.save(last_completed_epoch, knobs, prev_loss)
        if dashboard is not None:
            dashboard.finish(stop_reason)
        self.diagnostics = FitDiagnostics(
            converged=stop_reason != "max_epochs",
            reason=stop_reason,
            iterations=epochs_run,
            best_loss=self.best.loss,
        )
        if run_log is not None:
            run_log.finish(
                reason=stop_reason, iterations=epochs_run, best_loss=float(self.best.loss)
            )
        return self.best.value

    def _run_epoch(
        self,
        epoch: int,
        knobs: dict[str, float],
        channels: WorkerChannels,
        n_points: int,
    ) -> EpochState:
        """Evaluate every batch, stepping the optimiser after each one.

        A batch in which any worker lost its particles stops the stepping, and the
        epoch's knobs revert to those it started from.
        """
        started = time.time()
        lr = self.scheduler(epoch % (self.config.stage_epochs or epoch + 1))
        loss = 0.0
        grad = np.zeros(len(self.knob_names))
        start_knobs = knobs
        particle_lost = False
        for batch in range(self.num_batches):
            channels.send_all(Evaluate(knobs, batch))
            reduced = reduce_replies(channels.recv_all(), len(self.knob_names))
            loss += reduced.loss
            grad += reduced.grad
            particle_lost = particle_lost or reduced.particle_lost
            if not particle_lost:
                knobs = self._step(knobs, reduced.grad * self.num_batches / n_points, lr)
                if self.gain_names:
                    assert reduced.extra_sum is not None
                    knobs = {**knobs, **self._best_gains(reduced.extra_sum)}

        if particle_lost:
            LOGGER.warning(
                "Epoch %d: particle loss detected — rejecting knob updates and restoring "
                "pre-epoch parameters",
                epoch,
            )
            knobs = start_knobs

        loss /= n_points
        grad /= n_points
        grad_norm = float(np.linalg.norm(grad[grad != 0.0]))
        return EpochState(epoch, knobs, loss, grad_norm, lr, started)

    def _step(self, knobs: dict[str, float], grad: np.ndarray, lr: float) -> dict[str, float]:
        """One optimiser step from ``knobs`` along ``grad``."""
        params = np.array([knobs[k] for k in self.knob_names])
        return {**knobs, **dict(zip(self.knob_names, self.optimiser.step(params, grad, lr)))}

    def _best_gains(self, sums: np.ndarray) -> dict[str, float]:
        """The least-squares gains from one batch's summed ``(Σ w m (y - m), Σ w m²)``."""
        numerator, curvature = sums
        curvature = curvature + self.gain_ridge / self.num_batches
        gains = np.divide(numerator, curvature, out=np.zeros_like(numerator), where=curvature > 0)
        return dict(zip(self.gain_names, gains.tolist()))

    def _select_best(self, state: EpochState) -> bool:
        """Record ``state`` as the best epoch if its loss is the lowest so far."""
        if state.selection_loss < self.best.loss:
            self.best.record(state.selection_loss, state.knobs.copy())
            return True
        return False

    def _should_stop(self, state: EpochState, prev_loss: float | None) -> str | None:
        """The reason to stop after ``state``, or ``None`` to carry on.

        Always updates the smoothed relative loss change, so call it once per epoch.
        Never stops a staged fit (``stage_epochs``) early.
        """
        if prev_loss is not None:
            loss = state.selection_loss
            change = abs(loss - prev_loss) / abs(prev_loss) if prev_loss != 0 else 0
            self.smoothed_loss_change = self._smooth(self.smoothed_loss_change, change)
            if (
                not self.config.stage_epochs
                and self.smoothed_loss_change < self.config.loss_change_tolerance
                and state.epoch
                > self.config.loss_change_min_epoch_fraction * self.config.max_epochs
            ):
                return "loss_converged"
        return None

    def _smooth(self, average: float, value: float) -> float:
        """Exponential moving average; the first value (``average`` exactly 0) starts it."""
        if average == 0.0:
            return value
        alpha = self.config.loss_change_alpha
        return alpha * average + (1.0 - alpha) * value

    def _true_diff(self, knobs: dict[str, float]) -> float:
        """Sum of absolute differences from the true strengths (the sum of the knobs without them)."""
        if not self.true_strengths:
            return sum(knobs.values())
        return np.sum([abs(knobs[k] - v) for k, v in self.true_strengths.items()])

    def _log_epoch(
        self, state: EpochState, run_log: RunLog | None, dashboard: FitDashboard | None
    ) -> None:
        """Record the epoch in the run log and on the dashboard."""
        epoch_time = time.time() - state.started
        if run_log is not None:
            run_log.log(
                state.epoch,
                loss=state.loss,
                validation_loss=state.validation_loss,
                grad_norm=state.grad_norm,
                learning_rate=state.lr,
                true_diff=float(state.true_diff),
                epoch_time=epoch_time,
                new_best=state.new_best,
            )
            run_log.record_knobs(state.epoch, state.knobs)
        if dashboard is not None:
            dashboard.update(
                epoch=state.epoch,
                loss=state.loss,
                validation_loss=state.validation_loss,
                grad_norm=state.grad_norm,
                lr=state.lr,
                epoch_time=epoch_time,
                true_diff=float(state.true_diff),
                knobs=state.knobs,
                new_best=state.new_best,
                note=state.note,
            )
