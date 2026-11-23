"""The tracking Levenberg-Marquardt objective: knobs, BPM gains and optionally the particle starts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import block_diag

from accelmo.fitting.bpm_gains import without_bpm_gains
from accelmo.fitting.lm_loop import LMPoint, LMProblem
from accelmo.fitting.priors import (
    ZERO_MEAN_SIGMA,
    add_quadratic_prior,
    inverse_variances,
    quadratic_prior,
)
from accelmo.fitting.protocol import Evaluate
from accelmo.fitting.reduction import reduce_replies

if TYPE_CHECKING:
    from collections.abc import Callable

    from accelmo.config import TrackingParameters
    from accelmo.tracking.session import TrackingSession

#: A refresh that moves no start coordinate by more than this (m, rad) leaves the objective unchanged.
LM_REFRESH_ATOL = 1e-12


@dataclass(frozen=True)
class StartGainMap:
    """Starts as a linear function of the BPM gains: ``s(g) = starts + jacobian @ (g - gains)``.

    ``starts`` ``(n_particles, 4)`` in :meth:`TrackingSession.send_init_condition_updates`
    order, ``jacobian`` ``(n_particles, 4, n_gains)`` and ``gains`` ``(n_gains,)`` in
    :attr:`TrackingFitter.gain_names` order. See docs/start_gain_chain_rule.md.
    """

    starts: np.ndarray
    jacobian: np.ndarray
    gains: np.ndarray

    def at(self, gains: np.ndarray) -> np.ndarray:
        """Starts at ``gains``."""
        return self.starts + self.jacobian @ (gains - self.gains)


def chain_starts_through_gains(
    grad: np.ndarray,
    hessian: np.ndarray,
    start_grad: np.ndarray,
    knob_start: np.ndarray,
    blocks: np.ndarray,
    jacobian: np.ndarray,
    gains: slice,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and Gauss-Newton Hessian of ``L(k, g, s(g))`` from those of ``L(k, g, s)``.

    ``start_grad`` (4P,), ``knob_start`` (n, 4P) and ``blocks`` (P, 4, 4) are the start
    blocks, ``jacobian`` (P, 4, n_gains) is ds/dg and ``gains`` the gains' slice of the
    n parameters. Derivation: docs/start_gain_chain_rule.md.
    """
    ds = jacobian.reshape(-1, jacobian.shape[-1])
    cross = knob_start @ ds
    grad, hessian = grad.copy(), np.array(hessian, dtype=float)
    grad[gains] += ds.T @ start_grad
    hessian[:, gains] += cross
    hessian[gains, :] += cross.T
    hessian[gains, gains] += ds.T @ (block_diag(blocks, format="csr") @ ds)
    return grad, hessian


class TrackingLMProblem(LMProblem):
    """Loss, gradient and Gauss-Newton Hessian of the tracking fit over every batch.

    The parameters are the ``names`` (magnet knobs, then the BPM gains, the last
    ``n_gains``). With ``fit_start_coords`` every training particle's start (x, px, y, py)
    follows them in session particle order and is pushed to the workers before each
    evaluation. With ``starts_follow_gains`` the starts are the callback's
    :class:`StartGainMap` at the trial gains.
    """

    def __init__(
        self,
        session: TrackingSession,
        names: list[str],
        n_gains: int,
        num_batches: int,
        parameters: TrackingParameters,
        knob_sigma: np.ndarray,
        gain_ridge: float,
        callback: Callable[[dict[str, float], dict[str, float]], np.ndarray | StartGainMap | None]
        | None,
        model_values: dict[str, float],
    ) -> None:
        self.session = session
        self.names = names
        self.n_knobs = len(names)
        self.gains = slice(self.n_knobs - n_gains, self.n_knobs)
        self.n_batches = num_batches
        self.fit_starts = parameters.fit_start_coords
        self.follow = parameters.starts_follow_gains
        self.callback = callback
        self.model_values = model_values
        counts = session.training.particle_counts
        self.n_train = sum(counts)
        if self.fit_starts and session.validation:
            raise ValueError("fit_start_coords needs a fit without validation workers")
        if self.follow and (self.fit_starts or not n_gains):
            raise ValueError("starts_follow_gains needs fitted gains and no fit_start_coords")

        ridge = np.zeros(self.n_knobs)
        ridge[self.gains] = gain_ridge
        # A gain named in knob_sigmas overrides bpm_gain_sigma.
        ridge = np.where(knob_sigma > 0, inverse_variances(knob_sigma), ridge)
        zero_mean = [
            np.array([i for i, name in enumerate(names) if name.startswith(f"bpmgain.{plane}.")])
            for plane in parameters.bpm_gain_zero_mean_planes
        ]
        self.prior_matrix = quadratic_prior(ridge, zero_mean, 1.0 / ZERO_MEAN_SIGMA**2)
        self.start_map: StartGainMap | None = None  # the callback's latest map
        self._pushed: np.ndarray | None = None  # the starts the workers hold after the last refresh

    def initial_params(self, knobs: dict[str, float]) -> np.ndarray:
        """Knobs in ``names`` order; with fitted starts, the workers' measured starts follow."""
        initial = np.array([knobs[name] for name in self.names], dtype=float)
        if not self.fit_starts:
            return initial
        *_, per_batch = self._gather(self.knobs(initial))
        starts = np.zeros((self.n_train, 4))
        for replies, row in zip(per_batch, self._start_slots(per_batch), strict=True):
            for reply, rows in zip(replies, row, strict=True):
                starts[rows] = reply.extra["start"]
        return np.concatenate([initial, starts.ravel()])

    def param_names(self) -> list[str]:
        """``names`` then one ``start_param{i}`` per fitted start coordinate."""
        n_starts = 4 * self.n_train if self.fit_starts else 0
        return self.names + [f"start_param{i}" for i in range(n_starts)]

    def starts(self, params: np.ndarray) -> np.ndarray:
        """The fitted ``(n_particles, 4)`` starts in ``params``; empty without fitted starts."""
        return params[self.n_knobs :].reshape(-1, 4)

    def knobs(self, params: np.ndarray) -> dict[str, float]:
        """The ``names`` entries of ``params`` as a knob dict."""
        return dict(zip(self.names, (float(v) for v in params[: self.n_knobs]), strict=True))

    def evaluate(self, params: np.ndarray, iteration: int) -> LMPoint:
        """Push the starts for ``params``, evaluate every batch and chain/prior the result."""
        knobs = self.knobs(params)
        if self.fit_starts:
            self.session.send_init_condition_updates(params[self.n_knobs :].reshape(-1, 4))
        if self.start_map is not None:
            self.session.send_init_condition_updates(self.start_map.at(params[self.gains]))
        loss, grad, hessian, lost, per_batch = self._gather(knobs)
        hessian = np.array(hessian)
        if self.start_map is not None and not lost:
            grad, hessian = chain_starts_through_gains(
                grad,
                hessian,
                *self._start_blocks(per_batch),
                self.start_map.jacobian[: self.n_train],
                self.gains,
            )
        if self.prior_matrix.any():  # Gaussian priors on gains and knobs, once for all batches
            loss, grad, hessian = add_quadratic_prior(
                loss, grad, hessian, params[: self.n_knobs], self.prior_matrix
            )

        n, k = len(params), self.n_knobs
        full_grad, full_hess = np.zeros(n), np.zeros((n, n))
        full_grad[:k], full_hess[:k, :k] = grad, hessian
        if self.fit_starts and not lost:
            start_grad, knob_start, blocks = self._start_blocks(per_batch)
            full_grad[k:] = start_grad
            full_hess[:k, k:] = knob_start
            full_hess[k:, :k] = knob_start.T
            for particle in range(self.n_train):
                cols = slice(k + 4 * particle, k + 4 * particle + 4)
                full_hess[cols, cols] = blocks[particle]
        return LMPoint(loss, full_grad, full_hess, lost)

    def refresh(self, params: np.ndarray) -> bool:
        """Rebuild the start conditions from the accepted knobs; True if they moved."""
        if self.callback is None:
            return False
        fitted = self.knobs(params)
        coords = self.callback(
            {**self.model_values, **without_bpm_gains(fitted)}, {**self.model_values, **fitted}
        )
        if coords is None or self.fit_starts:  # fitted starts: the callback only observes
            return False
        if isinstance(coords, StartGainMap):
            if self.start_map is not None:  # the workers hold the old map at these gains
                self._pushed = self.start_map.at(params[self.gains])
            self.start_map = coords
            coords = coords.at(params[self.gains])
        self.session.send_init_condition_updates(coords)
        moved = self._pushed is None or not np.allclose(
            coords, self._pushed, rtol=0.0, atol=LM_REFRESH_ATOL
        )
        self._pushed = coords
        return moved

    def _gather(self, knobs: dict[str, float]) -> tuple[float, np.ndarray, np.ndarray, bool, list]:
        """Summed knob loss/grad/Hessian and, per batch, the replies in worker order."""
        channels = self.session.training.channels
        n = self.n_knobs
        loss, grad, hessian, lost, per_batch = 0.0, np.zeros(n), np.zeros((n, n)), False, []
        for batch in range(self.n_batches):
            channels.send_all(Evaluate(knobs, batch, curvature=True))
            replies = channels.recv_all()
            reduced = reduce_replies(replies, self.n_knobs)
            loss += reduced.loss
            grad = grad + reduced.grad
            if reduced.normal is None:
                raise ValueError("The workers returned no Gauss-Newton matrix")
            hessian = hessian + 2.0 * reduced.normal
            lost = lost or reduced.particle_lost
            per_batch.append(replies)
        return loss, grad, hessian, lost, per_batch

    def _start_slots(self, per_batch: list) -> list[list[slice]]:
        """Global particle rows of each (batch, worker) reply."""
        counts = self.session.training.particle_counts
        worker_first = np.concatenate([[0], np.cumsum(counts)[:-1]])
        done = np.zeros(len(counts), dtype=int)
        slots = []
        for replies in per_batch:
            row = []
            for w, reply in enumerate(replies):
                n = len(reply.extra["start"])
                row.append(slice(worker_first[w] + done[w], worker_first[w] + done[w] + n))
                done[w] += n
            slots.append(row)
        return slots

    def _start_blocks(self, per_batch: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Training particles' dL/ds (4P), d2L/dk ds (knobs, 4P) and d2L/ds2 (P, 4, 4)."""
        start_grad = np.zeros(4 * self.n_train)
        knob_start = np.zeros((self.n_knobs, 4 * self.n_train))
        blocks = np.zeros((self.n_train, 4, 4))
        for replies, row in zip(per_batch, self._start_slots(per_batch), strict=True):
            for reply, rows in zip(replies, row, strict=True):
                cols = slice(4 * rows.start, 4 * rows.stop)
                start_grad[cols] = reply.extra["grad"].ravel()
                knob_start[:, cols] = reply.extra["knob_start"].reshape(self.n_knobs, -1)
                blocks[rows] = reply.extra["start_start"]
        return start_grad, knob_start, blocks
