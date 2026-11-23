"""Summing the workers' replies to one evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from accelmo.fitting.protocol import GradReply

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class Reduced:
    """The sum over the workers that returned a valid loss.

    ``normal`` is ``None`` when the workers send none (no curvature asked for).
    ``particle_lost`` is set when any worker lost its particles or closed orbit;
    those workers are left out of every sum and of ``n_valid``.
    """

    loss: float
    grad: np.ndarray
    normal: np.ndarray | None
    n_valid: int
    particle_lost: bool
    #: Sum of the replies' array ``extra`` (tracking: the BPM-gain sums), else ``None``.
    extra_sum: np.ndarray | None = None


def reduce_replies(replies: Sequence[object], n_knobs: int) -> Reduced:
    """Sum the :class:`~accelmo.fitting.protocol.GradReply` of every worker.

    A NaN loss marks a worker that lost its particles or closed orbit; an infinite
    one a worker error, which is raised.
    """
    if not replies:
        raise RuntimeError("No workers returned results")
    with_curvature = isinstance(replies[0], GradReply) and replies[0].normal is not None
    loss = 0.0
    grad = np.zeros(n_knobs, dtype=float)
    normal = np.zeros((n_knobs, n_knobs)) if with_curvature else None
    n_valid = 0
    particle_lost = False
    extra_sum = None
    for reply in replies:
        if not isinstance(reply, GradReply):
            raise RuntimeError(f"Unexpected worker reply: {reply!r}")
        if reply.loss == float("inf"):
            raise RuntimeError("Worker error detected during optimisation")
        if math.isnan(float(reply.loss)):
            particle_lost = True
            continue
        grad += np.asarray(reply.grad, dtype=float).ravel()
        if with_curvature:
            normal += np.asarray(reply.normal, dtype=float)
        if isinstance(reply.extra, np.ndarray):
            extra_sum = reply.extra.copy() if extra_sum is None else extra_sum + reply.extra
        loss += float(reply.loss)
        n_valid += 1
    return Reduced(loss, grad, normal, n_valid, particle_lost, extra_sum)
