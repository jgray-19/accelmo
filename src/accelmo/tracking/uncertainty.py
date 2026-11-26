"""Stopping the training workers and folding their uncertainty parts into ``(A, B)``."""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import psutil

from accelmo.fitting.protocol import STOP, SetUncertaintyMode, UncertaintyProgress, WorkerChannels
from accelmo.tracking.dispatch.setup import KickPlane

if TYPE_CHECKING:
    from collections.abc import Callable

    from accelmo.fitting.pool import WorkerPool
    from accelmo.tracking.dispatch.setup import TrackingWorkerConfig

LOGGER = logging.getLogger(__name__)

# Second-order TPSA memory of one MAD-NG process: the strengths and flow TPSAs the tracker keeps
# (upper estimate), and the factor for the temporaries of a TPSA operation.
SECOND_ORDER_SHARED_TPSAS = 100
SECOND_ORDER_TEMPORARIES = 2.0
# Per particle of a batch: six map coordinates and the contraction accumulator.
SECOND_ORDER_TPSAS_PER_PARTICLE = 7
# Fraction of the free memory the uncertainty stage may fill.
MEMORY_FRACTION = 0.5
# Seconds between progress lines of the second-order pass.
PROGRESS_SECONDS = 15.0


@dataclass
class UncertaintyPart:
    """One worker's contribution to the propagated knob covariance.

    ``normal`` is ``Σ w J Jᵀ``. Each row ``r`` of ``sensitivities`` is the change in
    that worker's gradient per unit noise on reading ``reading_ids[r]`` (an
    observation, a start coordinate, or both), with ``variances[r]`` its declared
    variance.
    """

    normal: np.ndarray  # Shape: (n_knobs, n_knobs)
    reading_ids: np.ndarray  # Shape: (n_rows,)
    sensitivities: np.ndarray  # Shape: (n_rows, n_knobs)
    variances: np.ndarray  # Shape: (n_rows,)

    @classmethod
    def empty(cls, n_knobs: int) -> UncertaintyPart:
        """A worker that contributes nothing (disabled, or uncertainty not requested)."""
        return cls(
            normal=np.zeros((n_knobs, n_knobs)),
            reading_ids=np.zeros(0, dtype=np.int64),
            sensitivities=np.zeros((0, n_knobs)),
            variances=np.zeros(0),
        )


def merge_uncertainty_parts(parts: list[UncertaintyPart], n_knobs: int) -> UncertaintyPart:
    """Sum the normal matrices and merge rows that share a reading id.

    ``G_e`` adds every part's sensitivity to reading ``e`` (shared between workers, or
    used as both an observation and a start), so :func:`noise_matrix` counts its noise
    once. Merging a merged part with further parts gives the same result.
    """
    normal = np.zeros((n_knobs, n_knobs), dtype=np.float64)
    for part in parts:
        normal += part.normal
    ids = np.concatenate([part.reading_ids for part in parts]) if parts else np.zeros(0, np.int64)
    if ids.size == 0:
        return UncertaintyPart(normal, ids, np.zeros((0, n_knobs)), np.zeros(0))

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    sensitivities = np.concatenate([part.sensitivities for part in parts])[order]
    return UncertaintyPart(
        normal=normal,
        reading_ids=sorted_ids[starts],
        sensitivities=np.add.reduceat(sensitivities, starts, axis=0),
        variances=np.concatenate([part.variances for part in parts])[order][starts],
    )


def noise_matrix(part: UncertaintyPart) -> np.ndarray:
    """``B = Σ_e σ_e² G_e G_eᵀ`` for a merged part; readings without a finite variance add nothing."""
    variances = np.where(np.isfinite(part.variances), part.variances, 0.0)
    return (part.sensitivities * variances[:, None]).T @ part.sensitivities


def second_order_bytes(n_params: int, batch_size: int) -> float:
    """Upper estimate of one MAD-NG process holding ``batch_size`` order-2 maps in ``n_params`` knobs.

    One TPSA has a coefficient per monomial of order <= 2 in 6 variables and ``n_params`` parameters.
    """
    monomials = (n_params + 7) * (n_params + 8) // 2
    tpsas = SECOND_ORDER_TPSAS_PER_PARTICLE * batch_size + SECOND_ORDER_SHARED_TPSAS
    return 8.0 * SECOND_ORDER_TEMPORARIES * monomials * tpsas


def _largest_second_order_bytes(pool: WorkerPool, n_params: int, num_batches: int) -> float:
    return max(
        (
            second_order_bytes(n_params, math.ceil(particles / num_batches))
            for particles in pool.particle_counts
        ),
        default=0.0,
    )


def second_order_blocker(pool: WorkerPool, n_params: int, num_batches: int) -> str | None:
    """Why the true Hessian cannot run, or ``None`` when one worker's second-order maps fit.

    ``n_params`` is an upper bound on the MAD-NG knobs. The check runs with the workers up, so
    the free memory already excludes their first-order footprint.
    """
    largest = _largest_second_order_bytes(pool, n_params, num_batches)
    available = psutil.virtual_memory().available
    LOGGER.debug(
        "Second-order uncertainty: <= %.1f MiB per worker, %.1f MiB available",
        largest / 2**20,
        available / 2**20,
    )
    if largest > MEMORY_FRACTION * available:
        return (
            f"one worker needs up to {largest / 2**30:.2f} GiB, but only "
            f"{MEMORY_FRACTION * available / 2**30:.2f} GiB of the free memory may be used "
            "(raise num_batches or drop knobs)"
        )
    return None


@dataclass
class UncertaintyCollection:
    """What the uncertainty stage returned.

    ``matrices`` maps each Hessian that ran to its normal matrix ``A`` and noise matrix ``B``;
    ``skipped`` maps each requested Hessian that did not run to the reason; ``seconds`` is the
    stage's wall time.
    """

    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def primary(self) -> tuple[np.ndarray, np.ndarray]:
        """``(A, B)`` of the first Hessian that ran."""
        return next(iter(self.matrices.values()))


class StageProgress:
    """Counts finished batches of the second-order pass, logging elapsed time and an ETA.

    ``on_update(done, total, elapsed, eta)`` is called at most every ``interval`` seconds and at the
    end, for the run log; ``eta`` is ``None`` until the first batch is done.
    """

    def __init__(
        self,
        total: int,
        on_update: Callable[[int, int, float, float | None], None] | None = None,
        interval: float = PROGRESS_SECONDS,
    ) -> None:
        self.total = total
        self.done = 0
        self.on_update = on_update
        self.interval = interval
        self.start = time.monotonic()
        self._last = self.start

    def advance(self, _message: UncertaintyProgress | None = None) -> None:
        """One more batch done."""
        self.done += 1
        now = time.monotonic()
        if self.done == self.total or now - self._last >= self.interval:
            self._last = now
            self._report(now - self.start)

    def _report(self, elapsed: float) -> None:
        eta = elapsed * (self.total - self.done) / self.done if self.done else None
        LOGGER.debug(
            "Second-order pass: %d/%d batches (%.0f%%), %.0f s elapsed, ETA %s",
            self.done,
            self.total,
            100.0 * self.done / max(self.total, 1),
            elapsed,
            "unknown" if eta is None else f"{eta:.0f} s",
        )
        if self.on_update is not None:
            self.on_update(self.done, self.total, elapsed, eta)


def drain_uncertainty(
    pool: WorkerPool[TrackingWorkerConfig],
    n_knobs: int,
    hessians: tuple[str, ...],
    second_order_batches: int | None = None,
    progress: StageProgress | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Stop ``pool`` in memory-bounded chunks; return each Hessian's normal matrix ``A`` and noise ``B``.

    Feed each pair to :func:`~accelmo.fitting.uncertainty.sandwich_uncertainties`.
    Reading ids are file-scoped, so once every worker of a file has reported, that
    file's merged readings become its ``B`` contribution and are dropped. Workers are
    drained in file order, so besides the chunk being received at most one partly
    merged file is held. No ``hessians`` stops the workers without computing anything.
    """
    normals = {h: np.zeros((n_knobs, n_knobs)) for h in hessians}
    noises = {h: np.zeros((n_knobs, n_knobs)) for h in hessians}
    files = [meta.file_idx for meta in pool.metadata]
    order = sorted(range(len(pool)), key=files.__getitem__)
    remaining = Counter(files)
    chunk_size = (
        _concurrency(pool, n_knobs, second_order_batches, len(hessians))
        if hessians
        else max(1, len(order))
    )

    pending: dict[tuple[int, str], UncertaintyPart] = {}
    for start in range(0, len(order), chunk_size):
        chunk = order[start : start + chunk_size]
        parts_by_file: dict[int, dict[str, list[UncertaintyPart]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for idx, by_hessian in zip(chunk, _drain_parts(pool, chunk, hessians, progress)):
            remaining[files[idx]] -= 1
            for hessian, part in by_hessian.items():
                parts_by_file[files[idx]][hessian].append(part)
        for file_idx, by_hessian in parts_by_file.items():
            for hessian, parts in by_hessian.items():
                if (file_idx, hessian) in pending:
                    parts.append(pending.pop((file_idx, hessian)))
                merged = merge_uncertainty_parts(parts, n_knobs)
                if remaining[file_idx]:
                    pending[file_idx, hessian] = merged
                else:
                    normals[hessian] += merged.normal
                    noises[hessian] += noise_matrix(merged)
    return {h: (normals[h], noises[h]) for h in hessians}


def _drain_parts(
    pool: WorkerPool,
    indices: list[int],
    hessians: tuple[str, ...],
    progress: StageProgress | None,
) -> list[dict[str, UncertaintyPart]]:
    """Stop the given workers and return each one's parts by Hessian, in order."""
    workers = [pool.workers[idx] for idx in indices]
    channels = WorkerChannels([pool.conns[idx] for idx in indices], workers)
    channels.ack_all(SetUncertaintyMode(hessians))
    channels.send_all(STOP)
    replies = channels.recv_all(None if progress is None else progress.advance)
    for reply in replies:
        if not isinstance(reply, dict) or not all(
            isinstance(part, UncertaintyPart) for part in reply.values()
        ):
            raise RuntimeError(f"Unexpected uncertainty payload from worker: {reply!r}")
    for worker in workers:
        worker.join()
    return replies


def _concurrency(
    pool: WorkerPool, n_knobs: int, second_order_batches: int | None, n_hessians: int
) -> int:
    """How many workers may compute and send their parts at once within half the free memory.

    A part has at most one row per observed point for up to two observables per
    kicked plane, plus one start row per plane and particle. Each row holds
    ``n_knobs`` sensitivities, an id and a variance, 8 bytes each, for each of ``n_hessians`` parts. With
    ``second_order_batches`` (the batches per worker) the worker's MAD-NG process also holds its
    order-2 maps (:func:`second_order_bytes`).
    """
    largest = 0
    for meta, particles in zip(pool.metadata, pool.particle_counts):
        planes = 2 if meta.kick_plane == KickPlane.XY else 1
        points = len(meta.bpm_names) * meta.geometry.n_turns
        rows = particles * planes * (2 * points + 1)
        mad = (
            0.0
            if second_order_batches is None
            else second_order_bytes(n_knobs, math.ceil(particles / second_order_batches))
        )
        largest = max(largest, n_hessians * rows * (n_knobs + 2) * 8 + mad)
    available = psutil.virtual_memory().available
    concurrency = max(1, min(len(pool), int(MEMORY_FRACTION * available // max(largest, 1))))
    LOGGER.info(
        "Uncertainty stage: largest worker <= %.1f MiB (part and second-order maps), "
        "%.1f MiB available, %d worker(s) at once",
        largest / 2**20,
        available / 2**20,
        concurrency,
    )
    return concurrency
