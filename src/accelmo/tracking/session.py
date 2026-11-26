"""The running workers of one tracking fit.

A :class:`TrackingSession` builds the worker payloads, spawns the training and
held-out validation :class:`~accelmo.fitting.pool.WorkerPool` s, and talks
to them until they are stopped: initial-condition updates, the
validation loss and the uncertainty drain. Worker ranges come from
:mod:`~accelmo.tracking.dispatch.setup`, payload arrays from
:mod:`~accelmo.tracking.dispatch.payloads`.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from accelmo.fitting.pool import WorkerPool
from accelmo.fitting.protocol import (
    LossReply,
    SetKnobs,
    Start,
    TotalChi2,
    UpdateInitCoords,
)
from accelmo.tracking.dispatch.payloads import WorkerPayloadBuilder
from accelmo.tracking.uncertainty import (
    StageProgress,
    UncertaintyCollection,
    drain_uncertainty,
    second_order_blocker,
)
from accelmo.tracking.worker import TrackingWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from accelmo.config import TrackingDataConfig, TrackingParameters, WorkerRuntime
    from accelmo.machine.accelerators import Accelerator
    from accelmo.tracking.data_manager import FileTracks
    from accelmo.tracking.dispatch.payloads import WorkerPayload
    from accelmo.tracking.dispatch.setup import TrackingWorkerConfig

LOGGER = logging.getLogger(__name__)


class TrackingSession:
    """The training and validation workers of one tracking fit, from start to shutdown.

    ``turn_batches`` and ``validation_turn_batches`` are the training and held-out
    turns, one list per worker batch; ``file_turn_map`` maps each turn to its
    measurement file and ``worker_plans`` each file to the workers one of its batches
    fans out to (:func:`~accelmo.tracking.dispatch.setup.plan_workers`). Pass an empty
    ``validation_turn_batches`` to run without validation workers.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        runtime: WorkerRuntime,
        tracking_data: TrackingDataConfig,
        parameters: TrackingParameters,
        *,
        turn_batches: list[list[int]],
        validation_turn_batches: list[list[int]],
        file_turn_map: dict[int, int],
        worker_plans: dict[int, list[TrackingWorkerConfig]],
        machine_deltaps: list[float],
        gain_bpms: tuple[str, ...] = (),
    ) -> None:
        self.runtime = runtime
        self.tracking_data = tracking_data
        self.parameters = parameters
        self.turn_batches = turn_batches
        self.validation_turn_batches = validation_turn_batches
        self.file_turn_map = file_turn_map
        self.worker_plans = worker_plans
        self.machine_deltaps = machine_deltaps
        self.payload_builder = WorkerPayloadBuilder(accelerator, gain_bpms)
        self.training: WorkerPool[TrackingWorkerConfig] = WorkerPool()
        self.validation: WorkerPool[TrackingWorkerConfig] = WorkerPool()

    def start(self, tracks: dict[int, FileTracks], initial_knobs: dict[str, float]) -> None:
        """Spawn the training and validation workers, each starting from ``initial_knobs``."""
        training_payloads, validation_payloads = self.build_payloads(tracks)
        LOGGER.debug(
            "Starting %d trn worker(s) + %d held-out val worker(s)",
            len(training_payloads),
            len(validation_payloads),
        )
        self.training = self._spawn(training_payloads, 0, initial_knobs, validation=False)
        self.validation = self._spawn(
            validation_payloads, len(training_payloads), initial_knobs, validation=True
        )
        if validation_payloads:
            covered = {
                (m.file_idx, m.geometry.bpm_range, m.geometry.sdir)
                for m in self.validation.metadata
            }
            LOGGER.debug(
                "Validation setup: payloads=%d, covered_ranges=%d, tracks=%d",
                len(validation_payloads),
                len(covered),
                sum(self.validation.particle_counts),
            )

    def build_payloads(
        self, tracks: dict[int, FileTracks]
    ) -> tuple[list[WorkerPayload], list[WorkerPayload]]:
        """Weighted training and held-out validation payloads from ``tracks``, from disjoint turns.

        Sets :attr:`n_training_points` and :attr:`n_validation_points`, the weighted points of each.
        """
        training = self.create_worker_payloads(tracks, self.turn_batches)
        validation = (
            self.create_worker_payloads(tracks, self.validation_turn_batches)
            if self.validation_turn_batches
            else []
        )
        momenta = self.tracking_data.optimise_momenta
        self.n_training_points = WorkerPayloadBuilder.attach_weights(
            training, optimise_momenta=momenta
        )
        self.n_validation_points = WorkerPayloadBuilder.attach_weights(
            validation, optimise_momenta=momenta
        )
        if not self.n_training_points or (validation and not self.n_validation_points):
            raise ValueError(
                "Every training or validation reading has zero weight (no finite variance)"
            )
        return training, validation

    def create_worker_payloads(
        self, tracks: dict[int, FileTracks], turn_batches: list[list[int]]
    ) -> list[WorkerPayload]:
        """One payload per (worker, turn batch); each batch belongs to one file."""
        payloads: list[WorkerPayload] = []
        for batch_idx, turn_batch in enumerate(turn_batches):
            if not turn_batch:
                raise ValueError(f"Empty turn batch {batch_idx}")
            for config in self.worker_plans[self.file_turn_map[turn_batch[0]]]:
                data = self.payload_builder.make_tracking_data(
                    turn_batch=turn_batch,
                    file_turn_map=self.file_turn_map,
                    config=config,
                    machine_deltaps=self.machine_deltaps,
                    tracks=tracks,
                )
                payloads.append((data, config))

        self._log_file_usage(
            payloads,
            frozenset(self.file_turn_map[turn] for batch in turn_batches for turn in batch),
        )
        return payloads

    def _log_file_usage(self, payloads: list[WorkerPayload], files_covered: frozenset[int]) -> None:
        """Log measurement-file usage and check that at least one worker exists."""
        num_files = len(self.worker_plans)
        file_usage = Counter(config.file_idx for _, config in payloads)
        LOGGER.debug(
            "Created %d workers using files: %s",
            len(payloads),
            ", ".join(f"file_{idx}={count} workers" for idx, count in sorted(file_usage.items())),
        )
        if len(files_covered) < num_files:
            LOGGER.warning(
                "Only %d/%d measurement files are being used by workers! "
                "This may lead to poor optimisation if different files have different deltap values.",
                len(files_covered),
                num_files,
            )
        if not file_usage:
            raise ValueError(
                "No worker payloads were created; check your input data and batch configuration"
            )

    def _spawn(
        self,
        payloads: list[WorkerPayload],
        first_id: int,
        initial_knobs: dict[str, float],
        *,
        validation: bool,
    ) -> WorkerPool[TrackingWorkerConfig]:
        """Start one worker per payload, send it the initial knobs and record its config."""
        pool: WorkerPool[TrackingWorkerConfig] = WorkerPool()
        for worker_id, (data, config) in enumerate(payloads, start=first_id):
            pool.spawn(
                TrackingWorker,
                worker_id,
                data,
                config,
                self.runtime,
                self.tracking_data,
                self.parameters,
                validation,
            ).send(Start(initial_knobs))
            pool.metadata.append(config)
            pool.particle_counts.append(len(data.init_coords))
            LOGGER.debug(
                "%s worker %d: file=%d, range=%s/%s, sdir=%d, kick_plane=%s, observed_bpms=%d, tracks=%d",
                "Val" if validation else "Trn",
                worker_id,
                config.file_idx,
                config.geometry.start_bpm,
                config.geometry.end_bpm,
                config.geometry.sdir,
                config.kick_plane,
                len(config.bpm_names),
                len(data.init_coords),
            )
        return pool

    def training_losses(self, knobs: dict[str, float]) -> list[float]:
        """Each training worker's chi-square at ``knobs``."""
        channels = self.training.channels
        channels.send_all(TotalChi2(knobs))
        replies = channels.recv_all()
        for reply in replies:
            if not isinstance(reply, LossReply):
                raise RuntimeError(f"Unexpected loss reply from worker: {reply!r}")
        return [reply.loss for reply in replies]

    def send_init_condition_updates(self, new_coords: np.ndarray) -> None:
        """Push updated initial ``x, px, y, py`` to every training and validation worker.

        ``new_coords`` must be a float64 array of shape ``(n_total_particles, 4)``
        whose columns are ``x, px, y, py`` and whose rows are ordered: training
        workers first (in creation order), then validation workers (in creation
        order), and within each worker in particle order.

        Positions travel with the momenta because the launch point can sit on a
        closed orbit that the fitted magnets themselves shape; see
        ``TrackingWorker._send_init_condition_update``. A caller with nothing new
        to say about position passes the current x/y straight back.

        Workers handle the update before processing the next gradient batch, so
        this method is safe to call between epochs (from the epoch-end hook).
        """
        pools = [pool for pool in (self.training, self.validation) if pool]
        expected = sum(sum(pool.particle_counts) for pool in pools)
        if new_coords.shape != (expected, 4):
            raise ValueError(
                f"new_coords must have shape ({expected}, 4) of x, px, y, py; "
                f"got {new_coords.shape}"
            )

        offset = 0
        for pool in pools:
            for conn, n in zip(pool.conns, pool.particle_counts):
                chunk = new_coords[offset : offset + n]
                conn.send(UpdateInitCoords(chunk))
                offset += n
            pool.channels.recv_all()

    def build_update_coords(self, tracks: dict[int, FileTracks]) -> np.ndarray:
        """The combined ``x, px, y, py`` start coordinates of the running workers, from ``tracks``.

        Returns a float64 array of shape ``(n_total_particles, 4)`` suitable for
        :meth:`send_init_condition_updates`: training worker rows first, followed
        by validation worker rows.
        """
        training, validation = self.build_payloads(tracks)
        return np.concatenate(
            [data.init_coords[:, :4] for data, _config in training + validation]
        ).astype(np.float64)

    def validation_loss(self, knobs: dict[str, float]) -> float | None:
        """Held-out chi-square per weighted point at ``knobs``, or ``None`` without validation workers.

        The validation workers track turns that were removed from training, so this
        is a genuine out-of-sample loss on the training loss's scale.
        """
        if not self.validation:
            return None

        channels = self.validation.channels
        channels.send_all(TotalChi2(knobs))
        results = channels.recv_all()
        for result in results:
            if not isinstance(result, LossReply):
                raise RuntimeError(f"Unexpected validation reply from worker: {result!r}")
        return sum(float(result.loss) for result in results) / self.n_validation_points

    def set_training_knobs(self, knobs: dict[str, float]) -> None:
        """Load ``knobs`` into every training worker, e.g. the best knobs before the uncertainty."""
        self.training.channels.ack_all(SetKnobs(knobs))

    def stop_and_collect_uncertainty(
        self,
        n_knobs: int,
        hessians: tuple[str, ...] | None = None,
        on_progress: Callable[[int, int, float, float | None], None] | None = None,
    ) -> UncertaintyCollection:
        """Stop every worker and return ``A`` and ``B`` of the training workers per Hessian.

        ``hessians`` defaults to the configured one; ``()`` stops without computing. The true
        Hessian is replaced by Gauss-Newton, with the reason in ``skipped``, when the backend has
        no second derivatives or one worker's second-order maps do not fit in memory.
        ``on_progress(done, total, elapsed, eta)`` follows the second-order pass.
        """
        start = time.monotonic()
        requested = (self.parameters.uncertainty_hessian,) if hessians is None else tuple(hessians)
        skipped: dict[str, str] = {}
        if "true" in requested:
            reason = self._true_hessian_blocker(n_knobs)
            if reason is not None:
                skipped["true"] = reason
                LOGGER.warning("True Hessian skipped, using Gauss-Newton: %s", reason)
        run = tuple(dict.fromkeys("gauss_newton" if h in skipped else h for h in requested))
        true_run = "true" in run
        progress = (
            StageProgress(len(self.training) * self.tracking_data.num_batches, on_progress)
            if true_run
            else None
        )
        LOGGER.debug("Terminating workers, computing the %s Hessian...", " and ".join(run))
        matrices = drain_uncertainty(
            self.training,
            n_knobs,
            run,
            self.tracking_data.num_batches if true_run else None,
            progress,
        )
        self.validation.stop()
        seconds = time.monotonic() - start
        LOGGER.debug("Uncertainty stage took %.1f s", seconds)
        return UncertaintyCollection(matrices, skipped, seconds)

    def _true_hessian_blocker(self, n_knobs: int) -> str | None:
        """Why the true Hessian cannot run on these workers, or ``None``."""
        backend = self.payload_builder.accelerator.backend
        if backend != "madng":
            return f"the {backend} backend has no second derivatives"
        return second_order_blocker(self.training, n_knobs, self.tracking_data.num_batches)

    def terminate(self) -> None:
        """Kill every worker without waiting, for aborting after an error or interrupt.

        Workers may be stuck in a failed simulation and not respond to a stop
        message, so SIGTERM is sent and the processes are reaped.
        """
        LOGGER.debug("Terminating workers...")
        self.training.kill()
        self.validation.kill()
