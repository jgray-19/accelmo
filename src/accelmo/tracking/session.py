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
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from accelmo.fitting.pool import WorkerPool
from accelmo.fitting.protocol import (
    Diagnose,
    LossReply,
    SetKnobs,
    Start,
    UpdateInitCoords,
    Validate,
)
from accelmo.tracking.dispatch.payloads import WorkerPayloadBuilder
from accelmo.tracking.uncertainty import drain_uncertainty
from accelmo.tracking.worker import TrackingWorker

if TYPE_CHECKING:
    from accelmo.config import SimulationConfig
    from accelmo.tracking.data_manager import FileTracks
    from accelmo.tracking.dispatch.payloads import WorkerPayload
    from accelmo.tracking.dispatch.setup import (
        WorkerObservationPlan,
        WorkerSetupHelper,
    )

LOGGER = logging.getLogger(__name__)


class TrackingSession:
    """The training and validation workers of one tracking fit, from start to shutdown.

    ``turn_batches`` and ``validation_turn_batches`` are the training and held-out
    turns, one list per worker batch; ``file_turn_map`` maps each turn to its
    measurement file and ``worker_plans`` each file to the workers one of its batches
    fans out to (:meth:`WorkerSetupHelper.plan_workers`). Pass an empty
    ``validation_turn_batches`` to run without validation workers.
    """

    def __init__(
        self,
        setup_helper: WorkerSetupHelper,
        simulation_config: SimulationConfig,
        *,
        turn_batches: list[list[int]],
        validation_turn_batches: list[list[int]],
        file_turn_map: dict[int, int],
        worker_plans: dict[int, list[WorkerObservationPlan]],
        machine_deltaps: list[float],
        gain_bpms: tuple[str, ...] = (),
    ) -> None:
        self.setup_helper = setup_helper
        self.simulation_config = simulation_config
        self.turn_batches = turn_batches
        self.validation_turn_batches = validation_turn_batches
        self.file_turn_map = file_turn_map
        self.worker_plans = worker_plans
        self.machine_deltaps = machine_deltaps
        self.payload_builder = WorkerPayloadBuilder(setup_helper.accelerator, gain_bpms)
        self.training: WorkerPool[WorkerObservationPlan] = WorkerPool()
        self.validation: WorkerPool[WorkerObservationPlan] = WorkerPool()

    def start(self, tracks: dict[int, FileTracks], initial_knobs: dict[str, float]) -> None:
        """Spawn the training and validation workers, each starting from ``initial_knobs``."""
        training_payloads, validation_payloads = self.build_payloads(tracks)
        LOGGER.info(
            "Starting %d trn worker(s) + %d held-out val worker(s)",
            len(training_payloads),
            len(validation_payloads),
        )
        # Converts a physical chi-square to the loop's loss units (see SGDLoop.run's prior).
        n_points = np.mean([data.position_comparisons.shape[1] for data, _, _ in training_payloads])
        self.chi2_to_loss = 1.0 / (training_payloads[0][0].precomputed_weights.scale * n_points)
        self.training = self._spawn(training_payloads, 0, initial_knobs, validation=False)
        self.validation = self._spawn(
            validation_payloads, len(training_payloads), initial_knobs, validation=True
        )
        if validation_payloads:
            covered = {(m.file_idx, m.range_spec) for m in self.validation.metadata}
            LOGGER.info(
                "Validation setup: payloads=%d, covered_ranges=%d, tracks=%d",
                len(validation_payloads),
                len(covered),
                sum(self.validation.particle_counts),
            )

    def build_payloads(
        self, tracks: dict[int, FileTracks]
    ) -> tuple[list[WorkerPayload], list[WorkerPayload]]:
        """Weighted training and held-out validation payloads from ``tracks``.

        The two sets come from disjoint turns. Weights are normalised across the
        *combined* set so training and validation losses live on the same scale.
        """
        training = self.create_worker_payloads(tracks, self.turn_batches)
        validation = (
            self.create_worker_payloads(tracks, self.validation_turn_batches)
            if self.validation_turn_batches
            else []
        )
        self.payload_builder.attach_global_weights(
            training + validation, optimise_momenta=self.simulation_config.optimise_momenta
        )
        return training, validation

    def create_worker_payloads(
        self, tracks: dict[int, FileTracks], turn_batches: list[list[int]]
    ) -> list[WorkerPayload]:
        """One payload per (worker plan, turn batch); each batch belongs to one file."""
        payloads: list[WorkerPayload] = []
        for batch_idx, turn_batch in enumerate(turn_batches):
            if not turn_batch:
                raise ValueError(f"Empty turn batch {batch_idx}")
            for plan in self.worker_plans[self.file_turn_map[turn_batch[0]]]:
                data = self.payload_builder.make_tracking_data(
                    turn_batch=turn_batch,
                    file_turn_map=self.file_turn_map,
                    plan=plan,
                    machine_deltaps=self.machine_deltaps,
                    tracks=tracks,
                    n_run_turns=self.simulation_config.n_run_turns,
                )
                payloads.append((data, self.setup_helper.make_worker_config(plan), plan))

        self._log_file_usage(
            payloads,
            frozenset(self.file_turn_map[turn] for batch in turn_batches for turn in batch),
        )
        return payloads

    def _log_file_usage(self, payloads: list[WorkerPayload], files_covered: frozenset[int]) -> None:
        """Log measurement-file usage and check that at least one worker exists."""
        num_files = len(self.setup_helper.interface_options_per_file)
        file_usage = Counter(plan.file_idx for _, _, plan in payloads)
        LOGGER.info(
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
    ) -> WorkerPool[WorkerObservationPlan]:
        """Start one worker per payload, send it the initial knobs and record its metadata."""
        pool: WorkerPool[WorkerObservationPlan] = WorkerPool()
        for worker_id, (data, config, plan) in enumerate(payloads, start=first_id):
            pool.spawn(
                TrackingWorker, worker_id, data, config, self.simulation_config, validation
            ).send(Start(initial_knobs))
            pool.metadata.append(plan)
            pool.particle_counts.append(len(data.init_coords))
            LOGGER.debug(
                "%s worker %d: file=%d, range=%s/%s, sdir=%d, kick_plane=%s, observed_bpms=%d, tracks=%d",
                "Val" if validation else "Trn",
                worker_id,
                plan.file_idx,
                config.tracking_start_bpm,
                config.tracking_end_bpm,
                config.sdir,
                config.kick_plane,
                len(plan.bpm_names),
                len(data.init_coords),
            )
        return pool

    def training_losses(self, knobs: dict[str, float]) -> list[float]:
        """Each training worker's loss per BPM point at ``knobs``."""
        channels = self.training.channels
        channels.send_all(Diagnose(knobs))
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
            [data.init_coords[:, :4] for data, _config, _file_idx in training + validation]
        ).astype(np.float64)

    def validation_loss(self, knobs: dict[str, float]) -> float | None:
        """The held-out loss at ``knobs``, or ``None`` without validation workers.

        The validation workers track turns that were removed from training, so this
        is a genuine out-of-sample loss; with ``None`` the caller falls back to the
        training loss.
        """
        if not self.validation:
            return None

        channels = self.validation.channels
        channels.send_all(Validate(knobs))
        results = channels.recv_all()
        losses: list[float] = []
        for result in results:
            if not isinstance(result, LossReply):
                raise RuntimeError(f"Unexpected validation reply from worker: {result!r}")
            losses.append(float(result.loss))

        # Each validation worker already reports a per-turn, per-BPM-point loss, so
        # combine them with an unweighted mean over workers -- the same reduction the
        # training loop uses. This keeps the validation number on the training scale.
        return float(np.mean(np.asarray(losses, dtype=np.float64)))

    def set_training_knobs(self, knobs: dict[str, float]) -> None:
        """Load ``knobs`` into every training worker, e.g. the best knobs before the uncertainty."""
        self.training.channels.ack_all(SetKnobs(knobs))

    def stop_and_collect_uncertainty(
        self, n_knobs: int, propagate_uncertainty: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stop every worker and return the normal matrix ``A`` and noise ``B`` of the training workers."""
        LOGGER.info("Terminating workers...")
        normal, noise = drain_uncertainty(
            self.training, n_knobs, propagate_uncertainty, self.simulation_config.n_run_turns
        )
        self.validation.stop()
        return normal, noise

    def terminate(self) -> None:
        """Kill every worker without waiting, for aborting after an error or interrupt.

        Workers may be stuck in a failed simulation and not respond to a stop
        message, so SIGTERM is sent and the processes are reaped.
        """
        LOGGER.info("Terminating workers...")
        self.training.kill()
        self.validation.kill()
