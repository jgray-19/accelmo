"""Abstract base class for all worker process types.

:meth:`AbstractWorker.run` owns the process lifecycle and the message loop of
:mod:`accelmo.fitting.protocol`; subclasses build their backend engine
(:mod:`accelmo.backend.protocol`) and answer :class:`~accelmo.fitting.protocol.Evaluate` and
:class:`~accelmo.fitting.protocol.Command` messages.
"""

from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from multiprocessing import Process
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from threadpoolctl import threadpool_limits

from accelmo.backend import get_backend
from accelmo.fitting.protocol import Command, ErrorReply, Evaluate, Stop

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from pathlib import Path

    from accelmo.config import SimulationConfig
    from accelmo.fitting.protocol import Ack, GradReply, LossReply
    from accelmo.machine.accelerators import Accelerator

LOGGER = logging.getLogger(__name__)

# Type variable for worker data type
WorkerDataType = TypeVar("WorkerDataType")


class KickPlane(StrEnum):
    """Kick-plane options for worker routing and payload selection."""

    X = "x"
    Y = "y"
    XY = "xy"


@dataclass
class WorkerConfig:
    """Configuration shared by all worker processes.

    The accelerator object bundles machine-specific setup, while the remaining
    fields describe the local BPM range, tracking direction, and optional input
    files needed by the worker.

    """

    accelerator: Accelerator
    tracking_start_bpm: str
    tracking_end_bpm: str
    magnet_range: str
    # Per-measurement keyword arguments forwarded to the MAD-NG interface, e.g.
    # machine_state, b2_errors.
    interface_options: dict[str, Any] = field(default_factory=dict)
    observation_range_start_bpm: str | None = None
    initial_condition_marker: str | None = None
    # Whether to cycle the sequence so tracking starts at this worker's init
    # marker. Closed-twiss workers fit the whole ring from ``$start`` and set it
    # False; every tracking plan cycles.
    cycle_sequence: bool = True
    sdir: int = 1
    kick_plane: KickPlane = KickPlane.XY
    bad_bpms: list[str] | None = None
    debug: bool = False
    mad_logfile: Path | None = None
    python_logfile: Path | None = None
    tracking_anchor_mode: str | None = None
    tracking_anchor_sources: list[str] | None = None
    observed_tracking_anchor_markers: list[str] | None = None
    cycle_marker: str | None = None

    @property
    def bpm_range(self) -> str:
        """Observed BPM range ``first/last``."""
        start = self.observation_range_start_bpm or self.tracking_start_bpm
        return f"{start}/{self.tracking_end_bpm}"

    @property
    def tracking_range(self) -> str | None:
        """Tracked range in tracking order; ``None`` tracks the full cycled sequence.

        Kicker mode (a marker supplies the start) tracks the whole ring for every turn,
        since a named range would treat elements outside it as drifts.
        """
        if self.initial_condition_marker is not None:
            return None
        if self.sdir < 0:
            return f"{self.tracking_end_bpm}/{self.tracking_start_bpm}"
        return self.bpm_range

    @property
    def cycle_target(self) -> str | None:
        """Element the sequence is cycled to before tracking, or None.

        The initial-condition marker (kicker/ACD) or the point where this worker's
        measured turn increment starts. For backward ranges that is the tracking end,
        because the payload initial coordinates are taken there. Validation workers
        must cycle identically or they track a different path.
        """
        if not self.cycle_sequence:
            return None
        tracking_init_bpm = self.tracking_start_bpm if self.sdir > 0 else self.tracking_end_bpm
        return self.cycle_marker or self.initial_condition_marker or tracking_init_bpm


class AbstractWorker(Process, ABC, Generic[WorkerDataType]):
    """Abstract base class for all worker process implementations.

    Subclasses implement :meth:`prepare_data`, :meth:`on_start` (which builds
    ``self.engine``) and :meth:`evaluate`, and override :meth:`on_start`, :meth:`handle_command`,
    :meth:`on_stop` and :meth:`close` where they need to.

    Type Parameters:
        WorkerDataType: The type of data structure this worker uses
    """

    def __init__(
        self,
        conn: Connection,
        worker_id: int,
        data: WorkerDataType,
        config: WorkerConfig,
        simulation_config: SimulationConfig,
    ) -> None:
        """Initialize the worker process.

        Args:
            conn: Pipe connection for communicating with main process
            worker_id: Unique identifier for this worker
            data: Worker-specific data (tracking or optics)
            config: Configuration parameters
            simulation_config: Simulation configuration settings
        """
        super().__init__()
        self.worker_id = worker_id
        self.conn = conn
        self.config = config
        self.simulation_config = simulation_config
        LOGGER.debug(
            "Initializing worker %d for BPM range %s -> %s",
            worker_id,
            config.tracking_start_bpm,
            config.tracking_end_bpm,
        )

        # Let subclasses process their specific data
        self.prepare_data(data)

    @abstractmethod
    def prepare_data(self, data: WorkerDataType) -> None:
        """Process and prepare worker-specific data.

        This method should extract relevant data from the input structure,
        compute weights, split into batches, etc.

        Args:
            data: Worker-specific data structure
        """

    @abstractmethod
    def on_start(self, knobs: dict[str, float]) -> None:
        """Build ``self.engine`` from the :class:`~accelmo.fitting.protocol.Start` knobs."""

    @abstractmethod
    def evaluate(self, message: Evaluate) -> GradReply:
        """Answer one :class:`~accelmo.fitting.protocol.Evaluate` message."""

    def handle_command(self, command: Command) -> Ack | LossReply:
        """Answer one :class:`~accelmo.fitting.protocol.Command` message."""
        raise ValueError(f"Worker {self.worker_id}: unsupported command {command!r}")

    def on_stop(self, failed: bool) -> None:
        """Called once the message loop ends, before the engine is closed."""

    def close(self) -> None:
        """Release resources other than the engine."""

    @property
    def backend(self):
        """The backend module of this worker's accelerator."""
        return get_backend(self.config.accelerator.backend)

    @property
    def engine_logfile(self) -> Path | None:
        """This worker's backend log file."""
        return self._resolve_per_worker_logfile(self.config.mad_logfile)

    def send_reply(self, reply: object) -> None:
        """Send one reply to the parent."""
        self.conn.send(reply)

    def run(self) -> None:
        """Run the worker: :class:`Start`, then evaluations and commands until :class:`Stop`."""
        self.engine = None
        try:
            self.configure_python_worker_logging()
            self.configure_worker_threads()
            message = self.conn.recv()
            if isinstance(message, Stop):
                return
            self.on_start(message.knobs)
            failed = False
            while True:
                message = self.conn.recv()
                if isinstance(message, Stop):
                    LOGGER.debug("Worker %s: received termination signal", self.worker_id)
                    break
                if isinstance(message, Command):
                    self.send_reply(self.handle_command(message))
                    continue
                if not isinstance(message, Evaluate):
                    raise ValueError(f"Worker {self.worker_id}: unexpected message {message!r}")
                try:
                    self.send_reply(self.evaluate(message))
                except Exception as exc:  # noqa: BLE001
                    self.send_error_payload(exc, phase="computation")
                    failed = True
                    break
            self.on_stop(failed)
        except Exception as exc:  # noqa: BLE001
            self.send_error_payload(exc, phase="startup")
        finally:
            LOGGER.debug("Worker %s: terminating", self.worker_id)
            self.close()
            if self.engine is not None:
                self.engine.close()

    def send_error_payload(self, exc: BaseException, *, phase: str) -> None:
        """Best-effort send of the failure to the parent."""
        reply = ErrorReply(
            worker_id=self.worker_id,
            phase=phase,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        LOGGER.error("Worker %s failed during %s: %s", self.worker_id, phase, reply.error)
        try:
            self.conn.send(reply)
        except (BrokenPipeError, EOFError, OSError):
            LOGGER.exception(
                "Worker %s could not send error payload to parent during %s",
                self.worker_id,
                phase,
            )

    def _resolve_per_worker_logfile(self, logfile_path):
        """Return a per-worker logfile path derived from a base logfile path."""
        if logfile_path is None:
            return None

        if logfile_path.suffix:
            return logfile_path.with_name(
                f"{logfile_path.stem}_worker_{self.worker_id}{logfile_path.suffix}"
            )
        return logfile_path.with_name(f"{logfile_path.name}_worker_{self.worker_id}")

    def configure_worker_threads(self) -> None:
        """Limit this worker process's BLAS threads (``SimulationConfig.worker_blas_threads``).

        Must run inside the worker process: the pool is already initialised in the forked
        parent, so the ``OPENBLAS_NUM_THREADS`` environment variable would be too late.
        """
        if self.simulation_config.worker_blas_threads is not None:
            threadpool_limits(limits=self.simulation_config.worker_blas_threads)

    def configure_python_worker_logging(self) -> None:
        """Attach a file handler so worker Python logs land in the worker logfile."""
        worker_logfile = self._resolve_per_worker_logfile(
            self.config.python_logfile or self.config.mad_logfile
        )
        if worker_logfile is None:
            return

        worker_logfile.parent.mkdir(parents=True, exist_ok=True)
        root_logger = logging.getLogger()
        level = self.simulation_config.worker_logging_level
        root_logger.setLevel(level)

        if any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == str(worker_logfile)
            for handler in root_logger.handlers
        ):
            return

        file_handler = logging.FileHandler(worker_logfile, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "PYTHON: %(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

        LOGGER.debug(
            "Worker %s Python logging attached to %s",
            self.worker_id,
            worker_logfile,
        )
