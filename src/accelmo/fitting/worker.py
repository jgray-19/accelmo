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
from dataclasses import dataclass
from multiprocessing import Process
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from threadpoolctl import threadpool_limits

from accelmo.backend import get_backend
from accelmo.fitting.protocol import Command, ErrorReply, Evaluate, Stop
from accelmo.fitting.worker_logging import attach_in_worker

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from pathlib import Path

    from accelmo.config import WorkerRuntime
    from accelmo.fitting.protocol import Ack, GradReply, LossReply
    from accelmo.machine.accelerators import Accelerator
    from accelmo.machine.machine_state import MachineState

LOGGER = logging.getLogger(__name__)

# Type variable for worker data type
WorkerDataType = TypeVar("WorkerDataType")
ConfigType = TypeVar("ConfigType", bound="WorkerConfig")


@dataclass
class WorkerConfig:
    """Configuration shared by all worker processes.

    The accelerator bundles machine-specific setup; the other fields are the model
    options and log files. Tracking workers add their geometry
    (:class:`~accelmo.tracking.dispatch.setup.TrackingWorkerConfig`).
    """

    accelerator: Accelerator
    magnet_range: str
    machine_state: MachineState | None = None
    b2_errors: Path | None = None
    bad_bpms: list[str] | None = None
    debug: bool = False
    mad_logfile: Path | None = None
    python_logfile: Path | None = None


class AbstractWorker(Process, ABC, Generic[WorkerDataType, ConfigType]):
    """Abstract base class for all worker process implementations.

    Subclasses implement :meth:`prepare_data`, :meth:`on_start` (which builds
    ``self.engine``) and :meth:`evaluate`, and override :meth:`on_start`, :meth:`handle_command`,
    :meth:`on_stop` and :meth:`close` where they need to.

    Type Parameters:
        WorkerDataType: The type of data structure this worker uses
    """

    engine: Any

    def __init__(
        self,
        conn: Connection,
        worker_id: int,
        data: WorkerDataType,
        config: ConfigType,
        runtime: WorkerRuntime,
    ) -> None:
        """Initialize the worker process.

        Args:
            conn: Pipe connection for communicating with main process
            worker_id: Unique identifier for this worker
            data: Worker-specific data (tracking or optics)
            config: Configuration parameters
            runtime: Worker process settings
        """
        super().__init__()
        self.worker_id = worker_id
        self.conn = conn
        self.config: ConfigType = config
        self.runtime = runtime
        LOGGER.debug("Initializing worker %d", worker_id)

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
        """Limit this worker process's BLAS threads (``WorkerRuntime.blas_threads``).

        Must run inside the worker process: the pool is already initialised in the forked
        parent, so the ``OPENBLAS_NUM_THREADS`` environment variable would be too late.
        """
        if self.runtime.blas_threads is not None:
            threadpool_limits(limits=self.runtime.blas_threads)

    def configure_python_worker_logging(self) -> None:
        """Log this worker to its own file; WARNING and above also reach the parent."""
        attach_in_worker(
            self._resolve_per_worker_logfile(self.config.python_logfile),
            self.runtime.logging_level,
        )
