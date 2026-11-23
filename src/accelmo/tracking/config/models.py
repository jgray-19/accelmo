"""Configuration dataclasses for fitter initialization.

These dataclasses group related parameters to reduce the number of
individual arguments passed to fitter constructors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from accelmo.machine.machine_state import MachineState


@dataclass
class MeasurementDetails:
    """Per-measurement model setup and momentum metadata.

    ``machine_state`` holds the model globals (tune knobs, corrector kicks, ...) the
    machine had during this measurement; ``b2_errors`` an LHC dipole b2 error table,
    which needs the tune knobs in ``machine_state``. The bunch structure is read from
    the ``bunch_number`` column of the measurement parquet, so it is not configured here.

    ``first_bpm`` names the BPM each recorded turn begins at, used to cycle the
    measurement data to the boundary it was generated from. Leave it ``None`` to
    use the file's own first recorded BPM; set it when that row order is
    unreliable (for example ACD marker rows written after the BPMs).
    """

    machine_state: MachineState | None = None
    b2_errors: Path | None = None
    machine_deltap: float = 0.0
    first_bpm: str | None = None


@dataclass
class MeasurementConfig:
    """Maps each measurement file to its :class:`MeasurementDetails`."""

    measurements: dict[Path, MeasurementDetails]

    def __post_init__(self) -> None:
        if not self.measurements:
            raise ValueError("MeasurementConfig requires at least one measurement file")

    @property
    def files(self) -> list[Path]:
        """Return the measurement files in insertion order."""
        return list(self.measurements.keys())

    @property
    def details(self) -> list[MeasurementDetails]:
        """Return the per-file details in measurement-file order."""
        return list(self.measurements.values())


@dataclass
class CheckpointConfig:
    """Checkpoint save/restore behaviour for optimisation runs."""

    checkpoint_path: Path
    checkpoint_every_n_epochs: int = 0
    restore_from_checkpoint: bool = False
