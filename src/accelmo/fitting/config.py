"""Configuration shared by every fitter: the lattice range and the output settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SequenceConfig:
    """Configuration for the sequence segment used during optimisation.

    The fields define the magnet range to expose to MAD-NG and any BPMs that
    should be ignored. Where measurement data starts (the BPM each recorded turn
    begins at) is a property of the measurement, not the sequence, so it lives on
    :class:`MeasurementDetails`.
    """

    magnet_range: str = "$start/$end"
    bad_bpms: list[str] | None = None


@dataclass
class OutputConfig:
    """Output and logging behaviour for optimisation runs.

    Attributes:
        write_run_log: Whether to write a run log (see :mod:`accelmo.runlog`).
        include_uncertainty: Whether to compute uncertainties. Disabling this
            skips worker-side Hessian estimation for faster execution.
        run_log_root: Directory under which each fit writes its run log.
        run_metadata: Free-form, JSON-serialisable description stored with the run
            log, e.g. what is needed to rebuild the model for replaying it.
        mad_logfile: Optional MAD log file path.
        python_logfile: Optional Python worker log file path.
    """

    write_run_log: bool = True
    include_uncertainty: bool = True
    run_log_root: Path = field(default_factory=lambda: Path("runs"))
    run_metadata: dict[str, Any] = field(default_factory=dict)
    mad_logfile: Path | None = None
    python_logfile: Path | None = None
