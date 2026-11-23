"""What the workers and the parent need from a model backend.

A backend module (``accelmo.backend.<name>``) provides :class:`TrackingEngine` and
:class:`ClosedOrbitEngine` implementations plus the parent-side :func:`open_model` and
:func:`closed_orbit_table`; :func:`accelmo.backend.get_backend` picks it from
``Accelerator.backend``. Only automatic differentiation computes derivatives.
MAD-NG is the production backend, xtrack the reference for other codes; the contract
and the acceptance tests are in ``docs/backends.rst``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from accelmo.fitting.worker import WorkerConfig
    from accelmo.machine.accelerators import Accelerator
    from accelmo.machine.machine_state import MachineState


@dataclass(frozen=True)
class TrackGeometry:
    """The model segment one tracking worker observes and tracks, fixed when its engine is built."""

    start_bpm: str
    end_bpm: str
    #: Marker supplying the start coordinates (kicker, AC dipole); the full ring is tracked.
    initial_condition_marker: str | None = None
    sdir: int = 1
    cycle_marker: str | None = None
    anchor_mode: str | None = None
    anchor_sources: tuple[str, ...] = ()
    #: Turns tracked from each start.
    n_turns: int = 1

    @property
    def init_point(self) -> str:
        """Where the start coordinates are read: the marker, else the first BPM in tracking order."""
        if self.initial_condition_marker is not None:
            return self.initial_condition_marker
        return self.start_bpm if self.sdir > 0 else self.end_bpm

    @property
    def bpm_range(self) -> str:
        """Observed BPM range ``first/last``."""
        return f"{self.start_bpm}/{self.end_bpm}"

    @property
    def tracking_range(self) -> str | None:
        """Tracked range in tracking order; ``None`` tracks the full cycled sequence.

        A marker start tracks the whole ring every turn, since a named range would treat
        elements outside it as drifts.
        """
        if self.initial_condition_marker is not None:
            return None
        if self.sdir < 0:
            return f"{self.end_bpm}/{self.start_bpm}"
        return self.bpm_range

    @property
    def cycle_target(self) -> str:
        """Element the sequence is cycled to before tracking.

        The initial-condition marker or where the measured turn increment starts: the
        tracking end for a backward range, where the start coordinates are taken.
        Validation workers cycle identically or they track a different path.
        """
        init_bpm = self.start_bpm if self.sdir > 0 else self.end_bpm
        return self.cycle_marker or self.initial_condition_marker or init_bpm


@dataclass(frozen=True)
class TrackRequest:
    """What one tracking worker tracks.

    ``turn_offset`` is the start turn minus the file's first turn (AC-kicker phase) and
    ``file_index`` the measurement file of every particle.
    """

    geometry: TrackGeometry
    observables: tuple[str, ...]
    batch_size: int
    num_batches: int
    validation: bool
    file_index: int
    turn_offset: int


@dataclass
class TrackResult:
    """One tracked batch.

    ``obs[o]`` is ``(P, N)``; ``d_dk[o]`` is ``(P, K, N)`` with ``K`` the knobs (then
    ``pt`` when the energy is fitted); ``d_dc0[o]`` is ``(P, 4, N)``, the derivatives by
    the start x, px, y, py. Validation engines fill ``obs`` only.
    """

    obs: dict[str, np.ndarray]
    d_dk: dict[str, np.ndarray] | None = None
    d_dc0: dict[str, np.ndarray] | None = None
    n_lost: int = 0
    n_total: int = 0


@dataclass
class ClosedSolution:
    """Closed orbit (and optics) at the BPMs in model order.

    ``values`` is ``(n_observables, n_bpms)`` and ``jacobian`` ``(n_observables, n_bpms,
    n_knobs)`` (``None`` from a ``jacobian=False`` solve), rows in the requested observable order.
    """

    names: list[str]
    values: np.ndarray
    jacobian: np.ndarray | None


class TrackingEngine(Protocol):
    """Tracks batches of particles with knob (and start-coordinate) derivatives.

    ``knob_names`` are the fitted knobs, ``pt`` last when the energy is fitted.
    """

    knob_names: list[str]
    nbpms: int

    def __init__(
        self,
        config: WorkerConfig,
        plan: TrackRequest,
        init_knobs: Mapping[str, float],
        *,
        worker_id: int,
        logfile: Path | None = None,
    ) -> None:
        """Load the model for ``config`` and prepare to track ``plan`` from ``init_knobs``."""

    def set_knobs(self, knobs: Mapping[str, float]) -> None:
        """Set the fitted knobs (and ``pt`` when the energy is fitted) to ``knobs``."""

    def load_particles(
        self, init_coords: list[list[list[float]]], init_pts: list[list[float]]
    ) -> None:
        """Set the start coordinates and ``pt`` offsets, both indexed ``[batch][particle]``."""

    def set_pts(self, pt: float, first_batch: int, last_batch: int) -> None:
        """Set ``pt`` of every particle in batches ``first..last`` (0-based) to ``pt`` plus its own offset."""

    def set_start_coords(
        self, x: np.ndarray, px: np.ndarray, y: np.ndarray, py: np.ndarray
    ) -> None:
        """Replace the start ``x, px, y, py`` of every particle, in batch order; ``pt`` is kept."""

    def preflight(self) -> dict[str, int | bool]:
        """One-particle dry run: ``{"observed", "expected", "lost"}``."""

    def track(self, batch: int, *, start_derivatives: bool) -> TrackResult:
        """Track batch ``batch`` (0-based); ``start_derivatives`` also fills ``d_dc0``."""

    def close(self) -> None:
        """Release the model."""


class ClosedOrbitEngine(Protocol):
    """Solves the closed orbit (and twiss) with knob derivatives."""

    knob_names: list[str]
    #: ``pt`` is fitted too: its derivative is the Jacobian column after the knobs.
    fit_pt: bool

    def __init__(
        self,
        config: WorkerConfig,
        init_knobs: Mapping[str, float],
        *,
        optics_names: Sequence[str],
        orbit_coords: Sequence[str],
        worker_id: int,
        logfile: Path | None = None,
        extra_knobs: Sequence[str] | None = None,
    ) -> None:
        """Load the model; ``solve`` returns ``optics_names`` and ``orbit_coords`` rows.

        ``extra_knobs`` (calibration fits; ``madng`` only) are zero-valued knobs after the magnet
        knobs, and ``pt`` is then never fitted.
        """

    def set_knobs(self, knobs: Mapping[str, float]) -> None:
        """Set the fitted knobs to ``knobs``."""

    def get_vars(self, names: Sequence[str]) -> dict[str, float]:
        """Current values of the global variables ``names`` (machine state, e.g. corrector kicks)."""

    def set_vars(self, values: Mapping[str, float]) -> None:
        """Set global variables or element attributes (a machine state)."""

    def set_pt(self, pt: float) -> None:
        """Set the momentum offset ``pt`` of the closed-orbit search."""

    def set_warm_start(self, key: str) -> None:
        """Start the next closed-orbit search from the last orbit solved under ``key``."""

    def solve(self, observables: Sequence[str], *, jacobian: bool = True) -> ClosedSolution | None:
        """``None`` when the closed orbit is lost.

        ``jacobian=False`` (loss-only evaluation, orbit observables only) leaves
        ``ClosedSolution.jacobian`` ``None`` and may skip the derivatives entirely.
        """

    def close(self) -> None:
        """Release the model."""


class Model(Protocol):
    """Parent-side model: knob discovery and BPM lists (no derivatives)."""

    knob_names: list[str]
    all_bpms: list[str]
    bpms_in_range: list[str]

    def cycle_to_start(self, marker: str) -> None:
        """Make ``marker`` the first element of the sequence."""

    def recount_bpms(self) -> None:
        """Refresh ``all_bpms`` and ``bpms_in_range`` after a cycle."""

    def apply_initial_model_values(self, values: Mapping[str, float]) -> None:
        """Set knobs and fixed strengths to ``values`` before the fit starts."""

    def knob_values(self) -> np.ndarray:
        """Current values of ``knob_names``, in that order."""

    def close(self) -> None:
        """Release the model."""


class Backend(Protocol):
    """The module ``accelmo.backend.<name>``."""

    TrackingEngine: type[TrackingEngine]
    ClosedOrbitEngine: type[ClosedOrbitEngine]

    def open_model(
        self,
        accelerator: Accelerator,
        *,
        magnet_range: str,
        machine_state: MachineState | None = None,
        b2_errors: Path | None = None,
        bad_bpms: list[str] | None = None,
        debug: bool = False,
        mad_logfile: Path | None = None,
        anchor_mode: str | None = None,
        anchor_markers: Sequence[str] = (),
    ) -> Model:
        """Load ``accelerator``, create its knobs over ``magnet_range`` and return the model."""

    def closed_orbit_table(
        self,
        accelerator: Accelerator,
        magnet_strengths: Mapping[str, float] | None,
        pt: float,
        machine_state: MachineState | None,
        columns: Sequence[str],
    ) -> pd.DataFrame:
        """Closed-orbit ``columns`` at the BPMs for these strengths, ``pt`` and state."""
