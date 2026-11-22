"""What the workers and the parent need from a model backend.

A backend module (``accelmo.backend.<name>``) provides :class:`TrackingEngine` and
:class:`ClosedOrbitEngine` implementations plus the parent-side :func:`open_model` and
:func:`closed_orbit_table`; :func:`accelmo.backend.get_backend` picks it from
``Accelerator.backend``. Only automatic differentiation computes derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from contextlib import AbstractContextManager
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from accelmo.fitting.worker import WorkerConfig
    from accelmo.machine.accelerators import Accelerator


@dataclass(frozen=True)
class TrackingPlan:
    """What one tracking worker tracks.

    ``tracking_range`` is ``(first, last)`` in tracking order, or ``None`` for the full
    (cycled) ring. ``turn_offset`` is the start turn minus the file's first turn
    (AC-kicker phase) and ``file_index`` the measurement file of every particle.
    """

    observables: tuple[str, ...]
    batch_size: int
    num_batches: int
    tracking_range: str | None
    n_run_turns: int
    start_on_first_turn: bool
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
    n_knobs)``, rows in the requested observable order.
    """

    names: list[str]
    values: np.ndarray
    jacobian: np.ndarray


class TrackingEngine(Protocol):
    """Tracks batches of particles with knob (and start-coordinate) derivatives.

    ``knob_names`` are the fitted knobs, ``pt`` last when the energy is fitted.
    """

    knob_names: list[str]
    nbpms: int

    def __init__(
        self,
        config: WorkerConfig,
        plan: TrackingPlan,
        init_knobs: Mapping[str, float],
        *,
        worker_id: int,
        logfile: Path | None = None,
    ) -> None: ...

    def set_knobs(self, knobs: Mapping[str, float]) -> None: ...

    def load_particles(
        self, init_coords: list[list[list[float]]], init_pts: list[list[float]]
    ) -> None: ...

    def set_pts(self, pt: float, first_batch: int, last_batch: int) -> None:
        """Set ``pt`` of every particle in batches ``first..last`` (0-based) to ``pt`` plus its own offset."""

    def set_start_coords(
        self, x: np.ndarray, px: np.ndarray, y: np.ndarray, py: np.ndarray
    ) -> None: ...

    def preflight(self) -> dict[str, int | bool]:
        """One-particle dry run: ``{"observed", "expected", "lost"}``."""

    def track(self, batch: int, *, start_derivatives: bool) -> TrackResult: ...

    def close(self) -> None: ...


class ClosedOrbitEngine(Protocol):
    """Solves the closed orbit (and twiss) with knob derivatives."""

    knob_names: list[str]
    n_knobs: int
    fit_pt: bool

    def __init__(
        self,
        config: WorkerConfig,
        init_knobs: Mapping[str, float],
        *,
        optics_names: Sequence[str],
        orbit_coords: Sequence[str],
        extra_knobs: Sequence[str] | None,
        worker_id: int,
        logfile: Path | None = None,
    ) -> None:
        """``extra_knobs``: zero-valued knobs after the magnet knobs (calibrated fit, never fits ``pt``)."""

    def set_knobs(self, knobs: Mapping[str, float]) -> None: ...

    def read_state(self, names: list[str]) -> dict[str, float]: ...

    def assign_state(self, state: Mapping[str, float]) -> None: ...

    def scale_globals(self, scaled: Sequence[tuple[str, float, float, str]]) -> None:
        """Set each global ``name`` to ``kick * (gain + knob)``, ``knob`` a zero-valued extra knob."""

    def set_pt(self, pt: float) -> None: ...

    def set_warm_start(self, key: str) -> None:
        """Start the next closed-orbit search from the last orbit solved under ``key``."""

    def solve(self, observables: Sequence[str]) -> ClosedSolution | None:
        """``None`` when the closed orbit is lost."""

    def plain_knobs(self) -> AbstractContextManager[None]:
        """Knobs as plain numbers inside the block (no derivatives)."""

    def solve_orbit_plain(self, n_bpms: int) -> np.ndarray | None:
        """Orbit values ``(n_coords, n_bpms)`` only; inside :meth:`plain_knobs`."""

    def close(self) -> None: ...


class Model(Protocol):
    """Parent-side model: knob discovery and BPM lists (no derivatives)."""

    knob_names: list[str]
    all_bpms: list[str]
    bpms_in_range: list[str]

    def cycle_to_start(self, marker: str) -> None: ...

    def recount_bpms(self) -> None:
        """Refresh ``all_bpms`` and ``bpms_in_range`` after a cycle."""

    def apply_initial_model_values(self, values: Mapping[str, float]) -> None: ...

    def knob_values(self) -> np.ndarray: ...

    def close(self) -> None: ...


class Backend(Protocol):
    """The module ``accelmo.backend.<name>``."""

    TrackingEngine: type[TrackingEngine]
    ClosedOrbitEngine: type[ClosedOrbitEngine]

    def open_model(
        self,
        accelerator: Accelerator,
        *,
        magnet_range: str,
        machine_state: object = None,
        b2_errors: Path | None = None,
        bad_bpms: list[str] | None = None,
        debug: bool = False,
        mad_logfile: Path | None = None,
        anchor_mode: str | None = None,
        anchor_markers: Sequence[str] = (),
    ) -> Model: ...

    def closed_orbit_table(
        self,
        accelerator: Accelerator,
        magnet_strengths: Mapping[str, float] | None,
        pt: float,
        machine_state: object,
        columns: Sequence[str],
    ) -> pd.DataFrame: ...
