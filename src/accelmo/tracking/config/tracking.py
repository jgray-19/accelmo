"""Tracking-mode planning for arc-by-arc, kicker- and AC-dipole-excited runs.

This module owns everything mode-specific about a training run: one
:class:`TrackingPlan` per mode fixes the simulation-config entries and start points
that are not free choices for the mode, and expands BPM points into worker ranges.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from accelmo.config import TrackingDataConfig
    from accelmo.tracking.ranges import BpmRanges


def extract_bpm_range_names(
    all_bpms: list[str],
    start_bpm: str,
    end_bpm: str,
    sdir: int,
) -> list[str]:
    """Extract BPM names between start and end BPMs, handling circular wrapping.

    Args:
        all_bpms: List of all BPM names
        start_bpm: Starting BPM name
        end_bpm: Ending BPM name
        sdir: Direction (1 for forward, -1 for reverse)

    Returns:
        List of BPM names in the range
    """
    if start_bpm not in all_bpms:
        raise ValueError(f"Start BPM '{start_bpm}' not found in BPM list")
    start_pos = all_bpms.index(start_bpm)

    if end_bpm not in all_bpms:
        raise ValueError(f"End BPM '{end_bpm}' not found in BPM list")
    end_pos = all_bpms.index(end_bpm) + 1

    if end_pos <= start_pos:
        # Circular wrapping
        extracted = all_bpms[start_pos:] + all_bpms[:end_pos]
    else:
        extracted = all_bpms[start_pos:end_pos]
    # Reverse for negative direction
    if sdir == -1:
        extracted = extracted[::-1]

    return extracted


def create_bpm_range_specs(
    bpm_start_points: list[str],
    bpm_end_points: list[str],
    use_fixed_bpm: bool,
    fixed_start: str | None = None,
    fixed_end: str | None = None,
) -> list[tuple[str, str, int]]:
    """Create BPM range specifications for optimisation workers.

    Args:
        bpm_start_points: List of starting BPM names
        bpm_end_points: List of ending BPM names
        use_fixed_bpm: If True, use fixed BPM pairs; if False, create cartesian product
        fixed_start: Fixed start BPM for backward tracking (only used if use_fixed_bpm=True)
        fixed_end: Fixed end BPM for forward tracking (only used if use_fixed_bpm=True)

    Returns:
        List of (start_bpm, end_bpm, sdir) tuples where sdir is 1 for forward, -1 for reverse
    """
    if use_fixed_bpm:
        logger.warning(
            "Using fixed BPM pairs for optimisation. This will create fewer, more constrained measurement combinations."
        )
        if fixed_start is None or fixed_end is None:
            raise ValueError("fixed_start and fixed_end must be provided when use_fixed_bpm=True")
        # Forward: start -> fixed_end; Backward: fixed_start -> end
        range_specs = [(s, fixed_end, 1) for s in bpm_start_points] + [
            (fixed_start, e, -1) for e in bpm_end_points
        ]
    else:
        # Cartesian product: every start with every end in both directions
        range_specs = [
            (s, e, sdir) for s in bpm_start_points for e in bpm_end_points for sdir in (1, -1)
        ]

    # MAD tracks "A/A" as the single element A: the worker would observe only its own
    # start point and constrain nothing.
    degenerate = sorted({s for s, e, _ in range_specs if s == e})
    if degenerate:
        raise ValueError(
            f"BPM ranges must not start and end at the same BPM: {degenerate}. "
            "Choose start and end points that differ (with use_fixed_bpm, the fixed start "
            "and end BPMs must also not appear among the end and start points)."
        )
    return range_specs


def _boundary_turns_for_track(track_turns: list, margin: int) -> list:
    """Return the boundary turns (first/last ``margin`` entries) for one track."""
    if len(track_turns) <= 2 * margin:
        return track_turns
    return track_turns[:margin] + track_turns[-margin:]


def _bpm_behind(all_bpms: list[str], bpm: str) -> str:
    """Return the BPM immediately behind ``bpm`` in ring order."""
    if bpm not in all_bpms:
        raise ValueError(f"Start BPM '{bpm}' not found in model BPM list")
    return all_bpms[all_bpms.index(bpm) - 1]


def _window_starts(
    bunch_turns_by_file: dict[int, dict[int, list[int]]], window: int
) -> tuple[dict[int, set[int]], list[int]]:
    """Start turns ``first, first + window, ...`` of every file whose track fits in the file.

    One bunch per file: the AC-kicker phase is counted from the file's first turn.
    """
    starts = []
    for file_idx, bunches in bunch_turns_by_file.items():
        if len(bunches) != 1:
            raise ValueError(f"File {file_idx} has {len(bunches)} bunches; ACD sinusoid needs one")
        (turns,) = bunches.values()
        starts.extend(range(min(turns), max(turns) - window + 1, window))
    return {file_idx: set() for file_idx in bunch_turns_by_file}, sorted(starts)


def _first_turn_per_bunch(
    bunch_turns_by_file: dict[int, dict[int, list[int]]],
) -> tuple[dict[int, set[int]], list[int]]:
    """One track per bunch, seeded from its first turn; no boundary turns."""
    boundary_turns_by_file = {file_idx: set() for file_idx in bunch_turns_by_file}
    start_turns = [
        min(bunch_turns)
        for bunches in bunch_turns_by_file.values()
        for bunch_turns in bunches.values()
    ]
    return boundary_turns_by_file, sorted(start_turns)


@dataclass(frozen=True)
class WorkerRangeSpec:
    """Logical BPM range assigned to a worker."""

    start_bpm: str
    end_bpm: str
    sdir: int

    @property
    def init_bpm(self) -> str:
        """Return the BPM used to initialise tracking for this direction."""
        return self.start_bpm if self.sdir > 0 else self.end_bpm


class TrackingPlan:
    """Mode-specific tracking policy; the base class is arc-by-arc BPM tracking.

    Subclasses override the class-level flags and the range-planning hooks below.
    """

    enable_validation: ClassVar[bool] = True
    #: Turns tracked from each start; an arc range is one turn.
    turns_per_track: ClassVar[int] = 1
    one_start_per_worker: ClassVar[bool] = False
    tracking_anchor_mode: ClassVar[str | None] = None

    @property
    def init_marker(self) -> str | None:
        """Element supplying tracking initial conditions instead of a range BPM."""
        return None

    @property
    def cycle_marker(self) -> str | None:
        """Optional unobserved element used only for sequence cycling."""
        return None

    @property
    def extra_markers(self) -> tuple[str, ...]:
        """Non-BPM markers that must be kept in measurement data."""
        return ()

    @property
    def start_points(self) -> tuple[str, ...]:
        """Fixed tracking start points of marker-anchored modes; empty means the caller's BPMs."""
        return ()

    @property
    def first_bpm_fallback(self) -> str | None:
        """Per-file fallback for the BPM the measurement turns are recorded from."""
        return None

    def configure(self, tracking_data: TrackingDataConfig) -> TrackingDataConfig:
        """``tracking_data`` with the entries this mode fixes; arc ranges fix none."""
        return tracking_data

    @property
    def tracking_anchor_sources(self) -> tuple[str, ...]:
        """Elements the model prepares for marker-anchored tracking."""
        return self.extra_markers

    def initial_condition_marker(self, range_spec: WorkerRangeSpec) -> str | None:
        """Return the measured marker used for a worker's initial coordinates."""
        del range_spec
        return self.init_marker

    def observed_bpms(self, bpms_in_range: list[str], all_bpms: list[str]) -> list[str]:
        """Return the BPMs compared against measurements."""
        return bpms_in_range

    def select_available_turns(
        self,
        *,
        bunch_turns_by_file: dict[int, dict[int, list[int]]],
        available_turns: list[int],
    ) -> tuple[dict[int, set[int]], list[int]]:
        """Return boundary turns and the filtered list of usable start turns.

        A start turn is unusable when the multi-turn track it seeds would cross a
        bunch boundary. Each bunch's first/last turns are therefore removed, using
        the per-file ``bunch_number`` grouping read from the measurement data.
        """
        margin = self.turns_per_track
        turns_to_remove = set()
        boundary_turns_by_file: dict[int, set[int]] = {}

        for file_idx, bunches in bunch_turns_by_file.items():
            for bunch_turns in bunches.values():
                boundary_turns = _boundary_turns_for_track(sorted(bunch_turns), margin)
                boundary_turns_by_file.setdefault(file_idx, set()).update(boundary_turns)
                turns_to_remove.update(boundary_turns)

        return boundary_turns_by_file, [t for t in available_turns if t not in turns_to_remove]

    def build_range_specs(self, ctx: BpmRanges) -> list[WorkerRangeSpec]:
        """Return the worker range specs for this plan."""
        return [
            WorkerRangeSpec(start_bpm, end_bpm, sdir)
            for start_bpm, end_bpm, sdir in create_bpm_range_specs(
                ctx.start_bpms, ctx.end_bpms, ctx.use_fixed_bpm, ctx.fixed_start, ctx.fixed_end
            )
        ]

    def get_range_bpm_names(
        self,
        *,
        all_bpms: list[str],
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        bad_bpms: list[str] | None,
    ) -> list[str]:
        """Return the BPMs in one logical observation range."""
        bpm_names = extract_bpm_range_names(all_bpms, start_bpm, end_bpm, sdir)
        excluded = set(bad_bpms or [])
        return [bpm for bpm in bpm_names if bpm not in excluded]


@dataclass(frozen=True)
class KickerTrackingPlan(TrackingPlan):
    """Forward-only tracking starting from a kicker initial-condition marker.

    The measured kicker marker supplies the initial coordinates, while the MAD
    sequence is cycled to an unobserved centre marker so the observed BPM order
    begins just after the kicker; only real BPMs are compared.
    """

    kicker_name: str
    #: Turns tracked from the kicker.
    turns_per_track: int

    enable_validation = False
    tracking_anchor_mode = "kicker"

    def __post_init__(self) -> None:
        if self.turns_per_track < 1:
            raise ValueError("turns_per_track must be >= 1")

    def configure(self, tracking_data: TrackingDataConfig) -> TrackingDataConfig:
        """One batch: each bunch gives a single track, so batches cannot split the turns."""
        return dataclasses.replace(tracking_data, num_batches=1)

    @property
    def start_points(self) -> tuple[str, ...]:
        """The kicker marker."""
        return (self.kicker_name,)

    @property
    def first_bpm_fallback(self) -> str:
        """Kicker files are recorded from the kicker marker."""
        return self.kicker_name

    @property
    def init_marker(self) -> str:
        """The kicker marker."""
        return self.kicker_name

    @property
    def extra_markers(self) -> tuple[str, ...]:
        """The kicker marker."""
        return (self.kicker_name,)

    @property
    def tracking_anchor_sources(self) -> tuple[str, ...]:
        """The kicker marker."""
        return (self.kicker_name,)

    @property
    def cycle_marker(self) -> str:
        """Marker at the kicker centre."""
        return f"{self.kicker_name}_centre"

    def observed_bpms(self, bpms_in_range: list[str], all_bpms: list[str]) -> list[str]:
        """Every BPM."""
        return all_bpms

    def select_available_turns(
        self,
        *,
        bunch_turns_by_file: dict[int, dict[int, list[int]]],
        available_turns: list[int],
    ) -> tuple[dict[int, set[int]], list[int]]:
        """Each bunch contributes exactly one track, seeded from its first turn."""
        return _first_turn_per_bunch(bunch_turns_by_file)

    def build_range_specs(self, ctx: BpmRanges) -> list[WorkerRangeSpec]:
        """One forward range from the first BPM round the ring."""
        start = ctx.all_bpms[0]
        return [WorkerRangeSpec(start, _bpm_behind(ctx.all_bpms, start), sdir=1)]


@dataclass(frozen=True)
class _AcdPlan(TrackingPlan):
    """Shared AC-dipole marker naming and MAD preparation mode."""

    acd_name: str

    tracking_anchor_mode = "acd"

    @property
    def acd_after(self) -> str:
        return f"{self.acd_name}_after"

    @property
    def acd_before(self) -> str:
        return f"{self.acd_name}_before"

    def get_range_bpm_names(
        self,
        *,
        all_bpms: list[str],
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        bad_bpms: list[str] | None,
    ) -> list[str]:
        """Observe physical BPMs only; the AC-dipole markers are never compared."""
        return super().get_range_bpm_names(
            all_bpms=all_bpms,
            start_bpm=start_bpm,
            end_bpm=end_bpm,
            sdir=sdir,
            bad_bpms=[*(bad_bpms or []), self.acd_after, self.acd_before],
        )


@dataclass(frozen=True)
class ACDTrackingPlan(_AcdPlan):
    """Bidirectional tracking initialised at AC-dipole markers."""

    @property
    def start_points(self) -> tuple[str, ...]:
        """The ``after`` and ``before`` markers."""
        return (self.acd_after, self.acd_before)

    @property
    def extra_markers(self) -> tuple[str, ...]:
        """The ``before`` and ``after`` markers."""
        return (self.acd_after, self.acd_before)

    def observed_bpms(self, bpms_in_range: list[str], all_bpms: list[str]) -> list[str]:
        """Every BPM."""
        return all_bpms

    def build_range_specs(self, ctx: BpmRanges) -> list[WorkerRangeSpec]:
        """Forward from ``after`` and backward from ``before``."""
        return [
            WorkerRangeSpec(start_bpm=self.acd_after, end_bpm=self.acd_before, sdir=sdir)
            for sdir in (1, -1)
        ]

    def initial_condition_marker(self, range_spec: WorkerRangeSpec) -> str:
        """Initialise each direction from the marker at its tracking start."""
        return range_spec.init_bpm


@dataclass(frozen=True)
class ACDSinusoidTrackingPlan(_AcdPlan):
    """Forward multi-turn tracking from the AC-dipole before-marker through the AC kickers.

    Each file is cut into back-to-back tracks of ``turns_per_track`` turns: one starts at the
    file's first turn and a new one every ``turns_per_track`` turns, each seeded from the
    measured before-marker at its start turn. The kick comes from the model AC kickers
    ``AC_KICKERS`` (PSB ``acd_drive_tunes``).
    """

    #: Turns tracked from each start, and the spacing of the starts.
    turns_per_track: int

    enable_validation = False
    one_start_per_worker = True

    def __post_init__(self) -> None:
        if self.turns_per_track < 1:
            raise ValueError("turns_per_track must be >= 1")

    def configure(self, tracking_data: TrackingDataConfig) -> TrackingDataConfig:
        """One batch: each worker holds one track (its own AC-kicker phase)."""
        return dataclasses.replace(tracking_data, num_batches=1)

    @property
    def start_points(self) -> tuple[str, ...]:
        """The ``before`` marker."""
        return (self.acd_before,)

    @property
    def init_marker(self) -> str:
        """The ``before`` marker."""
        return self.acd_before

    @property
    def extra_markers(self) -> tuple[str, ...]:
        """The ``before`` and ``after`` markers."""
        return (self.acd_after, self.acd_before)

    def observed_bpms(self, bpms_in_range: list[str], all_bpms: list[str]) -> list[str]:
        """BPMs in the range."""
        return bpms_in_range

    def select_available_turns(
        self,
        *,
        bunch_turns_by_file: dict[int, dict[int, list[int]]],
        available_turns: list[int],
    ) -> tuple[dict[int, set[int]], list[int]]:
        """Start turns every ``turns_per_track`` turns of each file."""
        return _window_starts(bunch_turns_by_file, self.turns_per_track)

    def build_range_specs(self, ctx: BpmRanges) -> list[WorkerRangeSpec]:
        """One forward range from ``before`` round the ring."""
        return [WorkerRangeSpec(self.acd_before, _bpm_behind(ctx.all_bpms, self.acd_before), 1)]


@dataclass(frozen=True)
class ACDArcByArcTrackingPlan(_AcdPlan):
    """Arc-by-arc tracking of AC-dipole data over ordinary BPM ranges.

    Unlike :class:`ACDTrackingPlan` (which tracks bidirectionally from the AC-dipole
    ``before``/``after`` markers), this plan tracks the caller's cartesian product of
    ``bpm_start_points`` x ``bpm_end_points``. The exciter is a driven, turn-varying
    element, so a range that crosses it would compare BPMs on opposite sides of a kick
    the free-oscillation model cannot reproduce.

    Rather than drop such a range, this plan **reroutes** it: two BPMs are joined by two
    arcs around the ring, and only one contains the AC dipole, so a crossing pair keeps
    ``start`` and ``end`` connected via the complementary long-way-round arc (in both
    tracking directions).

    The ``before``/``after`` monitors are installed (``tracking_anchor_mode`` == ``"acd"``)
    only so they appear in ``all_bpms`` and mark the exciter's ring position for the
    crossing test; they are not observed against the measurement.
    """

    def observed_bpms(self, bpms_in_range: list[str], all_bpms: list[str]) -> list[str]:
        """BPMs in the range, without the AC-dipole markers."""
        # Keep the exciter markers out of the measurement comparison; they exist only
        # to locate the AC dipole for the crossing test.
        markers = {self.acd_after, self.acd_before}
        return [bpm for bpm in bpms_in_range if bpm not in markers]

    def _range_crosses_acd(self, all_bpms: list[str], start_bpm: str, end_bpm: str) -> bool:
        """Return whether the forward span ``start_bpm`` -> ``end_bpm`` straddles the exciter.

        The ``before``/``after`` monitors sit adjacent in ring order, bracketing the AC
        dipole, so a contiguous forward span crosses the exciter exactly when it contains
        both of them.
        """
        if self.acd_before not in all_bpms or self.acd_after not in all_bpms:
            return False
        span = extract_bpm_range_names(all_bpms, start_bpm, end_bpm, 1)
        return self.acd_before in span and self.acd_after in span

    def _reroute_pair(self, all_bpms: list[str], start_bpm: str, end_bpm: str) -> tuple[str, str]:
        """Return the (start, end) whose forward span avoids the AC dipole.

        If the natural ``start`` -> ``end`` arc crosses the exciter, swap the endpoints so
        the range follows the complementary long-way-round arc instead. The AC dipole lies
        in exactly one of the two arcs joining a pair, so the swapped span never crosses.
        """
        if self._range_crosses_acd(all_bpms, start_bpm, end_bpm):
            return end_bpm, start_bpm
        return start_bpm, end_bpm

    def build_range_specs(self, ctx: BpmRanges) -> list[WorkerRangeSpec]:
        """Arc ranges, rerouted round the ring when they contain the AC dipole."""
        specs = super().build_range_specs(ctx)
        rerouted: list[WorkerRangeSpec] = []
        seen: set[tuple[str, str, int]] = set()
        n_rerouted = 0
        for spec in specs:
            start, end = self._reroute_pair(ctx.all_bpms, spec.start_bpm, spec.end_bpm)
            if (start, end) != (spec.start_bpm, spec.end_bpm):
                n_rerouted += 1
            key = (start, end, spec.sdir)
            if key in seen:
                continue
            seen.add(key)
            rerouted.append(WorkerRangeSpec(start_bpm=start, end_bpm=end, sdir=spec.sdir))
        if n_rerouted:
            logger.warning(
                "Rerouted %d ACD-crossing range(s) of %d the long way round the ring; "
                "ranges that would straddle the AC dipole (%s) are tracked via the "
                "complementary arc instead.",
                n_rerouted,
                len(specs),
                self.acd_name,
            )
        return rerouted
