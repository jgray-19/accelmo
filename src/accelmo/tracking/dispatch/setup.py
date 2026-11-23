"""Worker planning: the config of every worker for a set of BPM ranges and files.

LHC and PSB BPMs measure both planes, so workers follow the data alone: a file
kicked in one plane gives single-plane workers. On a machine with single-plane
BPMs (SPS ``BPH``/``BPV``) each worker additionally observes only the BPMs that
see its plane, ranges are planned per plane, and dual-plane data is split into
separate x and y workers.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from accelmo.backend.protocol import TrackGeometry
from accelmo.fitting.worker import WorkerConfig

if TYPE_CHECKING:
    from accelmo.machine.accelerators import Accelerator
    from accelmo.tracking.config.tracking import WorkerRangeSpec
    from accelmo.tracking.ranges import BpmRanges


LOGGER = logging.getLogger(__name__)


class KickPlane(StrEnum):
    """Kick-plane options for worker routing and payload selection."""

    X = "x"
    Y = "y"
    XY = "xy"


@dataclass(kw_only=True)
class TrackingWorkerConfig(WorkerConfig):
    """One tracking worker: model options, the segment it tracks, its file and observed BPMs."""

    geometry: TrackGeometry
    file_idx: int
    bpm_names: list[str]
    kick_plane: KickPlane = KickPlane.XY


def _merge_bad_bpms(*bad_bpm_lists: list[str] | None) -> list[str] | None:
    """Merge bad-BPM lists, keeping first-occurrence order."""
    merged: list[str] = []
    for bpm_list in bad_bpm_lists:
        for bpm in bpm_list or []:
            if bpm not in merged:
                merged.append(bpm)
    return merged or None


def _measures(accelerator: Accelerator, ranges: BpmRanges, bpm: str, plane: KickPlane) -> bool:
    """Whether ``bpm`` measures ``plane`` (``XY`` means both planes)."""
    # Tracking markers carry reconstructed coordinates in both planes.
    if bpm in ranges.plan.extra_markers:
        return True
    monitor = accelerator.infer_monitor_plane(bpm)
    if plane == KickPlane.X:
        return "H" in monitor
    if plane == KickPlane.Y:
        return "V" in monitor
    return "H" in monitor and "V" in monitor


def _single_plane_bpms(accelerator: Accelerator, ranges: BpmRanges) -> bool:
    return not all(_measures(accelerator, ranges, bpm, KickPlane.XY) for bpm in ranges.all_bpms)


def build_range_specs(accelerator: Accelerator, ranges: BpmRanges) -> list[WorkerRangeSpec]:
    """The logical worker ranges of ``ranges.plan``."""
    plan = ranges.plan
    # Marker-anchored plans (ACD markers, kicker) start on markers that carry both
    # planes; their workers are split by plane instead.
    if not _single_plane_bpms(accelerator, ranges) or plan.start_points:
        return plan.build_range_specs(ranges)

    # A range must start and end on BPMs of the plane it tracks.
    specs: list[WorkerRangeSpec] = []
    for plane in (KickPlane.X, KickPlane.Y):
        plane_bpms = [b for b in ranges.all_bpms if _measures(accelerator, ranges, b, plane)]
        starts = [bpm for bpm in ranges.start_bpms if bpm in plane_bpms]
        ends = [bpm for bpm in ranges.end_bpms if bpm in plane_bpms]
        if not starts or not ends:
            LOGGER.warning(
                "No %s-plane start and end BPMs; no %s-plane ranges", plane.value, plane.value
            )
            continue
        plane_ranges = dataclasses.replace(
            ranges,
            start_bpms=starts,
            end_bpms=ends,
            all_bpms=plane_bpms,
            fixed_start=starts[0] if ranges.use_fixed_bpm else ranges.fixed_start,
            fixed_end=ends[0] if ranges.use_fixed_bpm else ranges.fixed_end,
        )
        specs.extend(plan.build_range_specs(plane_ranges))
    return specs


def plan_workers(
    base: WorkerConfig,
    ranges: BpmRanges,
    file_idx: int,
    kick_plane: str | KickPlane,
    available_bpms: set[str],
) -> list[TrackingWorkerConfig]:
    """Every worker one turn batch of file ``file_idx`` fans out to.

    ``base`` holds the file's model options; ``kick_plane`` is the plane the file excites.
    """
    return [
        config
        for spec in build_range_specs(base.accelerator, ranges)
        for config in range_workers(base, ranges, spec, file_idx, kick_plane, available_bpms)
    ]


def range_workers(
    base: WorkerConfig,
    ranges: BpmRanges,
    range_spec: WorkerRangeSpec,
    file_idx: int,
    kick_plane: str | KickPlane,
    available_bpms: set[str],
) -> list[TrackingWorkerConfig]:
    """The worker(s) of one range and measurement file: one per fitted plane."""
    single_plane = _single_plane_bpms(base.accelerator, ranges)
    data_plane = KickPlane(kick_plane)
    planes = (
        (KickPlane.X, KickPlane.Y) if data_plane == KickPlane.XY and single_plane else (data_plane,)
    )
    configs = (
        _worker(base, ranges, range_spec, file_idx, plane, available_bpms, single_plane)
        for plane in planes
    )
    return [config for config in configs if config is not None]


def _worker(
    base: WorkerConfig,
    ranges: BpmRanges,
    range_spec: WorkerRangeSpec,
    file_idx: int,
    plane: KickPlane,
    available_bpms: set[str],
    single_plane: bool,
) -> TrackingWorkerConfig | None:
    """One worker's config, or ``None`` when the range is unusable.

    BPMs the worker cannot use are added to its bad-BPM list so the model stops
    observing them; if the initial-condition marker is missing from the file
    there is nothing to track from and the worker is dropped.
    """
    plan = ranges.plan
    bad_bpms = base.bad_bpms
    if single_plane and plane != KickPlane.XY:
        # Every blind BPM in the ring, not only those in the range: MAD's
        # observation flags are set on the whole sequence.
        blind = [b for b in ranges.all_bpms if not _measures(base.accelerator, ranges, b, plane)]
        bad_bpms = _merge_bad_bpms(bad_bpms, blind)

    bpm_names = plan.get_range_bpm_names(
        all_bpms=ranges.all_bpms,
        start_bpm=range_spec.start_bpm,
        end_bpm=range_spec.end_bpm,
        sdir=range_spec.sdir,
        bad_bpms=bad_bpms,
    )
    init_marker = plan.initial_condition_marker(range_spec)
    label = (
        f"File {file_idx} range {range_spec.start_bpm}/{range_spec.end_bpm} sdir={range_spec.sdir}"
    )

    missing = [bpm for bpm in bpm_names if bpm not in available_bpms]
    if missing:
        LOGGER.warning(
            "%s: %d BPMs missing from measurement data; adding to bad BPMs", label, len(missing)
        )
        bad_bpms = _merge_bad_bpms(bad_bpms, missing)
        bpm_names = [bpm for bpm in bpm_names if bpm in available_bpms]
    if init_marker is not None and init_marker not in available_bpms:
        LOGGER.warning("%s: init marker %s missing from measurement data", label, init_marker)
        return None
    if not bpm_names:
        return None
    if init_marker is None and range_spec.init_bpm not in bpm_names:
        return None

    shared = {f.name: getattr(base, f.name) for f in dataclasses.fields(WorkerConfig)}
    shared["bad_bpms"] = bad_bpms
    return TrackingWorkerConfig(
        **shared,
        geometry=TrackGeometry(
            start_bpm=range_spec.start_bpm,
            end_bpm=range_spec.end_bpm,
            initial_condition_marker=init_marker,
            sdir=range_spec.sdir,
            cycle_marker=plan.cycle_marker,
            anchor_mode=plan.tracking_anchor_mode,
            anchor_sources=tuple(plan.tracking_anchor_sources),
            n_turns=plan.turns_per_track,
        ),
        file_idx=file_idx,
        bpm_names=bpm_names,
        kick_plane=plane,
    )
