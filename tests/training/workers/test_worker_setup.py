"""Worker planning: ranges and per-file worker configs."""

from __future__ import annotations

import random
from itertools import product
from typing import TYPE_CHECKING

import pytest

from accelmo.config import TrackingDataConfig
from accelmo.fitting.worker import WorkerConfig
from accelmo.machine.accelerators import PSB, SPS
from accelmo.tracking.config.tracking import ACDTrackingPlan, TrackingPlan, WorkerRangeSpec
from accelmo.tracking.dispatch.setup import build_range_specs, plan_workers, range_workers
from accelmo.tracking.dispatch.turn_planner import WorkerTurnPlanner
from accelmo.tracking.ranges import BpmRanges

if TYPE_CHECKING:
    from pathlib import Path

ARC_BPMS = ["BR3.BPM1L3", "BR3.BPM2L3", "BR3.BPM3L3", "BR3.BPM4L3"]
ACD_BPMS = [
    "BR3.BPM1L3",
    "BR3.BPM2L3",
    "HACMAP_before",
    "HACMAP_after",
    "BR3.BPM3L3",
    "BR3.BPM4L3",
]
# SPS monitors see one plane each: BPH horizontal, BPV vertical.
SPS_BPMS = ["BPH.13008", "BPV.13108", "BPH.13208", "BPV.13308", "BPH.13408", "BPV.13508"]


def _psb(tmp_path: Path) -> PSB:
    seq_file = tmp_path / "machine.seq"
    seq_file.write_text("! placeholder sequence\n")
    return PSB(ring=3, sequence_file=seq_file)


def _sps(tmp_path: Path) -> SPS:
    seq_file = tmp_path / "sps.seq"
    seq_file.write_text("! placeholder sequence\n")
    return SPS(sequence_file=seq_file)


def _base(accelerator, machine_state: Path | None = None) -> WorkerConfig:
    return WorkerConfig(
        accelerator=accelerator, magnet_range="$start/$end", machine_state=machine_state
    )


def _ranges(
    all_bpms: list[str],
    plan,
    *,
    start_bpms: list[str] | None = None,
    end_bpms: list[str] | None = None,
    use_fixed_bpm: bool = True,
    fixed_start: str | None = None,
    fixed_end: str | None = None,
) -> BpmRanges:
    return BpmRanges(
        plan=plan,
        start_bpms=start_bpms or [],
        end_bpms=end_bpms or [],
        all_bpms=all_bpms,
        use_fixed_bpm=use_fixed_bpm,
        fixed_start=all_bpms[0] if fixed_start is None else fixed_start,
        fixed_end=all_bpms[-1] if fixed_end is None else fixed_end,
    )


def _sps_ranges(**kwargs) -> BpmRanges:
    return _ranges(SPS_BPMS, TrackingPlan(), **{"use_fixed_bpm": False, **kwargs})


def test_fixed_bpm_ranges_pair_each_start_with_the_fixed_end(tmp_path: Path) -> None:
    ranges = _ranges(
        ARC_BPMS, TrackingPlan(), start_bpms=["BR3.BPM1L3", "BR3.BPM2L3"], end_bpms=["BR3.BPM3L3"]
    )

    assert build_range_specs(_psb(tmp_path), ranges) == [
        WorkerRangeSpec("BR3.BPM1L3", "BR3.BPM4L3", 1),
        WorkerRangeSpec("BR3.BPM2L3", "BR3.BPM4L3", 1),
        WorkerRangeSpec("BR3.BPM1L3", "BR3.BPM3L3", -1),
    ]


def test_worker_takes_its_kick_plane_from_the_measurement_file(tmp_path: Path) -> None:
    base, ranges = _base(_psb(tmp_path)), _ranges(ARC_BPMS, TrackingPlan())
    spec = WorkerRangeSpec("BR3.BPM1L3", "BR3.BPM3L3", 1)

    [single_plane] = range_workers(base, ranges, spec, 0, "x", set(ARC_BPMS))
    [dual_plane] = range_workers(base, ranges, spec, 1, "xy", set(ARC_BPMS))

    assert (single_plane.kick_plane, single_plane.file_idx) == ("x", 0)
    assert (dual_plane.kick_plane, dual_plane.file_idx) == ("xy", 1)
    assert single_plane.bpm_names == ["BR3.BPM1L3", "BR3.BPM2L3", "BR3.BPM3L3"]
    assert single_plane.bad_bpms is None
    assert single_plane.geometry.init_point == "BR3.BPM1L3"


def test_bpms_missing_from_a_file_are_unobserved_rather_than_dropping_the_worker(
    tmp_path: Path,
) -> None:
    [config] = range_workers(
        _base(_psb(tmp_path)),
        _ranges(ARC_BPMS, TrackingPlan()),
        WorkerRangeSpec("BR3.BPM1L3", "BR3.BPM3L3", 1),
        0,
        "x",
        {"BR3.BPM1L3", "BR3.BPM3L3"},
    )

    assert config.bpm_names == ["BR3.BPM1L3", "BR3.BPM3L3"]
    assert config.bad_bpms == ["BR3.BPM2L3"]


def test_worker_config_carries_its_own_files_measurement_artifacts(tmp_path: Path) -> None:
    options = tmp_path / "corr1_state.txt"

    [config] = range_workers(
        _base(_psb(tmp_path), options),
        _ranges(ARC_BPMS, TrackingPlan()),
        WorkerRangeSpec("BR3.BPM1L3", "BR3.BPM3L3", 1),
        1,
        "xy",
        set(ARC_BPMS),
    )

    assert config.machine_state == options
    assert config.magnet_range == "$start/$end"
    assert config.kick_plane == "xy"


def test_acd_markers_supply_initial_conditions_and_are_not_observed(tmp_path: Path) -> None:
    """Each ACD direction initialises from its own marker and compares real BPMs only."""
    psb = _psb(tmp_path)
    ranges = _ranges(
        ACD_BPMS,
        ACDTrackingPlan(acd_name="HACMAP"),
        start_bpms=["HACMAP_after", "HACMAP_before"],
        use_fixed_bpm=False,
        fixed_start="$start",
        fixed_end="$end",
    )

    specs = build_range_specs(psb, ranges)
    assert specs == [
        WorkerRangeSpec("HACMAP_after", "HACMAP_before", 1),
        WorkerRangeSpec("HACMAP_after", "HACMAP_before", -1),
    ]

    [forward], [backward] = (
        range_workers(_base(psb), ranges, spec, 0, "xy", set(ACD_BPMS)) for spec in specs
    )

    assert forward.bpm_names == ["BR3.BPM3L3", "BR3.BPM4L3", "BR3.BPM1L3", "BR3.BPM2L3"]
    assert backward.bpm_names == ["BR3.BPM2L3", "BR3.BPM1L3", "BR3.BPM4L3", "BR3.BPM3L3"]
    assert forward.geometry.initial_condition_marker == "HACMAP_after"
    assert backward.geometry.initial_condition_marker == "HACMAP_before"
    assert backward.geometry.init_point == "HACMAP_before"


def test_single_plane_bpm_ranges_start_and_end_on_bpms_of_one_plane(tmp_path: Path) -> None:
    ranges = _sps_ranges(start_bpms=["BPH.13008", "BPV.13108"], end_bpms=["BPH.13408", "BPV.13508"])

    specs = build_range_specs(_sps(tmp_path), ranges)

    assert {(s.start_bpm, s.end_bpm) for s in specs} == {
        ("BPH.13008", "BPH.13408"),
        ("BPV.13108", "BPV.13508"),
    }


def test_single_plane_bpms_split_dual_plane_data_into_x_and_y_workers(tmp_path: Path) -> None:
    base, ranges = _base(_sps(tmp_path)), _sps_ranges()
    horizontal = WorkerRangeSpec("BPH.13008", "BPH.13408", 1)
    vertical = WorkerRangeSpec("BPV.13108", "BPV.13508", 1)

    # Each range yields only the worker whose plane its start BPM can see.
    [x_worker] = range_workers(base, ranges, horizontal, 1, "xy", set(SPS_BPMS))
    [y_worker] = range_workers(base, ranges, vertical, 1, "xy", set(SPS_BPMS))

    assert (x_worker.kick_plane, y_worker.kick_plane) == ("x", "y")
    assert x_worker.bpm_names == ["BPH.13008", "BPH.13208", "BPH.13408"]
    # MAD unobserves every blind BPM in the ring, not only those inside the range.
    assert x_worker.bad_bpms == ["BPV.13108", "BPV.13308", "BPV.13508"]


def test_single_plane_bpms_plan_acd_marker_ranges_and_split_workers_by_plane(
    tmp_path: Path,
) -> None:
    plan = ACDTrackingPlan(acd_name="ZKHA.21991")
    all_bpms = [*SPS_BPMS[:3], plan.acd_before, plan.acd_after, *SPS_BPMS[3:]]
    base = _base(_sps(tmp_path))
    ranges = _ranges(
        all_bpms, plan, start_bpms=[plan.acd_after, plan.acd_before], use_fixed_bpm=False
    )

    # The markers carry both planes, so the ranges are not split per plane.
    specs = build_range_specs(base.accelerator, ranges)
    assert specs == [
        WorkerRangeSpec(plan.acd_after, plan.acd_before, 1),
        WorkerRangeSpec(plan.acd_after, plan.acd_before, -1),
    ]

    # Dual-plane data gives one worker per plane, each blind to the other plane's BPMs.
    x_worker, y_worker = range_workers(base, ranges, specs[0], 1, "xy", set(all_bpms))
    assert (x_worker.kick_plane, y_worker.kick_plane) == ("x", "y")
    assert x_worker.bad_bpms == ["BPV.13108", "BPV.13308", "BPV.13508"]
    assert y_worker.bad_bpms == ["BPH.13008", "BPH.13208", "BPH.13408"]
    # A file driven in x only gives only the x worker.
    [x_only] = range_workers(base, ranges, specs[0], 0, "x", set(all_bpms))
    assert x_only.kick_plane == "x"


def test_single_plane_file_drops_workers_starting_on_the_blind_plane(tmp_path: Path) -> None:
    spec = WorkerRangeSpec("BPV.13108", "BPV.13508", 1)

    assert range_workers(_base(_sps(tmp_path)), _sps_ranges(), spec, 0, "x", set(SPS_BPMS)) == []


@pytest.mark.parametrize(("use_fixed_bpm", "kick_plane"), list(product((False, True), ("x", "xy"))))
def test_single_plane_bpms_spawn_the_requested_number_of_workers(
    tmp_path: Path, use_fixed_bpm: bool, kick_plane: str
) -> None:
    """With plenty of turns, the planned turn batches fill ``num_workers`` exactly."""
    ranges = _sps_ranges(
        start_bpms=SPS_BPMS[:2], end_bpms=SPS_BPMS[-2:], use_fixed_bpm=use_fixed_bpm
    )
    turns = list(range(100))
    tracking_data = TrackingDataConfig(num_batches=1)
    workers = plan_workers(_base(_sps(tmp_path)), ranges, 0, kick_plane, set(SPS_BPMS))
    batches = WorkerTurnPlanner(
        ranges.plan, tracking_data, 16, shuffle_turns=random.Random(0).shuffle
    ).build_turn_batches(
        available_turns=turns,
        file_map=dict.fromkeys(turns, 0),
        num_files=1,
        workers_per_batch=len(workers),
    )

    assert len(batches) * len(workers) == 16
