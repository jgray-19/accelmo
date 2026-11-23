"""Tests for tracking-mode planning in accelmo.tracking.config.tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from accelmo.config import OptimiserConfig, TrackingDataConfig, TrackingParameters, WorkerRuntime
from accelmo.machine.accelerators import PSB
from accelmo.tracking.config.tracking import (
    ACDArcByArcTrackingPlan,
    ACDSinusoidTrackingPlan,
    ACDTrackingPlan,
    KickerTrackingPlan,
    TrackingPlan,
)
from accelmo.tracking.fitter import ArcByArcFitter, FitterOptions
from accelmo.tracking.ranges import BpmRanges

if TYPE_CHECKING:
    from pathlib import Path

RING = [f"BPM.{i}" for i in range(8)]


BASE_TRACKING_DATA = TrackingDataConfig(num_batches=1)


@pytest.fixture
def psb_accelerator(seq_psb: Path) -> PSB:
    return PSB(ring=3, sequence_file=seq_psb)


class TestACDArcByArcReroute:
    """``ACDArcByArcTrackingPlan`` must reroute ranges that straddle the exciter."""

    def _plan(self) -> ACDArcByArcTrackingPlan:
        return ACDArcByArcTrackingPlan(acd_name="ACD")

    def _ctx(self, *, start_bpms: list[str], end_bpms: list[str]) -> BpmRanges:
        # The exciter markers sit adjacent between BPM.3 and BPM.4.
        all_bpms = [*RING[:4], "ACD_before", "ACD_after", *RING[4:]]
        return BpmRanges(
            plan=TrackingPlan(),
            start_bpms=start_bpms,
            end_bpms=end_bpms,
            all_bpms=all_bpms,
            use_fixed_bpm=False,
            fixed_start=all_bpms[0],
            fixed_end=all_bpms[-1],
        )

    def test_range_crossing_exciter_is_rerouted(self) -> None:
        plan = self._plan()
        ctx = self._ctx(start_bpms=["BPM.2"], end_bpms=["BPM.5"])

        specs = plan.build_range_specs(ctx)

        # BPM.2 -> BPM.5 forward crosses ACD_before/ACD_after, so it must be
        # swapped to the complementary long-way-round arc (BPM.5 -> BPM.2).
        forward_specs = [s for s in specs if s.sdir == 1]
        assert all((s.start_bpm, s.end_bpm) == ("BPM.5", "BPM.2") for s in forward_specs)

    def test_range_not_crossing_exciter_is_unchanged(self) -> None:
        plan = self._plan()
        ctx = self._ctx(start_bpms=["BPM.5"], end_bpms=["BPM.6"])

        specs = plan.build_range_specs(ctx)

        forward_specs = [s for s in specs if s.sdir == 1]
        assert all((s.start_bpm, s.end_bpm) == ("BPM.5", "BPM.6") for s in forward_specs)

    def test_range_specs_deduplicate_after_reroute(self) -> None:
        plan = self._plan()
        # Rerouted ranges that land on the same arc collapse to one range per direction.
        ctx = self._ctx(start_bpms=["BPM.2", "BPM.1"], end_bpms=["BPM.5"])

        specs = plan.build_range_specs(ctx)

        assert len(specs) == len(set(specs))

    def test_observed_bpms_excludes_exciter_markers(self) -> None:
        plan = self._plan()
        all_bpms = [*RING[:4], "ACD_before", "ACD_after", *RING[4:]]

        observed = plan.observed_bpms(all_bpms, all_bpms)

        assert "ACD_before" not in observed
        assert "ACD_after" not in observed


class TestPlainArcByArcPlan:
    def test_plain_arc_by_arc_ranges_are_untouched(self) -> None:
        plan = TrackingPlan()
        all_bpms = RING
        ctx = BpmRanges(
            plan=TrackingPlan(),
            start_bpms=["BPM.2"],
            end_bpms=["BPM.5"],
            all_bpms=all_bpms,
            use_fixed_bpm=False,
            fixed_start=all_bpms[0],
            fixed_end=all_bpms[-1],
        )

        specs = plan.build_range_specs(ctx)

        forward_specs = [s for s in specs if s.sdir == 1]
        assert all((s.start_bpm, s.end_bpm) == ("BPM.2", "BPM.5") for s in forward_specs)

    @pytest.mark.parametrize(
        ("use_fixed_bpm", "start_bpms", "end_bpms"),
        [
            # Cartesian product pairs BPM.5 with itself.
            (False, ["BPM.2", "BPM.5"], ["BPM.5", "BPM.6"]),
            # Fixed pairs: a start equal to the fixed end (the first end point).
            (True, ["BPM.2", "BPM.5"], ["BPM.5", "BPM.6"]),
            # Fixed pairs: an end equal to the fixed start (the first start point).
            (True, ["BPM.2", "BPM.3"], ["BPM.6", "BPM.2"]),
        ],
    )
    def test_range_starting_and_ending_at_the_same_bpm_is_rejected(
        self, use_fixed_bpm: bool, start_bpms: list[str], end_bpms: list[str]
    ) -> None:
        ctx = BpmRanges(
            plan=TrackingPlan(),
            start_bpms=start_bpms,
            end_bpms=end_bpms,
            all_bpms=RING,
            use_fixed_bpm=use_fixed_bpm,
            fixed_start=start_bpms[0],
            fixed_end=end_bpms[0],
        )

        with pytest.raises(ValueError, match="same BPM"):
            TrackingPlan().build_range_specs(ctx)


class TestModeConfiguration:
    """Each plan fixes the tracking-data entries and start points its mode needs."""

    def test_arc_ranges_are_single_turn(self) -> None:
        for plan in (TrackingPlan(), ACDArcByArcTrackingPlan(acd_name="ACD")):
            assert plan.configure(BASE_TRACKING_DATA) == BASE_TRACKING_DATA
            assert plan.turns_per_track == 1
            assert plan.start_points == ()

    def test_kicker_is_one_batch_of_the_kicked_turns_from_the_marker(self) -> None:
        plan = KickerTrackingPlan(kicker_name="KICKER1", turns_per_track=5)

        config = plan.configure(BASE_TRACKING_DATA)

        assert (config.num_batches, plan.turns_per_track) == (1, 5)
        assert plan.start_points == ("KICKER1",)
        assert plan.first_bpm_fallback == "KICKER1"

    def test_track_plans_reject_non_positive_turns(self, psb_accelerator: PSB) -> None:
        with pytest.raises(ValueError, match="turns_per_track"):
            KickerTrackingPlan(kicker_name="KICKER1", turns_per_track=0)
        with pytest.raises(ValueError, match="turns_per_track"):
            ACDSinusoidTrackingPlan(psb_accelerator.ac_dipole_name, turns_per_track=0)

    def test_acd_marker_tracks_one_turn_from_both_markers(self, psb_accelerator: PSB) -> None:
        plan = ACDTrackingPlan(acd_name=psb_accelerator.ac_dipole_name)

        assert plan.turns_per_track == 1
        assert plan.start_points == (
            psb_accelerator.acd_marker_name("after"),
            psb_accelerator.acd_marker_name("before"),
        )
        assert plan.first_bpm_fallback is None

    def test_acd_sinusoid_is_one_batch_from_the_before_marker(self, psb_accelerator: PSB) -> None:
        plan = ACDSinusoidTrackingPlan(acd_name=psb_accelerator.ac_dipole_name, turns_per_track=30)

        config = plan.configure(TrackingDataConfig(num_batches=4))

        assert (config.num_batches, plan.turns_per_track) == (1, 30)
        assert plan.start_points == (psb_accelerator.acd_marker_name("before"),)


@pytest.mark.parametrize(
    ("starts", "ends", "missing"),
    [([], ["BPM.1"], "bpm_start_points"), (["BPM.1"], [], "bpm_end_points")],
)
def test_arc_by_arc_fitter_requires_start_and_end_points(
    starts: list[str], ends: list[str], missing: str
) -> None:
    with pytest.raises(ValueError, match=missing):
        ArcByArcFitter(None, None, WorkerRuntime(2), BASE_TRACKING_DATA, None, None, starts, ends)


@pytest.mark.parametrize(
    "parameters",
    [
        TrackingParameters(knob_sigmas={"dk1l": 1e-3}),
        TrackingParameters(fit_start_coords=True),
        TrackingParameters(starts_follow_gains=True),
        TrackingParameters(bpm_gain_sigma=1e-2, bpm_gain_zero_mean_planes=("x",)),
    ],
)
def test_lm_only_parameters_refuse_the_sgd_solver(parameters: TrackingParameters) -> None:
    sgd = OptimiserConfig(
        max_epochs=1, warmup_epochs=0, warmup_lr_start=1e-6, max_lr=1e-6, min_lr=1e-6
    )
    with pytest.raises(ValueError, match="need the LM solver"):
        ArcByArcFitter(
            None,
            sgd,
            WorkerRuntime(2),
            BASE_TRACKING_DATA,
            None,
            None,
            ["BPM.1"],
            ["BPM.2"],
            options=FitterOptions(parameters=parameters),
        )
