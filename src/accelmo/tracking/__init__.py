"""Tracking fits: magnet knobs from turn-by-turn BPM data by mini-batch gradient descent.

The public entry points are the three tracking fitters -- :class:`ArcByArcFitter`,
:class:`ACDMarkerFitter` and :class:`KickerFitter`. They share the machine setup,
worker pool and results of :mod:`accelmo.fitting` with the closed-orbit and
closed-twiss fitters of :mod:`accelmo.poco`.
"""

from accelmo.tracking.config import (
    ACDArcByArcTrackingPlan,
    ACDSinusoidTrackingPlan,
    ACDTrackingPlan,
    CheckpointConfig,
    KickerTrackingPlan,
    MeasurementConfig,
    MeasurementDetails,
    TrackingPlan,
    WorkerRangeSpec,
    single_measurement_config,
)
from accelmo.tracking.config.tracking import extract_bpm_range_names
from accelmo.tracking.data_manager import DataManager, FileTracks
from accelmo.tracking.fitter import (
    ACDMarkerFitter,
    ACDSinusoidFitter,
    ArcByArcFitter,
    FitterOptions,
    KickerFitter,
    TrackingFitter,
)
from accelmo.tracking.session import TrackingSession
from accelmo.tracking.sgd.loop import EpochState, SGDLoop
from accelmo.tracking.sgd.scheduler import LRScheduler

__all__ = [
    "ACDArcByArcTrackingPlan",
    "ACDMarkerFitter",
    "ACDSinusoidFitter",
    "ACDSinusoidTrackingPlan",
    "ACDTrackingPlan",
    "ArcByArcFitter",
    "CheckpointConfig",
    "DataManager",
    "EpochState",
    "FileTracks",
    "FitterOptions",
    "KickerFitter",
    "KickerTrackingPlan",
    "LRScheduler",
    "MeasurementConfig",
    "MeasurementDetails",
    "SGDLoop",
    "TrackingFitter",
    "TrackingPlan",
    "TrackingSession",
    "WorkerRangeSpec",
    "extract_bpm_range_names",
    "single_measurement_config",
]
