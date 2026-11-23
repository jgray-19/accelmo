"""A tracking worker sends exactly one reply per request, including after a failure."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from accelmo.backend.protocol import TrackGeometry
from accelmo.config import TrackingDataConfig, TrackingParameters, WorkerRuntime
from accelmo.machine.accelerators import PSB
from accelmo.tracking.dispatch.setup import TrackingWorkerConfig
from accelmo.tracking.worker import PrecomputedTrackingWeights, TrackingData, TrackingWorker


def _worker(tmp_path, conn) -> TrackingWorker:
    seq_file = tmp_path / "psb.seq"
    seq_file.write_text("! placeholder sequence\n")
    shape = (2, 3)
    data = TrackingData(
        position_comparisons=np.zeros((*shape, 2)),
        momentum_comparisons=np.zeros((*shape, 2)),
        position_variances=np.ones((*shape, 2)),
        momentum_variances=np.ones((*shape, 2)),
        init_coords=np.zeros((2, 6)),
        init_pts=np.zeros(2),
        reading_ids=np.zeros(shape, dtype=np.int64),
        init_reading_ids=np.zeros(2, dtype=np.int64),
        init_variances=np.ones((2, 4)),
        start_turn_offsets=np.zeros(2, dtype=np.int64),
        precomputed_weights=PrecomputedTrackingWeights(
            x=np.ones(shape), y=np.ones(shape), px=np.ones(shape), py=np.ones(shape)
        ),
    )
    config = TrackingWorkerConfig(
        accelerator=PSB(ring=3, sequence_file=seq_file),
        magnet_range="$start/$end",
        geometry=TrackGeometry(start_bpm="BR3.BPM1L3", end_bpm="BR3.BPM3L3"),
        file_idx=0,
        bpm_names=["BR3.BPM1L3", "BR3.BPM2L3", "BR3.BPM3L3"],
    )
    return TrackingWorker(
        conn, 0, data, config, WorkerRuntime(1), TrackingDataConfig(1), TrackingParameters()
    )


def test_failed_training_worker_sends_no_uncertainty_part(tmp_path) -> None:
    """After its error reply a failed worker must not also send an empty part."""
    parent, child = mp.Pipe()
    worker = _worker(tmp_path, child)
    worker.n_reply_knobs = 2  # set by ``on_start``, which needs MAD-NG
    worker.on_stop(failed=True)

    assert not parent.poll(0.1)
