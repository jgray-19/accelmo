from __future__ import annotations

import multiprocessing
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest

from accelmo.backend.protocol import TrackGeometry
from accelmo.config import TrackingDataConfig, TrackingParameters, WorkerRuntime
from accelmo.fitting.pool import WorkerPool
from accelmo.fitting.protocol import STOP, Ack, UncertaintyProgress
from accelmo.fitting.worker import WorkerConfig
from accelmo.machine.accelerators import PSB
from accelmo.tracking.config.tracking import TrackingPlan
from accelmo.tracking.data_manager import FileTracks
from accelmo.tracking.dispatch.setup import KickPlane, TrackingWorkerConfig, plan_workers
from accelmo.tracking.ranges import BpmRanges
from accelmo.tracking.session import TrackingSession
from accelmo.tracking.uncertainty import (
    StageProgress,
    UncertaintyPart,
    drain_uncertainty,
    eliminate_fitted_starts,
    second_order_blocker,
)
from accelmo.tracking.worker import PrecomputedTrackingWeights, TrackingData, TrackingWorker

if TYPE_CHECKING:
    from pathlib import Path


class _FakeConn:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.sent: list[dict[str, object]] = []
        self._responses = responses

    def send(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)

    def recv(self) -> dict[str, object]:
        return self._responses.pop(0)


class _FakeWorker:
    def __init__(self) -> None:
        self.join_calls = 0
        self.terminate_calls = 0
        self.exitcode = 0
        self.pid = 1234

    def join(self, timeout: float | None = None) -> None:
        self.join_calls += 1

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        self.terminate_calls += 1


class _ConnThatMustNotPoll:
    def send(self, payload: object) -> None:
        self.sent = payload

    def poll(self, timeout: float | None = None) -> bool:
        raise AssertionError(f"Validation cleanup should not poll for payloads, got {timeout}")

    def recv(self) -> object:
        raise AssertionError("Validation cleanup should not receive a final payload")

    def close(self) -> None:
        pass


def _start_pipe_worker(
    child_conn: multiprocessing.connection.Connection, responses: list[object]
) -> threading.Thread:
    """Start a daemon thread that acts as a fake worker over a real pipe.

    The thread receives one binary message per entry of ``responses`` (via recv_bytes), then
    sends the reply back via send(); an entry that is a list is several replies to that one
    message (a worker's progress messages and final payload follow the ``STOP``). This matches
    the WorkerChannels protocol where the parent sends via send_bytes and receives via recv.
    """

    def _run() -> None:
        for response in responses:
            child_conn.recv_bytes()
            for reply in response if isinstance(response, list) else [response]:
                child_conn.send(reply)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


BPMS = ["BR3.BPM1L3", "BR3.BPM2L3", "BR3.BPM3L3", "BR3.BPM4L3"]


def _make_psb(tmp_path: Path) -> PSB:
    seq_file = tmp_path / "psb.seq"
    seq_file.write_text("! placeholder sequence\n")
    return PSB(ring=3, sequence_file=seq_file)


def _plan_file_workers(tmp_path: Path, file_idx: int) -> list[TrackingWorkerConfig]:
    """Workers of one file over the first three BPMs, each file with its own machine state."""
    bpms = BPMS[:3]
    ranges = BpmRanges(
        TrackingPlan(),
        [bpms[0]],
        [bpms[1]],
        bpms,
        use_fixed_bpm=True,
        fixed_start=bpms[0],
        fixed_end=bpms[-1],
    )
    base = WorkerConfig(
        accelerator=_make_psb(tmp_path),
        magnet_range="$start/$end",
        machine_state=tmp_path / f"corr{file_idx}_state.txt",
    )
    return plan_workers(base, ranges, file_idx, "xy", set(bpms))


def _make_tracks(all_bpms: list[str], turns: list[int]) -> FileTracks:
    grid = np.array(
        [[10.0 * (t + 1) + (b + 1) for b in range(len(all_bpms))] for t in range(len(turns))]
    )
    values = {
        "x": grid,
        "px": grid * 0.01,
        "y": grid * 0.1,
        "py": grid * 0.001,
        "var_x": np.ones_like(grid),
        "var_px": np.ones_like(grid),
        "var_y": np.ones_like(grid),
        "var_py": np.ones_like(grid),
    }
    return FileTracks(turns=np.array(turns), markers=list(all_bpms), values=values, kick_plane="xy")


def test_session_payloads_take_per_file_artifacts_from_the_file_turn_map(tmp_path: Path) -> None:
    session = TrackingSession(
        _make_psb(tmp_path),
        WorkerRuntime(2),
        TrackingDataConfig(1, optimise_momenta=False),
        TrackingParameters(),
        turn_batches=[[2], [202]],
        validation_turn_batches=[],
        file_turn_map={2: 0, 202: 1},
        worker_plans={f: _plan_file_workers(tmp_path, f) for f in (0, 1)},
        machine_deltaps=[0.0, 1e-3],
    )

    payloads = session.create_worker_payloads(
        tracks={
            0: _make_tracks(BPMS[:3], [1, 2, 3]),
            1: _make_tracks(BPMS[:3], [201, 202, 203]),
        },
        turn_batches=[[2], [202]],
    )

    assert [config.file_idx for _, config in payloads] == [0, 0, 1, 1]
    assert [config.machine_state for _, config in payloads] == [
        tmp_path / "corr0_state.txt",
        tmp_path / "corr0_state.txt",
        tmp_path / "corr1_state.txt",
        tmp_path / "corr1_state.txt",
    ]

    init_pts = [float(data.init_pts[0]) for data, _ in payloads]
    assert init_pts[0] == init_pts[1]
    assert init_pts[2] == init_pts[3]
    assert init_pts[0] != init_pts[2]


def test_validation_worker_keeps_all_held_out_turns_with_nondividing_batch_count(
    tmp_path: Path,
) -> None:
    accelerator = _make_psb(tmp_path)
    n_tracks = 10
    n_points = 3
    shape = (n_tracks, n_points)
    data = TrackingData(
        position_comparisons=np.zeros((n_tracks, n_points, 2), dtype=np.float64),
        momentum_comparisons=np.zeros((n_tracks, n_points, 2), dtype=np.float64),
        position_variances=np.ones((n_tracks, n_points, 2), dtype=np.float64),
        momentum_variances=np.ones((n_tracks, n_points, 2), dtype=np.float64),
        init_coords=np.zeros((n_tracks, 6), dtype=np.float64),
        init_pts=np.zeros((n_tracks,), dtype=np.float64),
        reading_ids=np.zeros(shape, dtype=np.int64),
        init_reading_ids=np.zeros((n_tracks,), dtype=np.int64),
        init_variances=np.ones((n_tracks, 4), dtype=np.float64),
        start_turn_offsets=np.zeros(n_tracks, dtype=np.int64),
        precomputed_weights=PrecomputedTrackingWeights(
            x=np.ones(shape, dtype=np.float64),
            y=np.ones(shape, dtype=np.float64),
            px=np.ones(shape, dtype=np.float64),
            py=np.ones(shape, dtype=np.float64),
        ),
    )
    config = TrackingWorkerConfig(
        accelerator=accelerator,
        magnet_range="$start/$end",
        kick_plane="xy",
        geometry=TrackGeometry(start_bpm="BPH.13008", end_bpm="BPH.13408", sdir=1),
        file_idx=0,
        bpm_names=["BPH.13008", "BPH.13408"],
    )

    worker = TrackingWorker(
        conn=_FakeConn([]),  # type: ignore[arg-type]
        worker_id=0,
        data=data,
        config=config,
        runtime=WorkerRuntime(1),
        tracking_data=TrackingDataConfig(8),
        parameters=TrackingParameters(),
        validation=True,
    )

    assert sum(len(batch) for batch in worker.init_coords) == n_tracks
    assert worker.num_batches == 5
    assert [len(batch) for batch in worker.init_coords] == [2, 2, 2, 2, 2]


def _uncertainty_pool(file_indices: list[int], responses: list[list[object]]) -> WorkerPool:
    """A pool whose workers are pipe threads, one per entry of ``file_indices``."""
    pool = WorkerPool()
    for worker_responses in responses:
        parent, child = multiprocessing.Pipe()
        _start_pipe_worker(child, worker_responses)
        pool.conns.append(parent)
    pool.workers = [_FakeWorker() for _ in file_indices]  # type: ignore[misc]
    pool.metadata = [
        TrackingWorkerConfig(
            accelerator=None,
            magnet_range="$start/$end",
            geometry=TrackGeometry(start_bpm=BPMS[0], end_bpm=BPMS[-1]),
            file_idx=file_idx,
            bpm_names=BPMS,
            kick_plane=KickPlane.XY,
        )
        for file_idx in file_indices
    ]
    pool.particle_counts = [1 for _ in file_indices]
    return pool


def test_drain_uncertainty_merges_shared_readings_per_file() -> None:
    # Workers 0 and 2 share reading 7 of file 0; worker 1 (file 1) is drained between them.
    g_a, g_b, g_c = np.array([1.0, 2.0]), np.array([3.0, -1.0]), np.array([0.5, 0.5])
    parts = [
        UncertaintyPart(np.eye(2), np.array([7]), g_a[None, :], np.array([0.1])),
        UncertaintyPart(2.0 * np.eye(2), np.array([8]), g_c[None, :], np.array([0.2])),
        UncertaintyPart(3.0 * np.eye(2), np.array([7]), g_b[None, :], np.array([0.1])),
    ]
    pool = _uncertainty_pool([0, 1, 0], [[Ack(0), {"gauss_newton": part}] for part in parts])

    normal, noise = drain_uncertainty(pool, 2, ("gauss_newton",))["gauss_newton"]

    shared = g_a + g_b
    np.testing.assert_allclose(normal, 6.0 * np.eye(2))
    np.testing.assert_allclose(noise, 0.1 * np.outer(shared, shared) + 0.2 * np.outer(g_c, g_c))
    assert [worker.join_calls for worker in pool.workers] == [1, 1, 1]


def test_terminate_kills_training_and_validation_workers(tmp_path: Path) -> None:
    session = TrackingSession(
        _make_psb(tmp_path),
        WorkerRuntime(1),
        TrackingDataConfig(1),
        TrackingParameters(),
        turn_batches=[],
        validation_turn_batches=[],
        file_turn_map={},
        worker_plans={},
        machine_deltaps=[],
    )
    training = [_FakeWorker(), _FakeWorker()]
    validation = [_FakeWorker()]
    session.training = WorkerPool(workers=training)  # type: ignore[arg-type]
    session.validation = WorkerPool(workers=validation)  # type: ignore[arg-type]

    session.terminate()

    for worker in (*training, *validation):
        assert worker.terminate_calls == 1
        assert worker.join_calls == 1


def test_pool_stop_does_not_wait_for_final_payload() -> None:
    worker = _FakeWorker()
    conn = _ConnThatMustNotPoll()
    pool = WorkerPool(conns=[conn], workers=[worker])  # type: ignore[list-item]

    pool.stop()

    assert conn.sent == STOP
    assert worker.join_calls == 1


def test_drain_uncertainty_disables_the_hessian_before_shutdown() -> None:
    # Two messages: ack for the mode, then the empty reply on termination
    pool = _uncertainty_pool([0], [[Ack(0), {}]])

    assert drain_uncertainty(pool, 2, ()) == {}
    assert pool.workers[0].join_calls == 1


def test_drain_uncertainty_keeps_each_hessian_apart_and_counts_progress() -> None:
    gauss_newton = UncertaintyPart(np.eye(2), np.array([7]), np.ones((1, 2)), np.array([0.1]))
    true = UncertaintyPart(3.0 * np.eye(2), np.array([7]), 2 * np.ones((1, 2)), np.array([0.1]))
    after_stop = [
        UncertaintyProgress(0),
        UncertaintyProgress(0),
        {"gauss_newton": gauss_newton, "true": true},
    ]
    pool = _uncertainty_pool([0], [[Ack(0), after_stop]])
    updates: list[tuple[int, int]] = []
    progress = StageProgress(2, lambda done, total, *_: updates.append((done, total)))

    matrices = drain_uncertainty(pool, 2, ("gauss_newton", "true"), 2, progress)

    np.testing.assert_allclose(matrices["gauss_newton"][0], np.eye(2))
    np.testing.assert_allclose(matrices["true"][0], 3.0 * np.eye(2))
    np.testing.assert_allclose(matrices["true"][1], 0.1 * 4.0 * np.ones((2, 2)))
    assert progress.done == 2
    assert updates[-1] == (2, 2)


def test_default_true_hessian_falls_back_to_gauss_newton_on_the_xtrack_backend(
    tmp_path: Path,
) -> None:
    seq_file = tmp_path / "psb.seq"
    seq_file.write_text("! placeholder sequence\n")
    parameters = TrackingParameters()
    assert parameters.uncertainty_hessian == "true"
    session = TrackingSession(
        PSB(ring=3, sequence_file=seq_file, backend="xtrack"),
        WorkerRuntime(1),
        TrackingDataConfig(1),
        parameters,
        turn_batches=[[1]],
        validation_turn_batches=[],
        file_turn_map={1: 0},
        worker_plans={},
        machine_deltaps=[0.0],
    )
    part = UncertaintyPart(np.eye(2), np.array([7]), np.ones((1, 2)), np.array([0.1]))
    session.training = _uncertainty_pool([0], [[Ack(0), {"gauss_newton": part}]])

    collection = session.stop_and_collect_uncertainty(2)

    assert collection.skipped == {"true": "the xtrack backend has no second derivatives"}
    assert list(collection.matrices) == ["gauss_newton"]
    np.testing.assert_allclose(collection.matrices["gauss_newton"][0], np.eye(2))


def test_fitted_start_elimination_is_the_schur_complement_of_the_full_hessian() -> None:
    rng = np.random.default_rng(1)
    n_particles, n_knobs, n_points = 3, 4, 12
    jacobian = rng.normal(size=(n_particles, n_knobs, n_points))
    d_dc0 = rng.normal(size=(n_particles, 4, n_points))
    weight = rng.uniform(0.5, 2.0, size=(n_particles, n_points))
    residual_kk = rng.normal(size=(n_knobs, n_knobs))
    residual_kk += residual_kk.T
    residual_ss = rng.normal(size=(n_particles, 4, 4))
    residual_ss += residual_ss.swapaxes(1, 2) + 10 * np.eye(4)
    residual_sk = rng.normal(size=(n_particles, 4, n_knobs))
    # A = Σ w J Jᵀ (Gauss-Newton) plus the residual term, as the true Hessian has it
    a_kk = np.einsum("pkn,pjn,pn->kj", jacobian, jacobian, weight) + residual_kk
    a_sk = np.einsum("pkn,pcn,pn->pck", jacobian, d_dc0, weight) + residual_sk
    a_ss = np.einsum("pcn,pdn,pn->pcd", d_dc0, d_dc0, weight) + residual_ss

    reduced, projection = eliminate_fitted_starts(a_kk, a_sk, a_ss)

    expected = a_kk - sum(a_sk[p].T @ np.linalg.solve(a_ss[p], a_sk[p]) for p in range(n_particles))
    np.testing.assert_allclose(reduced, expected, rtol=1e-12)
    np.testing.assert_allclose(projection, np.linalg.solve(a_ss, a_sk).transpose(0, 2, 1))


def _session_with_pool(tmp_path: Path, backend: str = "madng") -> TrackingSession:
    seq_file = tmp_path / "psb.seq"
    seq_file.write_text("! placeholder sequence\n")
    session = TrackingSession(
        PSB(ring=3, sequence_file=seq_file, backend=backend),
        WorkerRuntime(1),
        TrackingDataConfig(1),
        TrackingParameters(),
        turn_batches=[[1]],
        validation_turn_batches=[],
        file_turn_map={1: 0},
        worker_plans={},
        machine_deltaps=[0.0],
    )
    session.training = _uncertainty_pool([0], [[Ack(0), {}]])
    return session


def test_unknown_hessian_names_are_rejected_before_any_worker_is_touched(tmp_path: Path) -> None:
    session = _session_with_pool(tmp_path)

    with pytest.raises(ValueError, match="exact"):
        session.stop_and_collect_uncertainty(2, ("exact",))

    assert session.training.workers[0].join_calls == 0


def test_second_order_preflight_counts_magnet_knobs_not_gains(tmp_path: Path) -> None:
    from accelmo.fitting.bpm_gains import bpm_gain_names

    session = _session_with_pool(tmp_path)
    gains = dict.fromkeys(bpm_gain_names([f"BPM{i}" for i in range(50000)]), 0.0)
    session.n_mad_knobs = 2

    # 100000 gains and two magnet knobs: the maps of two knobs fit in memory
    assert second_order_blocker(session.training, len(gains) + 2, 1) is not None
    assert session._true_hessian_blocker() is None
