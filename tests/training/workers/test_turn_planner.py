"""Turn planning must train every file, whatever the worker cap."""

from __future__ import annotations

from types import SimpleNamespace

from accelmo.tracking.dispatch.turn_planner import WorkerTurnPlanner


class _Plan:
    """A tracking plan with one batch per group of workers."""

    one_start_per_worker = False


def _plan(num_workers: int, num_files: int, turns_per_file: int = 50):
    file_map = {}
    for f in range(num_files):
        for t in range(turns_per_file):
            file_map[f * turns_per_file + t] = f
    planner = WorkerTurnPlanner(
        _Plan(),
        SimpleNamespace(num_workers=num_workers, num_batches=1, use_fixed_bpm=True),
        shuffle_turns=lambda _turns: None,
    )
    batches = planner.build_turn_batches(
        available_turns=list(file_map),
        file_map=file_map,
        num_files=num_files,
        workers_per_batch=2,  # ACD markers: forward + backward
    )
    trained = {file_map[batch[0]] for batch in batches}
    return batches, trained


def test_worker_cap_below_file_count_still_trains_every_file():
    # psb_md's default: 8 workers, 2 range specs -> 4 batches for 15 files.
    batches, trained = _plan(num_workers=8, num_files=15)
    assert trained == set(range(15))
    assert len(batches) == 15
    assert sum(len(b) for b in batches) == 15 * 50


def test_worker_cap_above_file_count_is_unchanged():
    batches, trained = _plan(num_workers=60, num_files=10)
    assert trained == set(range(10))
    assert len(batches) == 30
