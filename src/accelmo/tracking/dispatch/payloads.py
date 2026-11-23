"""Worker payload construction.

Turns :class:`~accelmo.tracking.data_manager.FileTracks` grids and
worker configs into the immutable arrays a tracking worker receives.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from accelmo.fitting.bpm_gains import point_gain_index
from accelmo.fitting.weights import variance_to_weight
from accelmo.tracking.worker import (
    PrecomputedTrackingWeights,
    TrackingData,
    active_observables,
)

if TYPE_CHECKING:
    from accelmo.machine.accelerators import Accelerator
    from accelmo.tracking.data_manager import FileTracks
    from accelmo.tracking.dispatch.setup import TrackingWorkerConfig


WorkerPayload: TypeAlias = tuple["TrackingData", "TrackingWorkerConfig"]


def observation_turn_offsets(cols: np.ndarray, sdir: int) -> np.ndarray:
    """Return the per-point turn offset implied by a range's column order.

    Markers are stored in ring order cycled to the file's turn boundary, so a
    contiguous tracking range is a monotonic column sequence. Each place the
    sequence jumps backwards (forwards, when tracking in reverse) is the ring
    wrap, i.e. the point where the track moves into the next (previous) turn.
    """
    steps = np.diff(cols)
    wrapped = steps < 0 if sdir == 1 else steps > 0
    return np.cumsum(np.concatenate([[0], wrapped.astype(np.int64)])) * sdir


#: (position, momentum) value columns of the x and y planes.
_PLANE_KEYS = (("x", "px"), ("y", "py"))


@dataclass
class _PayloadArrays:
    """The arrays of one payload, filled file by file; inactive planes keep infinite variance."""

    pos: np.ndarray
    mom: np.ndarray
    pos_var: np.ndarray
    mom_var: np.ndarray
    init_coords: np.ndarray
    init_variances: np.ndarray
    reading_ids: np.ndarray
    init_reading_ids: np.ndarray

    @classmethod
    def empty(cls, n_turns: int, n_points: int) -> _PayloadArrays:
        return cls(
            pos=np.zeros((n_turns, n_points, 2)),
            mom=np.zeros((n_turns, n_points, 2)),
            pos_var=np.full((n_turns, n_points, 2), np.inf),
            mom_var=np.full((n_turns, n_points, 2), np.inf),
            init_coords=np.zeros((n_turns, 6)),
            init_variances=np.full((n_turns, 4), np.inf),
            reading_ids=np.zeros((n_turns, n_points), dtype=np.int64),
            init_reading_ids=np.zeros(n_turns, dtype=np.int64),
        )

    def fill_init(self, batch_rows, values, rows, col, active) -> None:
        """Start coordinates and variances of the active planes at grid column ``col``."""
        for plane, (pos_key, mom_key) in enumerate(_PLANE_KEYS):
            if not active[plane]:
                continue
            for offset, key in enumerate((pos_key, mom_key)):
                self.init_coords[batch_rows, 2 * plane + offset] = values[key][rows, col]
                self.init_variances[batch_rows, 2 * plane + offset] = values[f"var_{key}"][
                    rows, col
                ]

    def fill_points(self, batch_rows, values, rows, cols, active) -> None:
        """Readings and variances of the active planes at every observed point."""
        for plane, (pos_key, mom_key) in enumerate(_PLANE_KEYS):
            if not active[plane]:
                continue
            self.pos[batch_rows, :, plane] = values[pos_key][rows, cols]
            self.mom[batch_rows, :, plane] = values[mom_key][rows, cols]
            self.pos_var[batch_rows, :, plane] = values[f"var_{pos_key}"][rows, cols]
            self.mom_var[batch_rows, :, plane] = values[f"var_{mom_key}"][rows, cols]


class WorkerPayloadBuilder:
    """Build tracking payload arrays and shared weights for workers."""

    def __init__(self, accelerator: Accelerator, gain_bpms: tuple[str, ...] = ()) -> None:
        self.accelerator = accelerator
        self.gain_bpms = gain_bpms

    def compute_pt(self, file_idx: int, machine_deltaps: list[float]) -> float:
        """Return the reference ``pt`` for a measurement file's energy offset."""
        return self.accelerator.dp2pt(machine_deltaps[file_idx])

    def make_tracking_data(
        self,
        turn_batch: list[int],
        file_turn_map: dict[int, int],
        config: TrackingWorkerConfig,
        machine_deltaps: list[float],
        tracks: dict[int, FileTracks],
    ) -> TrackingData:
        """Build the serialisable tracking payload for one worker.

        Both transverse planes are always allocated; the ones the file does not
        excite keep their infinite variance and so contribute nothing to the loss.
        """
        bpm_names = config.bpm_names
        if not bpm_names:
            raise ValueError(f"No observed BPMs for worker range {config.geometry.bpm_range}")

        sdir = config.geometry.sdir
        track_turns = config.geometry.n_turns
        init_marker = config.geometry.init_point
        n_turns = len(turn_batch)
        n_points = len(bpm_names) * track_turns

        arrays = _PayloadArrays.empty(n_turns, n_points)
        pts = np.empty(n_turns)
        start_turn_offsets = np.zeros(n_turns, dtype=np.int64)

        active = (config.kick_plane in ("x", "xy"), config.kick_plane in ("y", "xy"))

        batch_rows_by_file: dict[int, list[int]] = defaultdict(list)
        turns_by_file: dict[int, list[int]] = defaultdict(list)
        for i, turn in enumerate(turn_batch):
            file_idx = file_turn_map[turn]
            batch_rows_by_file[file_idx].append(i)
            turns_by_file[file_idx].append(turn)
            pts[i] = self.compute_pt(file_idx, machine_deltaps)
            start_turn_offsets[i] = turn - tracks[file_idx].first_turn

        for file_idx, batch_rows in batch_rows_by_file.items():
            file_tracks = tracks[file_idx]
            if init_marker not in file_tracks.marker_col:
                raise ValueError(
                    f"Init marker '{init_marker}' not found in tracking data for file {file_idx}"
                )
            values = file_tracks.values

            cols = file_tracks.cols(bpm_names * track_turns)
            start_rows = file_tracks.rows(turns_by_file[file_idx])
            rows = start_rows[:, None] + observation_turn_offsets(cols, sdir)[None, :]

            init_col = file_tracks.marker_col[init_marker]
            # One id per (file, turn, marker) grid cell, shared by every worker.
            file_base = np.int64(int(file_idx) << 32)
            n_markers = len(file_tracks.markers)
            arrays.reading_ids[batch_rows] = file_base + rows * n_markers + cols
            arrays.init_reading_ids[batch_rows] = file_base + start_rows * n_markers + init_col
            arrays.fill_init(batch_rows, values, start_rows, init_col, active)
            arrays.init_coords[batch_rows, 5] = pts[batch_rows]

            # An all-zero initial state means the marker row was missing from the
            # measurement: tracking from it would silently fit noise.
            dead = np.all(arrays.init_coords[batch_rows, :4] == 0.0, axis=1)
            if dead.any():
                turn = turns_by_file[file_idx][int(np.argmax(dead))]
                raise ValueError(
                    f"Initial coordinates for turn {turn} at marker {init_marker} are all zero"
                )
            arrays.fill_points(batch_rows, values, rows, cols, active)

        for array in (arrays.pos, arrays.mom, arrays.pos_var, arrays.mom_var):
            array.setflags(write=False)

        return TrackingData(
            position_comparisons=arrays.pos,
            momentum_comparisons=arrays.mom,
            position_variances=arrays.pos_var,
            momentum_variances=arrays.mom_var,
            init_coords=arrays.init_coords,
            init_pts=pts,
            reading_ids=arrays.reading_ids,
            init_reading_ids=arrays.init_reading_ids,
            init_variances=arrays.init_variances,
            start_turn_offsets=start_turn_offsets,
            precomputed_weights=None,
            gain_bpms=self.gain_bpms,
            point_gain_index=point_gain_index(bpm_names * track_turns, self.gain_bpms),
        )

    @staticmethod
    def attach_weights(payloads: list[WorkerPayload], *, optimise_momenta: bool = True) -> int:
        """Attach inverse-variance weights to every payload; returns the number of weighted points."""
        n_points = 0
        for data, config in payloads:
            active = active_observables(config.kick_plane, optimise_momenta)
            weights = PrecomputedTrackingWeights(
                x=variance_to_weight(data.position_variances[:, :, 0]),
                y=variance_to_weight(data.position_variances[:, :, 1]),
                px=variance_to_weight(data.momentum_variances[:, :, 0]),
                py=variance_to_weight(data.momentum_variances[:, :, 1]),
            )
            n_points += sum(int(np.count_nonzero(getattr(weights, name))) for name in active)
            data.precomputed_weights = weights
        return n_points
