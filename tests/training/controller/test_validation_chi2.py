"""Training and held-out validation losses are both chi² per weighted point: ≈ 1 at the truth."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from accelmo.config import OptimiserConfig, TrackingDataConfig, WorkerRuntime
from accelmo.fitting.config import OutputConfig, SequenceConfig
from accelmo.machine.machine_state import merge_machine_states
from accelmo.tracking.config.helpers import single_measurement_config
from accelmo.tracking.fitter import ArcByArcFitter, FitterOptions
from tests.training.controller.test_uncertainty_calibration import (
    BPM_END_POINTS,
    BPM_START_POINTS,
    _start_rows,
)
from tests.training.controller_test_utils import (
    DPP_VALUE,
    FLATTOP_TURNS,
    POSITION_STD_DEV,
    _generate_nonoise_track,
)

if TYPE_CHECKING:
    from pathlib import Path

    from accelmo.backend.madng.interface import MadInterface

pytestmark = pytest.mark.serial

#: Standard deviations of chi²/N allowed around 1 (each reading may be seen by two workers).
N_SIGMA = 5.0


def test_training_and_validation_chi2_per_point_are_one_at_the_truth(
    tmp_path: Path, loaded_interface: MadInterface
) -> None:
    """Readings carry Gaussian noise of their declared σ and the starts are exact, so at the
    true knobs both the training and the held-out chi² per weighted point are 1 within noise."""
    track = tmp_path / "track.parquet"
    corrector, _, tune_knobs = _generate_nonoise_track(
        loaded_interface, FLATTOP_TURNS, track, DPP_VALUE
    )
    frame = pd.read_parquet(track)
    start_rows = _start_rows(frame)
    frame.loc[start_rows, ["var_x", "var_y"]] = np.inf
    rng = np.random.default_rng(20261123)
    for column in ("x", "y"):
        frame.loc[~start_rows, column] += rng.normal(
            0.0, POSITION_STD_DEV, int((~start_rows).sum())
        )
    frame.to_parquet(track, index=False)

    fitter = ArcByArcFitter(
        loaded_interface.accelerator.copy_with(optimise_energy=True),
        OptimiserConfig(
            max_epochs=1, warmup_epochs=0, warmup_lr_start=1e-8, max_lr=1e-8, min_lr=1e-8
        ),
        WorkerRuntime(num_workers=3),
        TrackingDataConfig(num_batches=2, optimise_momenta=False, validation_fraction=0.3),
        SequenceConfig(magnet_range="BPM.9R2.B1/BPM.9L3.B1"),
        single_measurement_config(track, machine_state=merge_machine_states(corrector, tune_knobs)),
        BPM_START_POINTS,
        BPM_END_POINTS,
        options=FitterOptions(
            output_config=OutputConfig(mad_logfile=tmp_path / "mad.log", write_run_log=False)
        ),
    )
    truth = {"pt": loaded_interface.accelerator.dp2pt(DPP_VALUE)}
    session = fitter.start_session(
        {**fitter.machine.initial_model_values, **truth}, enable_validation=True
    )
    try:
        training = sum(session.training_losses(truth)) / session.n_training_points
        validation = session.validation_loss(truth)
        n_train, n_val = session.n_training_points, session.n_validation_points
    finally:
        session.terminate()

    print(f"chi2/N training {training:.4f} (N={n_train}), validation {validation:.4f} (N={n_val})")
    assert n_val > 0
    assert training == pytest.approx(1.0, abs=N_SIGMA * np.sqrt(4.0 / n_train))
    assert validation == pytest.approx(1.0, abs=N_SIGMA * np.sqrt(4.0 / n_val))
