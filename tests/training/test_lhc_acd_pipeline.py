"""LHC ACD marker pipeline: xsuite truth tracking -> ACDMarkerFitter -> worker loss.

Exact (noise-free) xsuite AC-dipole data on real LHC beam 1 with injected quadrupole errors
is fed straight to ``ACDMarkerFitter``. The MAD-NG workers must reproduce it at the truth
strengths and not at the nominal ones.

pytest /afs/cern.ch/work/j/jmgray/private/sgd-magnet-tuner/tests/training/test_lhc_acd_pipeline.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
import tfs
from xtrack_tools.acd import run_ac_dipole_tracking
from xtrack_tools.env import initialise_env
from xtrack_tools.monitors import process_tracking_data

from accelmo.backend.madng import MadInterface
from accelmo.config import SimulationConfig
from accelmo.fitting.config import OutputConfig, SequenceConfig
from accelmo.machine.accelerators import LHC
from accelmo.tracking.config.models import MeasurementConfig, MeasurementDetails
from accelmo.tracking.fitter import ACDMarkerFitter, FitterOptions
from tests.training.controller_test_utils import evaluate_controller_worker_loss, make_lm_config
from tests.training.helpers import TRACK_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.serial

KINETIC_ENERGY = 6800.0
QUAD_REL_RMS = 5e-4
RAMP_TURNS = 100
FLATTOP_TURNS = 20
DRIVEN_TUNE_OFFSETS = (-0.01, 0.01)
HORIZONTAL_EXCITATION = 0.000371554879506
VERTICAL_EXCITATION = 0.000415765635123
BPM_PATTERN = r"(?i)^bpm.*$"
NUM_WORKERS = 2
FITTED_QUAD_PREFIX = "MQ."
# measured 2026-11-22: 1.0e-4 (initial 5.9e-13, true 6.1e-17)
TRUE_LOSS_OVER_INITIAL_LOSS = 3e-4


def _accelerator(seq_b1: Path) -> LHC:
    return LHC(beam=1, sequence_file=seq_b1, kinetic_energy=KINETIC_ENERGY, errors={"quad": {"k1"}})


@pytest.fixture(scope="module")
def lhc_acd_track(tmp_path_factory: pytest.TempPathFactory, seq_b1: Path):
    """Truth quad strengths and the xsuite ACD tracking parquet, built once."""
    root = tmp_path_factory.mktemp("lhc_acd_pipeline")
    accelerator = _accelerator(seq_b1)
    iface = MadInterface(accelerator, create_knobs=True)
    try:
        perturbed, _ = iface.apply_magnet_perturbations(
            rel_error=QUAD_REL_RMS, seed=24, magnet_type="q"
        )
        truth = {k: v for k, v in perturbed.items() if k.upper().startswith(FITTED_QUAD_PREFIX)}
        twiss = iface.run_twiss(observe=1, deltap=0.0, coupling=True, method=6)
    finally:
        iface.close()
    driven_tunes = [
        float(twiss.headers["q1"]) % 1.0 + DRIVEN_TUNE_OFFSETS[0],
        float(twiss.headers["q2"]) % 1.0 + DRIVEN_TUNE_OFFSETS[1],
    ]

    env = initialise_env(
        matched_tunes={},
        magnet_strengths=truth,
        corrector_table=tfs.TfsDataFrame({"name": [], "kind": [], "hkick": [], "vkick": []}),
        sequence_file=seq_b1,
        seq_name=accelerator.seq_name,
        kinetic_energy=KINETIC_ENERGY,
        strict_set=False,
    )
    line = env[accelerator.seq_name.lower()].copy()
    monitored = run_ac_dipole_tracking(
        line=line,
        acd_marker=accelerator.ac_dipole_name,
        sequence_name=accelerator.seq_name,
        tws=line.twiss(method="4d"),
        ramp_turns=RAMP_TURNS,
        flattop_turns=FLATTOP_TURNS,
        driven_tunes=driven_tunes,
        bpm_pattern=BPM_PATTERN,
        state_markers=True,
        horizontal_excitation=HORIZONTAL_EXCITATION,
        vertical_excitation=VERTICAL_EXCITATION,
    )
    tracked = process_tracking_data(
        monitored, ramp_turns=RAMP_TURNS, flattop_turns=FLATTOP_TURNS, add_variance_columns=True
    )
    tracked["bunch_number"] = 0
    tracked = tracked.loc[:, TRACK_COLUMNS].copy()
    tracked["name"] = (
        tracked["name"]
        .astype(str)
        .replace(
            {
                accelerator.acd_marker_name(side).upper(): accelerator.acd_marker_name(side)
                for side in ("before", "after")
            }
        )
    )
    path = root / "lhc_acd_track.parquet"
    tracked.to_parquet(path, index=False)
    return path, truth


def test_acd_markers_and_bpms_are_tracked_for_every_flattop_turn(
    seq_b1: Path, lhc_acd_track
) -> None:
    path, _ = lhc_acd_track
    tracked = pd.read_parquet(path)
    accelerator = _accelerator(seq_b1)
    for side in ("before", "after"):
        rows = tracked[tracked["name"] == accelerator.acd_marker_name(side)]
        assert len(rows) == FLATTOP_TURNS, side
    bpm_rows = tracked[tracked["name"].str.match(BPM_PATTERN)]
    assert bpm_rows["name"].nunique() > 500
    assert bpm_rows.groupby("name")["turn"].nunique().eq(FLATTOP_TURNS).all()


def test_worker_loss_prefers_the_true_quadrupole_strengths(
    tmp_path: Path, seq_b1: Path, lhc_acd_track
) -> None:
    path, truth = lhc_acd_track
    fitter = ACDMarkerFitter(
        accelerator=_accelerator(seq_b1),
        optimiser_config=make_lm_config(),
        simulation_config=SimulationConfig(num_workers=NUM_WORKERS, num_batches=2),
        sequence_config=SequenceConfig(magnet_range="$start/$end"),
        measurement_config=MeasurementConfig(
            {path: MeasurementDetails(interface_options={}, machine_deltap=0.0)}
        ),
        options=FitterOptions(
            initial_knob_strengths={},
            true_strengths=truth,
            output_config=OutputConfig(
                write_run_log=False,
                include_uncertainty=False,
                mad_logfile=tmp_path / "acd_mad.log",
            ),
        ),
    )

    # Errors are injected only into the fitted family, so every true strength is a fit knob.
    fitted = set(fitter.machine.initial_knobs)
    assert fitted
    assert set(fitter.machine.true_strengths) == fitted == {k for k in truth if k.endswith(".dk1l")}

    initial_loss = evaluate_controller_worker_loss(fitter, fitter.machine.initial_knobs)
    true_loss = evaluate_controller_worker_loss(fitter, fitter.machine.true_strengths)
    assert true_loss < initial_loss * TRUE_LOSS_OVER_INITIAL_LOSS, (
        f"initial={initial_loss:.3e}, true={true_loss:.3e}"
    )
