"""Loss-only trials of the closed-orbit fit: a rejected Levenberg-Marquardt step costs plain orbits.

On both backends, with one reference orbit shared by every series (a reference worker) and
solved by each series itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from accelmo.backend import get_backend
from accelmo.config import WorkerRuntime
from accelmo.fitting.config import SequenceConfig
from accelmo.fitting.pool import WorkerPool
from accelmo.fitting.priors import inverse_variances
from accelmo.fitting.protocol import Start
from accelmo.machine.accelerators import PSB
from accelmo.optimisers.levenberg_marquardt import LevenbergMarquardtConfig
from accelmo.poco import ClosedOrbitFitter, ClosedOrbitMeasurement, ClosedOrbitSeries
from accelmo.poco.fitter import _KnobProblem
from accelmo.poco.workers.closed_orbit import ClosedOrbitWorker

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.serial

BACKENDS = ("madng", "xtrack")
CORRECTORS = ("kbr3dhz2l4", "kbr3dvt3l4", "kbr3dhz2l4")
KICKS = (1e-4, -1e-4, -2e-4)
KNOB_SPREAD = 2e-3  # integrated dk1l [1/m]
KNOB_SIGMA = 1e-3
BPM_ERROR = 1e-5
ORBIT_SCALE = 1e-4  # [m]; random targets, so every residual is far from zero
SEED = 3
TOL_LOSS = 1e-9  # relative
TOL_FIT = 1e-8  # relative
MAX_ITERATIONS = 8


def _accelerator(seq_psb: Path, backend: str) -> PSB:
    return PSB(ring=3, sequence_file=seq_psb, errors={"quad": {"k1"}}, backend=backend)


def _series(seq_psb: Path, backend: str) -> list[ClosedOrbitSeries]:
    model = get_backend(backend).open_model(
        _accelerator(seq_psb, backend), magnet_range="$start/$end"
    )
    bpms = list(model.all_bpms)
    model.close()
    rng = np.random.default_rng(SEED)

    def orbit() -> pd.DataFrame:
        columns = {}
        for plane in ("X", "Y"):
            columns[plane] = rng.normal(0.0, ORBIT_SCALE, len(bpms))
            columns[f"ERR{plane}"] = np.full(len(bpms), BPM_ERROR)
        return pd.DataFrame(columns, index=bpms)

    reference = orbit()
    return [
        ClosedOrbitSeries(
            measurements=(ClosedOrbitMeasurement(orbit(), reference=reference),),
            machine_state={name: kick},
            label=f"{name}-{index}",
        )
        for index, (name, kick) in enumerate(zip(CORRECTORS, KICKS, strict=True))
    ]


def _fitter(seq_psb: Path, backend: str, *, shared_reference: bool, loss_only_trials: bool):
    series = _series(seq_psb, backend)
    model = get_backend(backend).open_model(
        _accelerator(seq_psb, backend), magnet_range="$start/$end"
    )
    names = list(model.knob_names)
    model.close()
    fitter = ClosedOrbitFitter(
        accelerator=_accelerator(seq_psb, backend),
        sequence_config=SequenceConfig(magnet_range="$start/$end"),
        series=series,
        lm_config=LevenbergMarquardtConfig(max_iterations=MAX_ITERATIONS),
        knob_sigmas=dict.fromkeys(names, KNOB_SIGMA),
        shared_reference=shared_reference,
        runtime=WorkerRuntime(num_workers=len(series)),
    )
    fitter.loss_only_trials = loss_only_trials
    return fitter


def _knobs(names: list[str], scale: float) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 1)
    return {name: float(scale * rng.uniform(-KNOB_SPREAD, KNOB_SPREAD)) for name in names}


@pytest.mark.parametrize("shared_reference", [True, False])
@pytest.mark.parametrize("backend", BACKENDS)
def test_trial_loss_equals_the_full_evaluation_loss(
    seq_psb: Path, backend: str, shared_reference: bool
) -> None:
    fitter = _fitter(seq_psb, backend, shared_reference=shared_reference, loss_only_trials=True)
    names = list(fitter.machine.knob_names)
    pool = WorkerPool()
    try:
        for worker_id, (config, data) in enumerate(fitter.worker_payloads):
            pool.spawn(ClosedOrbitWorker, worker_id, data, config, fitter.runtime)
        pool.channels.send_all(Start(dict.fromkeys(names, 0.0)))
        problem = _KnobProblem(
            names,
            lambda knobs, **kwargs: fitter._evaluate_workers(pool.channels, knobs, **kwargs),
            np.diag(inverse_variances(fitter.knob_sigma)),
            [],
            loss_only_trials=True,
        )
        problem.evaluate(np.zeros(len(names)), iteration=0)  # a full evaluation aligns the series
        for scale in (0.0, 1.0, 1.0 + 1e-6, 0.3, 2.0):
            params = np.array(list(_knobs(names, scale).values()))
            trial = problem.trial_loss(params)
            full = problem.evaluate(params, iteration=1).loss
            assert trial == pytest.approx(full, rel=TOL_LOSS), (backend, shared_reference, scale)
    finally:
        pool.stop()
        fitter.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_fit_is_the_same_with_and_without_loss_only_trials(seq_psb: Path, backend: str) -> None:
    results = {}
    for loss_only_trials in (True, False):
        fitter = _fitter(seq_psb, backend, shared_reference=True, loss_only_trials=loss_only_trials)
        try:
            result = fitter.run()
        finally:
            fitter.close()
        results[loss_only_trials] = result
    on, off = results[True], results[False]
    assert on.diagnostics.iterations > on.diagnostics.accepted_evaluations  # steps were rejected
    assert on.diagnostics.iterations == off.diagnostics.iterations
    assert on.diagnostics.best_loss == pytest.approx(off.diagnostics.best_loss, rel=TOL_FIT)
    fitted_on = np.array([on.knobs[name] for name in sorted(on.knobs)])
    fitted_off = np.array([off.knobs[name] for name in sorted(off.knobs)])
    scale = np.abs(fitted_off).max()
    assert np.abs(fitted_on - fitted_off).max() / scale < TOL_FIT
