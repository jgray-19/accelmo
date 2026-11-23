"""Fast contracts for the reusable closed-orbit fitter framework."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from accelmo.backend.madng import MadInterface
from accelmo.config import WorkerRuntime
from accelmo.fitting.protocol import Evaluate, GradReply
from accelmo.fitting.worker import WorkerConfig
from accelmo.machine.accelerators import PSB
from accelmo.poco import (
    ClosedOrbitMeasurement,
    ClosedOrbitSeries,
)
from accelmo.poco.fitter import _KnobProblem
from accelmo.poco.workers.closed_orbit import (
    ClosedOrbitMeasurementData,
    ClosedOrbitSeriesData,
    ClosedOrbitWorker,
)
from accelmo.poco.workers.closed_twiss import Observable

if TYPE_CHECKING:
    from pathlib import Path


def _frame(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"X": [value, value], "ERRX": [1.0, 1.0]},
        index=["BPM1", "BPM2"],
    )


def test_each_measurement_keeps_its_own_target_and_momenta() -> None:
    series = ClosedOrbitSeries(
        measurements=(
            ClosedOrbitMeasurement(_frame(1.0), pt=-1e-3, reference_pt=0.0),
            ClosedOrbitMeasurement(_frame(2.0), pt=2e-3, reference_pt=1e-3),
        ),
        machine_state={"kick": 5e-5},
    )
    assert [item.orbit.iloc[0, 0] for item in series.measurements] == [1.0, 2.0]
    assert [item.pt for item in series.measurements] == [-1e-3, 2e-3]
    assert [item.reference_pt for item in series.measurements] == [0.0, 1e-3]


def test_knob_prior_adds_to_the_workers_chi2() -> None:
    """The workers send chi²; the LM point is the MAP chi² ``chi² + k^T P k`` and its derivatives."""
    rng = np.random.default_rng(0)
    jac, target = rng.normal(size=(12, 3)), rng.normal(size=12)
    weights = rng.uniform(1.0, 4.0, 12)
    sigma = np.array([0.5, 0.0, 2.0])
    prior = np.diag(np.divide(1.0, sigma**2, out=np.zeros(3), where=sigma > 0))
    k = rng.normal(size=3)

    residual = jac @ k - target
    normal = jac.T @ (weights[:, None] * jac)
    reply = GradReply(
        0,
        float(residual @ (weights * residual)),
        2.0 * jac.T @ (weights * residual),
        normal,
    )
    names = ["q1.dk1l", "q2.dk1l", "q3.dk1l"]
    point = _KnobProblem(names, lambda _knobs: [reply], prior, []).evaluate(k, iteration=1)

    chi2 = float(residual @ (weights * residual)) + float(k @ prior @ k)
    assert point.loss == pytest.approx(chi2, rel=1e-14)
    assert point.grad == pytest.approx(
        2.0 * jac.T @ (weights * residual) + 2.0 * prior @ k, rel=1e-13
    )
    assert point.hessian == pytest.approx(2.0 * (normal + prior), rel=1e-13)


def test_measurements_in_one_series_add_up_like_separate_series(seq_psb: Path) -> None:
    """Measurements batched into one series are evaluated as independently as separate series.

    Two momenta measured against the same on-momentum reference: one series holding
    both must give the loss, gradient and normal matrix of the two one-measurement
    series summed. A signal orbit leaking between momenta, or a reference solved at
    the wrong momentum, breaks the equality.
    """
    accelerator = PSB(ring=3, sequence_file=seq_psb, misalignments={"quad": {"dx"}})
    iface = MadInterface(accelerator, create_knobs=True)
    bpms = list(iface.all_bpms)
    knobs = {
        name: 1e-5 * (i + 1) for i, name in enumerate(n for n in iface.knob_names if n != "pt")
    }
    iface.close()
    config = WorkerConfig(
        accelerator=accelerator,
        magnet_range="$start/$end",
    )

    def measurement(pt: float, target: float) -> ClosedOrbitMeasurementData:
        observable = Observable("x", np.full(len(bpms), target), np.full(len(bpms), 1e-8))
        return ClosedOrbitMeasurementData([observable], pt=pt, reference_pt=0.0)

    def evaluate(*measurements: ClosedOrbitMeasurementData) -> GradReply:
        series = ClosedOrbitSeriesData(bpm_names=bpms, measurements=list(measurements))
        worker = ClosedOrbitWorker(None, 0, [series], config, WorkerRuntime(num_workers=1))
        worker.on_start(knobs)
        try:
            return worker.evaluate(Evaluate(knobs))
        finally:
            worker.engine.close()

    low, high = measurement(1e-3, 1e-4), measurement(2e-3, -2e-4)
    together = evaluate(low, high)
    apart = [evaluate(low), evaluate(high)]

    assert together.loss == pytest.approx(sum(r.loss for r in apart), rel=1e-12)
    np.testing.assert_allclose(together.grad, sum(r.grad for r in apart), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(together.normal, sum(r.normal for r in apart), rtol=1e-12, atol=0.0)
