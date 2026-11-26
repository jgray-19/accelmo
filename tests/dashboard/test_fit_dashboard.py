"""The fit dashboard draws the numbers it is given."""

from __future__ import annotations

import io
import logging
import sys

import numpy as np
import pytest
from rich.console import Console

from accelmo.dashboard import FitDashboard
from accelmo.dashboard.plot import BRAILLE_BASE, LABEL_WIDTH, Series, log_plot

N_POINTS = 1000
N_PARAMS = 50
SIGMA = 1e-3


def _console() -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    return Console(file=stream, force_terminal=True, width=110, color_system=None), stream


def _dashboard(console: Console, gain_names=()) -> FitDashboard:
    names = ["a", "b", "c", *gain_names]
    dashboard = FitDashboard("test fit", console=console)
    dashboard.begin_fit(
        max_epochs=10,
        n_points=N_POINTS,
        n_params=N_PARAMS,
        start_knobs=dict.fromkeys(names, 0.0),
        knob_sigmas={"a": SIGMA, "b": SIGMA, "c": 0.0},
        gain_names=gain_names,
    )
    return dashboard


def test_decades_fall_on_their_rows_and_the_line_runs_corner_to_corner() -> None:
    values = [10.0**-k for k in range(9)]
    lines = log_plot([Series(values, "")], width=20, height=8, reference=None)
    assert len(lines) == 9
    top, bottom = lines[0].plain[LABEL_WIDTH:], lines[7].plain[LABEL_WIDTH:]
    assert top[0] != " " and bottom[-1] != " "
    assert top[-1] == " " and bottom[0] == " "
    assert all(ord(c) >= BRAILLE_BASE or c == " " for c in top + bottom)
    labels = [line.plain[:LABEL_WIDTH].strip() for line in lines[:8]]
    assert [lab for lab in labels if lab] == [f"1e{-k}" for k in range(1, 8)] or "1e-7" in labels


def test_reference_line_only_inside_the_data_range() -> None:
    def rows(values: list[float], reference: float | None) -> list[str]:
        return [line.plain for line in log_plot([Series(values, "")], 10, 4, reference=reference)]

    inside, inside_bare = rows([10.0, 0.1], 1.0), rows([10.0, 0.1], None)
    assert sum(a != b for a, b in zip(inside, inside_bare, strict=True)) == 1
    assert rows([1e-3, 1e-5], 1.0) == rows([1e-3, 1e-5], None)


def test_reduced_chi2_and_best_panel_numbers() -> None:
    console, stream = _console()
    with _dashboard(console, gain_names=["g"]) as dashboard:
        knobs = {"a": 2.5 * SIGMA, "b": 0.5 * SIGMA, "c": 7.0, "g": 4e-3}
        dashboard.update(
            epoch=3,
            loss=2.0,
            validation_loss=2.5,
            grad_norm=1e-6,
            lr=1e-3,
            epoch_time=1.0,
            true_diff=1e-4,
            knobs=knobs,
            new_best=True,
        )
    text = stream.getvalue()
    expected = 2.0 * N_POINTS / (N_POINTS - N_PARAMS)
    assert f"{expected:.3e}" in text
    assert "1/2 moved >1σ" in text
    assert f"rms {np.sqrt((2.5**2 + 0.5**2) / 2):.2f}σ" in text
    assert "at epoch 3" in text
    assert "max 4.0e-03" in text


def test_stalled_and_diverging_status() -> None:
    console, stream = _console()
    with _dashboard(console) as dashboard:
        knobs = {"a": 0.0, "b": 0.0, "c": 0.0}
        for epoch, loss in enumerate([1.0, 0.5, 0.5, 20.0]):
            dashboard.update(
                epoch=epoch,
                loss=loss,
                validation_loss=None,
                grad_norm=1.0,
                lr=1.0,
                epoch_time=1.0,
                true_diff=0.0,
                knobs=knobs,
                new_best=epoch < 2,
            )
    assert "diverging" in stream.getvalue()


def test_not_a_terminal_prints_nothing() -> None:
    stream = io.StringIO()
    dashboard = FitDashboard("test fit", console=Console(file=stream, force_terminal=False))
    dashboard.begin_fit(max_epochs=1, n_points=10, n_params=1, start_knobs={}, knob_sigmas={})
    with dashboard, dashboard.phase("data"):
        dashboard.update(
            epoch=0,
            loss=1.0,
            validation_loss=None,
            grad_norm=1.0,
            lr=1.0,
            epoch_time=1.0,
            true_diff=0.0,
            knobs={},
            new_best=True,
        )
    assert stream.getvalue() == ""
    assert not dashboard.chi2


@pytest.mark.parametrize(
    ("first", "best", "text"),
    [(1.0, 0.59, "↓ 41%"), (1.0, 1e-3, "↓ ÷1,000"), (1.0, 0.0, "↓ to zero")],
)
def test_change_text(first: float, best: float, text: str) -> None:
    from accelmo.dashboard.fit import format_change

    assert format_change(first, best).plain == text


def test_strip_shows_each_stage_and_the_facts() -> None:
    console, stream = _console()
    dashboard = FitDashboard("test fit", console=console)
    dashboard.fact("points", "5,000")
    dashboard.skip("uncertainty")
    with dashboard.phase("data"), dashboard.phase("model"):
        pass
    with pytest.raises(RuntimeError), dashboard.phase("workers"):
        raise RuntimeError
    dashboard.close()
    last_frame = stream.getvalue().split("✓ data")[-1]
    assert "✓ model" in last_frame
    assert "✗ workers" in last_frame
    assert "○ fit" in last_frame
    assert "uncertainty" not in last_frame
    assert "points" in last_frame
    assert "5,000" in last_frame


def test_console_log_handlers_are_detached_while_the_dashboard_is_live() -> None:
    console, _ = _console()
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    dashboard = FitDashboard("test fit", console=console)
    try:
        assert handler in root.handlers
        with dashboard.phase("data"):
            assert handler not in root.handlers
        assert handler not in root.handlers
        dashboard.close()
        assert handler in root.handlers
    finally:
        dashboard.close()
        root.removeHandler(handler)


def test_lm_loop_feeds_the_dashboard_every_accepted_iteration() -> None:
    from accelmo.fitting.lm_loop import LMPoint, LMProblem, run_levenberg_marquardt
    from accelmo.optimisers.levenberg_marquardt import LevenbergMarquardtConfig

    target = np.array([1.5, -0.5])

    class Quadratic(LMProblem):
        def evaluate(self, params: np.ndarray, iteration: int) -> LMPoint:
            residual = params - target
            return LMPoint(float(residual @ residual), 2 * residual, 2 * np.eye(2))

    console, stream = _console()
    dashboard = FitDashboard("lm fit", console=console)
    dashboard.begin_fit(
        max_epochs=20,
        n_points=None,
        n_params=2,
        start_knobs={"a": 0.0, "b": 0.0},
        knob_sigmas={"a": 1.0, "b": 1.0},
        loss_is_total=True,
        unit="iter",
        rate_name="λ",
    )
    with dashboard.phase("data"):
        optimiser, diagnostics = run_levenberg_marquardt(
            np.zeros(2),
            Quadratic(),
            LevenbergMarquardtConfig(max_iterations=20),
            param_names=["a", "b"],
            dashboard=dashboard,
        )
    dashboard.close()
    assert len(dashboard.chi2) == diagnostics.accepted_evaluations
    assert dashboard.chi2[0] == pytest.approx(float(target @ target))
    assert dashboard.best is not None
    assert dashboard.best.chi2 == pytest.approx(optimiser.best.loss)
    assert dashboard.best.knobs["a"] == pytest.approx(optimiser.best.value[0])
    text = stream.getvalue()
    assert "iter" in text
    assert "χ² " in text
    assert "stopped:" in text


def test_rejected_steps_advance_the_bar_and_are_counted() -> None:
    console, stream = _console()
    with _dashboard(console) as dashboard:
        dashboard.update(
            epoch=0,
            loss=1.0,
            validation_loss=None,
            grad_norm=1.0,
            lr=1e-3,
            epoch_time=1.0,
            true_diff=None,
            knobs={"a": 0.0, "b": 0.0, "c": 0.0},
            new_best=True,
        )
        dashboard.rejected(epoch=1, loss=3.0, grad_norm=1.0, lr=1e-2, epoch_time=1.0)
        dashboard.rejected(epoch=2, loss=float("nan"), grad_norm=0.0, lr=1e-1, epoch_time=1.0)
        dashboard.finish("max_iterations")
    text = stream.getvalue()
    assert dashboard.epoch == 2
    assert dashboard.rejected_steps == 2
    assert len(dashboard.chi2) == 1
    assert "3/10" in text
    assert "rejected (orbit lost)" in text
    assert "stopped: max_iterations · 2 rejected" in text


def test_resource_line_counts_a_child_process() -> None:
    import subprocess
    import time

    from accelmo.dashboard.resources import ResourceMonitor

    monitor = ResourceMonitor()
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]) as child:
        try:
            time.sleep(0.2)
            monitor.sample()
            assert monitor.processes >= 2
            assert monitor.memory > 0
            assert monitor.peak_memory >= monitor.memory
        finally:
            child.kill()
    console, stream = _console()
    with _dashboard(console):
        pass
    assert "cpu " in stream.getvalue()
    assert " GB (peak " in stream.getvalue()
