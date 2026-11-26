"""The live terminal dashboard of a gradient-descent fit."""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from accelmo.dashboard.plot import Series, log_plot
from accelmo.dashboard.resources import GIB, ResourceMonitor

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path
    from types import TracebackType

#: Share of the terminal width the progress panel takes.
LEFT_WIDTH_FRACTION = 0.66
MIN_WIDTH = 80
PLOT_HEIGHT = 8
#: Fits shorter than this many steps get a smaller plot.
SHORT_FIT_STEPS = 30
SHORT_PLOT_HEIGHT = 4
WARNING_LINES = 3
#: Redraws per second, so the resource line moves during a long step.
REFRESH_PER_SECOND = 1
#: Epochs averaged for the epoch time and ETA.
ETA_WINDOW = 10
#: Epochs without a new best before the status turns "stalled".
STALL_EPOCHS = 10
#: A loss this many times the best one is flagged as diverging.
DIVERGENCE_FACTOR = 10.0
#: A knob counts as moved when it is this many prior widths from its start.
MOVED_SIGMAS = 1.0
#: A reduction of the loss beyond this is printed as a factor, not a percentage.
FACTOR_BELOW_RATIO = 0.01

GOOD, WARN, BAD = "green", "yellow", "red"
TRAIN_STYLE, VALIDATION_STYLE = "cyan", "magenta"
LABEL_STYLE = "dim"
#: The stages of a run, in order; the strip at the top shows each one's state.
PHASES = ("data", "model", "workers", "fit", "uncertainty")
PHASE_MARKS = {"pending": ("○", LABEL_STYLE), "running": ("●", WARN), "done": ("✓", GOOD)}
PHASE_FAILED = ("✗", BAD)


def format_duration(seconds: float) -> str:
    """``73`` -> ``1m13s``; ``4000`` -> ``1h06m``."""
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def format_change(first: float, best: float) -> Text:
    """How far ``best`` is below ``first``: a percentage, or a factor when it is large."""
    ratio = best / first if first > 0 else 1.0
    if ratio >= 1.0:
        return Text("no improvement", style=WARN)
    if best <= 0.0:
        return Text("↓ to zero", style=GOOD)
    if ratio < FACTOR_BELOW_RATIO:
        return Text(f"↓ ÷{1 / ratio:,.0f}", style=GOOD)
    return Text(f"↓ {100 * (1 - ratio):.0f}%", style=GOOD)


@dataclass(frozen=True)
class Best:
    """The best epoch so far."""

    epoch: int
    chi2: float
    validation: float | None
    true_diff: float | None
    knobs: dict[str, float]


@dataclass
class PhaseState:
    """One stage of the run: whether it has started or finished, and how long it took."""

    status: str = "pending"
    detail: str = ""
    started: float = 0.0
    seconds: float = 0.0
    skipped: bool = False


def _writes_to_terminal(handler: logging.Handler) -> bool:
    """Whether ``handler`` prints to stdout or stderr (a file handler does not)."""
    return (
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        and handler.stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
    )


class _WarningTail(logging.Handler):
    """Keeps the last few WARNING+ records for the dashboard's bottom strip."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.lines: deque[Text] = deque(maxlen=WARNING_LINES)

    def emit(self, record: logging.LogRecord) -> None:
        style = BAD if record.levelno >= logging.ERROR else WARN
        message = f"{record.name.rsplit('.', 1)[-1]}: {record.getMessage()}"
        self.lines.append(Text(f"⚠ {message}", style=style, no_wrap=True, overflow="ellipsis"))


class FitDashboard:
    """The live terminal view of a fit: a strip of stages, progress, loss history and best-so-far.

    Does nothing when the output is not a terminal; the run log has the numbers then.
    The display starts at the first :meth:`phase` (or on entering the ``with`` block) and
    stays on screen after :meth:`close`. Mark each stage of the run with :meth:`phase`,
    state what the run is made of with :meth:`fact`, call :meth:`begin_fit` before the epoch
    loop and :meth:`update` after every epoch.
    """

    def __init__(self, title: str, *, console: Console | None = None, enabled: bool = True):
        self.title = title
        self.console = console or Console()
        self.enabled = enabled and self.console.is_terminal
        self.phases = {name: PhaseState() for name in PHASES}
        self.facts: dict[str, str] = {}
        self.run_log_path: Path | None = None
        self.max_epochs = 0
        self.loss_scale = 1.0
        self.chi2_label = "χ²ᵣ"
        self.unit = "epoch"
        self.rate_name = "lr"
        self.start_knobs: dict[str, float] = {}
        self.sigmas: dict[str, float] = {}
        self.gain_names: list[str] = []
        self.stage_epochs: int | None = None
        self.chi2: list[float] = []
        self.validation: list[float | None] = []
        self.epoch_times: list[float] = []
        self.epoch = -1
        self.best: Best | None = None
        self.last_best_epoch = -1
        self.stop_reason: str | None = None
        self.rejected_steps = 0
        self.resources = ResourceMonitor()
        self._warnings = _WarningTail()
        self._live = Live(
            console=self.console,
            get_renderable=self._render,
            auto_refresh=True,
            refresh_per_second=REFRESH_PER_SECOND,
            transient=False,
        )
        self._running = False
        self._console_handlers: list[logging.Handler] = []
        self._row = Text()

    def begin_fit(
        self,
        *,
        max_epochs: int,
        n_points: int | None,
        n_params: int,
        start_knobs: Mapping[str, float],
        knob_sigmas: Mapping[str, float],
        gain_names: Sequence[str] = (),
        stage_epochs: int | None = None,
        loss_is_total: bool = False,
        unit: str = "epoch",
        rate_name: str = "lr",
    ) -> None:
        """Start the fit stage.

        ``n_points`` and ``n_params`` give the degrees of freedom of the reduced chi-square
        (``loss`` is chi-square per point); ``n_points=None`` shows the total chi-square. ``knob_sigmas`` are the prior widths of the
        magnet knobs, ``start_knobs`` their starting values, ``gain_names`` the BPM gains.
        With ``loss_is_total`` the loss given to :meth:`update` is the chi-square itself, not per
        point. ``unit`` names a step (``epoch``, ``iter``) and ``rate_name`` the step-size
        parameter shown beside the gradient (``lr``, ``λ``).
        """
        self.max_epochs = max_epochs
        if n_points is None:
            self.loss_scale, self.chi2_label = 1.0, "χ²"
        else:
            ndf = n_points - n_params if n_points > n_params else n_points
            self.loss_scale = 1 / ndf if loss_is_total else n_points / ndf
        self.unit = unit
        self.rate_name = rate_name
        self.start_knobs = dict(start_knobs)
        self.sigmas = {k: s for k, s in knob_sigmas.items() if s > 0 and k not in gain_names}
        self.gain_names = list(gain_names)
        self.stage_epochs = stage_epochs
        fit = self.phases["fit"]
        fit.status, fit.started = "running", time.monotonic()
        self._refresh()

    def fact(self, label: str, value: object) -> None:
        """Show ``label: value`` in the run panel."""
        self.facts[label] = str(value)
        self._refresh()

    def skip(self, name: str) -> None:
        """Hide stage ``name``: this run does not have it."""
        self.phases[name].skipped = True
        self._refresh()

    def set_detail(self, name: str, detail: str) -> None:
        """Text shown after stage ``name`` in the strip."""
        self.phases[name].detail = detail
        self._refresh()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Mark stage ``name`` running for the duration of the block, then done (or failed)."""
        state = self.phases[name]
        state.status, state.started = "running", time.monotonic()
        self._start()
        self._refresh()
        try:
            yield
        except BaseException:
            state.status, state.seconds = "failed", time.monotonic() - state.started
            self._refresh()
            raise
        state.status, state.seconds = "done", time.monotonic() - state.started
        self._refresh()

    def uncertainty_progress(self, done: int, total: int, eta: float | None) -> None:
        """Show how far the uncertainty stage has got."""
        eta_text = "" if eta is None else f" ≤ {format_duration(eta)} left"
        self.set_detail("uncertainty", f"{done}/{total}{eta_text}")

    def _start(self) -> None:
        if self.enabled and not self._running:
            self._running = True
            root = logging.getLogger()
            self._console_handlers = [h for h in root.handlers if _writes_to_terminal(h)]
            for handler in self._console_handlers:
                root.removeHandler(handler)
            root.addHandler(self._warnings)
            self._live.start()

    def close(self) -> None:
        """Stop redrawing; the last frame stays on screen."""
        if self._running:
            self._running = False
            root = logging.getLogger()
            root.removeHandler(self._warnings)
            for handler in self._console_handlers:
                root.addHandler(handler)
            self._console_handlers = []
            self._live.stop()

    def __enter__(self) -> Self:
        self._start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def finish(self, reason: str) -> None:
        """The fit stage is over: show why the loop ended."""
        self.stop_reason = reason
        self._row.append(f" · stopped: {reason}", style=LABEL_STYLE)
        if self.rejected_steps:
            self._row.append(f" · {self.rejected_steps} rejected", style=WARN)
        fit = self.phases["fit"]
        fit.status, fit.seconds = "done", time.monotonic() - fit.started
        self._refresh()

    def update(
        self,
        *,
        epoch: int,
        loss: float,
        validation_loss: float | None,
        grad_norm: float,
        lr: float,
        epoch_time: float,
        true_diff: float | None,
        knobs: Mapping[str, float],
        new_best: bool,
        note: str | None = None,
    ) -> None:
        """Record one finished epoch and redraw."""
        if not self.enabled:
            return
        self.epoch = epoch
        self.chi2.append(loss * self.loss_scale)
        self.validation.append(validation_loss)
        self.epoch_times.append(epoch_time)
        if new_best:
            self.last_best_epoch = epoch
            self.best = Best(epoch, self.chi2[-1], validation_loss, true_diff, dict(knobs))
        self._row = self._status_row(grad_norm, lr, note)
        self._refresh()

    def rejected(
        self, *, epoch: int, loss: float, grad_norm: float, lr: float, epoch_time: float
    ) -> None:
        """Record a step the optimiser turned down (``loss`` is NaN when the orbit was lost)."""
        if not self.enabled:
            return
        self.epoch = epoch
        self.rejected_steps += 1
        self.epoch_times.append(epoch_time)
        outcome = (
            "orbit lost" if np.isnan(loss) else f"{self.chi2_label} {loss * self.loss_scale:.3e}"
        )
        self._row = Text.assemble(
            (f"grad {grad_norm:.1e} · {self.rate_name} {lr:.1e} · ", LABEL_STYLE),
            (f"● rejected ({outcome}), retrying from best · {self.rejected_steps} so far", WARN),
        )
        self._refresh()

    def _refresh(self) -> None:
        if self._running:
            self._live.refresh()

    def _render(self) -> Group:
        width = max(self.console.width, MIN_WIDTH)
        left_width = int(width * LEFT_WIDTH_FRACTION)
        layout = Table.grid(expand=True)
        layout.add_column(width=left_width)
        layout.add_column(ratio=1)
        if self.phases["fit"].status == "pending":
            layout.add_row(self._facts_panel(), "")
        else:
            layout.add_row(
                self._progress_panel(left_width),
                self._best_panel() if self.best else self._facts_panel(),
            )
        footer = (
            [Text(f"run log {self.run_log_path}", style=LABEL_STYLE)] if self.run_log_path else []
        )
        return Group(self._strip(), layout, self._resources_line(), *footer, *self._warnings.lines)

    def _resources_line(self) -> Text:
        self.resources.sample()
        usage = self.resources
        return Text(
            f"cpu {usage.cores:.1f}/{usage.machine_cores} cores · "
            f"mem {usage.memory / GIB:.1f} GB (peak {usage.peak_memory / GIB:.1f}) · "
            f"{usage.processes} processes",
            style=LABEL_STYLE,
        )

    def _strip(self) -> Text:
        """One line: each stage with its mark, detail and time."""
        parts = []
        for name, state in self.phases.items():
            if state.skipped:
                continue
            mark, style = PHASE_FAILED if state.status == "failed" else PHASE_MARKS[state.status]
            seconds = (
                time.monotonic() - state.started if state.status == "running" else state.seconds
            )
            text = Text.assemble((f"{mark} ", style), name)
            if state.detail:
                text.append(f" {state.detail}")
            if state.status != "pending" and name != "fit":
                text.append(f" {format_duration(seconds)}", style=LABEL_STYLE)
            elif name == "fit" and state.status == "running":
                text.append(f" {self.epoch + 1}/{self.max_epochs}")
            parts.append(text)
        return Text(" · ").join(parts)

    def _facts_panel(self) -> Panel:
        table = Table.grid(padding=(0, 2))
        table.add_column(style=LABEL_STYLE)
        table.add_column()
        for label, value in self.facts.items():
            table.add_row(label, value)
        return Panel(table, title="run", title_align="left")

    def _stage_title(self) -> str:
        if not self.stage_epochs:
            return self.title
        stages = -(-self.max_epochs // self.stage_epochs)
        return f"{self.title} · stage {max(self.epoch, 0) // self.stage_epochs + 1}/{stages}"

    def _progress_panel(self, panel_width: int) -> Panel:
        done = self.epoch + 1
        recent = self.epoch_times[-ETA_WINDOW:]
        eta = (
            f"≤ {format_duration((self.max_epochs - done) * float(np.mean(recent)))} left"
            if recent and not self.stop_reason
            else ""
        )
        bar = Table.grid(expand=True)
        bar.add_column(ratio=1)
        bar.add_column(no_wrap=True)
        bar.add_row(
            ProgressBar(total=self.max_epochs, completed=done),
            Text(f"  {done}/{self.max_epochs}  {eta}"),
        )
        inner = panel_width - 4
        plot_width = max(inner - 7, 10)
        plot = log_plot(
            [
                Series(self.chi2, TRAIN_STYLE),
                Series(self.validation, VALIDATION_STYLE),
            ],
            plot_width,
            PLOT_HEIGHT if self.max_epochs > SHORT_FIT_STEPS else SHORT_PLOT_HEIGHT,
        )
        header = Text.assemble(
            (f"{self.chi2_label} ", LABEL_STYLE),
            (f"{self.chi2[-1]:.3e}" if self.chi2 else "—", TRAIN_STYLE),
        )
        if any(v is not None for v in self.validation):
            last = next(v for v in reversed(self.validation) if v is not None)
            header.append_text(
                Text.assemble(("   val ", LABEL_STYLE), (f"{last:.3e}", VALIDATION_STYLE))
            )
        return Panel(
            Group(bar, header, *plot, self._row),
            title=self._stage_title(),
            title_align="left",
        )

    def _status_row(self, grad_norm: float, lr: float, note: str | None) -> Text:
        since_best = self.epoch - self.last_best_epoch
        best_chi2 = self.best.chi2 if self.best else self.chi2[-1]
        if self.chi2[-1] > DIVERGENCE_FACTOR * best_chi2:
            status = ("● diverging", BAD)
        elif since_best == 0:
            status = ("● new best", GOOD)
        elif since_best >= STALL_EPOCHS:
            status = (f"● stalled {since_best} {self.unit}s", WARN)
        else:
            status = ("● improving" if self.last_best_epoch >= 0 else "● running", GOOD)
        row = Text.assemble(
            (
                f"grad {grad_norm:.1e} · {self.rate_name} {lr:.1e} · {np.mean(self.epoch_times[-ETA_WINDOW:]):.1f}s/{self.unit} · ",
                LABEL_STYLE,
            ),
            status,
        )
        if note:
            row.append(f" · {note}", style=LABEL_STYLE)
        return row

    def _best_panel(self) -> Panel:
        assert self.best is not None
        best = self.best
        table = Table.grid(padding=(0, 2))
        table.add_column(style=LABEL_STYLE)
        table.add_column()
        table.add_row(self.chi2_label, Text(f"{best.chi2:.3e}", style=f"bold {GOOD}"))
        if best.validation is not None:
            table.add_row("validation", f"{best.validation:.3e}")
        if best.true_diff is not None:
            table.add_row("true_diff", f"{best.true_diff:.3e}")
        table.add_row("vs start", format_change(self.chi2[0], best.chi2))
        table.add_row("found", f"at {self.unit} {best.epoch}")
        knobs = best.knobs
        if self.sigmas:
            moves = np.array([(knobs[k] - self.start_knobs[k]) / s for k, s in self.sigmas.items()])
            moved = int(np.count_nonzero(np.abs(moves) > MOVED_SIGMAS))
            table.add_row("", "")
            table.add_row("knobs", f"{moved}/{len(moves)} moved >{MOVED_SIGMAS:g}σ")
            table.add_row("", f"rms {np.sqrt(np.mean(moves**2)):.2f}σ")
        if self.gain_names:
            gains = np.array([knobs[g] for g in self.gain_names])
            table.add_row("", "")
            table.add_row("gains", f"rms {np.sqrt(np.mean(gains**2)):.1e}")
            table.add_row("", f"max {np.abs(gains).max():.1e}")
        return Panel(table, title=f"best · {self.unit} {best.epoch}", title_align="left")
