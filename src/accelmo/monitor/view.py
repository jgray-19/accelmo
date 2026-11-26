"""What a panel gets to draw from."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd
    from bokeh.models import DataRange1d, Range1d

    from accelmo.runlog import StageLog


@dataclass
class MonitorView:
    """The selected stage and step, and everything known about them so far.

    Optics-related fields are ``None`` (or empty) until the background workers
    have computed them, or when the monitor runs without an optics source; a
    panel should draw what it can and say what it is waiting for.
    """

    stage: StageLog
    #: The selected step; -1 is the start of the stage.
    step: int
    #: Logged scalars, indexed by step.
    history: pd.DataFrame
    #: Knob values, indexed by step, one column per knob.
    knobs: pd.DataFrame
    #: The source's knob-independent tables (measurement, design, ...).
    references: dict[str, pd.DataFrame] | None
    #: The optics at :attr:`step`.
    optics: pd.DataFrame | None
    #: The source's summary of every step computed so far, indexed by step.
    summaries: pd.DataFrame
    #: Shared x-range for plots against longitudinal position, so zooming one
    #: zooms all and survives a refresh.
    s_range: Range1d | DataRange1d


def uncertainty_status(info: dict[str, Any] | None) -> str | None:
    """One status line for the uncertainty stage recorded in ``uncertainty.json``, or ``None``."""
    if info is None:
        return None
    hessian = info.get("hessian", "?")
    note = f" ({info['note']})" if info.get("note") else ""
    match info.get("state"):
        case "running" if info.get("total"):
            eta = info.get("eta")
            when = "ETA unknown" if eta is None else f"ETA {eta:.0f} s"
            return (
                f"Uncertainty ({hessian}{note}): {info['done']}/{info['total']} batches, "
                f"{info['elapsed']:.0f} s elapsed, {when}"
            )
        case "running":
            return f"Uncertainty ({hessian}{note}): running"
        case "done":
            return f"Uncertainty ({hessian}{note}): done in {info['seconds']:.1f} s"
        case "failed":
            return f"Uncertainty failed after {info['seconds']:.1f} s: {info['error']}"
    return None
