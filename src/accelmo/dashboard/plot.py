"""A log-scale line plot drawn with braille characters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Sequence

BRAILLE_BASE = 0x2800
#: Braille dot bit for (dot column in the cell, dot row in the cell).
DOT_BITS = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))
DOTS_PER_CELL_X = 2
DOTS_PER_CELL_Y = 4
LABEL_WIDTH = 7
REFERENCE_STYLE = "dim"
AXIS_STYLE = "dim"


@dataclass(frozen=True)
class Series:
    """Values per epoch (``nan`` or non-positive = no point) and the style they are drawn in."""

    values: Sequence[float | None]
    style: str


def _decade_label(low: float, high: float) -> str:
    """The label of the whole decade in ``[low, high)`` (log10 units), else blank."""
    decade = int(np.ceil(low))
    return f"1e{decade}" if decade < high else ""


def _dots(series: Series, count: int, rows: int, cols: int, top: float, span: float) -> np.ndarray:
    """The dot grid of one series: its points joined by straight segments."""
    grid = np.zeros((rows, cols), dtype=bool)
    values = np.array([np.nan if v is None else v for v in series.values], dtype=float)
    valid = np.flatnonzero(np.isfinite(values) & (values > 0))
    xs = np.rint(valid * (cols - 1) / max(count - 1, 1)).astype(int)
    ys = np.rint((top - np.log10(values[valid])) / span * (rows - 1)).astype(int)
    ys = np.clip(ys, 0, rows - 1)
    for x0, y0, x1, y1 in zip(xs[:-1], ys[:-1], xs[1:], ys[1:], strict=True):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        grid[
            np.rint(np.linspace(y0, y1, steps + 1)).astype(int),
            np.rint(np.linspace(x0, x1, steps + 1)).astype(int),
        ] = True
    if len(valid) == 1:
        grid[ys[0], xs[0]] = True
    return grid


def log_plot(
    series: Sequence[Series], width: int, height: int, *, reference: float | None = 1.0
) -> list[Text]:
    """Draw ``series`` on a log y-axis, ``width`` x ``height`` cells, one :class:`Text` per row.

    Later series are drawn over earlier ones. A dashed line is drawn at ``reference``
    when it lies inside the data's range. The last row is the x axis.
    """
    count = max((len(s.values) for s in series), default=0)
    logs = np.log10(
        [v for s in series for v in s.values if v is not None and np.isfinite(v) and v > 0]
    )
    if count == 0 or len(logs) == 0:
        return [Text(" " * (LABEL_WIDTH + width)) for _ in range(height + 1)]
    low, high = float(logs.min()), float(logs.max())
    if high - low < 1.0:
        middle = (high + low) / 2
        low, high = middle - 0.5, middle + 0.5
    pad = 0.05 * (high - low)
    low, high = low - pad, high + pad
    span = high - low

    rows, cols = height * DOTS_PER_CELL_Y, width * DOTS_PER_CELL_X
    layers = [(_dots(s, count, rows, cols, high, span), s.style) for s in series]
    if reference is not None and low <= np.log10(reference) <= high:
        line = np.zeros((rows, cols), dtype=bool)
        y = round((high - np.log10(reference)) / span * (rows - 1))
        line[y, ::2] = True
        layers.insert(0, (line, REFERENCE_STYLE))

    lines = []
    for row in range(height):
        label = _decade_label(high - (row + 1) * span / height, high - row * span / height)
        text = Text(label.rjust(LABEL_WIDTH - 1) + " ", style=AXIS_STYLE)
        for col in range(width):
            bits, style = 0, ""
            for grid, layer_style in layers:
                cell = grid[
                    row * DOTS_PER_CELL_Y : (row + 1) * DOTS_PER_CELL_Y,
                    col * DOTS_PER_CELL_X : (col + 1) * DOTS_PER_CELL_X,
                ]
                if cell.any():
                    bits |= sum(DOT_BITS[dx][dy] for dy, dx in zip(*np.nonzero(cell), strict=True))
                    style = layer_style
            text.append(chr(BRAILLE_BASE + bits) if bits else " ", style=style)
        lines.append(text)
    end = f" epoch {count - 1}"
    axis = Text(
        " " * (LABEL_WIDTH - 1) + "└" + "─" * (width - 1 - len(end)) + end, style=AXIS_STYLE
    )
    lines.append(axis)
    return lines
