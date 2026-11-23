"""Pure utilities for grouping accelerator magnets and their knob values."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

_GROUPED_QFO_KNOB_RE = re.compile(r"^(BR\.QFO)CELL(\d+)(\.(?:dk1l|dx|dy|tilt))$", re.IGNORECASE)
_CELL_TILT_KNOB_RE = re.compile(r"^(BR\.)QCELL(\d+)(\.tilt)$", re.IGNORECASE)


def expand_psb_grouped_quadrupole_knobs(
    values: dict[str, float], *, differential_tilt: bool = False
) -> dict[str, float]:
    """Expand shared PSB QFO-cell values to physical per-element knob names.

    ``differential_tilt``: ``BR.QCELL<c>.tilt`` rolls both QFO<c> by +v and QDE<c> by -v.
    """
    expanded: dict[str, float] = {}
    for name, value in values.items():
        if differential_tilt and (cell_tilt := _CELL_TILT_KNOB_RE.fullmatch(name)):
            prefix, cell, suffix = cell_tilt.groups()
            expanded[f"{prefix}QFO{cell}1{suffix}"] = float(value)
            expanded[f"{prefix}QFO{cell}2{suffix}"] = float(value)
            expanded[f"{prefix}QDE{cell}{suffix}"] = -float(value)
            continue
        match = _GROUPED_QFO_KNOB_RE.fullmatch(name)
        if match is None:
            expanded[name] = float(value)
            continue
        prefix, cell, suffix = match.groups()
        expanded[f"{prefix}{cell}1{suffix}"] = float(value)
        expanded[f"{prefix}{cell}2{suffix}"] = float(value)
    return expanded


_PHYSICAL_QFO_KNOB_RE = re.compile(r"^(BR\.QFO)(\d+)([12])(\.(?:dk1l|dx|dy|tilt))$", re.IGNORECASE)
_PHYSICAL_QUAD_TILT_RE = re.compile(r"^(BR\.)(?:QFO(\d+)[12]|QDE(\d+))(\.tilt)$", re.IGNORECASE)


def collapse_psb_grouped_quadrupole_knobs(
    values: dict[str, float], *, differential_tilt: bool = False
) -> dict[str, float]:
    """Collapse equal physical QFO pairs into PSB cell-grouped knobs.

    ``differential_tilt``: QFO<c>1/2 and QDE<c> tilts (QDE = -QFO) collapse to ``BR.QCELL<c>.tilt``.
    """
    collapsed: dict[str, float] = {}
    grouped: dict[str, list[tuple[str, float]]] = {}
    cells: dict[str, list[tuple[str, float]]] = {}
    for name, value in values.items():
        if differential_tilt and (tilt := _PHYSICAL_QUAD_TILT_RE.fullmatch(name)):
            prefix, qfo_cell, qde_cell, suffix = tilt.groups()
            sign = 1.0 if qfo_cell else -1.0
            cells.setdefault(f"{prefix}QCELL{qfo_cell or qde_cell}{suffix}", []).append(
                (name, sign * float(value))
            )
            continue
        match = _PHYSICAL_QFO_KNOB_RE.fullmatch(name)
        if match is None:
            collapsed[name] = float(value)
            continue
        prefix, cell, _magnet, suffix = match.groups()
        grouped_name = f"{prefix}CELL{cell}{suffix}"
        grouped.setdefault(grouped_name, []).append((name, float(value)))

    for grouped_name, members in grouped.items():
        if len(members) != 2:
            raise ValueError(
                f"Grouped initial knob {grouped_name} needs both physical QFO magnets, "
                f"got {[name for name, _ in members]}"
            )
        first, second = members
        if not np.isclose(first[1], second[1], rtol=1e-12, atol=1e-15):
            raise ValueError(
                f"Grouped initial knob {grouped_name} has unequal physical values: "
                f"{first[0]}={first[1]:.15g}, {second[0]}={second[1]:.15g}"
            )
        collapsed[grouped_name] = first[1]

    for cell_name, members in cells.items():
        if len(members) != 3:
            raise ValueError(
                f"Cell tilt knob {cell_name} needs both QFO and the QDE tilt, "
                f"got {[name for name, _ in members]}"
            )
        signed = [v for _, v in members]
        if not np.allclose(signed, signed[0], rtol=1e-12, atol=1e-15):
            raise ValueError(
                f"Cell tilt knob {cell_name} is not differential (QFO = -QDE): "
                + ", ".join(f"{n}={v:.15g}" for n, v in values.items() if n in dict(members))
            )
        collapsed[cell_name] = signed[0]
    return collapsed


def normalise_lhcbend_magnets(
    true_strengths_dict: dict[str, float], magnet_lengths: dict[str, float]
) -> dict[str, float]:
    """Normalise LHC bend names and length-weight-average grouped strengths.

    S-bends are grouped regardless of sign. R-bends are grouped separately for
    positive and negative strengths.
    """
    patterns = [
        (r"(MB\.)([ABCD])([0-9]+[LR][1-8]\.B[12])\.dk0l", "sbend"),
        (r"(MB[RXWAL]\w*\.)([A-G]?)([0-9]+[LR][1-8].*)\.dk0l", "rbend"),
    ]

    normalised: dict[str, Any] = {}
    counts = {"sbend": 0, "rbend": 0}

    for key, value in true_strengths_dict.items():
        matched = False
        for pattern, bend_type in patterns:
            if match := re.match(pattern, key):
                base_key = f"{match.group(1)}{match.group(3)}.dk0l"
                if bend_type == "rbend":
                    sign_suffix = "_p" if value >= 0 else "_n"
                    new_key = base_key.replace(".dk0l", f"{sign_suffix}.dk0l")
                else:
                    new_key = base_key

                if new_key not in normalised:
                    normalised[new_key] = []
                normalised[new_key].append((value, magnet_lengths[key]))
                counts[bend_type] += 1
                matched = True
                break

        if not matched:
            normalised[key] = value

    LOGGER.info("Normalised %d sbends and %d rbends.", counts["sbend"], counts["rbend"])

    def length_weighted_average(k_list: list[tuple[float, float]]) -> float:
        total_length = sum(length for _, length in k_list)
        weighted_sum = sum(k * length for k, length in k_list)
        return weighted_sum / total_length if total_length != 0 else 0.0

    return {
        key: length_weighted_average(value) if isinstance(value, list) else value
        for key, value in normalised.items()
    }
