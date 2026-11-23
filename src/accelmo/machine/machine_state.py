"""The machine state a model is put in: name/value pairs applied to the model's globals.

A name is either a MAD-X variable (``kbrqf``, ``kqtf.a12b1``, a corrector's ``kbr3dhz2l4``) or an
element attribute written ``<element>.<attribute>`` (``br3.dhz2l4.kick``). Element names are
upper case in the loaded sequence and matched case-insensitively. Assigning an element attribute
replaces its deferred expression (a kicker's ``kick := kbr3dhz2l4``) with the number.

A state is given as a mapping, as a ``name<TAB>value`` knobs file, or as a TFS corrector table
(columns ``ename``, ``kind``, ``hkick``/``vkick``), which becomes one ``<ename>.<attribute>`` entry per kicker.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

import tfs
from pymadng_utils.io.utils import read_knobs

MachineState: TypeAlias = Mapping[str, float] | str | Path

#: Kicker kinds of a corrector table: the attribute set and the table column holding its value.
_KICKER_ATTRIBUTES: dict[str, tuple[tuple[str, str], ...]] = {
    "hkicker": (("kick", "hkick"),),
    "vkicker": (("kick", "vkick"),),
    "tkicker": (("hkick", "hkick"), ("vkick", "vkick")),
}
_TABLE_COLUMNS = {"ename", "kind", "hkick", "hkick_old", "vkick", "vkick_old"}


def resolve_machine_state(state: MachineState) -> dict[str, float]:
    """Name/value pairs of *state*, whether given directly, as a knobs file or as a TFS corrector table."""
    if not isinstance(state, str | Path):
        return {str(name): float(value) for name, value in state.items()}
    path = Path(state)
    if path.suffix.lower() == ".tfs":
        return _corrector_table_state(tfs.read(path))
    return {name: float(value) for name, value in read_knobs(path).items()}


def merge_machine_states(*states: MachineState | None) -> dict[str, float] | None:
    """One state from several, later ones overriding earlier ones; ``None`` when none is given."""
    given = [state for state in states if state is not None]
    if not given:
        return None
    merged: dict[str, float] = {}
    for state in given:
        merged.update(resolve_machine_state(state))
    return merged


def _corrector_table_state(table: tfs.TfsDataFrame) -> dict[str, float]:
    """One ``<ename>.<attribute>`` entry per kicker whose strength differs from ``*_old``."""
    missing = _TABLE_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(
            f"Corrector table is missing required columns: {', '.join(sorted(missing))}"
        )
    changed = (table["hkick"] != table["hkick_old"]) | (table["vkick"] != table["vkick_old"])
    state = {}
    for row in table[changed].itertuples():  # ty:ignore[unresolved-attribute]
        for attribute, column in _KICKER_ATTRIBUTES.get(row.kind, ()):
            state[f"{row.ename}.{attribute}"] = float(getattr(row, column))
    return state
