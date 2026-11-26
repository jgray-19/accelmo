"""The xsuite line of one accelerator: knobs as line variables, observed BPMs, machine state.

Element names are upper case on the accelmo side (as in MAD-NG) and lower case in xsuite.
Knobs keep the MAD-NG names (``<ELEMENT>.dk1l``, ``.dk0l``, ``.kick``) and the integrated
convention: ``dk1l`` adds ``dk1l / L`` to the quadrupole ``k1``.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xtrack as xt
from xtrack_tools.env import create_xsuite_environment

from accelmo.machine.machine_state import MachineState, resolve_machine_state
from accelmo.machine.patterns import lua_to_regex

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from accelmo.machine.accelerators import Accelerator

LOGGER = logging.getLogger(__name__)

# Worker output (progress bars, xtrack's own deprecated names) would print over the dashboard.
xt.settings.progress_indicator = "suppress"
warnings.filterwarnings(
    "ignore",
    message=r"(SRotation|XRotation|YRotation|XYShift) is deprecated",
    category=FutureWarning,
)

QUAD_MODEL = "mat-kick-mat-exact"
QUAD_INTEGRATOR = "yoshida-6"
QUAD_KICKS = 1  # yoshida-6 rounds up to a full 7-kick slice
#: Thick magnets that are exact drifts when every strength is zero, and their strength attributes.
_ZERO_FIELD_STRENGTHS = {
    "Quadrupole": ("k1", "k1s"),
    "Sextupole": ("k2", "k2s"),
    "Octupole": ("k3", "k3s"),
}
_ZERO_FIELD_ATTRIBUTES = (
    "knl",
    "ksl",
    "knl_rel",
    "ksl_rel",
    "shift_x",
    "shift_y",
    "shift_s",
    "rot_s_rad",
)

#: MAD-X base element kinds the knob families select.
_KINDS = {
    "quadrupole",
    "sbend",
    "rbend",
    "hkicker",
    "vkicker",
    "kicker",
    "multipole",
    "sextupole",
    "monitor",
    "marker",
}
_DECLARATION = re.compile(r"^\s*([A-Za-z_][\w.$]*)\s*:\s*([A-Za-z_][\w.$]*)", re.MULTILINE)
#: (kind, attribute) -> knob suffix; the only knob types this backend builds. xsuite thin
#: ``Multipole`` correctors have no TPSA fields, so corrector kicks are machine state only.
_SUPPORTED = {
    ("quadrupole", "k1"): "dk1l",
    ("sbend", "k0"): "dk0l",
    ("rbend", "k0"): "dk0l",
}


def madx_element_kinds(sequence_file: Path) -> dict[str, str]:
    """Base MAD-X kind of every element declared in a sequence file (upper-case names)."""
    parents = {
        name.upper(): parent.lower()
        for name, parent in _DECLARATION.findall(sequence_file.read_text())
    }
    kinds = {}
    for name, first in parents.items():
        parent, seen = first, {name}
        while parent not in _KINDS and parent.upper() in parents and parent.upper() not in seen:
            seen.add(parent.upper())
            parent = parents[parent.upper()]
        kinds[name] = parent
    return kinds


def _range_indices(names: Sequence[str], span: str) -> tuple[int, int]:
    """``first/last`` (``$start``/``$end`` allowed) as inclusive indices into ``names``."""
    first, last = span.split("/")
    upper = [name.upper() for name in names]
    start = 0 if first == "$start" else upper.index(first.upper())
    stop = len(names) - 1 if last == "$end" else upper.index(last.upper())
    return start, stop


class XtrackModel:
    """One xsuite line with knobs; ``line`` is rebuilt by :meth:`cycle_to_start`."""

    def __init__(
        self,
        accelerator: Accelerator,
        *,
        magnet_range: str = "$start/$end",
        bpm_range: str | None = None,
        bad_bpms: Sequence[str] | None = None,
        machine_state: MachineState | None = None,
        initial_model_values: Mapping[str, float] | None = None,
    ) -> None:
        if accelerator.seq_dir != 1:
            raise NotImplementedError(
                "xtrack backend: backward sequences (LHC beam 2) are not supported"
            )
        if accelerator.optimise_energy or accelerator.misalignments:
            raise NotImplementedError("xtrack backend: pt and misalignment knobs are not supported")
        self.accelerator = accelerator
        self.magnet_range = magnet_range
        self.bpm_range = bpm_range or magnet_range
        self.bad_bpms = {name.upper() for name in bad_bpms or ()}
        sequence_file = Path(accelerator.sequence_file)
        self.kinds = madx_element_kinds(sequence_file)
        self.env = create_xsuite_environment(
            sequence_file, kinetic_energy=accelerator.kinetic_energy, seq_name=accelerator.seq_name
        )
        full = self.env[accelerator.seq_name.lower()]
        self.particle_ref = full.particle_ref.copy()
        table = full.get_table()
        # MAD-NG tracks with a 100 m aperture; the TPSA kernels have no aperture elements.
        self.names = [
            name
            for name, kind in zip(full.element_names, table.element_type)
            if not str(kind).startswith("Limit")
        ]
        self.line = self._make_line(self.names)
        if machine_state is not None:
            self.set_vars(resolve_machine_state(machine_state))
        self.knob_names: list[str] = []
        self._create_knobs()
        if initial_model_values:
            self.apply_initial_model_values(initial_model_values)
        self.recount_bpms()

    def _make_line(self, names: list[str]) -> xt.Line:
        line = self.env.new_line(components=names)
        line.particle_ref = self.particle_ref.copy()
        line.configure_quadrupole_model(
            model=QUAD_MODEL, integrator=QUAD_INTEGRATOR, num_multipole_kicks=QUAD_KICKS
        )
        line.config.XTRACK_MULTIPOLE_NO_SYNRAD = True
        line.build_tracker()
        return line

    # ------------------------------------------------------------------ knobs

    def _create_knobs(self) -> None:
        accelerator = self.accelerator
        if getattr(accelerator, "group_quadrupoles_by_cell", False) or (
            getattr(accelerator, "normalise_bends", False) and accelerator.optimises("bend", "k0")
        ):
            raise NotImplementedError(
                "xtrack backend: grouped or normalised knobs are not supported"
            )
        start, stop = _range_indices(self.names, self.magnet_range)
        selected = self.names[start : stop + 1]
        for spec in accelerator.get_knob_specs():
            suffix = _SUPPORTED.get((spec.kind, spec.attribute))
            if suffix is None:
                raise NotImplementedError(f"xtrack backend: no {spec.kind} {spec.attribute} knobs")
            regex = lua_to_regex(spec.pattern)
            for name in selected:
                upper = name.upper()
                if self.kinds.get(upper) != spec.kind or not regex.search(upper):
                    continue
                element = self.line[name]
                if (
                    spec.nonzero_attr is not None
                    and self._strength(element, spec.nonzero_attr) == 0
                ):
                    continue
                knob = f"{upper}.{suffix}"
                if knob in self.knob_names:
                    continue
                self._install_knob(name, spec.kind, knob)
                self.knob_names.append(knob)
        LOGGER.debug("xtrack backend: created %d knobs", len(self.knob_names))

    @staticmethod
    def _strength(element, attribute: str) -> float:
        value = getattr(element, attribute, 0.0)
        if value == "from_h":
            return float(element.h)
        return float(value or 0.0)

    def _install_knob(self, name: str, kind: str, knob: str) -> None:
        env, element, refs = self.env, self.line[name], self.line.element_refs[name]
        if kind == "quadrupole":
            env.vars[knob] = 0.0
            refs.k1 = float(element.k1) + env.vars[knob] / float(element.length)
        elif kind in ("sbend", "rbend"):
            env.vars[knob] = 0.0
            refs.k0 = self._strength(element, "k0") + env.vars[knob] / float(element.length)

    def set_knob_values(self, values: Mapping[str, float]) -> None:
        """Set knob variables of the environment by name."""
        for name, value in values.items():
            self.env.vars[name] = float(value)

    def knob_values(self) -> np.ndarray:
        """Current values of ``knob_names``, in that order."""
        return np.array([float(self.env.vars[name]._value) for name in self.knob_names])

    def apply_initial_model_values(self, values: Mapping[str, float]) -> None:
        """Knob values; ``pt`` is not a model value here."""
        unknown = [name for name in values if name != "pt" and name not in self.knob_names]
        if unknown:
            raise NotImplementedError(
                f"xtrack backend: fixed model values {unknown[:5]} are not supported"
            )
        self.set_knob_values({k: v for k, v in values.items() if k != "pt"})

    # ------------------------------------------------------------------ state

    def _target(self, name: str):
        element, _, attribute = name.rpartition(".")
        if name.lower() in self.env.vars:
            return None, name.lower()
        if element and element.lower() in self.env.element_dict:
            return element.lower(), attribute
        raise ValueError(
            f"machine_state names {name!r}, which is neither a variable nor an element attribute"
        )

    def get_vars(self, names: Sequence[str]) -> dict[str, float]:
        """Current values of variables or ``<element>.<attribute>`` names."""
        values = {}
        for name in names:
            element, attribute = self._target(name)
            if element is None:
                values[name] = float(self.env.vars[attribute]._value)
            else:
                values[name] = (
                    self._kick(element)
                    if attribute == "kick"
                    else float(getattr(self.env[element], attribute))
                )
        return values

    def _kick(self, element: str) -> float:
        kind = self.kinds[element.upper()]
        return (
            -float(self.env[element].knl[0])
            if kind == "hkicker"
            else float(self.env[element].ksl[0])
        )

    def set_vars(self, state: Mapping[str, float]) -> None:
        """Set variables or ``<element>.<attribute>`` names (a machine state)."""
        for name, value in state.items():
            element, attribute = self._target(name)
            if element is None:
                self.env.vars[attribute] = float(value)
            elif attribute == "kick":
                kind = self.kinds[element.upper()]
                field = "knl" if kind == "hkicker" else "ksl"
                getattr(self.env.element_refs[element], field)[0] = (
                    -float(value) if kind == "hkicker" else float(value)
                )
            else:
                setattr(self.env.element_refs[element], attribute, float(value))

    # ------------------------------------------------------------------ BPMs

    def recount_bpms(self) -> None:
        """Observed BPMs (pattern, not bad) in line order: all, and inside ``bpm_range``."""
        regex = lua_to_regex(self.accelerator.bpm_pattern)
        names = self.line.element_names
        observed = [
            name.upper()
            for name in names
            if self.kinds.get(name.upper()) == "monitor"
            and regex.search(name.upper())
            and name.upper() not in self.bad_bpms
        ]
        self.all_bpms = observed
        start, stop = _range_indices(names, self.bpm_range)
        in_range = {name.upper() for name in names[start : stop + 1]}
        self.bpms_in_range = [name for name in observed if name in in_range]
        self.nbpms = len(self.bpms_in_range)

    def cycle_to_start(self, marker: str) -> None:
        """Start the line at ``marker``; a thick element becomes drift, marker, drift (as MAD-NG)."""
        names = list(self.line.element_names)
        index = [name.upper() for name in names].index(marker.upper())
        name = names[index]
        length = float(getattr(self.line[name], "length", 0.0) or 0.0)
        if length > 0.0:
            half = f"{name}_half_drift"
            centre = f"{name}_centre_marker"
            if half not in self.env.element_dict:
                self.env.new(half, "Drift", length=length / 2)
                self.env.new(centre, "Marker")
            names[index : index + 1] = [half, centre, half]
            index += 1
        self.line = self._make_line(names[index:] + names[:index])

    def remove_zero_field_magnets(self) -> int:
        """Replace thick zero-field magnets by exact drifts and merge neighbouring drifts; returns how many removed.

        As the MAD-NG ``build_zero_field_removal_script``: call it once machine state and knobs are
        installed: a magnet whose strength comes from a variable later changed (not a knob) would
        stay a drift. A magnet carrying a knob, or misaligned, is kept.
        """
        line = self.line
        knobbed = {knob.rpartition(".")[0] for knob in self.knob_names}
        off: set[str] = set()
        for name in dict.fromkeys(line.element_names):
            element = line.element_dict[name]
            strengths = _ZERO_FIELD_STRENGTHS.get(type(element).__name__)
            if strengths is None or not float(element.length) > 0.0:
                continue
            attributes = (*strengths, *_ZERO_FIELD_ATTRIBUTES)
            if any(np.any(np.asarray(getattr(element, a)) != 0.0) for a in attributes):
                continue
            if name.upper() in knobbed:
                continue
            off.add(name)
        for name in off:
            self.env.new(f"{name}_off", "Drift", length=float(line[name].length), model="exact")
        names = [f"{name}_off" if name in off else name for name in line.element_names]
        self.line = self._make_line(names)
        self.line.discard_tracker()
        self.line.merge_consecutive_drifts(keep=[bpm.lower() for bpm in self.all_bpms])
        self.line.build_tracker()
        self.names = list(self.line.element_names)
        return len(off)

    def close(self) -> None:
        """Nothing to release: the line lives in this process."""
