"""MAD-NG side of the accelerator definitions: Lua run around knob creation, per-kind track methods."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from pymadng_utils.mad.ac_kickers import AC_KICKERS

from accelmo.machine.accelerators import LHC, PSB
from accelmo.machine.accelerators.magnet_grouping import normalise_lhcbend_magnets

if TYPE_CHECKING:
    from accelmo.backend.madng.interface import MadInterface
    from accelmo.machine.accelerators import Accelerator, KnobSpec

TILT_SEED = 1e-9  # Should discuss this with Laurent.
# Every offset is zeroed (not just the selected one) so the deferred table never reads nil.
_MISALIGN_PREPARATION = (
    "e.dx = e.dx or 0\ne.dy = e.dy or 0\ne.ds = e.ds or 0\n"
    "e.misalign = MAD.typeid.deferred{dx =\\->e.dx, dy =\\->e.dy, ds =\\->e.ds}"
)
_KNOB_PREPARATION = {
    "dx": _MISALIGN_PREPARATION,
    "dy": _MISALIGN_PREPARATION,
    "ds": _MISALIGN_PREPARATION,
    # MAD-NG drops a rotation whose scalar angle is zero (mad_dynmap.cpp:345),
    # silently zeroing the knob's Jacobian column. Seeding above ``minang``
    # (1e-10 rad) keeps the derivative; the knob inherits the seed as its value.
    "tilt": f"e.tilt = (e.tilt or 0) + {TILT_SEED:.15e}",
}
#: Per-kind track ``method`` overrides that keep the method=6 accuracy.
_TRACK_METHODS: dict[type, dict[str, int]] = {LHC: {"sextupole": 4}}
_PSB_GROUP_SUFFIXES = {
    "k1": "dk1l",
    "k0s": "dk0sl",
    "k1s": "dk1sl",
    "dx": "dx",
    "dy": "dy",
    "ds": "ds",
    "tilt": "tilt",
}


def track_methods(accelerator: Accelerator) -> dict[str, int]:
    """Per-element-kind MAD-NG track ``method`` overrides for ``accelerator``."""
    for kind, methods in _TRACK_METHODS.items():
        if isinstance(accelerator, kind):
            return methods
    return {}


def prepare_knob_creation(
    iface: MadInterface, accelerator: Accelerator, specs: list[KnobSpec]
) -> None:
    """Install PSB AC kickers, run each attribute's ``_KNOB_PREPARATION``, read LHC bend lengths."""
    if isinstance(accelerator, PSB) and accelerator.acd_drive_tunes is not None:
        iface.install_ac_kickers((0.0,) * len(AC_KICKERS), accelerator.acd_drive_tunes[0])
    grouped: dict[tuple[str, str], list[str]] = {}
    for spec in specs:
        body = _KNOB_PREPARATION.get(spec.attribute)
        if body is not None:
            grouped.setdefault((spec.kind, body), []).append(spec.pattern)
    for (element_kind, body), patterns in grouped.items():
        _prepare_matching_elements(iface, element_kind, tuple(dict.fromkeys(patterns)), body)
    if (
        isinstance(accelerator, LHC)
        and accelerator.optimises("bend", "k0")
        and accelerator.normalise_bends
    ):
        _normalise_lhc_bends(iface, accelerator)


def _prepare_matching_elements(
    iface: MadInterface, element_kind: str, patterns: tuple[str, ...], body: str
) -> None:
    """Run Lua ``body`` once per element of ``element_kind`` matching ``patterns`` (seen as ``e``)."""
    iface.mad.send(f"""
        local element_kind = {iface.py_name}:recv()
        local patterns = {iface.py_name}:recv()
        for i, e in loaded_sequence:siter(magnet_range) do
            if e.kind == element_kind then
                for _, pattern in ipairs(patterns) do
                    if string.match(e.name, pattern) then
{textwrap.indent(body, " " * 24)}
                        break
                    end
                end
            end
        end
        """)
    iface.mad.send(element_kind).send(patterns)


def _normalise_lhc_bends(iface: MadInterface, accelerator: LHC) -> None:
    iface.mad.send(f"""
            bend_dict = {{}}
            bend_lengths = {{}}
            for i, e in loaded_sequence:siter(magnet_range) do
                if (e.kind == "sbend" or e.kind == "rbend") and e.k0 ~= 0 then
                    bend_dict[e.name .. ".dk0l"] = e.k0
                    bend_lengths[e.name .. ".dk0l"] = e.l
                end
            end
            {iface.py_name}:send(bend_dict, true)
            {iface.py_name}:send(bend_lengths, true)
            bend_dict = {iface.py_name}:recv()
            """)
    true_strengths_dict: dict[str, float] = iface.mad.recv()
    accelerator.bend_lengths = iface.mad.recv()
    iface.mad.send(normalise_lhcbend_magnets(true_strengths_dict, accelerator.bend_lengths))


def knob_expansion_block(accelerator: Accelerator) -> str:
    """Lua run after knob creation that may edit ``knob_names`` / ``spos_list``.

    PSB with several ACD files: each kicker's volt (a or b) becomes one knob per file.
    """
    if not isinstance(accelerator, PSB) or accelerator.acd_files == 1:
        return ""
    names = ", ".join(f"'{name}.volt'" for name in AC_KICKERS)
    return f"""
for _, k in ipairs {{{names}}} do
    for idx, name in ipairs(knob_names) do
        if name == k then
            table.remove(knob_names, idx)
            local spos = table.remove(spos_list, idx)
            for f = 0, {accelerator.acd_files - 1} do
                loaded_sequence[k .. '_f' .. f] = loaded_sequence[k]
                table.insert(knob_names, k .. '_f' .. f)
                table.insert(spos_list, spos)
            end
            break
        end
    end
end
"""


def mad_attr_spec(accelerator: Accelerator, kind: str, attribute: str) -> dict[str, str]:
    """Lua ``name_expr`` / ``sign_expr`` / ``mad_value`` overrides for one element attribute."""
    if isinstance(accelerator, PSB):
        return _psb_attr_spec(accelerator, kind, attribute)
    if isinstance(accelerator, LHC):
        if not accelerator.normalise_bends or kind != "sbend" or attribute != "k0":
            return {}
        return {
            "name_expr": 'string.gsub(e.name, "(MB%.)([ABCD])([0-9]+[LR][1-8]%.B[12])", "%1%3") .. ".dk0l"',
            "mad_value": "bend_dict[k_str_name]",
        }
    return {}


def _psb_attr_spec(accelerator: PSB, kind: str, attribute: str) -> dict[str, str]:
    """Share the two QFO knobs in each cell when native grouping is enabled."""
    if not accelerator.group_quadrupoles_by_cell or kind != "quadrupole":
        return {}
    suffix = _PSB_GROUP_SUFFIXES.get(attribute)
    if suffix is None:
        return {}
    if attribute == "tilt" and accelerator.differential_cell_tilt:
        return {
            "name_expr": (
                '(string.gsub((string.gsub(e.name, "^BR%.QFO(%d+)%d$", "BR.QCELL%1")), '
                '"^BR%.QDE(%d+)$", "BR.QCELL%1")) .. ".tilt"'
            ),
            "sign_expr": '(e.name:match("^BR%.QDE") and -1 or 1)',
        }
    return {
        "name_expr": (f'string.gsub(e.name, "^(BR%.QFO)(%d+)(%d)$", "%1CELL%2") .. ".{suffix}"')
    }
