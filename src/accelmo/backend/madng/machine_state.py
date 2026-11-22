"""Read and assign machine-state globals (see :mod:`accelmo.machine.machine_state`) in a MAD-NG session."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pymadng import MAD


def _key(name: str) -> str:
    """Lua expression for the ``MADX.__var`` key of *name*: MAD-NG stores a dotted variable (``k_hcor_qf2a.56``) with ``.`` as ``_``,
    although ``MADX['k_hcor_qf2a.56']`` maps it on access (a raw ``__var`` lookup does not)."""
    return repr(name.replace(".", "_"))


def check_state_names(mad: MAD, py_name: str, names: list[str]) -> None:
    """Raise ``ValueError`` for a name that is neither a MAD-X variable nor an element attribute.

    Indexing ``MADX`` with an unknown name silently creates it as zero, so existence is tested first.
    """
    for name in names:
        element, _, attribute = name.rpartition(".")
        mad.send(
            f"{py_name}:send(MADX.__var[{_key(name)}] ~= nil or "
            f"('{element}' ~= '' and loaded_sequence[('{element}'):upper()] ~= nil and loaded_sequence[('{element}'):upper()]['{attribute}'] ~= nil))"
        )
        if not mad.recv():
            raise ValueError(
                f"machine_state names {name!r}, which is neither a MAD-X variable nor an element attribute"
            )


def read_state(mad: MAD, py_name: str, names: list[str]) -> dict[str, float]:
    """The current value of each name (checked first, see :func:`check_state_names`)."""
    check_state_names(mad, py_name, names)
    values = {}
    for name in names:
        element, _, attribute = name.rpartition(".")
        mad.send(
            f"{py_name}:send(MADX.__var[{_key(name)}] ~= nil and MADX['{name}'] or loaded_sequence[('{element}'):upper()]['{attribute}'])"
        )
        values[name] = float(mad.recv())
    return values


def assign_state(mad: MAD, state: Mapping[str, float]) -> None:
    """Set every name of *state*: an existing variable, else an element attribute, else a new variable."""
    if state:
        mad.send("\n".join(_assignment(name, value) for name, value in state.items()))


def _assignment(name: str, value: float) -> str:
    element, _, attribute = name.rpartition(".")
    target = f"loaded_sequence[('{element}'):upper()]"
    return (
        f"if MADX.__var[{_key(name)}] ~= nil then MADX['{name}'] = {value!r} "
        f"elseif '{element}' ~= '' and {target} ~= nil and {target}['{attribute}'] ~= nil then {target}['{attribute}'] = {value!r} "
        f"else MADX['{name}'] = {value!r} end"
    )
