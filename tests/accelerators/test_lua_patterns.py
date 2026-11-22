"""``lua_to_regex`` selects exactly the elements MAD-NG's ``string.match`` selects.

Every family pattern and BPM pattern of the LHC and PSB is matched against every element
name of the real sequence, in MAD-NG and as a regex.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from accelmo.machine.accelerators import LHC, PSB
from accelmo.machine.patterns import lua_to_regex

if TYPE_CHECKING:
    from accelmo.backend.madng import AbaMadInterface


def _patterns(accelerator) -> list[str]:
    patterns = {accelerator.bpm_pattern}
    for family in accelerator.FAMILIES.values():
        patterns |= set(family.patterns.values()) | set(family.attr_patterns.values())
    return sorted(patterns)


def _compare(interface: AbaMadInterface, patterns: list[str]) -> None:
    mad = interface.mad
    mad.send(f"""
local names = {{}}
for _, e in loaded_sequence:iter() do names[#names + 1] = e.name end
{interface.py_name}:send(names, true)
""")
    names = list(mad.recv())
    assert len(names) > 100
    for pattern in patterns:
        mad.send(f"""
local pattern = {interface.py_name}:recv()
local hits = {{}}
for _, e in loaded_sequence:iter() do
    if string.match(e.name, pattern) then hits[#hits + 1] = e.name end
end
{interface.py_name}:send(hits, true)
""")
        mad.send(pattern)
        lua_hits = list(mad.recv())
        regex = lua_to_regex(pattern)
        assert [name for name in names if regex.search(name)] == lua_hits, pattern


def test_lhc_patterns_match_madng(loaded_interface: AbaMadInterface, seq_b1) -> None:
    _compare(loaded_interface, _patterns(LHC(beam=1, sequence_file=seq_b1)))


def test_psb_patterns_match_madng(loaded_psb_interface: AbaMadInterface, seq_psb) -> None:
    _compare(loaded_psb_interface, _patterns(PSB(ring=3, sequence_file=seq_psb)))


@pytest.mark.parametrize("pattern", ["%b()", "%f[%w]", "(a)%1"])
def test_unsupported_lua_constructs_raise(pattern: str) -> None:
    with pytest.raises(ValueError):
        lua_to_regex(pattern)
