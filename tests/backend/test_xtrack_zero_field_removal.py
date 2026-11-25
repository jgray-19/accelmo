"""Zero-field magnets of the xtrack model become exact drifts, as in the MAD-NG backend.

The LHC beam-1 test sequence has 1527 thick quadrupoles, sextupoles and octupoles at zero
strength; the MAD-NG removal script leaves 734 quadrupoles, 344 sextupoles and no octupoles.
"""

from __future__ import annotations

import collections
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xtrack as xt

from accelmo.backend.xtrack.model import XtrackModel
from accelmo.machine.accelerators import LHC

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.serial

REMOVED = 1527
ELEMENTS = 13349  # of the unmodified line
REMAINING = {"Quadrupole": 734, "Sextupole": 344}
KICK = 1e-5  # [rad]
KNOB = 1e-4  # integrated dk1l [1/m]
TOL_ORBIT = 1e-11  # relative to the largest orbit value
MAX_SECONDS = 10.0


def _model(seq_b1: Path) -> XtrackModel:
    accelerator = LHC(beam=1, sequence_file=seq_b1, errors={"quad": {"k1"}}, backend="xtrack")
    return XtrackModel(accelerator)


def _orbit(model: XtrackModel) -> np.ndarray:
    """x and y at every BPM of the closed orbit with one horizontal corrector kicked."""
    names = set(model.line.element_names)
    corrector = next(
        name for name, kind in model.kinds.items() if kind == "hkicker" and name.lower() in names
    )
    model.set_vars({f"{corrector}.kick": KICK})
    model.set_knob_values(dict.fromkeys(model.knob_names[::10], KNOB))
    line = model.line
    ref = model.particle_ref
    delta0 = float(xt.Particles(p0c=ref.p0c[0], mass0=ref.mass0, q0=ref.q0, ptau=0.0).delta[0])
    orbit = line.find_closed_orbit(method="4d", delta0=delta0)
    bpms = [name.lower() for name in model.all_bpms]
    line.track(orbit.copy(), multi_element_monitor_at=bpms)
    data = np.asarray(line.tracker.record_multi_element_last_track.data)
    return data[0, 0, [0, 2]]


def test_removal_matches_the_madng_script_and_is_fast(seq_b1: Path) -> None:
    model = _model(seq_b1)
    start = time.perf_counter()
    removed = model.remove_zero_field_magnets()
    elapsed = time.perf_counter() - start
    kinds = collections.Counter(
        type(model.line.element_dict[name]).__name__ for name in model.line.element_names
    )
    assert removed == REMOVED
    assert len(model.line.element_names) < ELEMENTS  # neighbouring drifts are merged
    bpms = {name.lower() for name in model.all_bpms}
    assert bpms <= set(model.line.element_names)
    assert {kind: kinds[kind] for kind in ("Quadrupole", "Sextupole", "Octupole")} == {
        **REMAINING,
        "Octupole": 0,
    }
    assert elapsed < MAX_SECONDS


def test_removal_does_not_change_the_closed_orbit(seq_b1: Path) -> None:
    kept = _orbit(_model(seq_b1))
    model = _model(seq_b1)
    model.remove_zero_field_magnets()
    removed = _orbit(model)
    assert np.abs(kept).max() > 1e-5  # the corrector moved the orbit
    assert np.abs(removed - kept).max() / np.abs(kept).max() < TOL_ORBIT


def test_a_magnet_with_a_knob_is_never_removed(seq_b1: Path) -> None:
    model = _model(seq_b1)
    knobbed = {knob.rpartition(".")[0] for knob in model.knob_names}
    model.remove_zero_field_magnets()
    kept = {name.upper() for name in model.line.element_names}
    assert knobbed <= kept
