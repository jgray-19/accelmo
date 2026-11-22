"""MAD-NG integration tests for PSB quadrupole longitudinal (``ds``) misalignment knobs.

``ds`` reaches the element through the deferred ``misalign`` table built by
``_MISALIGN_PREPARATION``. A ``ds`` that never makes it into that table raises
nothing, so these tests check both that the knobs exist and that moving one
actually changes the optics.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from accelmo.backend.madng.interface import MadInterface
from accelmo.machine.accelerators import PSB

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

pytestmark = pytest.mark.serial

#: Longitudinal offset used throughout (m).
DS = 5e-2
TWISS_COLUMNS = ("name", "x", "y", "beta11", "beta22", "mu1", "mu2")


def _twiss(iface: MadInterface) -> pd.DataFrame:
    iface.mad.send("dstws = twiss{sequence=loaded_sequence, observe=1}")
    return iface.mad.dstws.to_df(columns=list(TWISS_COLUMNS)).set_index("name")


def _ds_knobs(iface: MadInterface) -> list[str]:
    return [k for k in iface.knob_names if k.endswith(".ds")]


@pytest.mark.parametrize("grouped", [False, True], ids=["per-magnet", "grouped-by-cell"])
def test_ds_knobs_are_created(seq_psb: Path, grouped: bool) -> None:
    """``misalignments={"quad": {"ds"}}`` creates one ``.ds`` knob per (grouped) quad."""
    accelerator = PSB(
        ring=3,
        sequence_file=seq_psb,
        misalignments={"quad": {"ds"}},
        group_quadrupoles_by_cell=grouped,
    )
    iface = MadInterface(accelerator=accelerator, discard_mad_output=True, create_knobs=True)
    try:
        knobs = _ds_knobs(iface)
        assert knobs == list(iface.knob_names), "only ds knobs should be created"
        assert len(knobs) > 2
        if grouped:
            assert "BR.QFOCELL1.ds" in knobs
            assert "BR.QFO11.ds" not in knobs
        else:
            assert "BR.QFO11.ds" in knobs
    finally:
        with contextlib.suppress(Exception):
            del iface


def test_nonzero_ds_changes_optics_like_a_static_misalignment(seq_psb: Path) -> None:
    """A nonzero ``ds`` knob moves the optics exactly as ``misalign = {ds = ...}`` does."""
    accelerator = PSB(ring=3, sequence_file=seq_psb, misalignments={"quad": {"ds"}})
    iface = MadInterface(accelerator=accelerator, discard_mad_output=True, create_knobs=True)
    try:
        knob = "BR.QFO11.ds"
        assert knob in iface.knob_names
        nominal = _twiss(iface)
        iface.update_knob_values({knob: DS})
        shifted = _twiss(iface)
    finally:
        with contextlib.suppress(Exception):
            del iface

    beta_change = np.max(np.abs(shifted["beta11"] - nominal["beta11"]))
    assert beta_change > 1e-6, (
        f"a {DS} m ds offset changed beta11 by only {beta_change:.3e} m; "
        "the deferred misalign table is not picking up ds"
    )

    reference_iface = MadInterface(
        accelerator=PSB(ring=3, sequence_file=seq_psb), discard_mad_output=True
    )
    try:
        reference_iface.mad.send(
            f"loaded_sequence['{knob.removesuffix('.ds')}'].misalign = {{ds = {DS:.15e}}}"
        )
        reference = _twiss(reference_iface)
    finally:
        with contextlib.suppress(Exception):
            del reference_iface

    for column in ("beta11", "beta22", "mu1", "mu2"):
        assert np.allclose(shifted[column], reference[column], rtol=1e-10, atol=1e-12), (
            f"'{column}' from the ds knob differs from a static ds misalignment by up to "
            f"{np.max(np.abs(shifted[column] - reference[column])):.3e}"
        )
