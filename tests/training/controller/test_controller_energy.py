"""
Energy-focused integration tests for controller logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.training.controller_test_utils import (
    DPP_VALUE,
    _build_energy_optimisation_case,
    _make_simulation_config_energy,
    _run_energy_optimisation_case,
    evaluate_controller_worker_losses,
    make_lm_config,
)

if TYPE_CHECKING:
    from pathlib import Path

    from accelmo.backend.madng.interface import MadInterface


pytestmark = pytest.mark.serial


@pytest.mark.parametrize("optimise_momenta", [False, True], ids=["position_only", "with_momenta"])
def test_controller_energy_opt(
    tmp_path: Path,
    seq_b1: Path,
    loaded_interface: MadInterface,
    optimise_momenta: bool,
    controller_test_mode: str,
) -> None:
    simulation_config = _make_simulation_config_energy(optimise_momenta)
    optimiser_config = make_lm_config()

    if controller_test_mode == "loss_regression":
        ctrl, true_knobs = _build_energy_optimisation_case(
            tmp_path=tmp_path,
            loaded_interface=loaded_interface,
            simulation_config=simulation_config,
            optimiser_config=optimiser_config,
            bpm_start_points=["BPM.9R2.B1", "BPM.10R2.B1", "BPM.11R2.B1"],
            bpm_end_points=["BPM.9L3.B1", "BPM.10L3.B1", "BPM.11L3.B1"],
            magnet_range="BPM.9R2.B1/BPM.9L3.B1",
            mad_log_name="controller_energy_opt.log",
        )
        initial_loss, true_loss = evaluate_controller_worker_losses(
            ctrl, [ctrl.machine.initial_knobs, true_knobs]
        )
        assert true_loss < initial_loss * 1e-2
        return

    result = _run_energy_optimisation_case(
        tmp_path=tmp_path,
        loaded_interface=loaded_interface,
        simulation_config=simulation_config,
        optimiser_config=optimiser_config,
        bpm_start_points=["BPM.9R2.B1", "BPM.10R2.B1", "BPM.11R2.B1"],
        bpm_end_points=["BPM.9L3.B1", "BPM.10L3.B1", "BPM.11L3.B1"],
        magnet_range="BPM.9R2.B1/BPM.9L3.B1",
        mad_log_name="controller_energy_opt.log",
    )
    estimate, unc = result.knobs, result.uncertainties

    true_pt = loaded_interface.dp2pt(DPP_VALUE)
    rel_error = abs(estimate.pop("pt") - true_pt) / abs(true_pt)
    # measured 2026-11-12: 8.8e-11 (position only), 2.3e-10 (with momenta)
    assert rel_error < 1e-9
    uncertainty = unc.pop("pt")
    assert 0 < uncertainty < 3e-6
    assert not estimate
    assert not unc
