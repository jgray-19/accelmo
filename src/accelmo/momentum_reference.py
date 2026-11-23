"""Fit bends and quads to closed orbit + phase, and hand back a closed-orbit reference.

Purpose
-------
A transverse-momentum reconstruction requires the closed-orbit angle ``px``/``py``
at every BPM. BPMs measure position only, so the angle must come from a model. A
nominal model without the machine's magnet errors gives an incorrect angle; on PSB
ring 3 with realistic dipole errors the nominal angle error equals the true angle.

Fitting the errors to a closed-orbit measurement corrects this, subject to three
requirements:

1. **Regularisation.** With 80 knobs against 17 BPMs at realistic resolution, an
   unregularised fit recovers an error vector ~6x larger than the true one, because
   noise is absorbed into compensating magnet errors, so ``knob_sigmas`` (a Gaussian
   prior of the expected error size) is required.
2. **Knob families matched to observables.** The closed orbit is insensitive to
   gradient errors (a quadrupole on a centred orbit produces no deflection), so
   free quadrupole knobs in an orbit-only fit worsen the result by ~2.6x. Phase
   advance responds to gradients; with phase included the same knobs improve it ~4x.
3. **More than one momentum.** At a single momentum the per-magnet Jacobians are
   degenerate.

Observables
-----------
The default observables are closed orbit and phase advance. Phase requires no
amplitude calibration and, unlike beta-from-amplitude or a model-derived
``DPX``/``DPY``, does not feed a modelled quantity back into the model fit. Beta
and dispersion can be selected through *observables* when the measurement supports
them.

Consuming the result
--------------------
:class:`MomentumReference` is plain data: fitted knob values and a per-BPM
closed-orbit frame with ``x``/``y``/``px``/``py``. A downstream reconstruction can
use it without importing from this package.

The reference is a momentum origin as well as an orbit. A reconstruction expressed
against :attr:`MomentumReference.closed_orbit` requires the offset *from*
:attr:`MomentumReference.reference_pt`, not the measurement's absolute ``pt``. For
a reference :math:`3\\times10^{-3}` off the origin, using the absolute ``pt`` as the
offset degrades the reconstructed ``px`` from 4.741e-4 to 7.702e-2, against
1.162e-3 for the offset. First order cancels either way; the error enters through
the second-order dispersion term and is therefore absent on a linear lattice.

The subtraction is not performed here.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from accelmo.backend import get_backend
from accelmo.fitting.config import SequenceConfig
from accelmo.machine.machine_state import resolve_machine_state
from accelmo.poco import ClosedOrbitFitter, LevenbergMarquardtConfig, closed_twiss_series

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import pandas as pd

    from accelmo.fitting.config import OutputConfig
    from accelmo.machine.accelerators.base import Accelerator

LOGGER = logging.getLogger(__name__)

#: Closed orbit and phase advance: the observables that need no amplitude
#: calibration and no model-derived intermediate. See the module docstring.
ORBIT_AND_PHASE: tuple[str, ...] = ("x", "y", "mu1", "mu2")

#: Observables that carry quadrupole-gradient information. The closed orbit does
#: not: a gradient error on a centred orbit produces no deflection.
GRADIENT_SENSITIVE: frozenset[str] = frozenset({"mu1", "mu2", "betx", "bety", "dx", "dy"})


#: Columns of the returned closed-orbit reference.
REFERENCE_COLUMNS: tuple[str, ...] = ("x", "y", "px", "py")


@dataclass(frozen=True)
class MomentumReference:
    """A fitted lattice and the closed-orbit reference it implies.

    Attributes:
        magnet_strengths: Fitted knob values, keyed by MAD knob name
            (``BR.BHZ11.dk0l``, ``BR.QFO11.dk1l``, ...). Apply these to a model
            of the same sequence to reproduce the fitted machine.
        closed_orbit: Per-BPM ``x``/``y``/``px``/``py`` of the fitted machine's
            closed orbit at :attr:`reference_pt`, indexed by BPM name. ``x``/``y`` are
            a cross-check against the measurement; ``px``/``py`` are the part
            that cannot be measured and is the reason this class exists.
        reference_pt: MAD-NG ``pt`` at which the reference was evaluated. This is
            the momentum origin of :attr:`closed_orbit`; momenta handed to a
            reconstruction against this reference are offsets from it, not
            absolute ``pt`` values.
        observables: Observables the fit actually used.
        knob_sigmas: Gaussian prior widths used.
    """

    magnet_strengths: dict[str, float]
    uncertainties: dict[str, float]
    closed_orbit: pd.DataFrame
    reference_pt: float
    observables: tuple[str, ...]
    knob_sigmas: dict[str, float]
    momentum_points: tuple[float, ...]
    bpm_coverage: dict[float, tuple[str, ...]]
    fit_settings: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def closed_orbit_at(
    accelerator: Accelerator,
    magnet_strengths: Mapping[str, float] | None,
    pt: float = 0.0,
    *,
    machine_state: Path | Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Closed orbit and its angles at the BPMs, for a model carrying *magnet_strengths*.

    ``pt`` is pinned on the initial condition, so this is the periodic solution at
    that momentum rather than an on-momentum orbit with a momentum offset added.
    """
    return get_backend(accelerator.backend).closed_orbit_table(
        accelerator, magnet_strengths, pt, machine_state, REFERENCE_COLUMNS
    )


def _check_observable_knob_match(accelerator: Accelerator, observables: Sequence[str]) -> None:
    """Reject knob families which the requested observables cannot constrain.

    Fitting gradients against the closed orbit alone is not merely uninformative,
    it is harmful: the extra knobs absorb measurement noise into compensating
    errors, which measured ~2.6x worse than leaving them out.
    """
    if not any(
        spec.kind == "quadrupole" and spec.attribute == "k1"
        for spec in accelerator.get_knob_specs()
    ):
        return
    if GRADIENT_SENSITIVE.intersection(observables):
        return
    raise ValueError(
        "Quadrupole knobs are enabled but none of the observables "
        f"{tuple(observables)} responds to a gradient error; the closed orbit is "
        "blind to gradients on a centred orbit. Add mu1/mu2 (or beta/dispersion), "
        "or stop fitting quadrupole k1."
    )


def fit_momentum_reference(
    accelerator: Accelerator,
    measurements: Mapping[float, pd.DataFrame],
    *,
    observables: Sequence[str] = ORBIT_AND_PHASE,
    knob_sigmas: Mapping[str, float],
    reference_pt: float = 0.0,
    sequence_config: SequenceConfig | None = None,
    lm_config: LevenbergMarquardtConfig | None = None,
    initial_knob_strengths: Mapping[str, float] | None = None,
    machine_state: Path | Mapping[str, float] | None = None,
    output_config: OutputConfig | None = None,
) -> MomentumReference:
    """Fit magnet errors to measured optics, and return the closed-orbit reference.

    Args:
        accelerator: Configured accelerator. Enable the knob families you intend
            to fit (bend ``k0`` and, if the observables can constrain them,
            quadrupole ``k1``) via its ``errors`` selection.
        measurements: Measured optics keyed by the known MAD-NG ``pt`` the
            measurement was taken at. At least two momenta are required: at one
            momentum the per-magnet Jacobians are degenerate.
        observables: What to fit. Defaults to closed orbit plus phase advance.
        knob_sigmas: Gaussian prior widths towards zero magnet error, by knob name or
            terminal attribute (see :func:`~accelmo.fitting.priors.knob_sigma_array`).
        reference_pt: MAD-NG ``pt`` at which to evaluate the returned closed orbit.
        sequence_config: Sequence range; defaults to the full ring.
        lm_config: Optimiser settings.
        initial_knob_strengths: Starting point, if resuming a previous fit.

    Returns:
        A :class:`MomentumReference`.

    Raises:
        ValueError: If fewer than two momenta are supplied.
    """
    momentum_points = tuple(sorted(float(pt) for pt in measurements))
    if len(set(momentum_points)) < 2:
        raise ValueError(
            "fit_momentum_reference needs measurements at at least two momenta; "
            f"got {list(momentum_points)}. At a single momentum the per-magnet "
            "Jacobians are degenerate and the fit cannot separate the magnets."
        )

    observables = tuple(observables)
    _check_observable_knob_match(accelerator, observables)

    fitter = ClosedOrbitFitter(
        accelerator=accelerator,
        sequence_config=sequence_config or SequenceConfig(magnet_range="$start/$end"),
        series=closed_twiss_series(measurements, observables),
        lm_config=lm_config,
        initial_knob_strengths=dict(initial_knob_strengths) if initial_knob_strengths else None,
        knob_sigmas=knob_sigmas,
        machine_state=machine_state,
        output_config=output_config,
    )
    try:
        result = fitter.run()
        magnet_strengths, uncertainties = result.knobs, result.uncertainties
    finally:
        # The workers are stopped by ``run``; this closes the fitter's own MAD
        # model (its ``MachineSetup``).
        fitter.close()

    LOGGER.info(
        "Fitted %d knobs against %s at momenta %s",
        len(magnet_strengths),
        observables,
        momentum_points,
    )

    reference = closed_orbit_at(
        accelerator,
        magnet_strengths,
        reference_pt,
        machine_state=machine_state,
    )
    uncertainty_map = {str(name): float(value) for name, value in (uncertainties or {}).items()}
    coverage = {
        float(pt): tuple(str(name) for name in frame.index) for pt, frame in measurements.items()
    }
    assert result.diagnostics is not None
    return MomentumReference(
        magnet_strengths=dict(magnet_strengths),
        uncertainties=uncertainty_map,
        closed_orbit=reference,
        reference_pt=float(reference_pt),
        observables=observables,
        knob_sigmas=dict(knob_sigmas),
        momentum_points=momentum_points,
        bpm_coverage=coverage,
        fit_settings={
            "sequence_range": (
                sequence_config or SequenceConfig(magnet_range="$start/$end")
            ).magnet_range,
            # The knobs themselves, not a path: they are tens of values, and a
            # path names a file that later runs rewrite -- which makes a saved
            # fit unreproducible from its own metadata.
            "machine_state": None
            if machine_state is None
            else dict(sorted(resolve_machine_state(machine_state).items())),
            "lm_config": repr(lm_config or fitter.lm_config),
        },
        diagnostics={**dataclasses.asdict(result.diagnostics), "n_knobs": len(magnet_strengths)},
    )


__all__ = [
    "GRADIENT_SENSITIVE",
    "ORBIT_AND_PHASE",
    "REFERENCE_COLUMNS",
    "MomentumReference",
    "closed_orbit_at",
    "fit_momentum_reference",
]
