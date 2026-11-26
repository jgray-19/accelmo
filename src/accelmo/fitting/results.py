"""What a fit returns."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FitDiagnostics:
    """How the optimisation ended.

    ``iterations`` counts epochs (tracking) or Levenberg-Marquardt iterations.
    ``gradient_norm``, ``damping`` and ``accepted_evaluations`` describe the last
    LM update and are ``None`` for the tracking fit.
    """

    converged: bool
    reason: str | None
    iterations: int
    best_loss: float
    gradient_norm: float | None = None
    damping: float | None = None
    accepted_evaluations: int | None = None


@dataclass(frozen=True)
class UncertaintyInfo:
    """How the uncertainties of a fit were computed."""

    #: ``"true"`` or ``"gauss_newton"``: the Hessian that was used.
    hessian: str
    #: Wall time of the uncertainty stage.
    seconds: float
    #: Why a requested Hessian was replaced, empty when it was not.
    note: str = ""


@dataclass(frozen=True)
class FitResult:
    """Fitted knobs, their 1-sigma uncertainties (by knob name) and how the fit ended."""

    knobs: dict[str, float]
    uncertainties: dict[str, float]
    diagnostics: FitDiagnostics | None = None
    #: Extra fitted parameters that are not model knobs (e.g. BPM and corrector gains).
    extra: dict[str, float] = field(default_factory=dict)
    #: 1-sigma uncertainties of ``extra``, by the same names.
    extra_uncertainties: dict[str, float] = field(default_factory=dict)
    #: ``None`` when the uncertainties were not computed or the stage failed.
    uncertainty: UncertaintyInfo | None = None
