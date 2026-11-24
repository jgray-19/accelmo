"""Levenberg-Marquardt base fitter of POCO and the measured-optics observables."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from accelmo.fitting.lifecycle import run_with_workers
from accelmo.fitting.lm_loop import LMPoint, LMProblem, run_levenberg_marquardt
from accelmo.fitting.pool import WorkerPool
from accelmo.fitting.priors import add_quadratic_prior, inverse_variances, knob_sigma_array
from accelmo.fitting.protocol import Evaluate, GradReply, Start
from accelmo.fitting.reduction import reduce_replies
from accelmo.fitting.results import FitDiagnostics, FitResult
from accelmo.fitting.setup import MachineSetup
from accelmo.fitting.uncertainty import hessian_uncertainties, warn_if_singular
from accelmo.fitting.worker import WorkerConfig
from accelmo.optimisers.levenberg_marquardt import LevenbergMarquardtConfig
from accelmo.poco.workers.closed_twiss import Observable, ObservableKind

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    import pandas as pd

    from accelmo.config import WorkerRuntime
    from accelmo.fitting.config import OutputConfig, SequenceConfig
    from accelmo.machine.accelerators import Accelerator
    from accelmo.machine.machine_state import MachineState

logger = logging.getLogger(__name__)


#: Measurement columns backing each fittable observable, as
#: ``observable -> (value column, error column)`` in the measured
#: optics frame.
#:
#: ``mu1``/``mu2`` are special: the measured frame carries the *cumulative* phase
#: and its cumulative variance, which are differenced into per-interval advances
#: (with the variance difference) by :func:`_advance_targets`.
#:
#: ``betx``/``bety``/``alfx``/``alfy`` are the physical-plane projections, never
#: ``beta11``/``beta22``/``alfa11``/``alfa22`` (MAD-NG's generalised-mode
#: columns, only meaningful with ``coupling=true`` -- see
#: ``run_closed_twiss_init.mad``). Phase advance has no such projected name in
#: MAD-NG (confirmed against its own optical-function name list: only
#: ``mu``/``dmu`` with a mode-index suffix exist, no ``mux``/``muy``), so
#: ``mu1``/``mu2`` remain mode-indexed on both sides here.
MEASUREMENT_COLUMNS: dict[str, tuple[str, str]] = {
    "x": ("X", "ERRX"),
    "y": ("Y", "ERRY"),
    "betx": ("BETX", "ERRBETX"),
    "bety": ("BETY", "ERRBETY"),
    "alfx": ("ALFX", "ERRALFX"),
    "alfy": ("ALFY", "ERRALFY"),
    "dx": ("DX", "ERRDX"),
    "dy": ("DY", "ERRDY"),
    "dpx": ("DPX", "ERRDPX"),
    "dpy": ("DPY", "ERRDPY"),
    # ``MUX``/``MUY`` are what ``build_twiss_from_measurements`` names the
    # cumulative phase it accumulates from the measured adjacent advances, paired
    # with the cumulative ``mu1_var``/``mu2_var``. Both are differenced back into
    # per-interval advances by :func:`_advance_targets`.
    "mu1": ("MUX", "mu1_var"),
    "mu2": ("MUY", "mu2_var"),
}

#: Observables fitted when the caller does not choose: everything the closed
#: twiss and an omc3 optics measurement independently have in common. All of
#: them are fitted by default: each family constrains a different combination of
#: the knobs, and inverse-variance weighting gives a noisy family little weight.
#: Narrow this only to isolate an effect, as the tests do.
#:
#: ``dpx``/``dpy`` are the one exclusion, and it is required for correctness:
#: omc3 does not measure them independently but *derives* them from
#: ``DX``/``DY`` through the model transfer matrix, so including them counts one
#: measurement twice. In the vertical it is worse than redundant - the derivation
#: assumes no vertical dispersion source exists anywhere, which is exactly the
#: hypothesis a ``Dy`` fit is testing. They remain available for callers whose
#: dispersion errors come from somewhere else.
DEFAULT_OBSERVABLES: tuple[str, ...] = (
    "x",
    "y",
    "betx",
    "bety",
    "alfx",
    "alfy",
    "mu1",
    "mu2",
    "dx",
    "dy",
)


class _KnobProblem(LMProblem):
    """The closed-orbit objective over the shared knobs, summed over the workers.

    The workers return the physical chi²; ``prior`` is ``diag(1/sigma²)``, centred on the ideal
    zero-error lattice (not on the optimisation start).
    """

    def __init__(
        self,
        knob_names: list[str],
        evaluate_workers: Callable[..., list[GradReply]],
        prior: np.ndarray,
        history: list,
        *,
        loss_only_trials: bool,
    ) -> None:
        self.knob_names = knob_names
        self.evaluate_workers = evaluate_workers
        self.prior = prior
        self.history = history
        self.loss_only_trials = loss_only_trials

    def knobs(self, params: np.ndarray) -> dict[str, float]:
        return dict(zip(self.knob_names, (float(v) for v in params), strict=True))

    def trial_loss(self, params: np.ndarray) -> float | None:
        reduced = reduce_replies(
            self.evaluate_workers(self.knobs(params), loss_only=True), len(self.knob_names)
        )
        if reduced.particle_lost:
            return None
        return reduced.loss + float(params @ self.prior @ params)

    def evaluate(self, params: np.ndarray, iteration: int) -> LMPoint:
        reduced = reduce_replies(self.evaluate_workers(self.knobs(params)), len(self.knob_names))
        if reduced.normal is None:
            raise ValueError("The workers returned no Gauss-Newton matrix")
        if reduced.particle_lost:
            return LMPoint(reduced.loss, reduced.grad, 2.0 * reduced.normal, failed=True)
        if iteration == 0:
            warn_if_singular(reduced.normal, self.knob_names)  # before the prior is added
        loss, grad, hessian = add_quadratic_prior(
            reduced.loss, reduced.grad, 2.0 * reduced.normal, params, self.prior
        )
        return LMPoint(loss, grad, hessian, failed=False)

    def on_accept(self, params: np.ndarray, point: LMPoint) -> None:
        self.history.append((self.knobs(params), float(point.loss)))


class LMFitter:
    """Shared lifecycle and Levenberg-Marquardt solve of the full-ring closed-orbit/closed-twiss fitters.

    Subclasses set ``worker_class`` and build :attr:`worker_payloads`, one
    ``(WorkerConfig, data)`` per worker, in their own ``__init__``.
    """

    worker_class: type
    log_suffix: str
    fit_label: str
    #: Whether ``accelerator.optimise_energy`` is supported: ``pt`` is then fitted as the last knob.
    supports_energy: bool = False
    #: Whether a rejected step costs a plain-orbit chi² instead of a full evaluation (orbit-only workers).
    loss_only_trials: bool = False

    def __init__(
        self,
        accelerator: Accelerator,
        sequence_config: SequenceConfig,
        *,
        runtime: WorkerRuntime,
        lm_config: LevenbergMarquardtConfig | None = None,
        initial_knob_strengths: dict[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        use_errors: bool = True,
        knob_sigmas: Mapping[str, float] | None = None,
        output_config: OutputConfig | None = None,
        b2_errors: Path | None = None,
        machine_state: MachineState | None = None,
        debug: bool = False,
    ) -> None:
        if accelerator.optimise_energy and not self.supports_energy:
            raise ValueError(
                f"{type(self).__name__} does not support accelerator.optimise_energy; "
                "measurement momenta are fixed pt inputs"
            )

        self.lm_config = lm_config or LevenbergMarquardtConfig()
        self.sequence_config = sequence_config
        self.diagnostics: FitDiagnostics | None = None
        #: ``(knobs, loss)`` of every accepted iteration, in order
        self.history: list[tuple[dict[str, float], float]] = []
        #: Posterior inverse covariance (data + prior) at the best knobs of the last run, in knob order.
        self.posterior_normal: np.ndarray | None = None
        self.machine = MachineSetup(
            accelerator,
            sequence_config,
            output_config=output_config,
            initial_knob_strengths=initial_knob_strengths,
            true_strengths=true_strengths,
            debug=debug,
        )
        self.worker_payloads: list = []
        self.runtime = runtime
        self.debug = debug
        #: Known LHC dipole b2 error table, applied to every worker's sequence; the interface
        #: also needs the tune knobs that compensate it, from ``machine_state``.
        self.b2_errors = b2_errors
        self.machine_state = machine_state

        self.use_errors = use_errors
        #: Gaussian prior width of each knob (0 = free); see :func:`knob_sigma_array`.
        self.knob_sigma = knob_sigma_array(self.machine.knob_names, knob_sigmas or {})

    def close(self) -> None:
        """Shut down the MAD process of the model; call once the fit (and any use of its model) is done."""
        self.machine.close()

    def run(self) -> FitResult:
        """Run the Levenberg-Marquardt fit on one worker per payload."""
        run_log = self.machine.make_run_log(self.log_suffix)
        pool = WorkerPool()
        knob_names = list(self.machine.knob_names)
        initial_knobs = self.machine.initial_knobs

        def body() -> tuple[dict[str, float], np.ndarray | None]:
            for worker_id, (config, data) in enumerate(self.worker_payloads):
                pool.spawn(self.worker_class, worker_id, data, config, self.runtime).send(
                    Start(dict(initial_knobs))
                )
            return self._solve(pool.channels, run_log)

        knobs, normal_matrix = run_with_workers(
            body,
            fallback=lambda: (dict(initial_knobs), None),
            stop_workers=pool.stop,
            run_log=run_log,
        )
        logger.info("%s optimisation complete.", self.fit_label)
        self.posterior_normal = normal_matrix
        if not self.machine.output_config.include_uncertainty:
            sigmas = np.zeros(len(knob_names))
        elif normal_matrix is None:
            sigmas = np.full(len(knob_names), np.nan)
        else:
            sigmas = hessian_uncertainties(normal_matrix)
        return FitResult(
            knobs=self.machine.accelerator.format_result_knobs(knobs),
            uncertainties=self.machine.accelerator.format_result_knobs(
                dict(zip(knob_names, sigmas.tolist(), strict=False))
            ),
            diagnostics=self.diagnostics,
            extra=self._extra_result(),
            extra_uncertainties=self._extra_uncertainties(),
        )

    def _extra_result(self) -> dict[str, float]:
        """Fitted parameters that are not model knobs."""
        return {}

    def _extra_uncertainties(self) -> dict[str, float]:
        """1-sigma uncertainties of :meth:`_extra_result`."""
        return {}

    def _worker_config(self, measured_bpms: list[str]) -> WorkerConfig:
        """Full-ring worker config; BPMs without a measurement are bad, so twiss does not observe them."""
        measured = set(measured_bpms)
        unmeasured = [bpm for bpm in self.machine.all_bpms if bpm not in measured]
        return WorkerConfig(
            accelerator=self.machine.accelerator,
            magnet_range=self.sequence_config.magnet_range,
            machine_state=self.machine_state if self.b2_errors is not None else None,
            b2_errors=self.b2_errors,
            bad_bpms=unmeasured + list(self.sequence_config.bad_bpms or []),
            debug=self.debug,
            mad_logfile=self.machine.output_config.mad_logfile,
            python_logfile=self.machine.output_config.python_logfile,
        )

    def _solve(self, channels, run_log) -> tuple[dict[str, float], np.ndarray | None]:
        """Levenberg-Marquardt solve over the shared knobs; returns the best knobs and the posterior normal matrix there.

        The closed twiss is close to linear in the knob strengths over the range
        a fit explores, so a curvature-preconditioned step converges in a few
        iterations and, unlike plain gradient descent, resolves the weakly
        conditioned directions made identifiable by the multi-delta measurements.
        """
        knob_names = list(self.machine.knob_names)
        initial = np.array(
            [float(self.machine.initial_knobs[name]) for name in knob_names], dtype=float
        )
        problem = _KnobProblem(
            knob_names,
            lambda knobs, **kwargs: self._evaluate_workers(channels, knobs, **kwargs),
            np.diag(inverse_variances(self.knob_sigma)),
            self.history,
            loss_only_trials=self.loss_only_trials and not self.machine.accelerator.optimise_energy,
        )
        optimiser, self.diagnostics = run_levenberg_marquardt(
            initial, problem, self.lm_config, run_log=run_log, param_names=knob_names
        )
        # Half the chi² Hessian (prior included) at the best point is the posterior inverse covariance.
        hessian = optimiser.best_hessian
        return problem.knobs(optimiser.best.value), None if hessian is None else 0.5 * hessian

    def _evaluate_workers(
        self, channels, knobs: dict[str, float], *, loss_only: bool = False
    ) -> list[GradReply]:
        """Send ``knobs`` to the workers and return the replies to be summed, in order."""
        channels.send_all(Evaluate(knobs, loss_only=loss_only))
        return channels.recv_all()


def load_measurement(measurement: pd.DataFrame, observables: tuple[str, ...]) -> pd.DataFrame:
    """Validate a measured optics set (DataFrame indexed by BPM name) for the fit.

    Args:
        measurement: DataFrame carrying the columns listed in :data:`MEASUREMENT_COLUMNS`.
        observables: Observable names that must be present.

    Returns:
        Copy of ``measurement``. Missing error columns are filled with NaN.
        Individual NaN errors drop their own points; an observable whose errors
        are *all* unusable is rejected by :func:`build_observable` unless the
        caller passes ``use_errors=False``, because a family with no variances
        cannot be weighted commensurably against families that have them.
    """
    measurement = measurement.copy()

    missing = [
        MEASUREMENT_COLUMNS[name][0]
        for name in observables
        if MEASUREMENT_COLUMNS[name][0] not in measurement.columns
    ]
    if missing:
        raise ValueError(f"Measurement is missing required columns: {sorted(missing)}")

    for name in observables:
        err_column = MEASUREMENT_COLUMNS[name][1]
        if err_column not in measurement.columns:
            measurement[err_column] = np.nan
    return measurement


def _advance_targets(
    measurement: pd.DataFrame, bpms: list[str], value_column: str, var_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-interval phase advance and its variance from a cumulative measurement.

    ``build_twiss_from_measurements`` accumulates the measured adjacent advances
    into a cumulative phase (and a cumulative variance), so differencing
    consecutive BPMs recovers exactly the independent advances that were measured
    - and their independent variances. The modulo keeps the result in ``[0, 1)``
    to match the wrapped model advance.
    """
    phase = measurement.loc[bpms, value_column].to_numpy(dtype=float)
    cumulative_var = measurement.loc[bpms, var_column].to_numpy(dtype=float)
    return np.mod(np.diff(phase), 1.0), np.diff(cumulative_var)


def measured_bpms(all_bpms: list[str], measurement: pd.DataFrame, label: str) -> list[str]:
    """The model BPMs that have a measurement, in model order; at least two."""
    bpms = [bpm for bpm in all_bpms if bpm in measurement.index]
    if len(bpms) < 2:
        raise ValueError(f"Fewer than two model BPMs have a measurement ({label}).")
    logger.info(
        "Fit (%s) over %d measured BPMs (%d model BPMs unobserved)",
        label,
        len(bpms),
        len(all_bpms) - len(bpms),
    )
    return bpms


def build_observable(
    name: str,
    measurement: pd.DataFrame,
    bpms: list[str],
    use_errors: bool,
    pt: float,
) -> Observable:
    """Extract one observable's targets and variances from a measurement frame."""
    value_column, err_column = MEASUREMENT_COLUMNS[name]
    observable = Observable(name=name, targets=np.empty(0), variances=np.empty(0))

    if observable.kind is ObservableKind.ADVANCE:
        # The phase "error" column is already a cumulative *variance*, so
        # differencing gives the per-interval variance directly.
        targets, variances = _advance_targets(measurement, bpms, value_column, err_column)
    else:
        targets = measurement.loc[bpms, value_column].to_numpy(dtype=float)
        errors = measurement.loc[bpms, err_column].to_numpy(dtype=float)
        variances = errors**2

    usable = np.isfinite(variances) & (variances > 0)
    if use_errors and not np.any(usable):
        finite = np.isfinite(variances)
        raise ValueError(
            f"Observable {name!r} (pt={pt:g}) has no usable measurement errors: "
            f"n={variances.size}, finite={int(np.count_nonzero(finite))}, "
            f"positive=0, zero={int(np.count_nonzero(finite & (variances == 0)))}, "
            f"negative={int(np.count_nonzero(finite & (variances < 0)))}. "
            f"Column {err_column!r} is empty or identically zero. A 1/<target^2> "
            "fallback weight is not used: it is in this family's own physical units, "
            "so against families with real variances it would effectively remove the "
            "family from the fit. Supply real errors, or set use_errors=False so every "
            "family is normalised the same way."
        )

    if not use_errors:
        # Weight each family by the inverse of its own mean square target. Without
        # this an unweighted fit would be dominated by whichever family happens to
        # carry the largest numbers (beta in metres against orbit in millimetres),
        # which is a units artefact rather than a statement about information.
        # Safe only because *no* family carries real variances in this branch.
        scale = float(np.mean(targets[np.isfinite(targets)] ** 2)) if targets.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        variances = np.full_like(targets, scale, dtype=float)
    elif not np.all(usable):
        logger.warning(
            "Observable '%s' (pt=%g): %d/%d points have no usable error and carry zero weight.",
            name,
            pt,
            int(np.count_nonzero(~usable)),
            usable.size,
        )

    return Observable(name=name, targets=targets, variances=variances)
