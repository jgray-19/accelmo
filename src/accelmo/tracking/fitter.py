"""Tracking fitters that recover magnet knob strengths from turn-by-turn data.

:class:`TrackingFitter` holds the mode-agnostic machinery (data + worker
management, the optimisation loop, and Hessian-based uncertainties). The public
entry points are its thin subclasses, one per tracking geometry:

* :class:`ArcByArcFitter` -- arc-by-arc ranges, with the AC dipole optionally
  accounted for (``acd_excited``).
* :class:`KickerFitter` -- forward-only tracking from a kicker marker.
* :class:`ACDMarkerFitter` -- bidirectional tracking from the AC-dipole markers.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from accelmo.config import TrackingParameters
from accelmo.dashboard import FitDashboard
from accelmo.fitting.bpm_gains import (
    bpm_gain_names,
    only_bpm_gains,
    without_bpm_gains,
)
from accelmo.fitting.config import OutputConfig
from accelmo.fitting.lifecycle import run_with_workers
from accelmo.fitting.lm_loop import run_levenberg_marquardt
from accelmo.fitting.priors import knob_sigma_array
from accelmo.fitting.results import FitDiagnostics, FitResult, UncertaintyInfo
from accelmo.fitting.setup import MachineSetup
from accelmo.fitting.uncertainty import sandwich_uncertainties
from accelmo.fitting.worker import WorkerConfig
from accelmo.fitting.worker_logging import WORKER_LOG_PATH
from accelmo.optimisers.levenberg_marquardt import LevenbergMarquardtConfig
from accelmo.tracking.config.tracking import (
    ACDArcByArcTrackingPlan,
    ACDSinusoidTrackingPlan,
    ACDTrackingPlan,
    KickerTrackingPlan,
    TrackingPlan,
)
from accelmo.tracking.data_manager import DataManager
from accelmo.tracking.dispatch.setup import TrackingWorkerConfig, plan_workers
from accelmo.tracking.lm_problem import StartGainMap, TrackingLMProblem
from accelmo.tracking.ranges import resolve_bpm_ranges
from accelmo.tracking.session import TrackingSession
from accelmo.tracking.sgd.loop import SGDLoop
from accelmo.tracking.uncertainty import UncertaintyCollection

#: LM stops after this many accepted steps in a row that each lower the chi² by less than this fraction.
LM_STALL_TOLERANCE = 1e-6
#: Below this chi² the stall test is absolute: ``abs(Δχ²) < 1e-12`` ends a noise-free fit.
LM_CHI2_FLOOR = 1e-6
LM_STALL_PATIENCE = 5
#: TrackingParameters fields only the LM solver fits.
LM_ONLY_PARAMETERS = (
    "knob_sigmas",
    "fit_start_coords",
    "starts_follow_gains",
    "bpm_gain_zero_mean_planes",
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from accelmo.analysis import DegeneracyReport
    from accelmo.config import OptimiserConfig, TrackingDataConfig, WorkerRuntime
    from accelmo.fitting.config import SequenceConfig
    from accelmo.machine.accelerators import Accelerator
    from accelmo.tracking.config.models import (
        CheckpointConfig,
        MeasurementConfig,
    )

    #: ``(current_knobs, best_knobs) -> new x, px, y, py start coordinates or None``; the LM fit
    #: with ``starts_follow_gains`` takes a :class:`StartGainMap` instead.
    InitialConditionsCallback = Callable[
        [dict[str, float], dict[str, float]], np.ndarray | StartGainMap | None
    ]
    #: ``(epoch, train_loss, validation_loss, grad_norm, true_diff)``
    LossCallback = Callable[[int, float, float | None, float, float], None]

logger = logging.getLogger(__name__)


def tracking_lm_config(max_iterations: int) -> LevenbergMarquardtConfig:
    """The tracking LM solver: stops after several stalled steps, since early steps can be tiny."""
    return LevenbergMarquardtConfig(
        max_iterations=max_iterations,
        loss_relative_tolerance=LM_STALL_TOLERANCE,
        loss_relative_floor=LM_CHI2_FLOOR,
        loss_stall_patience=LM_STALL_PATIENCE,
    )


@dataclass(frozen=True)
class FitterOptions:
    """Optional inputs shared by every tracking fitter.

    Attributes:
        initial_knob_strengths: Starting knob values (optimisation space; a
            ``deltap`` becomes ``pt``). Knobs left out keep the model value; fixed
            strengths outside the fitted knobs are set in every model.
        true_strengths: Known answer, as a dict or knob file, for diagnostics.
        debug: Run MAD-NG in debug mode.
        output_config: Logging, TensorBoard and uncertainty settings.
        checkpoint_config: Periodic checkpointing and restore.
        initial_conditions_callback: Epoch-end hook that returns refreshed start
            coordinates for every worker, or ``None`` to keep them.
        loss_callback: Called after every epoch with its losses.
        initial_starts: Start (x, px, y, py) per particle in session order, replacing
            the measured starts before the first evaluation.
        parameters: BPM gains, start coordinates and prior widths fitted with the knobs.
    """

    initial_knob_strengths: dict[str, float] | None = None
    true_strengths: Path | dict[str, float] | None = None
    debug: bool = False
    output_config: OutputConfig = field(default_factory=OutputConfig)
    checkpoint_config: CheckpointConfig | None = None
    initial_conditions_callback: InitialConditionsCallback | None = None
    loss_callback: LossCallback | None = None
    initial_starts: np.ndarray | None = None
    parameters: TrackingParameters = field(default_factory=TrackingParameters)


class TrackingFitter:
    """Recovers magnet knob strengths from turn-by-turn tracking, via MAD-NG.

    Composes a :class:`~accelmo.fitting.setup.MachineSetup` (model,
    BPM ranges, starting knobs), the measurement :class:`DataManager`, a
    :class:`~accelmo.tracking.session.TrackingSession` per run and the
    :class:`~accelmo.tracking.sgd.loop.SGDLoop`. It is not constructed
    directly: the ``plan`` comes from one of the public subclasses
    (:class:`ArcByArcFitter`, :class:`KickerFitter`, :class:`ACDMarkerFitter`,
    :class:`ACDSinusoidFitter`). Marker-anchored plans fix their own start points.

    All inputs and results are in optimisation space.
    """

    def __init__(
        self,
        plan: TrackingPlan,
        *,
        accelerator: Accelerator,
        solver: OptimiserConfig | LevenbergMarquardtConfig,
        runtime: WorkerRuntime,
        tracking_data: TrackingDataConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        bpm_start_points: list[str] | None = None,
        bpm_end_points: list[str] | None = None,
        use_fixed_bpm: bool = False,
        options: FitterOptions | None = None,
    ):
        options = options if options is not None else FitterOptions()
        if not isinstance(solver, LevenbergMarquardtConfig) and (
            lm_only := [name for name in LM_ONLY_PARAMETERS if getattr(options.parameters, name)]
        ):
            raise ValueError(f"{lm_only} need the LM solver")
        tracking_data = plan.configure(tracking_data)
        accelerator.log_optimisation_targets()
        self.dashboard = FitDashboard("tracking fit")
        self.dashboard.fact("observables", "x y px py" if tracking_data.optimise_momenta else "x y")

        self.tracking_plan = plan
        self.options = options
        self.runtime = runtime
        self.tracking_data = tracking_data
        self.parameters = options.parameters
        self.measurement_config = measurement_config
        details = measurement_config.details
        self.details = details
        self.machine_deltaps = [d.machine_deltap for d in details]
        # Per-file BPM the measurement turns are recorded from; in kicker mode,
        # fall back to the marker when a file does not set its own.
        self.first_bpms = [d.first_bpm or plan.first_bpm_fallback for d in details]

        with self.dashboard.phase("model"):
            self.machine = MachineSetup(
                accelerator,
                sequence_config,
                output_config=options.output_config,
                cycle_marker=plan.cycle_marker,
                anchor_mode=plan.tracking_anchor_mode,
                anchor_markers=plan.tracking_anchor_sources,
                initial_knob_strengths=options.initial_knob_strengths,
                true_strengths=options.true_strengths,
                debug=options.debug,
                # The fitter's own model matches the workers': the first file that sets each.
                machine_state=next((d.machine_state for d in details if d.machine_state), None),
                b2_errors=next((d.b2_errors for d in details if d.b2_errors), None),
            )
            self.ranges = resolve_bpm_ranges(
                self.machine,
                plan,
                list(plan.start_points) or list(bpm_start_points or []),
                list(bpm_end_points or []),
                use_fixed_bpm,
            )
        #: Per file, the workers one of its turn batches fans out to.
        self.worker_plans: dict[int, list[TrackingWorkerConfig]]
        with self.dashboard.phase("data"):
            self.data_manager: DataManager | None = self._load_data()
        # Every BPM the fit reads gets an x and a y gain; markers are not BPMs.
        self.gain_bpms: tuple[str, ...] = (
            tuple(b for b in self.machine.all_bpms if b not in self.tracking_plan.extra_markers)
            if self.parameters.bpm_gain_sigma is not None
            else ()
        )
        self.gain_names = bpm_gain_names(self.gain_bpms)
        #: Every fitted parameter, in uncertainty order: magnet knobs, then gains.
        self.knob_names = list(self.machine.knob_names) + self.gain_names
        self.lm_config: LevenbergMarquardtConfig | None = None
        self.loop: SGDLoop | None = None
        #: Prior width per fitted parameter (0 = none), from ``knob_sigmas`` as POCO resolves it.
        self.knob_sigma = knob_sigma_array(self.knob_names, self.parameters.knob_sigmas or {})
        if isinstance(solver, LevenbergMarquardtConfig):
            self.lm_config = solver
        else:
            self.loop = SGDLoop(
                self.machine.knob_names,
                self.machine.true_strengths,
                solver,
                self.tracking_data.num_batches,
            )
        #: How the last run ended.
        self.diagnostics: FitDiagnostics | None = None
        self._lm_best: dict[str, float] = {}
        #: Start (x, px, y, py) per particle from the last LM run with ``fit_start_coords``.
        self.fitted_starts = np.zeros((0, 4))
        #: LM chi-square Gauss-Newton Hessian at the best point (knobs, then starts), priors included.
        self.lm_hessian = np.zeros((0, 0))
        #: Prior inverse covariance (1/σ²) of the last LM run, added to the uncertainty ``A``.
        self._prior = np.zeros((0, 0))
        #: The workers of the current or last run.
        self.session: TrackingSession | None = None

    def run(self) -> FitResult:
        """Run the optimisation; knobs and uncertainties are in optimisation space."""
        checkpoint_config = self.options.checkpoint_config
        run_log = self.machine.make_run_log(
            "tracking_opt",
            resume=checkpoint_config is not None and checkpoint_config.restore_from_checkpoint,
        )
        include_uncertainty = self.machine.output_config.include_uncertainty
        start_knobs = self._with_gains(self.machine.worker_start_knobs)
        if run_log is not None:
            self.dashboard.run_log_path = run_log.directory
            self._log_workers_in(run_log.directory / WORKER_LOG_PATH)
        if not include_uncertainty:
            self.dashboard.skip("uncertainty")

        def body() -> tuple[dict[str, float], UncertaintyCollection | None]:
            with self.dashboard.phase("workers"):
                session = self.start_session(start_knobs)
                if self.options.initial_starts is not None:
                    session.send_init_condition_updates(self.options.initial_starts)
            self._show_run_facts(session)
            # The workers hold their own copies of the data now.
            self.data_manager = None
            gc.collect()

            if self.lm_config is not None:
                final_knobs = self._solve_lm(session, run_log, self.lm_config)
            else:
                assert self.loop is not None
                initial_knobs = self._with_gains(self.machine.initial_knobs)
                self.dashboard.begin_fit(
                    max_epochs=self.loop.config.max_epochs,
                    n_points=session.n_training_points,
                    n_params=len(self.knob_names),
                    start_knobs=initial_knobs,
                    knob_sigmas=dict(zip(self.knob_names, self.knob_sigma, strict=True)),
                    gain_names=self.gain_names,
                    stage_epochs=self.loop.config.stage_epochs,
                )
                final_knobs = self.loop.run(
                    initial_knobs,
                    session.training.channels,
                    n_points=session.n_training_points,
                    run_log=run_log,
                    dashboard=self.dashboard,
                    checkpoint_config=self.options.checkpoint_config,
                    validation_loss_fn=session.validation_loss,
                    loss_callback=self.options.loss_callback,
                    epoch_end_hook=self._epoch_end_hook(session),
                    gain_names=self.gain_names,
                    gain_ridge=self._gain_ridge(),
                )

            # Workers still hold the last batch's knobs; the uncertainty must be
            # propagated at the knobs we return.
            if include_uncertainty:
                session.set_training_knobs(final_knobs)
            with self.dashboard.phase("uncertainty") if include_uncertainty else nullcontext():
                return final_knobs, self._collect_uncertainty(
                    session, final_knobs, run_log, include_uncertainty
                )

        try:
            final_knobs, collection = run_with_workers(
                body,
                fallback=lambda: (
                    self._lm_best if self.loop is None else self.loop.best.value,
                    None,
                ),
                stop_workers=self._terminate_session,
                run_log=run_log,
            )
        finally:
            self.dashboard.close()
        uncertainties = self._uncertainties(final_knobs, collection)
        return FitResult(
            knobs=without_bpm_gains(final_knobs),
            uncertainties=without_bpm_gains(uncertainties),
            diagnostics=self.diagnostics if self.loop is None else self.loop.diagnostics,
            extra=only_bpm_gains(final_knobs),
            extra_uncertainties=only_bpm_gains(uncertainties),
            uncertainty=self._uncertainty_info(collection),
        )

    def _log_workers_in(self, logfile: Path) -> None:
        """Give workers that have no ``python_logfile`` of their own ``logfile`` (per worker)."""
        for idx, plans in self.worker_plans.items():
            self.worker_plans[idx] = [
                plan
                if plan.python_logfile is not None
                else dataclasses.replace(plan, python_logfile=logfile)
                for plan in plans
            ]

    def _show_run_facts(self, session: TrackingSession) -> None:
        """Put what the run is made of on the dashboard."""
        validation = (
            f" (val {session.n_validation_points:,})" if session.n_validation_points else ""
        )
        self.dashboard.fact("points", f"{session.n_training_points:,}{validation}")
        self.dashboard.fact(
            "knobs", f"{len(self.machine.knob_names)}  gains {len(self.gain_names)}"
        )
        self.dashboard.fact("batches", self.tracking_data.num_batches)
        self.dashboard.fact(
            "workers",
            f"{len(session.training.metadata)} train + {len(session.validation.metadata)} val",
        )

    def _collect_uncertainty(
        self,
        session: TrackingSession,
        final_knobs: dict[str, float],
        run_log,
        include_uncertainty: bool,
    ) -> UncertaintyCollection | None:
        """Run the uncertainty stage, logging its progress; ``None`` if it fails, so the fit survives.

        The fitted knobs are recorded in the run log before the stage starts.
        """
        hessians = (self.parameters.uncertainty_hessian,) if include_uncertainty else ()
        if run_log is not None:
            run_log.record_uncertainty(
                state="running", hessian=hessians[0] if hessians else None, knobs=final_knobs
            )

        def on_progress(done: int, total: int, elapsed: float, eta: float | None) -> None:
            self.dashboard.uncertainty_progress(done, total, eta)
            if run_log is not None:
                run_log.record_uncertainty(
                    state="running",
                    hessian="true",
                    knobs=final_knobs,
                    done=done,
                    total=total,
                    elapsed=elapsed,
                    eta=eta,
                )

        start = time.monotonic()
        try:
            collection = session.stop_and_collect_uncertainty(
                len(final_knobs), hessians, on_progress
            )
        except Exception as exc:
            seconds = time.monotonic() - start
            logger.exception(
                "Uncertainty stage failed after %.1f s; the fit result is kept", seconds
            )
            if run_log is not None:
                run_log.record_uncertainty(
                    state="failed",
                    knobs=final_knobs,
                    seconds=seconds,
                    error=f"{type(exc).__name__}: {exc}"[:500],
                )
            return None
        if run_log is not None and hessians:
            run_log.record_uncertainty(
                state="done",
                knobs=final_knobs,
                hessian=next(iter(collection.matrices)),
                seconds=collection.seconds,
                note="; ".join(f"{k} skipped: {v}" for k, v in collection.skipped.items()),
            )
        return collection

    @staticmethod
    def _uncertainty_info(collection: UncertaintyCollection | None) -> UncertaintyInfo | None:
        if collection is None or not collection.matrices:
            return None
        return UncertaintyInfo(
            hessian=next(iter(collection.matrices)),
            seconds=collection.seconds,
            note="; ".join(f"{k} skipped: {v}" for k, v in collection.skipped.items()),
        )

    def recompute_uncertainties(
        self,
        knobs: dict[str, float],
        hessians: tuple[str, ...] = ("gauss_newton", "true"),
    ) -> dict[str, dict[str, float]]:
        """Recompute the uncertainties at fixed ``knobs`` with each Hessian, in one worker session.

        ``knobs`` is every optimisation parameter: the fit's ``knobs`` and its gains (``extra``),
        or a run log's knobs at a step. Returns the 1-sigma uncertainties by knob name for each
        Hessian that ran (a requested true Hessian that cannot run is replaced, see
        :meth:`TrackingSession.stop_and_collect_uncertainty`) and logs their ratio.
        """
        start_knobs = self._with_gains(knobs)
        session = self.start_session(start_knobs)
        try:
            starts = self.fitted_starts if self.fitted_starts.size else self.options.initial_starts
            if starts is not None:
                session.send_init_condition_updates(starts)
            collection = session.stop_and_collect_uncertainty(len(start_knobs), hessians)
        except BaseException:
            self._terminate_session()
            raise
        sigmas = {
            hessian: self._uncertainties(start_knobs, UncertaintyCollection({hessian: matrices}))
            for hessian, matrices in collection.matrices.items()
        }
        logger.info("Uncertainties by Hessian (%s), %.1f s", ", ".join(sigmas), collection.seconds)
        if len(sigmas) == 2:
            (name_a, sig_a), (name_b, sig_b) = sigmas.items()
            ratios = np.array([sig_b[k] / sig_a[k] for k in sig_a if sig_a[k] > 0.0])
            if ratios.size:
                logger.info(
                    "%s / %s sigma ratio: min %.4g, median %.4g, max %.4g over %d knobs",
                    name_b,
                    name_a,
                    ratios.min(),
                    np.median(ratios),
                    ratios.max(),
                    ratios.size,
                )
        return sigmas

    def _solve_lm(
        self, session: TrackingSession, run_log, lm_config: LevenbergMarquardtConfig
    ) -> dict[str, float]:
        """Levenberg-Marquardt over every batch; returns the best knobs.

        With ``fit_start_coords`` every particle's start (x, px, y, py) joins the
        parameters after the knobs (see :class:`TrackingLMProblem`); the fitted starts are
        kept in :attr:`fitted_starts`.
        """
        problem = TrackingLMProblem(
            session,
            list(self.knob_names),
            len(self.gain_names),
            self.tracking_data.num_batches,
            self.parameters,
            self.knob_sigma,
            self._gain_ridge(),
            self.options.initial_conditions_callback,
            self.machine.initial_model_values,
        )
        self._prior = problem.prior_matrix

        initial = problem.initial_params(self._with_gains(self.machine.initial_knobs))
        self._lm_best = problem.knobs(initial)
        names = problem.param_names()
        self.dashboard.begin_fit(
            max_epochs=lm_config.max_iterations,
            n_points=session.n_training_points,
            n_params=len(names),
            start_knobs=dict(zip(names, initial.tolist(), strict=True)),
            knob_sigmas=dict(zip(self.knob_names, self.knob_sigma, strict=True)),
            gain_names=self.gain_names,
            loss_is_total=True,
            unit="iter",
            rate_name="λ",
        )
        optimiser, self.diagnostics = run_levenberg_marquardt(
            initial,
            problem,
            lm_config,
            run_log=run_log,
            param_names=names,
            dashboard=self.dashboard,
        )
        best = optimiser.best.value
        self._lm_best = problem.knobs(best)
        self.fitted_starts = problem.starts(best)
        best_hessian = optimiser.best_hessian
        self.lm_hessian = np.zeros((0, 0)) if best_hessian is None else best_hessian
        if problem.fit_starts:
            session.send_init_condition_updates(self.fitted_starts)
        return self._lm_best

    def _with_gains(self, knobs: dict[str, float]) -> dict[str, float]:
        """``knobs`` plus every fitted gain at its starting value: ``initial_knob_strengths``, else zero."""
        initial = self.options.initial_knob_strengths or {}
        return {**knobs, **{name: float(initial.get(name, 0.0)) for name in self.gain_names}}

    def _gain_ridge(self) -> float:
        """The gain prior ``1/σ²``, in the chi-square units of the workers' gain sums."""
        sigma = self.parameters.bpm_gain_sigma
        if not self.gain_names or sigma is None:
            return 0.0
        return 1.0 / sigma**2

    def start_session(
        self,
        start_knobs: dict[str, float] | None = None,
        *,
        enable_validation: bool | None = None,
    ) -> TrackingSession:
        """Start the workers, as :meth:`run` does, and return their session.

        ``start_knobs`` defaults to :attr:`MachineSetup.worker_start_knobs`;
        ``enable_validation`` to the tracking plan's choice. The data are reloaded
        if a previous run released them. Stop the session (or
        :meth:`TrackingSession.terminate` it) when done.
        """
        if self.data_manager is None:
            self.data_manager = self._load_data()
        if enable_validation is None:
            enable_validation = self.tracking_plan.enable_validation
        data = self.data_manager
        if start_knobs is None:
            start_knobs = self._with_gains(self.machine.worker_start_knobs)
        self.session = TrackingSession(
            self.machine.accelerator,
            self.runtime,
            self.tracking_data,
            self.parameters,
            turn_batches=data.turn_batches,
            validation_turn_batches=data.validation_turn_batches if enable_validation else [],
            file_turn_map=data.file_map,
            worker_plans=self.worker_plans,
            machine_deltaps=self.machine_deltaps,
            gain_bpms=self.gain_bpms,
        )
        self.session.start(data.tracks, start_knobs)
        return self.session

    def build_initial_normal_matrix(
        self, start_knobs: dict[str, float] | None = None
    ) -> tuple[np.ndarray, list[str]]:
        """Accumulate the Gauss-Newton normal matrix ``A = JᵀWJ`` at the initial knobs
        (or at ``start_knobs``).

        This starts the tracking workers exactly as :meth:`run` does, then requests
        the worker-side Hessians *without taking a single optimisation step*, so the
        returned matrix describes the problem the optimiser is about to face. Workers
        are shut down before returning; the fitter can still :meth:`run` afterwards.

        Returns:
            Tuple of (normal matrix, knob names) with the knob names in the row/column
            order of the matrix.
        """
        try:
            session = self.start_session(start_knobs)
            normal, _ = session.stop_and_collect_uncertainty(
                len(self.knob_names), ("gauss_newton",)
            ).primary
        except BaseException:
            self._terminate_session()
            raise
        return normal, list(self.knob_names)

    def check_degeneracy(self, **analyse_kwargs) -> DegeneracyReport:
        """Diagnose unconstrained knob directions before optimising.

        Builds the normal matrix at the initial knobs and analyses its eigenspectrum.
        Keyword arguments are forwarded to
        :func:`accelmo.analysis.analyse_degeneracy` (e.g. ``rel_tol``, ``scale``).
        """
        from accelmo.analysis import analyse_degeneracy

        normal_matrix, knob_names = self.build_initial_normal_matrix()
        return analyse_degeneracy(normal_matrix, knob_names, **analyse_kwargs)

    def _terminate_session(self) -> None:
        if self.session is not None:
            self.session.terminate()

    def _epoch_end_hook(
        self, session: TrackingSession
    ) -> Callable[[dict[str, float], dict[str, float]], str | None] | None:
        """The epoch-end hook refreshing the workers' initial conditions, or ``None`` without a callback."""
        callback = self.options.initial_conditions_callback
        if callback is None:
            return None
        return initial_conditions_hook(
            callback, self.machine.initial_model_values, session.send_init_condition_updates
        )

    def _uncertainties(
        self,
        final_knobs: dict[str, float],
        collection: UncertaintyCollection | None,
    ) -> dict[str, float]:
        """1-sigma uncertainty of each optimisation knob; zero when not computed.

        ``(A, B)`` are the normal matrix and the measurement noise propagated through
        every observation and start coordinate, ``Cov = A⁻¹ B A⁻¹``. With starts that follow
        the gains, ``A`` holds the starts fixed and misses the directions they cancel, so the
        LM's chained Hessian (priors included) replaces it.
        """
        follow = self.parameters.starts_follow_gains and self.lm_hessian.size
        if collection is None or not collection.matrices:
            uncertainties = np.zeros(len(final_knobs), dtype=np.float64)
        else:
            normal, noise = collection.primary
            if follow:
                uncertainties = sandwich_uncertainties(0.5 * self.lm_hessian, noise)
            elif self._prior.any():  # a prior is one more measurement of the knobs
                uncertainties = sandwich_uncertainties(normal + self._prior, noise + self._prior)
            else:
                uncertainties = sandwich_uncertainties(normal, noise)
        return dict(zip(self.knob_names, uncertainties.tolist(), strict=True))

    def _load_data(self) -> DataManager:
        """Load the measurement files, plan the workers, batch the turns and fix ``num_batches``."""
        machine = self.machine
        data = DataManager(
            self.tracking_plan.observed_bpms(machine.bpms_in_range, machine.all_bpms),
            machine.all_bpms,
            self.tracking_data,
            self.runtime.num_workers,
            self.measurement_config.files,
            tracking_plan=self.tracking_plan,
            first_bpms=self.first_bpms,
            extra_markers=list(self.tracking_plan.extra_markers),
        )
        data.load_track_data()
        output_config = machine.output_config
        self.worker_plans = {
            idx: plan_workers(
                WorkerConfig(
                    accelerator=machine.accelerator,
                    magnet_range=machine.sequence_config.magnet_range,
                    machine_state=self.details[idx].machine_state,
                    b2_errors=self.details[idx].b2_errors,
                    bad_bpms=machine.sequence_config.bad_bpms,
                    debug=self.options.debug,
                    mad_logfile=output_config.mad_logfile,
                    python_logfile=output_config.python_logfile,
                ),
                self.ranges,
                idx,
                tracks.kick_plane,
                set(tracks.marker_col),
            )
            for idx, tracks in data.tracks.items()
        }
        data.prepare_turn_batches(max(len(plans) for plans in self.worker_plans.values()))

        # No batch may need more turns than the smallest worker holds.
        min_turns_per_batch = min(len(batch) for batch in data.turn_batches)
        self.tracking_data = dataclasses.replace(
            self.tracking_data,
            num_batches=min(self.tracking_data.num_batches, min_turns_per_batch),
        )
        data.tracking_data = self.tracking_data
        return data


def initial_conditions_hook(
    callback: InitialConditionsCallback,
    fixed_model_values: dict[str, float],
    push: Callable[[np.ndarray], None],
) -> Callable[[dict[str, float], dict[str, float]], str | None]:
    """An epoch-end hook that pushes ``callback``'s new start coordinates with ``push``.

    The optimiser only carries this stage's knobs, but the workers track through the
    full model: ``fixed_model_values`` (fixed strengths and ``pt`` from
    ``initial_knob_strengths``) are merged into the knobs the callback sees, or it
    would rebuild initial conditions from the bare model defaults. An empty
    ``best_knobs`` stays empty so the callback can skip early epochs. The best
    knobs keep the fitted ``bpmgain.*`` entries so the callback can correct the
    readings it reconstructs from; the current knobs carry none.

    The hook returns a fragment for the epoch log line: ``dic`` is the mean absolute
    change per component since the previous update and ``dic0`` the mean absolute
    drift since the first one. They separate an update that is converging (``dic``
    falling, ``dic0`` settling), one that never moved the conditions (both zero), and
    a callback that keeps returning ``None``, which emits no fragment.
    """
    fixed = dict(fixed_model_values)
    # First and previous pushed coordinates, set on the first successful update so
    # a failing first call cannot become the drift baseline.
    pushed: dict[str, np.ndarray] = {}

    def hook(current_knobs: dict[str, float], best_knobs: dict[str, float]) -> str | None:
        current_knobs = without_bpm_gains(current_knobs)
        new_coords = callback(
            {**fixed, **current_knobs}, {**fixed, **best_knobs} if best_knobs else best_knobs
        )
        if new_coords is None:
            return None
        if isinstance(new_coords, StartGainMap):
            raise TypeError("A StartGainMap callback needs the LM fit with starts_follow_gains")
        push(new_coords)

        first = pushed.setdefault("first", new_coords)
        previous = pushed.get("previous", new_coords)
        pushed["previous"] = new_coords
        # Mean over particles and components, so the number stays comparable
        # whatever the tracking mode's particle count is.
        step = float(np.abs(new_coords - previous).mean())
        drift = float(np.abs(new_coords - first).mean())
        return f"dic={step:.2e}, dic0={drift:.2e}"

    return hook


class ArcByArcFitter(TrackingFitter):
    """Fit magnet strengths from arc-by-arc BPM ranges.

    Tracks the caller's ``bpm_start_points`` x ``bpm_end_points`` ranges. With
    ``use_fixed_bpm`` (default) every start pairs with one fixed end and every end with
    one fixed start, from ``magnet_range``; without it every start pairs with every end
    in both directions (many more, more correlated ranges). Set
    ``acd_excited`` when the data was AC-dipole excited: the exciter markers are then
    installed and any range that would straddle the AC dipole is rerouted the long
    way round the ring. Leave it ``False`` for free-oscillation data.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        solver: OptimiserConfig | LevenbergMarquardtConfig,
        runtime: WorkerRuntime,
        tracking_data: TrackingDataConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        *,
        acd_excited: bool = False,
        use_fixed_bpm: bool = True,
        options: FitterOptions | None = None,
    ):
        if not bpm_start_points:
            raise ValueError("Arc-by-arc mode requires bpm_start_points.")
        if not bpm_end_points:
            raise ValueError("Arc-by-arc mode requires bpm_end_points.")
        super().__init__(
            ACDArcByArcTrackingPlan(accelerator.ac_dipole_name) if acd_excited else TrackingPlan(),
            accelerator=accelerator,
            solver=solver,
            runtime=runtime,
            tracking_data=tracking_data,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            bpm_start_points=bpm_start_points,
            bpm_end_points=bpm_end_points,
            use_fixed_bpm=use_fixed_bpm,
            options=options,
        )


class KickerFitter(TrackingFitter):
    """Fit magnet strengths from kicker-excited turn-by-turn data.

    Tracks forward-only from the kicker marker, which supplies the tracking initial
    conditions, for ``turns_per_track`` turns; ``num_workers`` should be the number of
    files (one per momentum).
    """

    def __init__(
        self,
        accelerator: Accelerator,
        solver: OptimiserConfig | LevenbergMarquardtConfig,
        runtime: WorkerRuntime,
        tracking_data: TrackingDataConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        kicker_name: str,
        turns_per_track: int,
        options: FitterOptions | None = None,
    ):
        super().__init__(
            KickerTrackingPlan(kicker_name, turns_per_track),
            accelerator=accelerator,
            solver=solver,
            runtime=runtime,
            tracking_data=tracking_data,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            options=options,
        )


class ACDMarkerFitter(TrackingFitter):
    """Fit magnet strengths from AC-dipole data, tracked from the exciter markers.

    Tracks bidirectionally from the AC-dipole ``before``/``after`` markers (which
    supply the initial conditions) and observes the whole ring. Use
    :class:`ArcByArcFitter` with ``acd_excited=True`` instead to track AC-dipole data
    over ordinary arc ranges.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        solver: OptimiserConfig | LevenbergMarquardtConfig,
        runtime: WorkerRuntime,
        tracking_data: TrackingDataConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        options: FitterOptions | None = None,
    ):
        super().__init__(
            ACDTrackingPlan(accelerator.ac_dipole_name),
            accelerator=accelerator,
            solver=solver,
            runtime=runtime,
            tracking_data=tracking_data,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            options=options,
        )


class ACDSinusoidFitter(TrackingFitter):
    """Fit AC-dipole data tracked forward over many turns through the model AC kickers.

    The accelerator must install the kickers (PSB ``acd_drive_tunes``); fitting the ``acd``
    family adds their (a, b) volts as knobs. Each file is cut into back-to-back tracks of
    ``turns_per_track`` turns (see :class:`ACDSinusoidTrackingPlan`).
    """

    def __init__(
        self,
        accelerator: Accelerator,
        solver: OptimiserConfig | LevenbergMarquardtConfig,
        runtime: WorkerRuntime,
        tracking_data: TrackingDataConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        turns_per_track: int,
        options: FitterOptions | None = None,
    ):
        super().__init__(
            ACDSinusoidTrackingPlan(accelerator.ac_dipole_name, turns_per_track),
            accelerator=accelerator,
            solver=solver,
            runtime=runtime,
            tracking_data=tracking_data,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            options=options,
        )
