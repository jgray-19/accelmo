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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.sparse import block_diag
from scipy.sparse.linalg import lsqr

from accelmo.fitting.bpm_gains import (
    ZERO_MEAN_SIGMA,
    bpm_gain_names,
    only_bpm_gains,
    without_bpm_gains,
)
from accelmo.fitting.config import OutputConfig
from accelmo.fitting.lifecycle import run_with_workers
from accelmo.fitting.lm_loop import LMPoint, run_levenberg_marquardt
from accelmo.fitting.protocol import Evaluate
from accelmo.fitting.reduction import reduce_replies
from accelmo.fitting.results import FitDiagnostics, FitResult
from accelmo.fitting.setup import MachineSetup
from accelmo.fitting.uncertainty import sandwich_uncertainties
from accelmo.optimisers.levenberg_marquardt import LevenbergMarquardtConfig
from accelmo.tracking.config.tracking import (
    TrackingModeSetup,
    acd_marker_setup,
    acd_sinusoid_setup,
    arc_by_arc_setup,
    kicker_setup,
)
from accelmo.tracking.data_manager import DataManager
from accelmo.tracking.dispatch.setup import WorkerObservationPlan, WorkerSetupHelper
from accelmo.tracking.ranges import resolve_bpm_ranges
from accelmo.tracking.session import TrackingSession
from accelmo.tracking.sgd.loop import SGDLoop

LM_OPTIMISER = "levenberg_marquardt"
#: A refresh that moves no start coordinate by more than this (m, rad) leaves the objective unchanged.
LM_REFRESH_ATOL = 1e-12
#: LM stops after this many accepted steps in a row that each lower the loss by less than this fraction.
LM_STALL_TOLERANCE = 1e-6
LM_STALL_PATIENCE = 5

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from accelmo.analysis import DegeneracyReport
    from accelmo.config import OptimiserConfig, SimulationConfig
    from accelmo.fitting.config import SequenceConfig
    from accelmo.machine.accelerators import Accelerator
    from accelmo.tracking.config.models import (
        CheckpointConfig,
        KickerConfig,
        MeasurementConfig,
    )

    #: ``(current_knobs, best_knobs) -> new x, px, y, py start coordinates or None``
    InitialConditionsCallback = Callable[[dict[str, float], dict[str, float]], np.ndarray | None]
    #: ``(epoch, train_loss, validation_loss, grad_norm, true_diff)``
    LossCallback = Callable[[int, float, float | None, float, float], None]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StartGainMap:
    """Starts as a linear function of the BPM gains: ``s(g) = starts + jacobian @ (g - gains)``.

    ``starts`` ``(n_particles, 4)`` in :meth:`TrackingSession.send_init_condition_updates`
    order, ``jacobian`` ``(n_particles, 4, n_gains)`` and ``gains`` ``(n_gains,)`` in
    :attr:`TrackingFitter.gain_names` order. See docs/start_gain_chain_rule.md.
    """

    starts: np.ndarray
    jacobian: np.ndarray
    gains: np.ndarray

    def at(self, gains: np.ndarray) -> np.ndarray:
        return self.starts + self.jacobian @ (gains - self.gains)


def lm_prior_matrix(ridge: np.ndarray, zero_mean: list[np.ndarray], weight: float) -> np.ndarray:
    """Quadratic prior ``p^T P p``: ``diag(ridge)`` plus ``weight * mean²`` of each index group in ``zero_mean``."""
    prior = np.diag(np.asarray(ridge, dtype=float))
    for group in zero_mean:
        a = np.zeros(len(prior))
        a[group] = 1.0 / len(group)
        prior += weight * np.outer(a, a)
    return prior


def chained_sandwich_uncertainties(
    hessian: np.ndarray,
    prior_hessian: np.ndarray,
    normal: np.ndarray,
    noise: np.ndarray,
    unchained: np.ndarray,
) -> np.ndarray:
    """1-sigma ``sqrt(diag(H⁻¹ B H⁻¹))`` with ``H`` the LM's chained loss Hessian (priors included).

    ``normal``/``noise`` are the fixed-start ``A``/``B`` in their own weight units; the scale to
    loss units comes from the ``unchained`` rows (no gains), where ``H - prior_hessian ∝ A``.
    """
    rows = np.ix_(unchained, unchained)
    scale = np.trace(normal[rows]) / np.trace((hessian - prior_hessian)[rows])
    return sandwich_uncertainties(scale * np.asarray(hessian, dtype=float), noise)


def chain_starts_through_gains(
    grad: np.ndarray,
    hessian: np.ndarray,
    start_grad: np.ndarray,
    knob_start: np.ndarray,
    blocks: np.ndarray,
    jacobian: np.ndarray,
    gains: slice,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and Gauss-Newton Hessian of ``L(k, g, s(g))`` from those of ``L(k, g, s)``.

    ``start_grad`` (4P,), ``knob_start`` (n, 4P) and ``blocks`` (P, 4, 4) are the start
    blocks, ``jacobian`` (P, 4, n_gains) is ds/dg and ``gains`` the gains' slice of the
    n parameters. Derivation: docs/start_gain_chain_rule.md.
    """
    ds = jacobian.reshape(-1, jacobian.shape[-1])
    cross = knob_start @ ds
    grad, hessian = grad.copy(), np.array(hessian, dtype=float)
    grad[gains] += ds.T @ start_grad
    hessian[:, gains] += cross
    hessian[gains, :] += cross.T
    hessian[gains, gains] += ds.T @ (block_diag(blocks, format="csr") @ ds)
    return grad, hessian


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
    """

    initial_knob_strengths: dict[str, float] | None = None
    true_strengths: Path | dict[str, float] | None = None
    debug: bool = False
    output_config: OutputConfig = field(default_factory=OutputConfig)
    checkpoint_config: CheckpointConfig | None = None
    initial_conditions_callback: InitialConditionsCallback | None = None
    loss_callback: LossCallback | None = None
    initial_starts: np.ndarray | None = None


class TrackingFitter:
    """Recovers magnet knob strengths from turn-by-turn tracking, via MAD-NG.

    Composes a :class:`~accelmo.fitting.setup.MachineSetup` (model,
    BPM ranges, starting knobs), the measurement :class:`DataManager`, a
    :class:`~accelmo.tracking.session.TrackingSession` per run and the
    :class:`~accelmo.tracking.sgd.loop.SGDLoop`. It is not constructed
    directly: the ``setup`` comes from one of the public subclasses
    (:class:`ArcByArcFitter`, :class:`KickerFitter`, :class:`ACDMarkerFitter`).

    All inputs and results are in optimisation space.
    """

    def __init__(
        self,
        setup: TrackingModeSetup,
        *,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        options: FitterOptions | None = None,
    ):
        options = options if options is not None else FitterOptions()
        accelerator.log_optimisation_targets()
        if setup.simulation_config.optimise_momenta:
            logger.info("Including momenta (px, py) in loss function")
        else:
            logger.info("Using position-only optimisation (x, y only)")
        optimiser_config.log_state()

        self.tracking_plan = setup.plan
        self.options = options
        self.measurement_config = measurement_config
        details = measurement_config.details
        self.interface_options = [d.interface_options for d in details]
        self.machine_deltaps = [d.machine_deltap for d in details]
        # Per-file BPM the measurement turns are recorded from; in kicker mode,
        # fall back to the marker when a file does not set its own.
        self.first_bpms = [d.first_bpm or setup.first_bpm_fallback for d in details]

        self.machine = MachineSetup(
            accelerator,
            setup.simulation_config,
            sequence_config,
            output_config=options.output_config,
            cycle_marker=setup.plan.cycle_marker,
            anchor_mode=setup.plan.tracking_anchor_mode,
            anchor_markers=setup.plan.tracking_anchor_sources,
            initial_knob_strengths=options.initial_knob_strengths,
            true_strengths=options.true_strengths,
            debug=options.debug,
            **_first_set(self.interface_options, ("machine_state", "b2_errors")),
        )
        self.ranges = resolve_bpm_ranges(
            self.machine, setup.plan, setup.bpm_start_points, setup.bpm_end_points
        )
        self.worker_setup: WorkerSetupHelper
        #: Per file, the workers one of its turn batches fans out to.
        self.worker_plans: dict[int, list[WorkerObservationPlan]]
        self.data_manager: DataManager | None = self._load_data()
        simulation_config = self.machine.simulation_config
        for worker_module in ("accelmo.fitting.worker", "accelmo.tracking.worker"):
            logging.getLogger(worker_module).setLevel(simulation_config.worker_logging_level)
        # Every BPM the fit reads gets an x and a y gain; markers are not BPMs.
        self.gain_bpms: tuple[str, ...] = (
            tuple(b for b in self.machine.all_bpms if b not in self.tracking_plan.extra_markers)
            if simulation_config.bpm_gain_sigma is not None
            else ()
        )
        self.gain_names = bpm_gain_names(self.gain_bpms)
        #: Every fitted parameter, in uncertainty order: magnet knobs, then gains.
        self.knob_names = list(self.machine.knob_names) + self.gain_names
        self.lm_config: LevenbergMarquardtConfig | None = None
        self.loop: SGDLoop | None = None
        #: Prior width per fitted parameter (0 = none), from ``knob_sigmas`` as POCO resolves it.
        self.knob_sigma = self._knob_sigma(simulation_config.knob_sigmas or {})
        if self.knob_sigma.any() and optimiser_config.optimiser_type != LM_OPTIMISER:
            raise ValueError("knob_sigmas needs the LM optimiser")
        if optimiser_config.optimiser_type == LM_OPTIMISER:
            # Tracking losses are far below 1, so the loss-change stop is purely relative; it needs
            # several stalled steps in a row because single steps can be tiny early in a fit.
            self.lm_config = LevenbergMarquardtConfig(
                max_iterations=optimiser_config.max_epochs,
                gradient_converged_value=optimiser_config.gradient_converged_value,
                loss_relative_tolerance=LM_STALL_TOLERANCE,
                loss_relative_floor=0.0,
                loss_stall_patience=LM_STALL_PATIENCE,
            )
        else:
            self.loop = SGDLoop(
                self.machine.knob_names,
                self.machine.true_strengths,
                optimiser_config,
                simulation_config.num_batches,
            )
        #: How the last run ended.
        self.diagnostics: FitDiagnostics | None = None
        self._lm_best: dict[str, float] = {}
        #: Start (x, px, y, py) per particle from the last LM run with ``fit_start_coords``.
        self.fitted_starts = np.zeros((0, 4))
        #: LM Gauss-Newton matrix at the best point (knobs, then starts), loss units.
        self.lm_hessian = np.zeros((0, 0))
        #: Chi-square per unit of LM loss, set by the last LM run.
        self._lm_prior_hessian = np.zeros((0, 0))
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

        def body() -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray] | None]:
            session = self.start_session(start_knobs)
            if self.options.initial_starts is not None:
                session.send_init_condition_updates(self.options.initial_starts)
            total_turns = self.data_manager.get_total_turns()
            # The workers hold their own copies of the data now.
            self.data_manager = None
            gc.collect()

            if self.lm_config is not None:
                final_knobs = self._solve_lm(session, run_log)
            else:
                final_knobs = self.loop.run(
                    self._with_gains(self.machine.initial_knobs),
                    session.training.channels,
                    total_turns=total_turns,
                    run_log=run_log,
                    checkpoint_config=self.options.checkpoint_config,
                    validation_loss_fn=session.validation_loss,
                    loss_callback=self.options.loss_callback,
                    epoch_end_hook=self._epoch_end_hook(session),
                    gain_names=self.gain_names,
                    gain_ridge=self._gain_ridge(session),
                )

            # Workers still hold the last batch's knobs; the uncertainty must be
            # propagated at the knobs we return.
            if include_uncertainty:
                session.set_training_knobs(final_knobs)
            normal_and_noise = session.stop_and_collect_uncertainty(
                len(final_knobs), propagate_uncertainty=include_uncertainty
            )
            return final_knobs, normal_and_noise

        final_knobs, normal_and_noise = run_with_workers(
            body,
            fallback=lambda: (
                self._lm_best if self.loop is None else self.loop.best.value,
                None,
            ),
            stop_workers=self._terminate_session,
            run_log=run_log,
        )
        uncertainties = self._uncertainties(final_knobs, normal_and_noise)
        logger.info("Optimisation complete.")
        return FitResult(
            knobs=without_bpm_gains(final_knobs),
            uncertainties=without_bpm_gains(uncertainties),
            diagnostics=self.diagnostics if self.loop is None else self.loop.diagnostics,
            extra=only_bpm_gains(final_knobs),
            extra_uncertainties=only_bpm_gains(uncertainties),
        )

    def _solve_lm(self, session: TrackingSession, run_log) -> dict[str, float]:
        """Levenberg-Marquardt over every batch; returns the best knobs.

        With ``fit_start_coords`` every particle's start (x, px, y, py) joins the
        parameters after the knobs, in session particle order, and is pushed to the
        workers before each evaluation; the fitted starts are kept in :attr:`fitted_starts`.
        """
        names = list(self.knob_names)  # magnet knobs, then the BPM gains
        n_knobs = len(names)
        gains = slice(n_knobs - len(self.gain_names), n_knobs)
        channels = session.training.channels
        n_batches = self.machine.simulation_config.num_batches
        # The LM loss is also per particle (worker scale 1/(points·particles)), unlike the gain sums.
        per_particle = np.mean(session.training.particle_counts) / n_batches
        ridge = np.zeros(n_knobs)
        ridge[gains] = self._gain_ridge(session) / per_particle
        prior = self.knob_sigma > 0
        ridge[prior] = session.chi2_to_loss / self.knob_sigma[prior] ** 2 / per_particle
        zero_mean = [
            np.array([i for i, name in enumerate(names) if name.startswith(f"bpmgain.{plane}.")])
            for plane in self.machine.simulation_config.bpm_gain_zero_mean_planes
        ]
        prior_matrix = lm_prior_matrix(
            ridge, zero_mean, session.chi2_to_loss / ZERO_MEAN_SIGMA**2 / per_particle
        )
        self._lm_prior_hessian = 2.0 * prior_matrix
        fit_starts = self.machine.simulation_config.fit_start_coords
        if fit_starts and session.validation:
            raise ValueError("fit_start_coords needs a fit without validation workers")
        follow = self.machine.simulation_config.starts_follow_gains
        if follow and (fit_starts or not self.gain_names):
            raise ValueError("starts_follow_gains needs fitted gains and no fit_start_coords")
        start_map: list[StartGainMap] = []  # the callback's latest map
        n_train = sum(session.training.particle_counts)

        def gather(knobs: dict[str, float]) -> tuple[float, np.ndarray, np.ndarray, bool, list]:
            """Summed knob loss/grad/Hessian and, per batch, the replies in worker order."""
            loss, grad, hessian, lost, per_batch = 0.0, np.zeros(n_knobs), 0.0, False, []
            for batch in range(n_batches):
                channels.send_all(Evaluate(knobs, batch, curvature=True))
                replies = channels.recv_all()
                reduced = reduce_replies(replies, n_knobs)
                loss += reduced.loss
                grad = grad + reduced.grad
                hessian = hessian + reduced.hessian
                lost = lost or reduced.particle_lost
                per_batch.append(replies)
            return loss, grad, hessian, lost, per_batch

        def start_slots(per_batch: list) -> list[list[slice]]:
            """Global particle rows of each (batch, worker) reply."""
            counts = session.training.particle_counts
            worker_first = np.concatenate([[0], np.cumsum(counts)[:-1]])
            done = np.zeros(len(counts), dtype=int)
            slots = []
            for replies in per_batch:
                row = []
                for w, reply in enumerate(replies):
                    n = len(reply.extra["start"])
                    row.append(slice(worker_first[w] + done[w], worker_first[w] + done[w] + n))
                    done[w] += n
                slots.append(row)
            return slots

        initial_knobs = self._with_gains(self.machine.initial_knobs)
        initial = np.array([initial_knobs[name] for name in names], dtype=float)
        knobs0 = dict(zip(names, initial.tolist(), strict=True))
        slots: list[list[slice]] = []
        basis = None
        if fit_starts:
            *_, per_batch = gather(knobs0)
            slots = start_slots(per_batch)
            n_particles = sum(session.training.particle_counts)
            starts0, ids = np.zeros((n_particles, 4)), np.zeros(n_particles, dtype=np.int64)
            for replies, row in zip(per_batch, slots, strict=True):
                for reply, rows in zip(replies, row, strict=True):
                    starts0[rows] = reply.extra["start"]
                    ids[rows] = reply.extra["ids"]
            basis = self._start_basis(ids, starts0)
            free0 = starts0.ravel() if basis is None else basis[1]
            initial = np.concatenate([initial, free0])

        def particle_starts(free: np.ndarray) -> np.ndarray:
            return (free if basis is None else basis[0] @ free).reshape(-1, 4)

        def start_blocks(per_batch: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Training particles' dL/ds (4P), d2L/dk ds (knobs, 4P) and d2L/ds2 (P, 4, 4)."""
            start_grad = np.zeros(4 * n_train)
            knob_start = np.zeros((n_knobs, 4 * n_train))
            blocks = np.zeros((n_train, 4, 4))
            for replies, row in zip(per_batch, start_slots(per_batch), strict=True):
                for reply, rows in zip(replies, row, strict=True):
                    cols = slice(4 * rows.start, 4 * rows.stop)
                    start_grad[cols] = reply.extra["grad"].ravel()
                    knob_start[:, cols] = reply.extra["knob_start"].reshape(n_knobs, -1)
                    blocks[rows] = reply.extra["start_start"]
            return start_grad, knob_start, blocks

        def evaluate(params: np.ndarray, iteration: int) -> LMPoint:
            del iteration
            knobs = dict(zip(names, (float(v) for v in params[:n_knobs]), strict=True))
            if fit_starts:
                session.send_init_condition_updates(particle_starts(params[n_knobs:]))
            if start_map:
                session.send_init_condition_updates(start_map[0].at(params[gains]))
            loss, grad, hessian, lost, per_batch = gather(knobs)
            hessian = np.array(hessian)
            if start_map and not lost:
                grad, hessian = chain_starts_through_gains(
                    grad, hessian, *start_blocks(per_batch), start_map[0].jacobian[:n_train], gains
                )
            if prior_matrix.any():  # Gaussian priors on gains and knobs, once for all batches
                p = params[:n_knobs]
                loss, grad = loss + p @ prior_matrix @ p, grad + 2.0 * prior_matrix @ p
                hessian += 2.0 * prior_matrix
            n = len(params)
            if not fit_starts or lost:
                full_grad, full_hess = np.zeros(n), np.zeros((n, n))
                full_grad[:n_knobs], full_hess[:n_knobs, :n_knobs] = grad, hessian
                return LMPoint(loss, full_grad, full_hess, lost)
            n_particles = n_train
            start_grad, knob_start, blocks = start_blocks(per_batch)
            full_grad = np.concatenate([grad, np.zeros(n - n_knobs)])
            full_hess = np.zeros((n, n))
            full_hess[:n_knobs, :n_knobs] = hessian
            if basis is None:
                full_grad[n_knobs:] = start_grad
                full_hess[:n_knobs, n_knobs:] = knob_start
                for particle in range(n_particles):
                    cols = slice(n_knobs + 4 * particle, n_knobs + 4 * particle + 4)
                    full_hess[cols, cols] = blocks[particle]
            else:
                t = basis[0]
                full_grad[n_knobs:] = t.T @ start_grad
                full_hess[:n_knobs, n_knobs:] = (t.T @ knob_start.T).T
                full_hess[n_knobs:, n_knobs:] = (
                    t.T @ (block_diag(blocks, format="csr") @ t)
                ).toarray()
            full_hess[n_knobs:, :n_knobs] = full_hess[:n_knobs, n_knobs:].T
            return LMPoint(loss, full_grad, full_hess, lost)

        callback = self.options.initial_conditions_callback
        pushed: list[np.ndarray] = []

        def refresh(params: np.ndarray) -> bool:
            """Rebuild the start conditions from the accepted knobs; True if they moved."""
            fitted = dict(zip(names, (float(v) for v in params[:n_knobs]), strict=True))
            coords = callback(
                {**self.machine.initial_model_values, **without_bpm_gains(fitted)},
                {**self.machine.initial_model_values, **fitted},
            )
            if coords is None or fit_starts:  # fitted starts: the callback only observes
                return False
            if follow:
                if start_map:  # the workers hold the old map at these gains
                    pushed[:] = [start_map[0].at(params[gains])]
                start_map[:] = [coords]
                coords = coords.at(params[gains])
            session.send_init_condition_updates(coords)
            moved = not pushed or not np.allclose(
                coords, pushed[-1], rtol=0.0, atol=LM_REFRESH_ATOL
            )
            pushed[:] = [coords]
            return moved

        self._lm_best = knobs0
        param_names = names + [f"start_param{i}" for i in range(len(initial) - n_knobs)]
        optimiser, self.diagnostics = run_levenberg_marquardt(
            initial,
            evaluate,
            self.lm_config,
            refresh=None if callback is None else refresh,
            run_log=run_log,
            param_names=param_names,
        )
        best = optimiser.best.value
        self._lm_best = dict(zip(names, (float(v) for v in best[:n_knobs]), strict=True))
        self.start_params = best[n_knobs:]
        self.fitted_starts = particle_starts(self.start_params) if fit_starts else np.zeros((0, 4))
        self.lm_hessian = optimiser.best_hessian
        if fit_starts:
            session.send_init_condition_updates(self.fitted_starts)
        return self._lm_best

    def _start_basis(
        self, ids: np.ndarray, starts: np.ndarray
    ) -> tuple[sparse.csr_matrix, np.ndarray] | None:
        """Linear map from free parameters to the particles' starts, and its initial value.

        ``ids`` are the particles' start reading ids, ``starts`` their measured starts (both
        in session particle order). ``None``: every particle's start is free.
        """
        del ids, starts
        return None

    def _knob_sigma(self, knob_sigmas: dict[str, float]) -> np.ndarray:
        """Width of each of :attr:`knob_names` by name, else by terminal attribute (lower case)."""
        sigmas = {name.lower(): float(sigma) for name, sigma in knob_sigmas.items()}
        if any(sigma <= 0 for sigma in sigmas.values()):
            raise ValueError("knob_sigmas must be > 0")
        names = [name.lower() for name in self.knob_names]
        attributes = [name.rpartition(".")[2] for name in names]
        if unknown := sorted(set(sigmas) - set(names) - set(attributes)):
            raise ValueError(f"knob_sigmas names knobs that are not optimised: {unknown[:5]}")
        return np.array(
            [sigmas.get(n, sigmas.get(a, 0.0)) for n, a in zip(names, attributes, strict=True)]
        )

    def _with_gains(self, knobs: dict[str, float]) -> dict[str, float]:
        """``knobs`` plus every fitted gain at its starting value: ``initial_knob_strengths``, else zero."""
        initial = self.options.initial_knob_strengths or {}
        return {**knobs, **{name: float(initial.get(name, 0.0)) for name in self.gain_names}}

    def _gain_ridge(self, session: TrackingSession) -> float:
        """The gain prior ``1/σ²`` in the units of the workers' gain sums.

        Those sums are ``Σ w ...`` over each worker's points divided by its point count,
        with ``w`` the normalised weights, so a physical chi-square is
        ``session.chi2_to_loss`` times them (exact when every worker observes the same
        number of points; the mean is used otherwise).
        """
        if not self.gain_names:
            return 0.0
        return session.chi2_to_loss / self.machine.simulation_config.bpm_gain_sigma**2

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
            self.worker_setup,
            self.machine.simulation_config,
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
                len(self.knob_names), propagate_uncertainty=True
            )
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
        normal_and_noise: tuple[np.ndarray, np.ndarray] | None,
    ) -> dict[str, float]:
        """1-sigma uncertainty of each optimisation knob; zero when not computed.

        ``(A, B)`` are the normal matrix and the measurement noise propagated through
        every observation and start coordinate, ``Cov = A⁻¹ B A⁻¹``. With starts that follow
        the gains, ``A`` holds the starts fixed and misses the directions they cancel, so the
        LM's chained Hessian (priors included) replaces it.
        """
        follow = self.machine.simulation_config.starts_follow_gains and self.lm_hessian.size
        if not self.machine.output_config.include_uncertainty or normal_and_noise is None:
            uncertainties = np.zeros(len(final_knobs), dtype=np.float64)
        elif follow:
            n_gains = len(self.gain_names)
            uncertainties = chained_sandwich_uncertainties(
                self.lm_hessian,
                self._lm_prior_hessian,
                *normal_and_noise,
                np.arange(len(final_knobs) - n_gains),
            )
        else:
            uncertainties = sandwich_uncertainties(*normal_and_noise)
        return dict(zip(self.knob_names, uncertainties.tolist(), strict=True))

    def _load_data(self) -> DataManager:
        """Load the measurement files, plan the workers, batch the turns and fix ``num_batches``."""
        machine = self.machine
        data = DataManager(
            self.tracking_plan.observed_bpms(machine.bpms_in_range, machine.all_bpms),
            machine.all_bpms,
            machine.simulation_config,
            self.measurement_config.files,
            tracking_plan=self.tracking_plan,
            first_bpms=self.first_bpms,
            extra_markers=list(self.tracking_plan.extra_markers),
        )
        data.load_track_data()
        output_config = machine.output_config
        self.worker_setup = WorkerSetupHelper(
            accelerator=machine.accelerator,
            all_bpms=machine.all_bpms,
            fixed_start=self.ranges.fixed_start,
            fixed_end=self.ranges.fixed_end,
            use_fixed_bpm=machine.simulation_config.use_fixed_bpm,
            bad_bpms=machine.sequence_config.bad_bpms,
            file_kick_planes=data.file_kick_planes,
            magnet_range=machine.sequence_config.magnet_range,
            interface_options_per_file=self.interface_options,
            debug=self.options.debug,
            mad_logfile=output_config.mad_logfile,
            python_logfile=output_config.python_logfile,
            tracking_plan=self.tracking_plan,
        )
        self.worker_plans = {
            idx: self.worker_setup.plan_workers(
                self.ranges.start_bpms, self.ranges.end_bpms, idx, set(tracks.marker_col)
            )
            for idx, tracks in data.tracks.items()
        }
        data.prepare_turn_batches(max(len(plans) for plans in self.worker_plans.values()))

        # No batch may need more turns than the smallest worker holds.
        min_turns_per_batch = min(len(batch) for batch in data.turn_batches)
        machine.simulation_config = dataclasses.replace(
            machine.simulation_config,
            num_batches=min(machine.simulation_config.num_batches, min_turns_per_batch),
        )
        data.simulation_config = machine.simulation_config
        return data


def initial_conditions_hook(
    callback: InitialConditionsCallback,
    fixed_model_values: dict[str, float],
    push: Callable[[np.ndarray], None],
) -> Callable[[dict[str, float], dict[str, float]], str]:
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


def _first_set(options_per_file: list[dict], keys: tuple[str, ...]) -> dict:
    """For each of ``keys``, the first file's non-``None`` value, so the fitter's model matches the workers'."""
    merged: dict = {}
    for key in keys:
        for options in options_per_file:
            if options.get(key) is not None:
                merged[key] = options[key]
                break
    return merged


class ArcByArcFitter(TrackingFitter):
    """Fit magnet strengths from arc-by-arc BPM ranges.

    Tracks the caller's ``bpm_start_points`` x ``bpm_end_points`` ranges. Set
    ``acd_excited`` when the data was AC-dipole excited: the exciter markers are then
    installed and any range that would straddle the AC dipole is rerouted the long
    way round the ring. Leave it ``False`` for free-oscillation data.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        *,
        acd_excited: bool = False,
        options: FitterOptions | None = None,
    ):
        setup = arc_by_arc_setup(
            accelerator=accelerator,
            simulation_config=simulation_config,
            bpm_start_points=bpm_start_points,
            bpm_end_points=bpm_end_points,
            acd_excited=acd_excited,
        )
        super().__init__(
            setup,
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            options=options,
        )


class KickerFitter(TrackingFitter):
    """Fit magnet strengths from kicker-excited turn-by-turn data.

    Runs a single worker forward-only from the kicker marker, which supplies the
    tracking initial conditions.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        kicker_config: KickerConfig,
        options: FitterOptions | None = None,
    ):
        setup = kicker_setup(kicker_config, simulation_config)
        super().__init__(
            setup,
            accelerator=accelerator,
            optimiser_config=optimiser_config,
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
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        options: FitterOptions | None = None,
    ):
        setup = acd_marker_setup(accelerator, simulation_config)
        super().__init__(
            setup,
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            options=options,
        )


class ACDSinusoidFitter(TrackingFitter):
    """Fit AC-dipole data tracked forward over many turns through the model AC kickers.

    The accelerator must install the kickers (PSB ``acd_drive_tunes``); fitting the ``acd``
    family adds their (a, b) volts as knobs.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        options: FitterOptions | None = None,
    ):
        setup = acd_sinusoid_setup(accelerator, simulation_config)
        super().__init__(
            setup,
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            options=options,
        )


class ACDTurnSinusoidFitter(TrackingFitter):
    """AC-dipole data, one turn per state, no AC dipole in the model.

    Every turn has one fitted state at the ``before`` marker; it is tracked backward from
    ``before`` and forward from ``after`` = ``before`` + the kick: a full turn each way,
    so every reading is fitted by two states. The kick is a sinusoid
    per file and plane, ``a cos(2 pi Q_d row) + b sin(2 pi Q_d row)`` (``drive_tunes`` per file),
    linear in (a, b). Needs ``fit_start_coords`` and the LM optimiser.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        drive_tunes: tuple[tuple[float, float], ...],
        options: FitterOptions | None = None,
    ):
        if not simulation_config.fit_start_coords:
            raise ValueError("ACDTurnSinusoidFitter fits the turn states: needs fit_start_coords")
        setup = acd_marker_setup(accelerator, simulation_config)
        super().__init__(
            setup,
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            options=options,
        )
        plan = setup.plan
        self.drive_tunes = drive_tunes
        self._grid = {}
        for file_idx, tracks in self.data_manager.tracks.items():
            cols = {name.upper(): col for name, col in tracks.marker_col.items()}
            self._grid[file_idx] = (
                len(tracks.markers),
                cols[plan.acd_before.upper()],
                cols[plan.acd_after.upper()],
            )
        if len(drive_tunes) != len(self._grid):
            raise ValueError(f"{len(drive_tunes)} drive tunes for {len(self._grid)} files")
        #: Fitted (a_x, b_x, a_y, b_y) kick per file, and the fitted turn states.
        self.kicks = np.zeros((len(self._grid), 4))
        self.turn_states = np.zeros((0, 4))

    def _start_basis(
        self, ids: np.ndarray, starts: np.ndarray
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        files = ids >> 32
        cells = ids & 0xFFFFFFFF
        keys, is_after = [], []
        for file_idx, cell in zip(files.tolist(), cells.tolist(), strict=True):
            n_markers, before, after = self._grid[file_idx]
            row, col = divmod(cell, n_markers)
            if col not in (before, after):
                raise ValueError(f"Start at marker column {col} is not an AC-dipole marker")
            keys.append((file_idx, row))
            is_after.append(col == after)
        states = sorted(set(keys))
        index = {key: i for i, key in enumerate(states)}
        n_states, n_files = len(states), len(self._grid)
        kick0 = 4 * n_states
        rows, cols, vals = [], [], []
        for p, (key, after) in enumerate(zip(keys, is_after, strict=True)):
            for c in range(4):
                rows.append(4 * p + c)
                cols.append(4 * index[key] + c)
                vals.append(1.0)
            if after:
                file_idx, row = key
                for plane, coord in ((0, 1), (1, 3)):
                    phase = 2 * np.pi * self.drive_tunes[file_idx][plane] * row
                    for k, v in enumerate((np.cos(phase), np.sin(phase))):
                        rows.append(4 * p + coord)
                        cols.append(kick0 + 4 * file_idx + 2 * plane + k)
                        vals.append(v)
        basis = sparse.csr_matrix((vals, (rows, cols)), shape=(4 * len(keys), kick0 + 4 * n_files))
        # Initial states: measured before markers (after markers where a turn has no before);
        # initial kick: least squares of the measured after - before momenta.
        free0 = np.zeros(kick0 + 4 * n_files)
        have = np.zeros(n_states, dtype=bool)
        for p in np.argsort(is_after, kind="stable"):
            i = index[keys[p]]
            if not have[i]:
                free0[4 * i : 4 * i + 4] = starts[p]
                have[i] = True
        residual = starts.ravel() - basis[:, :kick0] @ free0[:kick0]
        kick_cols = basis[:, kick0:]
        free0[kick0:] = lsqr(kick_cols, residual)[0]
        self._n_states = n_states
        return basis, free0

    def _solve_lm(self, session, run_log):
        knobs = super()._solve_lm(session, run_log)
        n = 4 * self._n_states
        self.turn_states = self.start_params[:n].reshape(-1, 4)
        self.kicks = self.start_params[n:].reshape(-1, 4)
        return knobs
