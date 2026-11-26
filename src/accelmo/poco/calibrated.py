"""Closed-orbit fitter that also fits BPM gains and corrector kick calibrations (see :mod:`accelmo.poco.calibration`)."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from accelmo.fitting.bpm_gains import bpm_gain_names
from accelmo.fitting.lm_loop import LMPoint, LMProblem, run_levenberg_marquardt
from accelmo.fitting.protocol import Evaluate, distribute
from accelmo.fitting.reduction import reduce_replies
from accelmo.fitting.shared_reference import attach_shared_block
from accelmo.machine.machine_state import resolve_machine_state
from accelmo.poco.calibration import (
    CalibrationBlocks,
    CalibrationSpec,
    apply_prior,
    back_substitute,
    corrector_gain_name,
    gain_covariance,
    reduce_blocks,
)
from accelmo.poco.closed_orbit import CLOSED_ORBIT_OBSERVABLES, ClosedOrbitFitter
from accelmo.poco.workers.calibrated import CalibratedClosedOrbitWorker

if TYPE_CHECKING:
    from collections.abc import Mapping
    from multiprocessing import shared_memory
    from pathlib import Path

    from accelmo.config import WorkerRuntime
    from accelmo.fitting.config import OutputConfig, SequenceConfig
    from accelmo.machine.accelerators import Accelerator
    from accelmo.optimisers.levenberg_marquardt import LevenbergMarquardtConfig
    from accelmo.poco.closed_orbit import ClosedOrbitSeries

LOGGER = logging.getLogger(__name__)


class _CalibratedProblem(LMProblem):
    """Knobs and corrector gains as LM parameters; the BPM gains are eliminated from each solve.

    The BPM gains are back-substituted from the best point's blocks after every step.
    """

    def __init__(
        self,
        fitter: CalibratedClosedOrbitFitter,
        channels,
        knob_names: list[str],
        g_names: list[str],
        b_names: list[str],
        sigma_q: np.ndarray,
        zero_mean: np.ndarray | None,
    ) -> None:
        self.fitter = fitter
        self.channels = channels
        self.knob_names = knob_names
        self.names = knob_names + g_names + b_names
        self.n_q, self.n_g = len(knob_names), len(g_names)
        self.sigma_q = sigma_q
        self.zero_mean = zero_mean
        self.loss_only_trials = fitter.loss_only_trials
        self.b = np.zeros(len(b_names))
        self.best_b = np.zeros(len(b_names))
        self.best_blocks: CalibrationBlocks | None = None

    def _knobs_at(self, params: np.ndarray) -> dict[str, float]:
        values = np.concatenate([params, self.b]).tolist()
        return dict(zip(self.names, values, strict=True))

    def _apply_prior(self, blocks, params: np.ndarray, **kwargs) -> None:
        apply_prior(
            blocks,
            params,
            self.b,
            self.fitter.sigma_corrector,
            self.fitter.sigma_bpm,
            self.sigma_q,
            zero_mean=self.zero_mean,
            **kwargs,
        )

    def trial_loss(self, params: np.ndarray) -> float | None:
        loss = self.fitter._collect_loss_only(self.channels, self._knobs_at(params))
        if loss is None:
            return None
        shell = SimpleNamespace(loss=loss, n_q=self.n_q)
        self._apply_prior(shell, params, loss_only=True)
        return shell.loss

    def evaluate(self, params: np.ndarray, iteration: int) -> LMPoint:
        n = self.n_q + self.n_g
        blocks = self.fitter._collect_calibrated(self.channels, self._knobs_at(params))
        if blocks is None:
            return LMPoint(float("nan"), np.zeros(n), np.zeros((n, n)), failed=True)
        self._apply_prior(blocks, params)
        grad, hess = reduce_blocks(blocks)
        return LMPoint(blocks.loss, grad, hess, extra=blocks)

    def on_accept(self, params: np.ndarray, point: LMPoint) -> None:
        self.best_b, self.best_blocks = self.b.copy(), point.extra
        knobs = dict(zip(self.knob_names, params[: self.n_q].tolist(), strict=True))
        self.fitter.history.append((knobs, float(point.loss)))

    def after_step(self, params: np.ndarray, optimiser) -> None:
        if self.best_blocks is not None:
            self.b = self.best_b + back_substitute(self.best_blocks, params - optimiser.best.value)

    def final_bpm_gains(self) -> np.ndarray:
        """The best point's BPM gains.

        Its own gain step has not been taken yet (it is the one at zero step in the corrector
        parameters): without it the gains of a fit that converges at once would be the
        starting zeros.
        """
        if self.best_blocks is None:
            return self.best_b
        return self.best_b + back_substitute(self.best_blocks, np.zeros(self.n_q + self.n_g))


class CalibratedClosedOrbitFitter(ClosedOrbitFitter):
    """:class:`ClosedOrbitFitter` with ``(1 + b_bpm)(1 + g_corrector)`` calibration parameters.

    Every ``<kick_prefix><corrector>`` kick a series' own ``machine_state`` sets (its change from the fitter's
    ``machine_state`` is worked out internally) gets a gain, named after the corrector; a series may set any number, none included. The BPM gains are
    eliminated by a Schur complement each iteration and recovered by back-substitution, so the Levenberg-Marquardt solve
    is over the magnet knobs and the corrector gains only. ``sigma_bpm`` / ``sigma_corrector`` are Gaussian priors that
    also fix the ``b``/``g`` scale degeneracy. ``knob_sigmas`` is the magnet-knob prior of :class:`LMFitter`. After the fit, :attr:`calibration_result` holds every gain.

    With ``fit_corrector_gains=False`` only the BPM gains are fitted: no kick is scaled and ``sigma_corrector`` is unused. This is the
    mode for absolute orbits, where every corrector's standing kick is part of the orbit and only the correctors some series sets could
    be given a gain. ``corrector_gain_pattern`` (regex, full match on the upper-case corrector name) gives a gain only to the
    correctors it matches; the others' kicks are taken as exact. ``zero_mean_gain_pattern`` holds the mean gain of the fitted
    correctors it matches at 0, removing the common scale those correctors share with the BPM gains (it goes to the BPMs).
    """

    worker_class = CalibratedClosedOrbitWorker

    def __init__(
        self,
        accelerator: Accelerator,
        sequence_config: SequenceConfig,
        series: list[ClosedOrbitSeries] | tuple[ClosedOrbitSeries, ...],
        observables: tuple[str, ...] = CLOSED_ORBIT_OBSERVABLES,
        lm_config: LevenbergMarquardtConfig | None = None,
        initial_knob_strengths: dict[str, float] | None = None,
        machine_state: Path | Mapping[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        use_errors: bool = True,
        knob_sigmas: Mapping[str, float] | None = None,
        output_config: OutputConfig | None = None,
        runtime: WorkerRuntime | None = None,
        *,
        sigma_bpm: float = 1e-2,
        sigma_corrector: float = 1e-2,
        loss_only_trials: bool = True,
        kick_prefix: str = "k_",
        fit_corrector_gains: bool = True,
        corrector_gain_pattern: str | None = None,
        zero_mean_gain_pattern: str | None = None,
        debug: bool = False,
    ) -> None:
        self.kick_prefix = kick_prefix
        self.fit_corrector_gains = bool(fit_corrector_gains)
        self.corrector_gain_pattern = corrector_gain_pattern
        self.zero_mean_gain_pattern = zero_mean_gain_pattern
        #: Evaluate each trial point with the loss alone (plain closed-orbit solves, no knob derivatives) and run the full
        #: gradient/Hessian evaluation only at points that improve on the best loss. Same accepted path, rejected trials ~6x cheaper.
        self.loss_only_trials = bool(loss_only_trials)
        self.sigma_bpm = float(sigma_bpm)
        self.sigma_corrector = float(sigma_corrector)
        self._attached: dict[
            str, shared_memory.SharedMemory
        ] = {}  # workers' shared blocks, mapped once
        self.calibration_result: dict[str, float] = {}
        self.calibration_uncertainty: dict[str, float] = {}
        self.calibration_spec: CalibrationSpec | None = None
        # The gains are grouped into every worker's batch, so the shared-reference worker never applies.
        super().__init__(
            accelerator,
            sequence_config,
            series,
            observables=observables,
            lm_config=lm_config,
            initial_knob_strengths=initial_knob_strengths,
            machine_state=machine_state,
            true_strengths=true_strengths,
            use_errors=use_errors,
            knob_sigmas=knob_sigmas,
            output_config=output_config,
            runtime=runtime,
            debug=debug,
        )

    def _group_payloads(self, payloads):
        """Always batch workers, each series tagged with the global gain layout."""
        bpms = tuple(payloads[0][1].bpm_names)
        if any(tuple(data.bpm_names) != bpms for _, data in payloads):
            raise ValueError("Calibration fits need every series to observe the same BPMs")
        # Every kick a series' own state sets gets a gain; a series that sets none (an off-momentum orbit, say) has none to fit.
        correctors = ()
        if self.fit_corrector_gains:
            kicks = (
                name
                for item in self.series
                for name in resolve_machine_state(item.machine_state)
                if name.startswith(self.kick_prefix)
            )
            correctors = tuple(
                corrector
                for corrector in dict.fromkeys(
                    name[len(self.kick_prefix) :].upper() for name in kicks
                )
                if self.corrector_gain_pattern is None
                or re.fullmatch(self.corrector_gain_pattern, corrector)
            )
        self.calibration_spec = CalibrationSpec(bpms, correctors, self.kick_prefix)
        for _, data in payloads:
            data.calibration = self.calibration_spec
        groups = distribute([len(item.measurements) for item in self.series], self.n_workers)
        LOGGER.info(
            "Calibration fit: %d BPMs x 2 planes, %d correctors, %d worker(s)",
            len(bpms),
            len(correctors),
            len(groups),
        )
        return [(payloads[group[0]][0], [payloads[i][1] for i in group]) for group in groups]

    def _attach(self, name: str) -> shared_memory.SharedMemory:
        """Map a worker's shared block once, without taking over its deletion."""
        if name not in self._attached:
            self._attached[name] = attach_shared_block(name)
        return self._attached[name]

    def _collect_calibrated(self, channels, knobs: dict[str, float]) -> CalibrationBlocks | None:
        """Sum every worker's blocks; ``None`` if any worker lost its closed orbit."""
        channels.send_all(Evaluate(knobs))
        replies = channels.recv_all()
        if reduce_replies(replies, 1).particle_lost:
            return None
        payloads = [reply.extra for reply in replies]
        first = payloads[0]
        shape = (first["n_q"], first["n_g"], first["n_b"])
        parts = [
            CalibrationBlocks(*shape, self._attach(p["shared_memory"]).buf, zero=False)
            for p in payloads
        ]
        total = CalibrationBlocks(*shape)
        total.loss = sum(p["loss"] for p in payloads)

        # Memory-bound sum of ~170 MB per worker: split by rows over threads (numpy releases the GIL).
        def add_rows(name: str, rows: slice) -> None:
            out = getattr(total, name)[rows]
            np.copyto(out, getattr(parts[0], name)[rows])
            for part in parts[1:]:
                np.add(out, getattr(part, name)[rows], out=out)

        step = 64
        jobs = [
            (name, slice(i, i + step))
            for name in CalibrationBlocks.ARRAYS
            for i in range(0, getattr(total, name).shape[0], step)
        ]
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda job: add_rows(*job), jobs))
        return total

    def _collect_loss_only(self, channels, knobs: dict[str, float]) -> float | None:
        """Sum every worker's chi² (no derivatives); ``None`` if any worker lost its closed orbit."""
        channels.send_all(Evaluate(knobs, loss_only=True))
        reduced = reduce_replies(channels.recv_all(), 1)
        return None if reduced.particle_lost else reduced.loss

    def _extra_result(self) -> dict[str, float]:
        return dict(self.calibration_result)

    def _extra_uncertainties(self) -> dict[str, float]:
        return dict(self.calibration_uncertainty)

    def _solve(self, channels, run_log):
        spec = self.calibration_spec
        if spec is None:
            raise RuntimeError("The calibration spec is built when the data is prepared")
        knob_names = list(self.machine.knob_names)
        n_q, n_g = len(knob_names), len(spec.correctors)
        g_names = [corrector_gain_name(c) for c in spec.correctors]
        b_names = bpm_gain_names(spec.bpms)

        u = np.zeros(n_q + n_g)
        u[:n_q] = [float(self.machine.initial_knobs[name]) for name in knob_names]
        zero_mean = None
        if self.zero_mean_gain_pattern is not None:
            zero_mean = np.array(
                [
                    i
                    for i, c in enumerate(spec.correctors)
                    if re.fullmatch(self.zero_mean_gain_pattern, c)
                ],
                dtype=int,
            )
            if not len(zero_mean):
                raise ValueError(
                    f"zero_mean_gain_pattern {self.zero_mean_gain_pattern!r} matches no fitted corrector"
                )

        problem = _CalibratedProblem(
            self, channels, knob_names, g_names, b_names, self.knob_sigma, zero_mean
        )
        self.begin_dashboard_fit(knob_names + g_names, u, gain_names=g_names)
        optimiser, self.diagnostics = run_levenberg_marquardt(
            u,
            problem,
            self.lm_config,
            run_log=run_log,
            param_names=knob_names + g_names,
            dashboard=self.dashboard,
        )
        best_u = optimiser.best.value
        self.calibration_result = dict(zip(g_names, best_u[n_q:].tolist(), strict=True))
        best_b = problem.final_bpm_gains()
        self.calibration_result.update(zip(b_names, best_b.tolist(), strict=True))
        normal = None
        if problem.best_blocks is not None:
            normal, sigma_u, sigma_b = gain_covariance(problem.best_blocks)
            self.calibration_uncertainty = dict(
                zip(g_names + b_names, [*sigma_u[n_q:], *sigma_b], strict=True)
            )
        return dict(zip(knob_names, best_u[:n_q].tolist(), strict=True)), normal
