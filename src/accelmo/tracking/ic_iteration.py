"""ACD sinusoid fits: single steps (IC or k free) and the joint fit with turn continuation.

Every step is one tracking LM fit with a different free set. The caller builds the fitter
for each step from a :class:`Step`; values outside the free set stay fixed at the current
state (``initial_knob_strengths``) and the starts are carried by ``initial_starts``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy as np

    from accelmo.tracking.fitter import TrackingFitter

logger = logging.getLogger(__name__)

CONTINUATION_TURNS = (10, 30, 100, 300, 1000)


@dataclass(frozen=True)
class Step:
    """What one fit frees and where it starts.

    ``errors`` are the fitted families (``Accelerator`` ``errors``); ``fit_start_coords``
    frees every start; ``n_turns`` limits the tracked turns (``None``: all).
    """

    errors: Mapping[str, set[str]]
    fit_start_coords: bool
    knobs: dict[str, float]
    starts: np.ndarray | None
    n_turns: int | None = None


@dataclass(frozen=True)
class State:
    """Current knobs, starts (x, px, y, py per particle; ``None``: measured) and loss.

    ``hessian`` is the LM Gauss-Newton matrix at the fit (free knobs, then starts). ``gains`` /
    ``gain_uncertainties`` are the BPM gains the fit returned (each fit restarts them at zero).
    """

    knobs: dict[str, float]
    starts: np.ndarray | None
    loss: float
    hessian: np.ndarray | None = None
    free_knobs: tuple[str, ...] = ()
    gains: dict[str, float] = field(default_factory=dict)
    gain_uncertainties: dict[str, float] = field(default_factory=dict)


def fit_step(
    make_fitter: Callable[[Step], TrackingFitter],
    state: State,
    errors: Mapping[str, set[str]],
    fit_start_coords: bool,
    n_turns: int | None = None,
) -> State:
    """One LM fit of ``errors`` (and the starts if ``fit_start_coords``) from ``state``."""
    fitter = make_fitter(Step(errors, fit_start_coords, state.knobs, state.starts, n_turns))
    result = fitter.run()
    starts = fitter.fitted_starts if fit_start_coords else state.starts
    loss = result.diagnostics.best_loss
    logger.info(f"step {sorted(errors)} starts={fit_start_coords} turns={n_turns}: loss {loss:.6e}")
    return State(
        {**state.knobs, **result.knobs},
        starts,
        loss,
        fitter.lm_hessian,
        tuple(result.knobs),
        result.extra,
        result.extra_uncertainties,
    )


def joint(
    make_fitter: Callable[[Step], TrackingFitter],
    state: State,
    ic_errors: Mapping[str, set[str]],
    k_errors: Mapping[str, set[str]],
    turns: tuple[int | None, ...] = (None,),
) -> list[State]:
    """Joint fit of starts, kick and k, once per entry of ``turns`` (continuation).

    Each entry is the window length, so the windows (starts) change between entries:
    later entries refit the starts from the measured ones and carry k and the kick.
    """
    errors = {**k_errors, **ic_errors}
    history = [state]
    for n_turns in turns:
        start = history[-1] if len(history) == 1 else replace(history[-1], starts=None)
        history.append(fit_step(make_fitter, start, errors, fit_start_coords=True, n_turns=n_turns))
    return history


def cold_joint(
    make_fitter: Callable[[Step], TrackingFitter],
    state: State,
    ic_errors: Mapping[str, set[str]],
    k_errors: Mapping[str, set[str]],
) -> list[State]:
    """Joint fit from the initialisation with turn continuation 10 -> 30 -> 100 -> all."""
    return joint(make_fitter, state, ic_errors, k_errors, (*CONTINUATION_TURNS, None))
