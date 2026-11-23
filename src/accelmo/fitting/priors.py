"""Gaussian priors shared by the tracking and closed-orbit LM fits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Width of the prior that holds a zero-mean gain group's mean at 0 (a near-exact constraint).
ZERO_MEAN_SIGMA = 1e-5


def knob_sigma_array(knob_names: Sequence[str], knob_sigmas: Mapping[str, float]) -> np.ndarray:
    """Prior width of each of ``knob_names`` (0 = free), by name, else by terminal attribute.

    ``knob_sigmas`` keys are case-insensitive: a knob name or a terminal attribute such as
    ``dk1l`` that applies to every knob ending in it.
    """
    sigmas = {name.lower(): float(sigma) for name, sigma in knob_sigmas.items()}
    if any(sigma <= 0 for sigma in sigmas.values()):
        raise ValueError("knob_sigmas must be > 0")
    names = [name.lower() for name in knob_names]
    attributes = [name.rpartition(".")[2] for name in names]
    if unknown := sorted(set(sigmas) - set(names) - set(attributes)):
        raise ValueError(f"knob_sigmas names knobs that are not optimised: {unknown[:5]}")
    return np.array(
        [sigmas.get(n, sigmas.get(a, 0.0)) for n, a in zip(names, attributes, strict=True)]
    )


def inverse_variances(sigmas: np.ndarray) -> np.ndarray:
    """``1/σ²`` of each prior width; 0 where ``σ`` is 0 (no prior)."""
    sigmas = np.asarray(sigmas, dtype=float)
    return np.divide(1.0, sigmas**2, out=np.zeros_like(sigmas), where=sigmas > 0)


def add_quadratic_prior(
    loss: float, grad: np.ndarray, hessian: np.ndarray, params: np.ndarray, prior: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """``loss``, ``grad`` and ``hessian`` plus those of ``params^T prior params``."""
    return (
        loss + float(params @ prior @ params),
        grad + 2.0 * prior @ params,
        hessian + 2.0 * prior,
    )


def quadratic_prior(ridge: np.ndarray, zero_mean: list[np.ndarray], weight: float) -> np.ndarray:
    """Quadratic prior ``p^T P p``: ``diag(ridge)`` plus ``weight * mean²`` of each index group in ``zero_mean``."""
    prior = np.diag(np.asarray(ridge, dtype=float))
    for group in zero_mean:
        a = np.zeros(len(prior))
        a[group] = 1.0 / len(group)
        prior += weight * np.outer(a, a)
    return prior
