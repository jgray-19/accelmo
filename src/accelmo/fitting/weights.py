"""Inverse-variance loss weights."""

from __future__ import annotations

import numpy as np


def variance_to_weight(variances: np.ndarray) -> np.ndarray:
    """Inverse-variance weights; a non-finite or non-positive variance gets weight zero."""
    weights = np.zeros_like(variances, dtype=np.float64)
    valid = np.isfinite(variances) & (variances > 0.0)
    np.divide(1.0, variances, out=weights, where=valid)
    return weights
