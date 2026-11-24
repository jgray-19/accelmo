"""The closed-orbit normal matrix has the same degenerate directions on both backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from accelmo.analysis import analyse_degeneracy
from accelmo.fitting.pool import WorkerPool
from accelmo.fitting.protocol import Start
from accelmo.fitting.reduction import reduce_replies
from accelmo.fitting.uncertainty import SINGULAR_REL_TOLERANCE, warn_if_singular
from accelmo.poco.workers.closed_orbit import ClosedOrbitWorker
from tests.training.test_closed_orbit_loss_only import _fitter

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.serial

N_WEAK = 16  # of 48 knobs, from the three series of the loss-only test
MIN_GAP = 1e5  # weakest constrained eigenvalue over strongest weak one; measured 2.7e5 and 3.0e5
TOL_SUBSPACE = 1e-4  # spectral norm of the projector difference; measured 1.8e-5
TOL_EIGENVALUES = 1e-4  # relative; measured 2.4e-5


def _normal_matrix(seq_psb: Path, backend: str) -> tuple[np.ndarray, list[str]]:
    fitter = _fitter(seq_psb, backend, shared_reference=False, loss_only_trials=False)
    names = list(fitter.machine.knob_names)
    pool = WorkerPool()
    try:
        for worker_id, (config, data) in enumerate(fitter.worker_payloads):
            pool.spawn(ClosedOrbitWorker, worker_id, data, config, fitter.runtime)
        knobs = dict.fromkeys(names, 0.0)
        pool.channels.send_all(Start(knobs))
        replies = fitter._evaluate_workers(pool.channels, knobs)
        return reduce_replies(replies, len(names)).normal, names
    finally:
        pool.stop()
        fitter.close()


def _unit_diagonal_eigen(matrix: np.ndarray):
    scale = np.sqrt(np.diag(matrix))
    return np.linalg.eigh(matrix / np.outer(scale, scale))


@pytest.fixture(scope="module")
def spectra(seq_psb: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Unit-diagonal eigenvalues (ascending) and eigenvectors of each backend's normal matrix."""
    out = {}
    for backend in ("madng", "xtrack"):
        matrix, names = _normal_matrix(seq_psb, backend)
        assert warn_if_singular(matrix, names) == N_WEAK, backend
        report = analyse_degeneracy(matrix, names)
        assert report.n_degenerate == N_WEAK, backend
        assert report.numerical_rank == len(names) - N_WEAK, backend
        out[backend] = _unit_diagonal_eigen(matrix)
    return out


def test_weak_directions_are_separated_from_the_constrained_ones(spectra) -> None:
    for backend, (values, _) in spectra.items():
        assert values[N_WEAK - 1] < SINGULAR_REL_TOLERANCE * values[-1], backend
        assert values[N_WEAK] / values[N_WEAK - 1] > MIN_GAP, backend


def test_both_backends_find_the_same_degenerate_subspace(spectra) -> None:
    (values_a, vectors_a), (values_b, vectors_b) = spectra["madng"], spectra["xtrack"]
    projector_a = vectors_a[:, :N_WEAK] @ vectors_a[:, :N_WEAK].T
    projector_b = vectors_b[:, :N_WEAK] @ vectors_b[:, :N_WEAK].T
    assert np.linalg.norm(projector_a - projector_b, 2) < TOL_SUBSPACE
    assert np.abs(values_a[N_WEAK:] / values_b[N_WEAK:] - 1).max() < TOL_EIGENVALUES
