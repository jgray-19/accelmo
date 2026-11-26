"""Memory analysis of the true-Hessian uncertainty stage and its configuration."""

from __future__ import annotations

from math import comb
from types import SimpleNamespace

import pytest

from accelmo.backend.protocol import TrackGeometry
from accelmo.config import TrackingParameters
from accelmo.fitting.pool import WorkerPool
from accelmo.tracking.dispatch.setup import KickPlane, TrackingWorkerConfig
from accelmo.tracking.uncertainty import (
    MEMORY_FRACTION,
    SECOND_ORDER_SHARED_TPSAS,
    SECOND_ORDER_TEMPORARIES,
    SECOND_ORDER_TPSAS_PER_PARTICLE,
    _concurrency,
    second_order_blocker,
    second_order_bytes,
)

BPMS = ["BR3.BPM1L3", "BR3.BPM2L3"]
GIB = 2**30


def _pool(particle_counts: list[int]) -> WorkerPool:
    pool = WorkerPool()
    pool.particle_counts = particle_counts
    pool.metadata = [
        TrackingWorkerConfig(
            accelerator=None,
            magnet_range="$start/$end",
            geometry=TrackGeometry(start_bpm=BPMS[0], end_bpm=BPMS[-1]),
            file_idx=0,
            bpm_names=BPMS,
            kick_plane=KickPlane.XY,
        )
        for _ in particle_counts
    ]
    pool.workers = [None] * len(particle_counts)  # type: ignore[list-item]
    return pool


def _available(monkeypatch, nbytes: float) -> None:
    monkeypatch.setattr(
        "accelmo.tracking.uncertainty.psutil.virtual_memory",
        lambda: SimpleNamespace(available=nbytes),
    )


@pytest.mark.parametrize("n_params", [0, 1, 50, 300])
def test_second_order_bytes_counts_every_monomial_of_order_two(n_params: int) -> None:
    # Monomials of order <= 2 in 6 variables and n parameters: C(6 + n + 2, 2).
    tpsas = SECOND_ORDER_TPSAS_PER_PARTICLE * 10 + SECOND_ORDER_SHARED_TPSAS

    assert second_order_bytes(n_params, 10) == (
        8.0 * SECOND_ORDER_TEMPORARIES * comb(6 + n_params + 2, 2) * tpsas
    )


def test_second_order_memory_grows_quadratically_in_the_knobs() -> None:
    ratio = second_order_bytes(20000, 10) / second_order_bytes(10000, 10)

    assert ratio == pytest.approx(4.0, rel=1e-3)


def test_preflight_passes_when_the_maps_fit(monkeypatch) -> None:
    _available(monkeypatch, 8 * GIB)

    assert second_order_blocker(_pool([64, 64]), 100, 2) is None


def test_preflight_names_the_memory_a_worker_needs(monkeypatch) -> None:
    needed = second_order_bytes(300, 32)
    available = 0.9 * needed / MEMORY_FRACTION
    _available(monkeypatch, available)

    reason = second_order_blocker(_pool([64]), 300, 2)

    assert reason is not None
    assert f"{needed / GIB:.2f} GiB" in reason
    assert f"{MEMORY_FRACTION * available / GIB:.2f} GiB" in reason


def test_preflight_uses_the_batch_size_not_the_particle_count(monkeypatch) -> None:
    # More batches shrink the per-batch maps: the same worker fits with 8 batches, not with 1.
    batch_of_8 = second_order_bytes(300, 8)
    _available(monkeypatch, 1.5 * batch_of_8 / MEMORY_FRACTION)

    assert second_order_blocker(_pool([64]), 300, 8) is None
    assert second_order_blocker(_pool([64]), 300, 1) is not None


def test_second_order_maps_limit_how_many_workers_finish_at_once(monkeypatch) -> None:
    per_worker = second_order_bytes(300, 32)
    _available(monkeypatch, 2.5 * per_worker / MEMORY_FRACTION)
    pool = _pool([64] * 6)

    assert _concurrency(pool, 300, second_order_batches=2, n_hessians=1) == 2
    assert _concurrency(pool, 300, second_order_batches=None, n_hessians=1) == 6


def test_uncertainty_hessian_defaults_to_the_true_hessian_and_rejects_unknown_names() -> None:
    assert TrackingParameters().uncertainty_hessian == "true"
    assert TrackingParameters(uncertainty_hessian="gauss_newton").uncertainty_hessian == (
        "gauss_newton"
    )
    with pytest.raises(ValueError, match="uncertainty_hessian"):
        TrackingParameters(uncertainty_hessian="exact")
