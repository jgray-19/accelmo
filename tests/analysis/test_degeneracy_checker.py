"""Tests for the degeneracy checker's pure-math core.

Each test constructs a normal matrix ``A = JᵀWJ`` with a known degeneracy
structure and asserts that :func:`analyse_degeneracy` recovers it: rank,
number of unconstrained directions, which knobs collide, and that unit-scale
disparity alone is not mistaken for degeneracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from accelmo.analysis import analyse_degeneracy


def _normal_matrix_from_jacobian(jacobian: np.ndarray) -> np.ndarray:
    """Return ``JᵀJ`` (unit weights) for a residual Jacobian ``J``."""
    return jacobian.T @ jacobian


def test_full_rank_problem_is_well_posed() -> None:
    rng = np.random.default_rng(0)
    jac = rng.standard_normal((50, 4))
    matrix = _normal_matrix_from_jacobian(jac)

    report = analyse_degeneracy(matrix, ["a", "b", "c", "d"])

    assert not report.is_degenerate
    assert report.numerical_rank == 4
    assert report.n_degenerate == 0
    assert report.degenerate_directions == []
    assert np.isfinite(report.condition_number)


def test_two_collinear_knobs_give_one_null_direction() -> None:
    # Knob "c" is an exact copy of knob "a": only the sum a+c is constrained,
    # so a-c is a flat direction.
    rng = np.random.default_rng(1)
    base = rng.standard_normal((60, 3))
    jac = np.column_stack([base[:, 0], base[:, 1], base[:, 0]])  # col 2 == col 0
    matrix = _normal_matrix_from_jacobian(jac)

    report = analyse_degeneracy(matrix, ["a", "b", "c"])

    assert report.is_degenerate
    assert report.n_degenerate == 1
    assert report.numerical_rank == 2
    assert not np.isfinite(report.condition_number)

    direction = report.degenerate_directions[0]
    contributing = {name for name, _ in direction.contributions}
    assert contributing == {"a", "c"}
    # a and c trade off with opposite sign and equal magnitude, b is absent.
    coeffs = dict(direction.contributions)
    assert coeffs["a"] * coeffs["c"] < 0
    assert coeffs["a"] == pytest.approx(-coeffs["c"], abs=1e-6)
    assert "b" not in coeffs


def test_zero_sensitivity_knob_is_flagged() -> None:
    # Knob "dead" has an all-zero Jacobian column: the data cannot see it at all.
    rng = np.random.default_rng(2)
    jac = np.column_stack([rng.standard_normal(40), np.zeros(40), rng.standard_normal(40)])
    matrix = _normal_matrix_from_jacobian(jac)

    report = analyse_degeneracy(matrix, ["live1", "dead", "live2"])

    assert report.zero_sensitivity_knobs == ["dead"]
    assert report.n_degenerate == 1
    # The dead knob's own direction is unconstrained.
    direction = report.degenerate_directions[0]
    assert direction.contributions[0][0] == "dead"
    # Its physical uncertainty is finite (floored), not NaN/inf.
    assert np.isfinite(report.uncertainties["dead"])


def test_unit_scale_disparity_is_not_degeneracy() -> None:
    # Two independent, well-constrained knobs whose sensitivities differ by ~1e3.
    # Raw analysis reports a huge condition number; scaled analysis sees full rank.
    rng = np.random.default_rng(3)
    jac = np.column_stack([rng.standard_normal(40) * 1.0, rng.standard_normal(40) * 1e3])
    matrix = _normal_matrix_from_jacobian(jac)

    scaled = analyse_degeneracy(matrix, ["small", "big"], scale=True)
    raw = analyse_degeneracy(matrix, ["small", "big"], scale=False)

    assert not scaled.is_degenerate
    assert scaled.numerical_rank == 2
    # Raw condition number is dominated by the ~1e6 scale ratio (1e3 squared)...
    assert raw.condition_number > 1e5
    # ...while the scaled one is modest (genuine geometry, not units).
    assert scaled.condition_number < 1e3


def test_scaled_eigenvalues_are_order_unity_for_constrained_knobs() -> None:
    rng = np.random.default_rng(4)
    jac = rng.standard_normal((100, 5))
    matrix = _normal_matrix_from_jacobian(jac)

    report = analyse_degeneracy(matrix, list("abcde"), scale=True)

    # Scaled normal matrix has unit diagonal -> trace == n, eigenvalues sum to n.
    assert report.eigenvalues.sum() == pytest.approx(5.0, rel=1e-9)
    assert report.eigenvalues[0] > report.eigenvalues[-1]  # descending


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="square"):
        analyse_degeneracy(np.zeros((2, 3)), ["a", "b"])
    with pytest.raises(ValueError, match="knob_names"):
        analyse_degeneracy(np.eye(3), ["a", "b"])
    with pytest.raises(ValueError, match="non-finite"):
        analyse_degeneracy(np.array([[1.0, np.nan], [np.nan, 1.0]]), ["a", "b"])


def test_all_zero_matrix_is_fully_degenerate() -> None:
    report = analyse_degeneracy(np.zeros((3, 3)), ["a", "b", "c"])
    assert report.numerical_rank == 0
    assert report.n_degenerate == 3
    assert set(report.zero_sensitivity_knobs) == {"a", "b", "c"}
    assert not np.isfinite(report.condition_number)


def test_single_knob() -> None:
    constrained = analyse_degeneracy(np.array([[4.0]]), ["solo"])
    assert not constrained.is_degenerate
    assert constrained.numerical_rank == 1
    assert constrained.condition_number == pytest.approx(1.0)

    dead = analyse_degeneracy(np.array([[0.0]]), ["solo"])
    assert dead.is_degenerate
    assert dead.zero_sensitivity_knobs == ["solo"]


def test_summary_mentions_degenerate_verdict() -> None:
    jac = np.column_stack([np.ones(10), np.ones(10)])  # identical columns
    matrix = _normal_matrix_from_jacobian(jac)
    report = analyse_degeneracy(matrix, ["a", "b"])
    text = report.summary()
    assert "DEGENERATE" in text
    assert "unconstrained" in text


def test_unscaled_directions_are_null_vectors_of_the_jacobian() -> None:
    rng = np.random.default_rng(5)
    base = rng.standard_normal((40, 3))
    jac = np.column_stack([base[:, 0], base[:, 1], 2.0 * base[:, 0] + 3.0 * base[:, 2], base[:, 2]])
    report = analyse_degeneracy(jac.T @ jac, list("abcd"), scale=False)

    assert report.n_degenerate == 1
    (direction,) = report.degenerate_directions
    assert np.linalg.norm(jac @ direction.components) < 1e-6 * np.linalg.norm(jac)


def test_scaled_directions_are_null_vectors_in_unit_scaled_knobs() -> None:
    rng = np.random.default_rng(6)
    base = rng.standard_normal((40, 2))
    units = np.array([1e-6, 1.0, 1e3])  # knob units differ by nine decades
    jac = np.column_stack([base[:, 0], base[:, 1], base[:, 0] + base[:, 1]]) * units
    matrix = jac.T @ jac
    report = analyse_degeneracy(matrix, list("abc"))

    assert report.n_degenerate == 1
    (direction,) = report.degenerate_directions
    knob_direction = direction.components / np.sqrt(np.diag(matrix))
    residual = np.linalg.norm(jac @ knob_direction)
    assert residual < 1e-9 * np.linalg.norm(jac) * np.linalg.norm(knob_direction)


def test_worst_constrained_knobs_are_ordered_by_uncertainty() -> None:
    report = analyse_degeneracy(np.diag([4.0, 1.0, 16.0]), ["a", "b", "c"])
    worst = report.worst_constrained_knobs(2)

    assert [name for name, _ in worst] == ["b", "a"]
    assert worst[0][1] / worst[1][1] == pytest.approx(2.0, rel=1e-9)  # sigma ~ 1/sqrt(A)


def test_summary_truncates_long_lists() -> None:
    n = 6
    report = analyse_degeneracy(np.zeros((n, n)), [f"k{i}" for i in range(n)])
    text = report.summary(max_directions=2, max_knobs=3)

    assert "(+3 more)" in text
    assert "(+4 more directions)" in text
    assert "k0, k1, k2" in text


def test_a_well_posed_summary_reports_the_rank() -> None:
    rng = np.random.default_rng(7)
    jac = rng.standard_normal((30, 3))
    text = analyse_degeneracy(jac.T @ jac, list("abc")).summary()

    assert "well-posed" in text
    assert "3/3" in text
    assert "zero-sensitivity" not in text
