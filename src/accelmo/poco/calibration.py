"""Measure map for BPM gain and corrector kick calibration in closed-orbit fits.

A measured orbit change is the model's change at the kick ``nominal + (1 + g_corrector) * delta`` (the corrector calibration acts on
the kick put into the lattice), scaled per BPM and plane by ``(1 + b)`` (a readout error). The corrector kick is a parameter of the
MAD-NG damap, so ``d(orbit)/d(kick)`` is exact and the orbit's nonlinearity in the kick is kept; the BPM gains are linear scalings
and their columns are analytic. BPM gain columns are one-hot (one column per BPM and plane), so the normal matrix is block-structured and the
BPM gains are eliminated by a Schur complement (:func:`schur_step`) instead of carrying a dense matrix.

Parameter blocks: ``u = (q, g)`` (magnet knobs then corrector gains, dense) and ``b`` (BPM gains, diagonal Hessian).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg.blas import dsyrk  # ty: ignore[missing-module-member]

from accelmo.fitting.priors import ZERO_MEAN_SIGMA, inverse_variances, quadratic_prior
from accelmo.fitting.uncertainty import covariance


#: Corrector gain names: ``corrgain.<CORRECTOR>`` (upper-case, as MAD-NG names them); BPM gains are
#: named by :mod:`accelmo.fitting.bpm_gains`.
def corrector_gain_name(corrector: str) -> str:
    """Fit-parameter name of the gain of ``corrector``."""
    return f"corrgain.{corrector}"


def corrector_gain_knob_name(corrector: str) -> str:
    """Model knob (constant part 0) added to a corrector's ``1 + g``, so its Jacobian column is d(orbit)/d(g)."""
    return f"dg.{corrector}"


@dataclass(frozen=True)
class CalibrationSpec:
    """Global gain layout shared by every series: BPM names (both planes), corrector names and the kick globals' prefix."""

    bpms: tuple[str, ...]
    correctors: tuple[str, ...]
    #: A corrector's kick global is ``<kick_prefix><corrector>`` (case-insensitive), e.g. ``k_dhz2l4``.
    kick_prefix: str = "k_"

    def corrector_of(self, global_name: str) -> str | None:
        """The upper-case corrector a kick global belongs to, or ``None`` if *global_name* is not a kick."""
        if not global_name.startswith(self.kick_prefix):
            return None
        return global_name[len(self.kick_prefix) :].upper()


class CalibrationBlocks:
    """Summed loss, gradient and Gauss-Newton Hessian blocks (``2 JᵀWJ`` convention, unnormalised)."""

    ARRAYS = ("grad_u", "hess_uu", "hess_ub", "hess_bb", "grad_b")
    grad_u: np.ndarray
    hess_uu: np.ndarray
    hess_ub: np.ndarray
    hess_bb: np.ndarray
    grad_b: np.ndarray

    def __init__(self, n_q: int, n_g: int, n_b: int, buffer=None, *, zero: bool = True) -> None:
        """Blocks backed by ``buffer`` (a shared-memory buffer of :meth:`nbytes`) or by fresh arrays; ``zero=False`` keeps a buffer's contents."""
        self.n_q, self.n_g, self.n_b = n_q, n_g, n_b
        self.loss = 0.0
        shapes = self.shapes(n_q, n_g, n_b)
        if buffer is None:
            buffer = np.zeros(self.nbytes(n_q, n_g, n_b) // 8)
        flat = np.frombuffer(buffer, dtype=np.float64, count=self.nbytes(n_q, n_g, n_b) // 8)
        offset = 0
        for name, shape in shapes.items():
            size = int(np.prod(shape))
            setattr(self, name, flat[offset : offset + size].reshape(shape))
            offset += size
        if zero:
            flat[:] = 0.0

    @classmethod
    def shapes(cls, n_q: int, n_g: int, n_b: int) -> dict[str, tuple[int, ...]]:
        """Array shape of each block."""
        n_u = n_q + n_g
        return {
            "grad_u": (n_u,),
            "hess_uu": (n_u, n_u),
            "hess_ub": (n_u, n_b),
            "hess_bb": (n_b,),
            "grad_b": (n_b,),
        }

    @classmethod
    def nbytes(cls, n_q: int, n_g: int, n_b: int) -> int:
        """Size in bytes of all blocks."""
        return 8 * sum(int(np.prod(shape)) for shape in cls.shapes(n_q, n_g, n_b).values())

    def add(self, other: CalibrationBlocks) -> None:
        """Add ``other`` to these blocks."""
        self.loss += other.loss
        self.grad_u += other.grad_u
        self.hess_uu += other.hess_uu
        self.hess_ub += other.hess_ub
        self.hess_bb += other.hess_bb
        self.grad_b += other.grad_b

    def to_payload(self) -> dict:
        """The scalars; the arrays travel through shared memory (a pickle would copy ~170 MB twice)."""
        return {"n_q": self.n_q, "n_g": self.n_g, "n_b": self.n_b, "loss": self.loss}

    def symmetrise(self) -> None:
        """Mirror the upper triangle of the magnet-knob block, which :func:`add_series` fills alone, onto the lower one."""
        h = self.hess_uu[: self.n_q, : self.n_q]
        h += np.triu(h, 1).T


def add_series(
    blocks: CalibrationBlocks,
    model: np.ndarray,
    jacobian: np.ndarray,
    d_gain: np.ndarray,
    targets,
    weights,
    bpm_gain: np.ndarray,
    bpm_cols: np.ndarray,
    corrector_indices: list[int],
) -> None:
    """Add one series' contribution.

    The magnet-knob block of ``hess_uu`` gets its upper triangle only; call :meth:`CalibrationBlocks.symmetrise` once all series are in.

    ``model`` is the orbit change with every corrector kick scaled by ``1 + g`` (reference subtracted) and ``jacobian`` its
    quad derivatives; ``d_gain[..., j]`` is its derivative with respect to the gain of corrector ``corrector_indices[j]``.
    Only the BPM gains scale the model: ``(1 + b) * model`` against the target.
    """
    n_q = blocks.n_q
    g_rows = n_q + np.asarray(corrector_indices, dtype=int)
    for o in range(model.shape[0]):
        scale = 1.0 + bpm_gain[o]
        mp = scale * model[o]
        jac = jacobian[
            o
        ]  # not scaled: the BPM gain scale is folded into the weights below, so no copy of the Jacobian
        d_g = scale[:, None] * d_gain[o]
        w = np.asarray(weights[o], dtype=float)
        w_scaled = w * scale**2
        r = np.where(w > 0.0, mp - np.asarray(targets[o], dtype=float), 0.0)
        wr = w * r
        d_b = model[o]
        cols = bpm_cols[o]

        blocks.loss += float(np.sum(w * r * r))
        blocks.grad_u[:n_q] += 2.0 * (scale * wr) @ jac
        root = (
            np.sqrt(2.0 * w_scaled)[:, None] * jac
        )  # sqrt(2 w scale^2) J: its Gram matrix is 2 J^T w scale^2 J
        blocks.hess_uu[:n_q, :n_q] += dsyrk(
            1.0, root.T, trans=0, lower=1
        ).T  # one triangle only (half the flops of a product)
        del root
        cross = (jac.T * (2.0 * w_scaled)) @ d_gain[o]
        blocks.grad_u[g_rows] += 2.0 * wr @ d_g
        blocks.hess_uu[np.ix_(g_rows, g_rows)] += 2.0 * (d_g.T * w) @ d_g
        blocks.hess_uu[:n_q, g_rows] += cross
        blocks.hess_uu[g_rows, :n_q] += cross.T
        blocks.grad_b[cols] += 2.0 * wr * d_b
        blocks.hess_bb[cols] += 2.0 * w * d_b * d_b
        blocks.hess_ub[:n_q, cols] += jac.T * (2.0 * scale * w * d_b)
        blocks.hess_ub[np.ix_(g_rows, cols)] += 2.0 * (d_g * (w * d_b)[:, None]).T


def apply_prior(
    blocks: CalibrationBlocks,
    u: np.ndarray,
    b: np.ndarray,
    sigma_g: float,
    sigma_b: float,
    sigma_q: np.ndarray | None = None,
    *,
    zero_mean: np.ndarray | None = None,
    loss_only: bool = False,
):
    """Gaussian priors ``g²/σ_g²``, ``b²/σ_b²`` and ``q²/σ_q²`` added to the chi² blocks.

    ``sigma_q`` holds one absolute width per magnet knob (the knob's units); an entry of 0 leaves that knob free.
    ``loss_only`` adds the priors to ``blocks.loss`` alone (``blocks`` then needs only ``loss`` and ``n_q``): a loss-only trial
    evaluation has no gradient or Hessian blocks. ``zero_mean`` (indices into ``g``) holds the mean of those gains at 0
    with width :data:`ZERO_MEAN_SIGMA`.
    """
    n_q = blocks.n_q
    ridge = np.zeros(len(u))
    if sigma_q is not None:
        ridge[:n_q] = inverse_variances(sigma_q)
    if sigma_g > 0:
        ridge[n_q:] = 1.0 / sigma_g**2
    groups = (
        [n_q + np.asarray(zero_mean, dtype=int)] if zero_mean is not None and len(zero_mean) else []
    )
    prior = quadratic_prior(ridge, groups, 1.0 / ZERO_MEAN_SIGMA**2)
    blocks.loss += float(u @ prior @ u)
    if not loss_only:
        blocks.grad_u += 2.0 * prior @ u
        blocks.hess_uu += 2.0 * prior
    if sigma_b > 0:
        blocks.loss += float(b @ b) / sigma_b**2
        if not loss_only:
            blocks.grad_b += 2.0 * b / sigma_b**2
            blocks.hess_bb += 2.0 / sigma_b**2


def reduce_blocks(blocks: CalibrationBlocks) -> tuple[np.ndarray, np.ndarray]:
    """Schur complement over ``b``: reduced ``(gradient, Hessian)`` of ``u``."""
    inv = 1.0 / blocks.hess_bb
    ub = blocks.hess_ub
    scaled = ub * np.sqrt(inv)
    hess = blocks.hess_uu - scaled @ scaled.T
    grad = blocks.grad_u - ub @ (inv * blocks.grad_b)
    return grad, hess


def gain_covariance(blocks: CalibrationBlocks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical normal matrix of ``u``, and the 1-sigma of ``u`` and of ``b``, from blocks that include the priors.

    The blocks hold the chi² Hessian, so ``hess / 2`` is the inverse-variance normal matrix and its inverse the posterior covariance (the priors are part of the fit). ``b`` is marginalised over ``u``:
    with the diagonal ``H_bb``, ``Cov_bb = H_bb⁻¹ + H_bb⁻¹ H_ubᵀ Cov_u H_ub H_bb⁻¹`` and only its diagonal is formed.
    """
    scale = 0.5
    normal_u = reduce_blocks(blocks)[1] * scale
    cov_u = covariance(normal_u)
    inv_bb = 1.0 / (blocks.hess_bb * scale)
    coupling = blocks.hess_ub * scale * inv_bb
    var_b = inv_bb + np.einsum("ij,ij->j", coupling, cov_u @ coupling)
    return normal_u, np.sqrt(np.clip(np.diag(cov_u), 0.0, None)), np.sqrt(np.clip(var_b, 0.0, None))


def back_substitute(blocks: CalibrationBlocks, delta_u: np.ndarray) -> np.ndarray:
    """Step of ``b`` that is optimal for the step ``delta_u`` (conditional Gauss-Newton minimiser)."""
    return -(blocks.grad_b + blocks.hess_ub.T @ delta_u) / blocks.hess_bb
