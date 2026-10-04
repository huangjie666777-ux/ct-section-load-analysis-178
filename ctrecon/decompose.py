"""Dual-energy material decomposition via exact 2x2 non-negative least squares.

The linear two-material model assumes the reconstructed attenuation at
energy e is mu_e = sum_m A[e, m] * rho_m, where A is the mass attenuation
coefficient matrix (mm^2/mg) and rho_m the partial density of material m
(mg/mm^3).
"""

from __future__ import annotations

import numpy as np

from .io_utils import ValidationError

MAX_CONDITION = 10000.0


def validate_materials(materials: object) -> tuple[str, str]:
    """Validate the two unique, non-empty material names."""
    if (
        not isinstance(materials, (list, tuple))
        or len(materials) != 2
        or any(not isinstance(m, str) or not m.strip() for m in materials)
    ):
        raise ValidationError("materials must be a list of two non-empty names")
    names = (materials[0].strip(), materials[1].strip())
    if names[0] == names[1]:
        raise ValidationError("material names must be unique")
    return names


def validate_mu_matrix(matrix: object) -> np.ndarray:
    """Validate the 2x2 positive finite mass attenuation matrix (mm^2/mg).

    Rows are the low/high energies, columns the two materials. A 2-norm
    condition number above MAX_CONDITION is rejected outright instead of
    being papered over with regularization.
    """
    try:
        arr = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"mu_matrix must be a 2x2 numeric matrix: {exc}") from exc
    if arr.shape != (2, 2):
        raise ValidationError("mu_matrix must be a 2x2 matrix")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValidationError("mu_matrix entries must be finite and strictly positive")
    cond = float(np.linalg.cond(arr, 2))
    if not np.isfinite(cond) or cond > MAX_CONDITION:
        raise ValidationError(
            f"mu_matrix 2-norm condition number {cond:.6g} exceeds limit {MAX_CONDITION:g}"
        )
    return arr


def nnls_2x2(
    matrix: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel exact NNLS: minimize ||A rho - b||^2 with rho >= 0.

    b stacks the two observed attenuation maps (mm^-1). Negative components
    of the unconstrained solution are *not* simply truncated: the exact
    constrained minimizer is chosen among the interior solution and the two
    axis-boundary (single-material) solutions by comparing squared
    residuals.
    """
    if low.shape != high.shape:
        raise ValidationError("low and high energy reconstructions must have the same shape")
    a = matrix
    obs = np.stack([low.ravel(), high.ravel()])  # (2, n_pixels)

    # Unconstrained least squares (A is square and well conditioned).
    unconstrained = np.linalg.solve(a, obs)

    # Boundary candidates: best non-negative fit with one material absent.
    col0, col1 = a[:, 0:1], a[:, 1:2]
    only0 = np.maximum((col0 * obs).sum(axis=0) / float(col0.T @ col0), 0.0)
    only1 = np.maximum((col1 * obs).sum(axis=0) / float(col1.T @ col1), 0.0)

    candidates = np.stack(
        [
            np.where((unconstrained >= 0.0).all(axis=0), unconstrained, np.inf),
            np.stack([only0, np.zeros_like(only0)]),
            np.stack([np.zeros_like(only1), only1]),
        ]
    )  # (3, 2, n_pixels)
    residuals = np.einsum("ij,cjn->cin", a, candidates) - obs[np.newaxis]
    cost = np.einsum("cin,cin->cn", residuals, residuals)
    n_pixels = obs.shape[1]
    pick = np.argmin(cost, axis=0)  # (n_pixels,)
    chosen = candidates[pick[:, None], np.arange(2)[None, :], np.arange(n_pixels)[:, None]]
    best = chosen.T  # (2, n_pixels)

    shape = low.shape
    return best[0].reshape(shape), best[1].reshape(shape)


def decompose(
    matrix: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (rho_1, rho_2, residual_low, residual_high).

    Densities are in mg/mm^3; residuals are A @ rho - observed per energy,
    in mm^-1, and keep their sign.
    """
    rho1, rho2 = nnls_2x2(matrix, low, high)
    pred_low = matrix[0, 0] * rho1 + matrix[0, 1] * rho2
    pred_high = matrix[1, 0] * rho1 + matrix[1, 1] * rho2
    return rho1, rho2, pred_low - low, pred_high - high
