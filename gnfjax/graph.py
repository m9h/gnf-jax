"""Laplacian eigensystems and the CHAOSS functional-connectivity prediction."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def laplacian_eigensystem(weights):
    """Eigen-decompose the (negative) combinatorial Laplacian of a weighted graph.

    Args:
        weights: (N, N) symmetric non-negative adjacency / weight matrix.

    Returns:
        ``(eigs, U)``: eigenvalues of ``-(D - W)`` sorted descending so that
        ``eigs[0] == 0`` is the constant mode and the rest are negative; ``U``
        holds the matching orthonormal eigenvectors as columns.

    The sign convention matches the original code (``one_dim_Laplacian_eigenvalues``):
    every kernel in :mod:`gnfjax.kernels` expects non-positive eigenvalues.
    """
    W = np.asarray(weights, dtype=np.float64)
    L = -(np.diag(W.sum(axis=1)) - W)
    s, U = np.linalg.eigh(L)
    s = s[::-1].copy()
    U = U[:, ::-1].copy()
    s[0] = 0.0  # numerically the constant mode is exactly zero
    return jnp.asarray(s), jnp.asarray(U)


def grid_laplacian_eigensystem(n: int, h: float = 1.0):
    """1-D chain graph with spacing ``h`` (edge weight ``1/h**2``), as in the original tests."""
    W = np.zeros((n, n))
    idx = np.arange(n - 1)
    W[idx, idx + 1] = W[idx + 1, idx] = 1.0 / h**2
    return laplacian_eigensystem(W)


def functional_connectivity(U, mode_power):
    """CHAOSS prediction of FC from the per-mode power (harmonic spectrum).

    Covariance ``U diag(P) U^T`` normalised to a correlation matrix.
    """
    cov = (U * mode_power[None, :]) @ U.T
    inv_sd = 1.0 / jnp.sqrt(jnp.diag(cov))
    return cov * inv_sd[:, None] * inv_sd[None, :]
