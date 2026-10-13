"""Stochastic Wilson-Cowan graph neural field: steady states, linearisation, spectra.

Model (Aqil et al. 2021, PLoS Comput Biol; Aqil 2026, Netw Neurosci)::

    tau_e dE/dt = -d_e E + S(alpha_ee K_ee * E - alpha_ie K_ie * I + P) + noise_e dW
    tau_i dI/dt = -d_i I + S(alpha_ei K_ei * E - alpha_ii K_ii * I + Q) + noise_i dW

where ``S`` is the logistic function and ``K_xy *`` is a graph kernel acting
through the Laplacian eigenbasis. In that basis every eigenmode ``k`` is an
independent 2-state linear system around the homogeneous steady state, so the
harmonic (per-mode variance) and temporal power spectra are closed form.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import expit

from .kernels import graph_kernel, diffusion_time


class GNFParams(NamedTuple):
    """Parameters of the graph Wilson-Cowan model (all scalars, jit/grad-friendly)."""

    alpha_ee: float = 1.0
    alpha_ie: float = 1.0
    alpha_ei: float = 1.0
    alpha_ii: float = 1.0
    d_e: float = 1.0
    d_i: float = 1.0
    p: float = 0.0
    q: float = 0.0
    sigma_ee: float = 10.0
    sigma_ie: float = 10.0
    sigma_ei: float = 10.0
    sigma_ii: float = 10.0
    tau_e: float = 1.0
    tau_i: float = 1.0
    noise_e: float = 1.0
    noise_i: float = 1.0
    diffusion: float = 1.0
    # damped-wave kernel coefficients (ignored by the other kernels)
    a_dw: float = 1.0
    b_dw: float = 1.0
    c_dw: float = 0.0


# ---------------------------------------------------------------------------
# Homogeneous steady state
# ---------------------------------------------------------------------------

def steady_state_residual(params: GNFParams, x):
    """``-d x + S(alpha x + input)`` for the spatially homogeneous state ``x = (E, I)``."""
    e, i = x[0], x[1]
    re = -params.d_e * e + expit(params.alpha_ee * e - params.alpha_ie * i + params.p)
    ri = -params.d_i * i + expit(params.alpha_ei * e - params.alpha_ii * i + params.q)
    return jnp.stack([re, ri])


def _newton(params, x0, n_iter=60):
    f = lambda x: steady_state_residual(params, x)
    jac = jax.jacfwd(f)

    def step(x, _):
        J = jac(x)
        dx = jnp.linalg.solve(J + 1e-12 * jnp.eye(2), f(x))
        x_new = x - dx
        # keep the iterate inside the admissible box to avoid escaping to infinity
        x_new = jnp.clip(x_new, -0.5, 1.5)
        return x_new, None

    x, _ = jax.lax.scan(step, x0, None, length=n_iter)
    return x


def steady_states(params: GNFParams, n_guesses: int = 30, key=None, tol: float = 1e-9, dedupe_tol: float = 1e-6):
    """Find all homogeneous steady states by multi-start Newton iteration.

    Mirrors ``H_Simple_Steady_State``: initial guesses are the origin,
    ``(1/(2 d_e), 1/(2 d_i))`` and uniform random points in the unit square;
    solutions are accepted when ``0.001 < E, I < 0.999`` and the residual is below
    ``tol``, then de-duplicated.

    Returns:
        ``(states, valid)``: ``states`` is (n_guesses, 2); ``valid`` marks the
        unique accepted solutions (duplicates are marked False).
    """
    key = jax.random.PRNGKey(0) if key is None else key
    x0 = jax.random.uniform(key, (n_guesses, 2))
    x0 = x0.at[0].set(jnp.zeros(2))
    x0 = x0.at[1].set(jnp.array([0.5 / params.d_e, 0.5 / params.d_i]))
    sols = jax.vmap(lambda x: _newton(params, x))(x0)
    resid = jax.vmap(lambda x: jnp.abs(steady_state_residual(params, x)).max())(sols)
    ok = (resid < tol) & jnp.all(sols > 0.001, axis=1) & jnp.all(sols < 0.999, axis=1)
    # de-duplicate: keep the first occurrence of each solution (pure JAX, jit/grad safe)
    dist = jnp.linalg.norm(sols[:, None, :] - sols[None, :, :], axis=-1)
    earlier = jnp.tril(jnp.ones((n_guesses, n_guesses), dtype=bool), k=-1)  # i < j
    dup = jnp.any(earlier & ok[:, None] & (dist < dedupe_tol), axis=0)
    return sols, ok & ~dup


# ---------------------------------------------------------------------------
# Linearisation about a steady state, per eigenmode
# ---------------------------------------------------------------------------

def _kernels(params: GNFParams, eigs, kernel: str):
    kw = dict(a=params.a_dw, b=params.b_dw, c=params.c_dw)
    D = params.diffusion
    k_ee = graph_kernel(eigs, diffusion_time(params.sigma_ee, D), kernel, **kw)
    k_ie = graph_kernel(eigs, diffusion_time(params.sigma_ie, D), kernel, **kw)
    k_ei = graph_kernel(eigs, diffusion_time(params.sigma_ei, D), kernel, **kw)
    k_ii = graph_kernel(eigs, diffusion_time(params.sigma_ii, D), kernel, **kw)
    return k_ee, k_ie, k_ei, k_ii


def linearise(params: GNFParams, eigs, ess, iss, kernel: str = "gaussian"):
    """Per-mode drift ``A_k`` and noise intensity ``D`` of ``d beta = -A beta dt + sqrt(D) dW``.

    ``A_k = -J_k`` where ``J_k`` is the Jacobian of the Wilson-Cowan field for
    eigenmode ``k``; the gain factors are ``a = d_e E*(1 - d_e E*)`` and
    ``b = d_i I*(1 - d_i I*)`` (derivative of the logistic at the steady state).

    Returns:
        ``A`` (N, 2, 2) and ``D`` (2, 2) with ``D = diag((noise_e/tau_e)**2, (noise_i/tau_i)**2)``.
    """
    k_ee, k_ie, k_ei, k_ii = _kernels(params, eigs, kernel)
    a = params.d_e * ess * (1.0 - params.d_e * ess)
    b = params.d_i * iss * (1.0 - params.d_i * iss)
    te, ti = params.tau_e, params.tau_i
    a00 = (params.d_e - a * params.alpha_ee * k_ee) / te
    a01 = (a * params.alpha_ie * k_ie) / te
    a10 = (-b * params.alpha_ei * k_ei) / ti
    a11 = (params.d_i + b * params.alpha_ii * k_ii) / ti
    A = jnp.stack([jnp.stack([a00, a01], -1), jnp.stack([a10, a11], -1)], -2)
    D = jnp.diag(jnp.array([(params.noise_e / te) ** 2, (params.noise_i / ti) ** 2]))
    return A, D


def jacobian_eigenvalues(params: GNFParams, eigs, ess, iss, kernel: str = "gaussian"):
    """Eigenvalues of each mode's 2x2 Jacobian, shape (N, 2), complex."""
    A, _ = linearise(params, eigs, ess, iss, kernel)
    J = -A
    tr = J[:, 0, 0] + J[:, 1, 1]
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    disc = jnp.sqrt((tr**2 - 4.0 * det).astype(jnp.result_type(tr, 1j)))
    return jnp.stack([(tr + disc) / 2.0, (tr - disc) / 2.0], -1)


def distance_to_instability(params: GNFParams, eigs, ess, iss, kernel: str = "gaussian"):
    """``-max Re(lambda)`` over all modes: positive for a stable steady state."""
    return -jnp.max(jacobian_eigenvalues(params, eigs, ess, iss, kernel).real)


def classify_steady_state(params: GNFParams, eigs, ess, iss, kernel: str = "gaussian", margin: float = 0.01):
    """Stability class as in ``GraphWC_Jacobian_TrDet``.

    0 unstable (some ``Re lambda >= -margin``), 1 strictly stable and non-oscillatory,
    2 stable with (more than ten) oscillatory modes, 4 stable, few oscillatory modes.
    """
    lam = jacobian_eigenvalues(params, eigs, ess, iss, kernel)
    unstable = jnp.any(lam.real >= -margin) | jnp.any(~jnp.isfinite(lam))
    n_osc = jnp.sum(lam.imag != 0)
    cls = jnp.where(n_osc == 0, 1, jnp.where(n_osc > 10, 2, 4))
    return jnp.where(unstable, 0, cls)


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def mode_covariance(A, D):
    """Stationary covariance of ``d x = -A x dt + sqrt(D) dW`` for a batch of 2x2 ``A``.

    Closed-form Lyapunov solution ``A S + S A^T = D``:
    ``S = (det(A) D + (A - tr(A) I) D (A - tr(A) I)^T) / (2 tr(A) det(A))``.
    """
    tr = A[..., 0, 0] + A[..., 1, 1]
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    B = A - tr[..., None, None] * jnp.eye(2)
    S = det[..., None, None] * D + B @ D @ jnp.swapaxes(B, -1, -2)
    return S / (2.0 * tr * det)[..., None, None]


def harmonic_spectrum(params: GNFParams, eigs, ess, iss, kernel: str = "gaussian"):
    """Per-mode stationary variance of the E and I fluctuations (the harmonic power spectrum).

    Returns:
        ``(P_E, P_I)``, each (N,).
    """
    A, D = linearise(params, eigs, ess, iss, kernel)
    S = mode_covariance(A, D)
    return S[:, 0, 0], S[:, 1, 1]


def temporal_spectrum(params: GNFParams, eigs, ess, iss, omegas, kernel: str = "gaussian"):
    """Two-sided power spectral density of each mode at angular frequencies ``omegas``.

    ``S_E(omega) = (D_ee (A_11^2 + omega^2) + D_ii A_01^2) / |det(A + i omega)|^2`` and
    the mirror expression for I. Shape (n_omega, N). The temporal power spectrum
    of the whole field is ``2 * sum_k S_k(omega)`` (see :func:`gnfjax.fit.model_temporal_spectrum`).
    """
    A, D = linearise(params, eigs, ess, iss, kernel)
    omegas = jnp.asarray(omegas)[:, None]
    a00, a01, a10, a11 = A[:, 0, 0][None], A[:, 0, 1][None], A[:, 1, 0][None], A[:, 1, 1][None]
    dee, dii = D[0, 0], D[1, 1]
    denom = (a00 * a11 - a01 * a10 - omegas**2) ** 2 + ((a00 + a11) * omegas) ** 2
    s_e = (dee * (a11**2 + omegas**2) + dii * a01**2) / denom
    s_i = (dii * (a00**2 + omegas**2) + dee * a10**2) / denom
    return s_e, s_i
