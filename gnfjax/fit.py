"""Fitting model spectra to empirical harmonic / temporal power spectra.

Reproduces the objective of the original ``Full_Analysis``: an affine map from the
model's E and I spectra to the empirical one is fitted (``c = x0 P_E + x1 P_I + x2``),
and the distance is ``(1 + ||log10 c - log10 target||_2)**p`` plus the same term on
the first differences (log-slopes), with ``p = 2`` for the harmonic spectrum and
``p = 3`` for the temporal one; the total is ``dist_t**2 + dist_s**2.4``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .model import GNFParams, steady_states, harmonic_spectrum, temporal_spectrum, distance_to_instability

PENALTY = 1e9


def model_temporal_spectrum(params, eigs, ess, iss, omegas, kernel="gaussian"):
    """Whole-field temporal spectrum ``2 * sum_k S_k(omega)`` for E and I."""
    s_e, s_i = temporal_spectrum(params, eigs, ess, iss, omegas, kernel)
    return 2.0 * s_e.sum(axis=1), 2.0 * s_i.sum(axis=1)


def affine_combination(x, p_e, p_i, floor=1e-20):
    """``x0 P_E + x1 P_I + x2`` clipped away from zero (``NF_to_empirical``)."""
    return jnp.maximum(x[0] * p_e + x[1] * p_i + x[2], floor)


def log_spectral_distance(model, target, power: int):
    """``(1 + ||log10 m - log10 t||)^p + (1 + ||d log10 m - d log10 t||)^p``."""
    lm, lt = jnp.log10(model), jnp.log10(target)
    d0 = jnp.linalg.norm(lm - lt)
    d1 = jnp.linalg.norm(jnp.diff(lm) - jnp.diff(lt))
    return (1.0 + d0) ** power + (1.0 + d1) ** power


def scale_to_empirical(p_e, p_i, target, x0=(1.0, 0.0, 0.0), n_iter: int = 100, damping: float = 1e-3):
    """Fit the affine map from model (E, I) spectra to an empirical spectrum (``find_scaling``).

    Minimises ``||log10(x0 P_E + x1 P_I + x2) - log10(target)||_2`` by Levenberg-
    Marquardt in pure JAX (jit/grad safe); ``x0`` is kept non-negative as in the
    original bounds.

    Returns:
        ``(x, fitted)`` with ``x = (scale_E, scale_I, offset)``.
    """
    p_e, p_i, target = jnp.asarray(p_e), jnp.asarray(p_i), jnp.asarray(target)
    lt = jnp.log10(target)
    resid = lambda x: jnp.log10(affine_combination(x, p_e, p_i)) - lt
    jac = jax.jacfwd(resid)

    def step(carry, _):
        x, lam = carry
        r, J = resid(x), jac(x)
        H = J.T @ J
        dx = jnp.linalg.solve(H + lam * jnp.diag(jnp.diag(H) + 1e-12), J.T @ r)
        x_new = x - dx
        x_new = x_new.at[0].set(jnp.maximum(x_new[0], 0.0))
        better = jnp.sum(resid(x_new) ** 2) < jnp.sum(r**2)
        x = jnp.where(better, x_new, x)
        lam = jnp.where(better, lam * 0.3, lam * 10.0)
        return (x, lam), None

    (x, _), _ = jax.lax.scan(step, (jnp.asarray(x0, dtype=p_e.dtype), jnp.asarray(damping, p_e.dtype)), None, length=n_iter)
    return x, affine_combination(x, p_e, p_i)


def spectral_loss(params: GNFParams, eigs, omegas, target_harmonic=None, target_temporal=None,
                  first_k: int = 2, last_k=None, kernel: str = "gaussian", steady_state=None,
                  margin: float = 0.01):
    """Distance between model and empirical spectra at the best stable steady state.

    For each admissible steady state the affine scaling is fitted
    (:func:`scale_to_empirical`), the composite distance is formed and the
    minimum over states is returned. Unstable states (``distance_to_instability
    <= margin``) and parameter sets with no steady state return ``PENALTY``.

    Jit- and grad-compatible: the steady-state search and the affine scaling
    are evaluated with ``stop_gradient`` (treated as constants for the gradient,
    as in the original two-level scheme). Use it as a black-box objective for a
    global optimiser (e.g. ``scipy.optimize.differential_evolution``, as the
    original fits did) or with ``jax.grad`` for local refinement.
    """
    if steady_state is None:
        states, valid = steady_states(params)
    else:
        states, valid = jnp.asarray(steady_state)[None, :], jnp.ones((1,), dtype=bool)
    last_k = eigs.shape[0] if last_k is None else last_k

    def one_state(ss):
        ess, iss = ss[0], ss[1]
        stable = distance_to_instability(params, eigs, ess, iss, kernel) > margin
        total = 0.0
        if target_harmonic is not None:
            p_e, p_i = harmonic_spectrum(params, eigs, ess, iss, kernel)
            p_e, p_i = p_e[first_k:last_k], p_i[first_k:last_k]
            x, _ = scale_to_empirical(jax.lax.stop_gradient(p_e), jax.lax.stop_gradient(p_i), target_harmonic)
            c = affine_combination(jax.lax.stop_gradient(x), p_e, p_i)
            total = total + log_spectral_distance(c, target_harmonic, 2) ** 2.4
        if target_temporal is not None:
            t_e, t_i = model_temporal_spectrum(params, eigs, ess, iss, omegas, kernel)
            x, _ = scale_to_empirical(jax.lax.stop_gradient(t_e), jax.lax.stop_gradient(t_i), target_temporal)
            c = affine_combination(jax.lax.stop_gradient(x), t_e, t_i)
            total = total + log_spectral_distance(c, target_temporal, 3) ** 2.0
        return jnp.where(stable & jnp.isfinite(total), total, PENALTY)

    losses = jax.vmap(one_state)(states)
    return jnp.min(jnp.where(valid, losses, PENALTY))
