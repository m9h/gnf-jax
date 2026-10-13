"""Euler-Maruyama simulation of the graph Wilson-Cowan field (``lax.scan``)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import expit

from .model import GNFParams, linearise, _kernels


def simulate_linearised(params: GNFParams, eigs, ess, iss, key, dt: float, n_steps: int,
                        downsample: int = 1, kernel: str = "gaussian", beta0=None):
    """Linearised dynamics in the Laplacian eigenbasis.

    Integrates ``d beta_k = -A_k beta_k dt + sqrt(D) dW`` for every mode with the
    Euler-Maruyama scheme (noise enters as ``noise/tau * sqrt(dt)``, as in the
    original ``Linearized_GLDomain_Wilson_Cowan_Model``).

    Returns:
        ``(beta_e, beta_i)`` each (n_steps // downsample, N): fluctuation
        coefficients sampled every ``downsample`` steps (plain decimation).
    """
    A, D = linearise(params, eigs, ess, iss, kernel)
    n = eigs.shape[0]
    sd = jnp.sqrt(jnp.diag(D)) * jnp.sqrt(dt)  # (2,)
    x0 = jnp.zeros((n, 2)) if beta0 is None else jnp.asarray(beta0)

    def block(x, k):
        def step(x, k):
            noise = jax.random.normal(k, (n, 2)) * sd
            x = x - dt * jnp.einsum("kij,kj->ki", A, x) + noise
            return x, None

        keys = jax.random.split(k, downsample)
        x, _ = jax.lax.scan(step, x, keys)
        return x, x

    keys = jax.random.split(key, n_steps // downsample)
    _, traj = jax.lax.scan(block, x0, keys)
    return traj[:, :, 0], traj[:, :, 1]


def simulate_nonlinear(params: GNFParams, eigs, U, ess, iss, key, dt: float, n_steps: int,
                       downsample: int = 1, kernel: str = "gaussian", e0=None, i0=None):
    """Full nonlinear graph Wilson-Cowan field in node space.

    The four kernel operators are applied spectrally (``U diag(K) U^T``), as in
    ``graph_WCM_propagators``. Dense ``U`` is fine up to a few thousand nodes;
    for surface-resolution graphs use the linearised simulator.

    Returns:
        ``(E, I)`` each (n_steps // downsample, N) activities.
    """
    k_ee, k_ie, k_ei, k_ii = _kernels(params, eigs, kernel)
    n = eigs.shape[0]
    Ut = U.T
    apply = lambda k, x: U @ (k * (Ut @ x))
    e = jnp.full(n, ess) if e0 is None else jnp.asarray(e0)
    i = jnp.full(n, iss) if i0 is None else jnp.asarray(i0)
    te, ti = params.tau_e, params.tau_i
    sd_e = params.noise_e * jnp.sqrt(dt) / te
    sd_i = params.noise_i * jnp.sqrt(dt) / ti

    def block(state, k):
        def step(state, k):
            e, i = state
            ke, ki = jax.random.split(k)
            de = -params.d_e * e + expit(params.alpha_ee * apply(k_ee, e) - params.alpha_ie * apply(k_ie, i) + params.p)
            di = -params.d_i * i + expit(params.alpha_ei * apply(k_ei, e) - params.alpha_ii * apply(k_ii, i) + params.q)
            e = e + dt / te * de + sd_e * jax.random.normal(ke, (n,))
            i = i + dt / ti * di + sd_i * jax.random.normal(ki, (n,))
            return (e, i), None

        keys = jax.random.split(k, downsample)
        state, _ = jax.lax.scan(step, state, keys)
        return state, state

    keys = jax.random.split(key, n_steps // downsample)
    _, (E, I) = jax.lax.scan(block, (e, i), keys)
    return E, I
