"""Detrended fluctuation analysis (Peng et al. 1994) for long-range temporal correlations."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


def _fluctuation(profile, n: int, order: int):
    """RMS residual of an order-``order`` polynomial detrend over non-overlapping windows of size ``n``."""
    m = profile.shape[-1] // n
    seg = profile[..., : m * n].reshape(profile.shape[:-1] + (m, n))
    t = jnp.arange(n, dtype=profile.dtype)
    V = jnp.stack([t**k for k in range(order + 1)], -1)  # (n, order+1)
    Q, R = jnp.linalg.qr(V)
    proj = Q @ Q.T
    resid = seg - seg @ proj.T
    return jnp.sqrt(jnp.mean(resid**2, axis=(-1, -2)))


def dfa_exponent(x, window_sizes, order: int = 1, return_fluctuations: bool = False):
    """DFA scaling exponent of ``x`` (last axis is time; leading axes are batched).

    Args:
        x: (..., T) signal(s).
        window_sizes: iterable of integer window lengths (static).
        order: detrending polynomial order (1 = DFA-1).

    Returns:
        The slope of ``log F(n)`` against ``log n`` (0.5 white noise, 1.0 pink
        noise, 1.5 Brownian). With ``return_fluctuations`` also returns ``F``.
    """
    x = jnp.asarray(x)
    profile = jnp.cumsum(x - x.mean(axis=-1, keepdims=True), axis=-1)
    sizes = [int(n) for n in window_sizes]
    F = jnp.stack([_fluctuation(profile, n, order) for n in sizes], -1)
    logn = jnp.log(jnp.asarray(sizes, dtype=F.dtype))
    logF = jnp.log(F)
    ln = logn - logn.mean()
    slope = jnp.sum(ln * (logF - logF.mean(axis=-1, keepdims=True)), axis=-1) / jnp.sum(ln**2)
    return (slope, F) if return_fluctuations else slope
