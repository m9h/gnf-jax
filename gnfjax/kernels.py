"""Graph kernels :math:`K(\\lambda, t)` applied to Laplacian eigenvalues.

Conventions follow the original code: the graph Laplacian is
:math:`\\mathcal{L} = -(D - W)` so its eigenvalues are non-positive, the
constant mode has :math:`\\lambda_0 = 0` and every kernel is 1 there.
"""
from __future__ import annotations

import jax.numpy as jnp

KERNELS = ("gaussian", "exponential", "mexican_hat", "damped_wave")


def diffusion_time(sigma, diffusion=1.0):
    """Kernel time from a spatial scale: ``t = sigma**2 / (2 D)``."""
    return 0.5 * sigma**2 / diffusion


def graph_kernel(eigs, t, kind: str = "gaussian", a=1.0, b=1.0, c=0.0):
    """Evaluate a graph kernel on Laplacian eigenvalues.

    Args:
        eigs: (N,) non-positive Laplacian eigenvalues.
        t: kernel parameter (diffusion time, or propagation time for the damped wave).
        kind: one of ``gaussian`` (heat kernel ``exp(t lambda)``), ``exponential``
            (``t / (t - lambda)``), ``mexican_hat`` (``-lambda exp(t lambda)``) or
            ``damped_wave`` (Green's function of ``a u'' + b u' = (L - c) u``).
        a, b, c: damped-wave coefficients (ignored otherwise).

    Returns:
        (N,) kernel values.
    """
    eigs = jnp.asarray(eigs)
    if kind == "gaussian":
        return jnp.exp(t * eigs)
    if kind == "exponential":
        return t / (t - eigs)
    if kind == "mexican_hat":
        return -eigs * jnp.exp(t * eigs)
    if kind == "damped_wave":
        return damped_wave_kernel(eigs, t, a, b, c)[0]
    raise ValueError(f"unknown kernel {kind!r}; choose from {KERNELS}")


def damped_wave_kernel(eigs, t, a=1.0, b=1.0, c=0.0):
    """Damped-wave propagator and its time derivative on the eigenvalues.

    Roots of ``a r**2 + b r - (lambda - c) = 0`` may be complex; the kernel is
    real for real ``t`` and is returned as such.
    """
    eigs = jnp.asarray(eigs, dtype=jnp.result_type(eigs, 1.0))
    delta = (b**2 + 4.0 * a * (eigs - c)).astype(jnp.complex128 if eigs.dtype == jnp.float64 else jnp.complex64)
    root = jnp.sqrt(delta)
    r1 = (-b + root) / (2.0 * a)
    r2 = (-b - root) / (2.0 * a)
    rdiff = r1 - r2
    k = (r1 * jnp.exp(r2 * t) - r2 * jnp.exp(r1 * t)) / rdiff
    k_prime = (jnp.exp(r1 * t) - jnp.exp(r2 * t)) / rdiff
    return k.real, k_prime.real
