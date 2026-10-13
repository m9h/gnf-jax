import numpy as np
import jax.numpy as jnp

from gnfjax.kernels import graph_kernel, diffusion_time


def test_diffusion_time_matches_original_convention():
    # t = sigma^2 / (2 D)
    assert float(diffusion_time(2.0, 1.0)) == 2.0
    assert float(diffusion_time(10.0, 4.0)) == 12.5


def test_gaussian_kernel_is_unity_on_constant_mode_and_decays():
    eigs = jnp.array([0.0, -0.5, -2.0])
    k = graph_kernel(eigs, 3.0, "gaussian")
    np.testing.assert_allclose(k, np.exp(3.0 * np.array([0.0, -0.5, -2.0])))
    assert k[0] == 1.0 and k[1] > k[2]


def test_exponential_kernel_matches_original_formula():
    eigs = jnp.array([0.0, -1.0, -4.0])
    t = 2.0
    np.testing.assert_allclose(graph_kernel(eigs, t, "exponential"), t / (t - np.array([0.0, -1.0, -4.0])))


def test_mexican_hat_kernel():
    eigs = jnp.array([0.0, -1.0])
    np.testing.assert_allclose(graph_kernel(eigs, 1.0, "mexican_hat"), -np.array([0.0, -1.0]) * np.exp(np.array([0.0, -1.0])))


def test_damped_wave_kernel_matches_original_closed_form():
    eigs = np.array([0.0, -0.3, -2.0, -10.0])
    a, b, c, t = 1.0, 1.0, 0.0, 1.5
    delta = b**2 + 4 * a * (eigs - c)
    r1 = (-b + np.lib.scimath.sqrt(delta)) / (2 * a)
    r2 = (-b - np.lib.scimath.sqrt(delta)) / (2 * a)
    expected = ((r1 * np.exp(r2 * t) - r2 * np.exp(r1 * t)) / (r1 - r2)).real
    got = graph_kernel(jnp.array(eigs), t, "damped_wave", a=a, b=b, c=c)
    np.testing.assert_allclose(got, expected, rtol=1e-10)
    assert np.isfinite(np.asarray(got)).all()
