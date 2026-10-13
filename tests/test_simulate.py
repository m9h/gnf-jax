import numpy as np
import jax
import jax.numpy as jnp

from gnfjax import harmonic_spectrum, temporal_spectrum
from gnfjax.simulate import simulate_linearised, simulate_nonlinear
from gnfjax.graph import grid_laplacian_eigensystem


def test_linearised_simulation_variance_matches_analytic_spectrum(ref, ref_params):
    eigs = jnp.array(ref["eigs"])
    dt, n_steps = 1e-3, 400_000
    beta_e, beta_i = simulate_linearised(
        ref_params, eigs, ref["Ess"], ref["Iss"], jax.random.PRNGKey(0), dt=dt, n_steps=n_steps, downsample=10
    )
    assert beta_e.shape == (n_steps // 10, 64)
    pe, pi = harmonic_spectrum(ref_params, eigs, ref["Ess"], ref["Iss"])
    var_e = np.var(np.asarray(beta_e[1000:]), axis=0)
    var_i = np.var(np.asarray(beta_i[1000:]), axis=0)
    # stochastic: ~400 correlation times per mode, so ~7 % sd on each variance estimate
    for got, exp in ((var_e, pe), (var_i, pi)):
        rel = np.abs(got / np.asarray(exp) - 1.0)
        assert np.median(rel) < 0.08 and rel.max() < 0.35, rel


def test_nonlinear_simulation_stays_near_steady_state_without_noise(ref, ref_params):
    eigs, U = grid_laplacian_eigensystem(64, 1.0)
    E, I = simulate_nonlinear(
        ref_params._replace(noise_e=0.0, noise_i=0.0), eigs, U, ref["Ess"], ref["Iss"],
        jax.random.PRNGKey(1), dt=1e-3, n_steps=2000,
    )
    assert E.shape == (2000, 64)
    np.testing.assert_allclose(E[-1], ref["Ess"], atol=1e-6)
    np.testing.assert_allclose(I[-1], ref["Iss"], atol=1e-6)


def test_nonlinear_simulation_small_noise_matches_linearised_mode_variance(ref, ref_params):
    eigs, U = grid_laplacian_eigensystem(64, 1.0)
    small = ref_params._replace(noise_e=1e-3, noise_i=0.7e-3)
    E, I = simulate_nonlinear(small, eigs, U, ref["Ess"], ref["Iss"], jax.random.PRNGKey(2), dt=1e-3, n_steps=300_000)
    beta_e = (E[2000:] - ref["Ess"]) @ U  # project fluctuations onto modes
    pe, _ = harmonic_spectrum(small, eigs, ref["Ess"], ref["Iss"])
    rel = np.abs(np.var(np.asarray(beta_e), axis=0) / np.asarray(pe) - 1.0)
    assert np.median(rel) < 0.08 and rel.max() < 0.35, rel
