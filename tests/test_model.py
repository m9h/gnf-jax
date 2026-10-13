import numpy as np
import jax
import jax.numpy as jnp

from gnfjax import (
    GNFParams,
    steady_states,
    linearise,
    jacobian_eigenvalues,
    distance_to_instability,
    mode_covariance,
    harmonic_spectrum,
    temporal_spectrum,
)


def test_steady_states_recovers_both_original_fixed_points(ref, ref_params):
    ss, valid = steady_states(ref_params)
    found = np.asarray(ss)[np.asarray(valid)]
    expected = ref["all_ss"].T  # (n_ss, 2)
    for e in expected:
        assert np.min(np.linalg.norm(found - e, axis=1)) < 1e-6, (found, e)
    # no duplicates
    assert len(np.unique(found.round(6), axis=0)) == len(expected)


def test_steady_state_residual_is_zero(ref_params):
    from gnfjax.model import steady_state_residual

    ss, valid = steady_states(ref_params)
    for s, v in zip(np.asarray(ss), np.asarray(valid)):
        if v:
            np.testing.assert_allclose(steady_state_residual(ref_params, s), 0.0, atol=1e-9)


def test_jacobian_eigenvalues_match_original(ref, ref_params):
    eigs = jnp.array(ref["eigs"])
    lam = jacobian_eigenvalues(ref_params, eigs, ref["Ess"], ref["Iss"])
    assert lam.shape == (64, 2)
    got = np.sort_complex(np.asarray(lam).ravel())
    exp = np.sort_complex(ref["jac_eigs"].ravel())
    np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-10)


def test_distance_to_instability_is_negative_max_real_part(ref, ref_params):
    eigs = jnp.array(ref["eigs"])
    d = distance_to_instability(ref_params, eigs, ref["Ess"], ref["Iss"])
    np.testing.assert_allclose(d, -ref["jac_eigs"].real.max(), rtol=1e-8)
    assert d > 0  # the fixture steady state is stable


def test_mode_covariance_solves_lyapunov(ref, ref_params):
    eigs = jnp.array(ref["eigs"])
    A, D = linearise(ref_params, eigs, ref["Ess"], ref["Iss"])
    S = mode_covariance(A, D)
    assert S.shape == (64, 2, 2)
    resid = jnp.einsum("kij,kjl->kil", A, S) + jnp.einsum("kij,klj->kil", S, A) - D[None]
    np.testing.assert_allclose(resid, 0.0, atol=1e-9)


def test_harmonic_spectrum_matches_original_gaussian_and_exponential(ref, ref_params):
    eigs = jnp.array(ref["eigs"])
    pe, pi = harmonic_spectrum(ref_params, eigs, ref["Ess"], ref["Iss"])
    np.testing.assert_allclose(pe, ref["sps_E"], rtol=1e-8)
    np.testing.assert_allclose(pi, ref["sps_I"], rtol=1e-8)
    pe2, pi2 = harmonic_spectrum(ref_params, eigs, ref["Ess"], ref["Iss"], kernel="exponential")
    np.testing.assert_allclose(pe2, ref["sps_exp_E"], rtol=1e-8)
    np.testing.assert_allclose(pi2, ref["sps_exp_I"], rtol=1e-8)


def test_temporal_spectrum_matches_original(ref, ref_params):
    eigs = jnp.array(ref["eigs"])
    omegas = jnp.array(ref["omegas"])
    se, si = temporal_spectrum(ref_params, eigs, ref["Ess"], ref["Iss"], omegas)
    assert se.shape == (len(omegas), 64)
    np.testing.assert_allclose(se, ref["tps_E"], rtol=1e-8)
    np.testing.assert_allclose(si, ref["tps_I"], rtol=1e-8)


def test_temporal_spectrum_integrates_to_mode_variance(ref, ref_params):
    # int_0^inf S(omega) d omega / pi  ==  stationary variance (Wiener-Khinchin)
    eigs = jnp.array(ref["eigs"])
    omegas = jnp.linspace(0.0, 2000.0, 400001)
    se, si = temporal_spectrum(ref_params, eigs, ref["Ess"], ref["Iss"], omegas)
    pe, pi = harmonic_spectrum(ref_params, eigs, ref["Ess"], ref["Iss"])
    var_e = np.trapezoid(np.asarray(se), np.asarray(omegas), axis=0) / np.pi
    var_i = np.trapezoid(np.asarray(si), np.asarray(omegas), axis=0) / np.pi
    np.testing.assert_allclose(var_e, pe, rtol=2e-3)
    np.testing.assert_allclose(var_i, pi, rtol=2e-3)


def test_spectra_are_jittable_and_differentiable(ref, ref_params):
    eigs = jnp.array(ref["eigs"])

    def loss(params):
        pe, _ = harmonic_spectrum(params, eigs, ref["Ess"], ref["Iss"])
        return jnp.sum(jnp.log(pe))

    g = jax.jit(jax.grad(loss))(ref_params)
    assert all(np.isfinite(np.asarray(v)).all() for v in g)
    assert float(g.alpha_ee) != 0.0
