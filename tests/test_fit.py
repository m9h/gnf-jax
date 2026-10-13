import numpy as np
import jax
import jax.numpy as jnp

from gnfjax import harmonic_spectrum, temporal_spectrum
from gnfjax.fit import scale_to_empirical, log_spectral_distance, spectral_loss, model_temporal_spectrum


def test_scale_to_empirical_recovers_affine_combination(ref, ref_params):
    eigs = jnp.array(ref["eigs"])
    pe, pi = harmonic_spectrum(ref_params, eigs, ref["Ess"], ref["Iss"])
    target = 3.0 * pe + 0.5 * pi + 0.1
    x, fitted = scale_to_empirical(pe, pi, target)
    np.testing.assert_allclose(x, [3.0, 0.5, 0.1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(fitted, target, rtol=1e-3)


def test_log_spectral_distance_is_zero_for_identical_and_positive_otherwise():
    s = jnp.linspace(1.0, 10.0, 20)
    assert float(log_spectral_distance(s, s, power=2)) == (1.0) ** 2 + (1.0) ** 2  # (1+0)^p twice
    assert float(log_spectral_distance(s, 2 * s, power=2)) > 2.0


def test_model_temporal_spectrum_sums_modes(ref, ref_params):
    eigs = jnp.array(ref["eigs"])
    omegas = jnp.array(ref["omegas"])
    te, ti = model_temporal_spectrum(ref_params, eigs, ref["Ess"], ref["Iss"], omegas)
    np.testing.assert_allclose(te, 2 * ref["tps_E"].sum(axis=1), rtol=1e-8)
    np.testing.assert_allclose(ti, 2 * ref["tps_I"].sum(axis=1), rtol=1e-8)


def test_spectral_loss_is_minimal_at_generating_parameters_and_differentiable(ref, ref_params):
    eigs = jnp.array(ref["eigs"])
    omegas = jnp.array(ref["omegas"])
    pe, pi = harmonic_spectrum(ref_params, eigs, ref["Ess"], ref["Iss"])
    te, _ = model_temporal_spectrum(ref_params, eigs, ref["Ess"], ref["Iss"], omegas)
    target_sp = (2.0 * pe + 0.3)[2:]  # empirical spectrum is given on modes first_k:last_k
    target_tp = 5.0 * te + 1.0
    l0 = spectral_loss(ref_params, eigs, omegas, target_sp, target_tp, first_k=2)
    perturbed = ref_params._replace(alpha_ee=ref_params.alpha_ee * 1.1)
    l1 = spectral_loss(perturbed, eigs, omegas, target_sp, target_tp, first_k=2)
    assert float(l0) < float(l1)
    g = jax.grad(lambda p: spectral_loss(p, eigs, omegas, target_sp, target_tp, first_k=2))(ref_params)
    assert all(np.isfinite(np.asarray(v)).all() for v in g)


def test_spectral_loss_penalises_unstable_parameters(ref, ref_params):
    eigs = jnp.array(ref["eigs"])
    omegas = jnp.array(ref["omegas"])
    pe, _ = harmonic_spectrum(ref_params, eigs, ref["Ess"], ref["Iss"])
    te, _ = model_temporal_spectrum(ref_params, eigs, ref["Ess"], ref["Iss"], omegas)
    # the fixture's *other* steady state is unstable; forcing it should give the large penalty
    unstable = ref["all_ss"][:, 0]
    l = spectral_loss(ref_params, eigs, omegas, pe[2:], te, first_k=2, steady_state=unstable)
    assert float(l) >= 1e9
