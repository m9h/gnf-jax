import numpy as np
import jax.numpy as jnp

from gnfjax.dfa import dfa_exponent


def test_white_noise_has_dfa_exponent_half():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2**15)
    h = dfa_exponent(jnp.array(x), window_sizes=np.unique(np.logspace(1.2, 3.3, 20).astype(int)))
    assert abs(float(h) - 0.5) < 0.06


def test_random_walk_has_dfa_exponent_three_halves():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(2**15))
    h = dfa_exponent(jnp.array(x), window_sizes=np.unique(np.logspace(1.2, 3.3, 20).astype(int)))
    assert abs(float(h) - 1.5) < 0.1


def test_dfa_is_batched_over_leading_axis():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 2**13))
    h = dfa_exponent(jnp.array(x), window_sizes=np.array([16, 32, 64, 128, 256, 512]))
    assert h.shape == (3,)
