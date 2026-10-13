"""Shared fixtures: reference values computed by the original numpy code.

``tests/data/gnf_reference.npz`` was produced by ``make_reference_fixture.py``
running Marco Aqil's ``core/analysis.py`` (Graph-Stochastic-Wilson-Cowan-Model,
branch ``spinoza``) on a 64-node 1-D grid Laplacian with a stable oscillatory
steady state. Every analytic quantity in gnfjax is checked against it.
"""
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

DATA = Path(__file__).parent / "data" / "gnf_reference.npz"


@pytest.fixture(scope="session")
def ref():
    z = np.load(DATA)
    return {k: z[k] for k in z.files}


@pytest.fixture(scope="session")
def ref_params(ref):
    from gnfjax import GNFParams

    p = ref["params"]
    return GNFParams(
        alpha_ee=p[0], alpha_ie=p[1], alpha_ei=p[2], alpha_ii=p[3],
        d_e=p[4], d_i=p[5], p=p[6], q=p[7],
        sigma_ee=p[8], sigma_ie=p[9], sigma_ei=p[10], sigma_ii=p[11],
        tau_e=p[12], tau_i=p[13],
        noise_e=ref["sigma_noise"][0], noise_i=ref["sigma_noise"][1],
    )
