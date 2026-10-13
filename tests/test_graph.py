import numpy as np
import jax.numpy as jnp

from gnfjax.graph import grid_laplacian_eigensystem, laplacian_eigensystem, functional_connectivity


def test_grid_laplacian_matches_original_eigenvalues(ref):
    eigs, U = grid_laplacian_eigensystem(64, 1.0)
    np.testing.assert_allclose(eigs, ref["eigs"], atol=1e-9)
    assert eigs[0] == 0.0 and np.all(np.asarray(eigs[1:]) < 0)
    np.testing.assert_allclose(np.asarray(U).T @ np.asarray(U), np.eye(64), atol=1e-9)


def test_laplacian_eigensystem_sign_convention_is_nonpositive_descending():
    rng = np.random.default_rng(0)
    W = rng.random((10, 10))
    W = np.triu(W, 1)
    W = W + W.T
    eigs, U = laplacian_eigensystem(W)
    assert eigs.shape == (10,) and U.shape == (10, 10)
    assert abs(float(eigs[0])) < 1e-9
    assert np.all(np.diff(np.asarray(eigs)) <= 1e-12)  # descending: 0, -..., -...
    # eigenvectors diagonalise -(D - W)
    L = -(np.diag(W.sum(1)) - W)
    np.testing.assert_allclose(np.asarray(U).T @ L @ np.asarray(U), np.diag(np.asarray(eigs)), atol=1e-8)


def test_functional_connectivity_matches_original(ref):
    FC = functional_connectivity(jnp.array(ref["U"]), jnp.array(ref["sps_E"]))
    np.testing.assert_allclose(FC, ref["FC"], atol=1e-10)
    np.testing.assert_allclose(np.diag(np.asarray(FC)), 1.0, atol=1e-10)
