"""gnf-jax: graph neural fields (stochastic Wilson-Cowan on connectome harmonics) in JAX.

JAX port of Marco Aqil's Graph-Stochastic-Wilson-Cowan-Model
(https://github.com/marcoaqil/Graph-Stochastic-Wilson-Cowan-Model, MIT).
"""
from .kernels import graph_kernel, diffusion_time, damped_wave_kernel, KERNELS
from .graph import laplacian_eigensystem, grid_laplacian_eigensystem, functional_connectivity
from .model import (
    GNFParams,
    steady_state_residual,
    steady_states,
    linearise,
    jacobian_eigenvalues,
    distance_to_instability,
    classify_steady_state,
    mode_covariance,
    harmonic_spectrum,
    temporal_spectrum,
)
from .simulate import simulate_linearised, simulate_nonlinear
from .fit import model_temporal_spectrum, scale_to_empirical, log_spectral_distance, spectral_loss, affine_combination
from .dfa import dfa_exponent

__version__ = "0.1.0"

__all__ = [
    "GNFParams", "KERNELS", "graph_kernel", "diffusion_time", "damped_wave_kernel",
    "laplacian_eigensystem", "grid_laplacian_eigensystem", "functional_connectivity",
    "steady_state_residual", "steady_states", "linearise", "jacobian_eigenvalues",
    "distance_to_instability", "classify_steady_state", "mode_covariance",
    "harmonic_spectrum", "temporal_spectrum", "simulate_linearised", "simulate_nonlinear",
    "model_temporal_spectrum", "scale_to_empirical", "log_spectral_distance", "spectral_loss",
    "affine_combination", "dfa_exponent",
]
