"""Reproduce the reference setting: spectra, stability and FC on a 64-node chain graph.

Run: python examples/grid_demo.py  (writes grid_demo.png next to this file)
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import gnfjax as g

eigs, U = g.grid_laplacian_eigensystem(64, 1.0)
params = g.GNFParams(alpha_ee=20.0, alpha_ie=15.0, alpha_ei=15.0, alpha_ii=5.0, p=-5.0, q=-8.0,
                     sigma_ee=2.0, sigma_ie=2.0, sigma_ei=2.0, sigma_ii=2.0, tau_e=1.0, tau_i=0.5)

states, valid = g.steady_states(params)
for s in np.asarray(states)[np.asarray(valid)]:
    cls = int(g.classify_steady_state(params, eigs, s[0], s[1]))
    print(f"steady state E*={s[0]:.4f} I*={s[1]:.4f}  class={cls}  "
          f"distance to instability={float(g.distance_to_instability(params, eigs, s[0], s[1])):.4f}")

ess, iss = max((s for s in np.asarray(states)[np.asarray(valid)]),
               key=lambda s: float(g.distance_to_instability(params, eigs, s[0], s[1])))
p_e, p_i = g.harmonic_spectrum(params, eigs, ess, iss)
omegas = jnp.linspace(0.0, 40.0, 801)
t_e, t_i = g.model_temporal_spectrum(params, eigs, ess, iss, omegas)
lam = g.jacobian_eigenvalues(params, eigs, ess, iss)
fc = g.functional_connectivity(U, p_e)

beta_e, _ = g.simulate_linearised(params, eigs, ess, iss, jax.random.PRNGKey(0), dt=1e-3, n_steps=200_000, downsample=10)
sim_var = np.var(np.asarray(beta_e[1000:]), axis=0)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("install matplotlib to draw the figure")

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
k = np.arange(1, 64)
ax[0].loglog(k, p_e[1:], "k-", label="E (analytic)")
ax[0].loglog(k, p_i[1:], "b-", label="I (analytic)")
ax[0].loglog(k, sim_var[1:], "r.", ms=4, label="E (simulation)")
ax[0].set(xlabel="eigenmode k", title="harmonic power spectrum"); ax[0].legend()
ax[1].loglog(omegas[1:] / (2 * np.pi), t_e[1:], "k-"); ax[1].loglog(omegas[1:] / (2 * np.pi), t_i[1:], "b-")
ax[1].set(xlabel="frequency (Hz)", title="temporal power spectrum")
ax[2].scatter(np.asarray(lam).real.ravel(), np.asarray(lam).imag.ravel(), s=6, c="k")
ax[2].axvline(0, color="r", lw=0.8); ax[2].set(xlabel="Re", ylabel="Im", title="Jacobian eigenvalues")
im = ax[3].imshow(np.asarray(fc), cmap="inferno", vmin=0, vmax=1); ax[3].set(title="predicted FC (CHAOSS)")
fig.colorbar(im, ax=ax[3])
fig.tight_layout()
out = __file__.replace(".py", ".png")
fig.savefig(out, dpi=120)
print("wrote", out)
