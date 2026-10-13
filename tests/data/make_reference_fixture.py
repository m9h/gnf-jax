import sys, types, numpy as np
# stub heavy/unneeded imports of the original module
for name in ("similaritymeasures","matplotlib","matplotlib.pyplot","matplotlib.colors","h5py"):
    m = types.ModuleType(name); sys.modules[name]=m
sys.modules["matplotlib.pyplot"].rcParams = {}
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
sys.modules["matplotlib"].colors = sys.modules["matplotlib.colors"]
sys.path.insert(0, ".")
import importlib.util
spec = importlib.util.spec_from_file_location("aqil_analysis", "analysis.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
np.random.seed(0)
eigs = A.one_dim_Laplacian_eigenvalues(64, 1.0)
P = dict(alpha_EE=20., alpha_IE=15., alpha_EI=15., alpha_II=5., d_e=1., d_i=1., P=-5., Q=-8.)
ss, ok = A.H_Simple_Steady_State(**P)
print("steady states", ok, ss)
kern = dict(sigma_EE=2., sigma_IE=2., sigma_EI=2., sigma_II=2., D=1., tau_e=1., tau_i=0.5)
chosen=None
for j in range(ss.shape[1]):
    typ, suit, jac = A.GraphWC_Jacobian_TrDet(eigs, 'Gaussian', ss[0,j], ss[1,j], P['alpha_EE'],P['alpha_IE'],P['alpha_EI'],P['alpha_II'],P['d_e'],P['d_i'], **kern)
    print(j, ss[:,j], "type", typ, "suitable", suit, "max re", jac.real.max())
    if suit and chosen is None: chosen=j
Ess, Iss = ss[0,chosen], ss[1,chosen]
omegas = np.linspace(0, 40, 801)
sps = A.Graph_WC_Spatiotemporal_PowerSpectrum(eigs,'Gaussian',Ess,Iss,P['alpha_EE'],P['alpha_IE'],P['alpha_EI'],P['alpha_II'],P['d_e'],P['d_i'],**kern, sigma_noise_e=1.0, sigma_noise_i=1.0, Spatial_Spectrum_Only=True)
E_tps, I_tps = A.Graph_WC_Spatiotemporal_PowerSpectrum(eigs,'Gaussian',Ess,Iss,P['alpha_EE'],P['alpha_IE'],P['alpha_EI'],P['alpha_II'],P['d_e'],P['d_i'],**kern, sigma_noise_e=1.0, sigma_noise_i=1.0, omegas=omegas, Spatial_Spectrum_Only=False)
typ, suit, jac = A.GraphWC_Jacobian_TrDet(eigs,'Gaussian',Ess,Iss,P['alpha_EE'],P['alpha_IE'],P['alpha_EI'],P['alpha_II'],P['d_e'],P['d_i'],**kern)
# exponential kernel too
sps_exp = A.Graph_WC_Spatiotemporal_PowerSpectrum(eigs,'Exponential',Ess,Iss,P['alpha_EE'],P['alpha_IE'],P['alpha_EI'],P['alpha_II'],P['d_e'],P['d_i'],**kern, sigma_noise_e=1.0, sigma_noise_i=1.0, Spatial_Spectrum_Only=True)
_, U = A.one_dim_Laplacian_eigenvalues(64, 1.0, vecs=True)
FC = A.Functional_Connectivity(U, sps[:,0,0])
np.savez("gnf_reference.npz", eigs=eigs, U=U, Ess=Ess, Iss=Iss, params=np.array([P['alpha_EE'],P['alpha_IE'],P['alpha_EI'],P['alpha_II'],P['d_e'],P['d_i'],P['P'],P['Q'],2.,2.,2.,2.,1.,0.5]), sigma_noise=np.array([1.0,1.0]), omegas=omegas, sps_E=sps[:,0,0], sps_I=sps[:,1,1], tps_E=E_tps, tps_I=I_tps, jac_eigs=jac, sps_exp_E=sps_exp[:,0,0], sps_exp_I=sps_exp[:,1,1], FC=FC, all_ss=ss)
print("saved; sps_E[:5]", sps[:5,0,0], "tps_E shape", E_tps.shape)
