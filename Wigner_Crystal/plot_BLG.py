import numpy as np
import matplotlib.pyplot as plt

nus = np.linspace(0.0, 0.2, 100)
E_meV = -0.7821 * nus**1.5 * 56.1 /np.sqrt(16) *np.sqrt(18)
E_meV_modified = -0.7821 * nus**1.5 * 56.1 /np.sqrt(16) *np.sqrt(18) /1.5

plt.figure(figsize=(6,4))
plt.plot(nus, E_meV, "--", label=r"Ref: $E=-0.7821\nu^{1.5}E_C$")
plt.plot(nus, E_meV_modified, "--", label=r"Ref: $E=-0.7821\nu^{1.5}E_C/1.5$")
plt.xlabel(r"$\nu$")
plt.xlim(0.0, 0.2)
#plt.ylim(-4, 0)
plt.ylabel(r"$E$ [meV]")
plt.title("Wigner crystal energy (reference)")
plt.legend()
plt.tight_layout()
plt.savefig("wc_ref_E.png", dpi=200)

mu_meV = -1.173 * nus**0.5* 56.1 /4.0 *np.sqrt(18)
mu_meV_modified = -1.173 * nus**0.5* 56.1 /4.0 *np.sqrt(18) /1.5

plt.figure(figsize=(4, 6))
plt.plot(nus, mu_meV, "--", label=r"Ref: $\mu=-1.173\nu^{0.5}E_C$")
plt.plot(nus, mu_meV_modified, "--", label=r"Ref: $\mu=-1.173\nu^{0.5}E_C/1.5$")
plt.xlabel(r"$\nu$")
plt.xlim(0.0, 0.2)
#plt.ylim(-4, 0)
plt.ylabel(r"$\mu$ [meV]")
plt.title("Wigner crystal energy (reference),")
plt.legend()
plt.tight_layout()
plt.savefig("wc_ref_mu.png", dpi=200)

# Compose mu(ν) from three connected segments and integrate to E(ν)
# Segments:
#  A: nu ∈ [0, 0.262]  -> mu_meV_modified1
#  B: nu ∈ [0.262, 0.746] from mu_BLG_N=1 (shifted by offset)
#  C: nu ∈ [0.746, 1.0]  -> mu_meV_modified2

#N=0, nu in [0, 1]
# Build segment A with finer sampling]
param0=8.0
param1=6.0
mu_BLG_N0 = np.loadtxt("mu_BLG_N=0.csv", delimiter=",", skiprows=1)
mu_BLG_N1 = np.loadtxt("mu_BLG_N=1.csv", delimiter=",", skiprows=1)

nuLN0=mu_BLG_N0[0, 0]
nuRN0=mu_BLG_N0[-1, 0]
nuLN1=mu_BLG_N1[0, 0]
nuRN1=mu_BLG_N1[-1, 0]
muLN0=mu_BLG_N0[0, 1]
muRN0=mu_BLG_N0[-1, 1]
muLN1=mu_BLG_N1[0, 1]
muRN1=mu_BLG_N1[-1, 1]
offsetN0= -(-1.173 * (1-nuRN0)**0.5* 56.1 /4.0 *np.sqrt(18) /param0-muRN0)
offsetN1= -(-1.173 * (1-nuRN1)**0.5* 56.1 /4.0 *np.sqrt(18) /param1-muRN1)

print(nuLN0, nuRN0, nuLN1, nuRN1, muLN0, muRN0, muLN1, muRN1)

nu_0_L = np.linspace(0.0, nuLN0, 400)
nu_0_R = np.linspace(nuRN0, 1.0, 400)
nu_1_L = np.linspace(0.0, nuLN1, 400)
nu_1_R = np.linspace(nuRN1, 1.0, 400)
#N=0, nu in [0, 1]
nu_a = np.linspace(0.0, nuLN0, 400)
mu_a = -1.173 * (nu_0_L**0.5 - nuLN0**0.5) * 56.1 / 4.0 * np.sqrt(18) / param0 + muLN0 + offsetN0
mask_b = (mu_BLG_N0[:, 0] >= nuLN0) & (mu_BLG_N0[:, 0] <= nuRN0)
nu_b = mu_BLG_N0[mask_b, 0]
mu_b = mu_BLG_N0[mask_b, 1] + offsetN0
nu_c = np.linspace(nuRN0, 1.0, 400)
mu_c = 1.173 * ((1 - nu_0_R)**0.5 - (1-nuRN0)**0.5) * 56.1 / 4.0 * np.sqrt(18) / param0 + muRN0 + offsetN0

#N=1, nu in [1, 2]
nu_d = np.linspace(0.0+1, nuLN1+1, 400)
mu_d = -1.173 * (nu_1_L**0.5 - nuLN1**0.5) * 56.1 / 4.0 * np.sqrt(18) / param1 + muLN1 + offsetN1
mask_e = (mu_BLG_N1[:, 0] >= nuLN1) & (mu_BLG_N1[:, 0] <= nuRN1)
nu_e = mu_BLG_N1[mask_e, 0]+1
mu_e = mu_BLG_N1[mask_e, 1] + offsetN1
nu_f = np.linspace(nuRN1+1, 1.0+1, 400)
mu_f = 1.173 * ((1 - nu_1_R)**0.5 - (1-nuRN1)**0.5) * 56.1 / 4.0 * np.sqrt(18) / param1 + muRN1 + offsetN1

#N=0, nu in [2, 3]
nu_g = np.linspace(0.0+2, nuLN0+2, 400)
mu_g = -1.173 * (nu_0_L**0.5 - nuLN0**0.5) * 56.1 / 4.0 * np.sqrt(18) / param0 + muLN0 + offsetN0
mask_h = (mu_BLG_N0[:, 0] >= nuLN0) & (mu_BLG_N0[:, 0] <= nuRN0)
nu_h = mu_BLG_N0[mask_h, 0]+2
mu_h = mu_BLG_N0[mask_h, 1] + offsetN0
nu_i = np.linspace(nuRN0+2, 1.0+2, 400)
mu_i = 1.173 * ((1 - nu_0_R)**0.5 - (1-nuRN0)**0.5) * 56.1 / 4.0 * np.sqrt(18) / param0 + muRN0 + offsetN0

#N=1, nu in [-1, 0]
nu_j = np.linspace(0.0-1, nuLN1-1, 400)
mu_j = -1.173 * (nu_1_L**0.5 - nuLN1**0.5) * 56.1 / 4.0 * np.sqrt(18) / param1 + muLN1 + offsetN1
mask_k = (mu_BLG_N1[:, 0] >= nuLN1) & (mu_BLG_N1[:, 0] <= nuRN1)
nu_k = mu_BLG_N1[mask_k, 0]-1
mu_k = mu_BLG_N1[mask_k, 1] + offsetN1
nu_l = np.linspace(nuRN1-1, 1.0-1, 400)
mu_l = 1.173 * ((1 - nu_1_R)**0.5 - (1-nuRN1)**0.5) * 56.1 / 4.0 * np.sqrt(18) / param1 + muRN1 + offsetN1

#N=0, nu in [-2, -1]
nu_m = np.linspace(0.0-2, nuLN0-2, 400)
mu_m = -1.173 * (nu_0_L**0.5 - nuLN0**0.5) * 56.1 / 4.0 * np.sqrt(18) / param0 + muLN0 + offsetN0
mask_n = (mu_BLG_N0[:, 0] >= nuLN0) & (mu_BLG_N0[:, 0] <= nuRN0)
nu_n = mu_BLG_N0[mask_n, 0]-2
mu_n = mu_BLG_N0[mask_n, 1] + offsetN0
nu_o = np.linspace(nuRN0-2, 1.0-2, 400)
mu_o = 1.173 * ((1 - nu_0_R)**0.5 - (1-nuRN0)**0.5) * 56.1 / 4.0 * np.sqrt(18) / param0 + muRN0 + offsetN0

#N=1, nu in [-3, -2]
nu_p = np.linspace(0.0-3, nuLN1-3, 400)
mu_p = -1.173 * (nu_1_L**0.5 - nuLN1**0.5) * 56.1 / 4.0 * np.sqrt(18) / param1 + muLN1 + offsetN1
mask_q = (mu_BLG_N1[:, 0] >= nuLN1) & (mu_BLG_N1[:, 0] <= nuRN1)
nu_q = mu_BLG_N1[mask_q, 0]-3
mu_q = mu_BLG_N1[mask_q, 1] + offsetN1
nu_r = np.linspace(nuRN1-3, 1.0-3, 400)
mu_r = 1.173 * ((1 - nu_1_R)**0.5 - (1-nuRN1)**0.5) * 56.1 / 4.0 * np.sqrt(18) / param1 + muRN1 + offsetN1



# Concatenate and sort to ensure monotonic ν
nu_all = np.concatenate([nu_a, nu_b, nu_c, nu_d, nu_e, nu_f, nu_g, nu_h, nu_i, nu_j, nu_k, nu_l, nu_m, nu_n, nu_o, nu_p, nu_q, nu_r])
mu_all = np.concatenate([mu_a, mu_b, mu_c, mu_d, mu_e, mu_f, mu_g, mu_h, mu_i, mu_j, mu_k, mu_l, mu_m, mu_n, mu_o, mu_p, mu_q, mu_r])
order = np.argsort(nu_all)
nu_all = nu_all[order]
mu_all = mu_all[order]

# Remove potential duplicate ν points (keep first occurrence)
unique_mask = np.concatenate([[True], np.diff(nu_all) > 1e-12])
nu_all = nu_all[unique_mask]
mu_all = mu_all[unique_mask]

# Local cumulative trapezoid (no SciPy dependency)
def _cumtrapz(y, x):
    dx = np.diff(x)
    incr = 0.5 * (y[1:] + y[:-1]) * dx
    return np.concatenate([[0.0], np.cumsum(incr)])

# Integrate: E(ν) = ∫_0^ν μ(ν') dν' (meV)
E_all = _cumtrapz(mu_all, nu_all)

# Shift baseline so that E(0) = 0 exactly
E_at_zero = np.interp(0.0, nu_all, E_all)
E_all = E_all - E_at_zero

# Save CSV
out = np.stack([nu_all, mu_all, E_all], axis=1)
np.savetxt("/Users/rikutofushio/00ARIP/Thomas-Fermi-Folder/Thomas-Fermi/data/0-data/Exc_BLG_01.csv", out, delimiter=",", header="nu,mu_meV,E_meV", comments="")

# Plot μ and E
plt.figure(figsize=(6, 6))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(nu_all, mu_all, label=r"$\mu(\nu)$")
ax1.set_xlabel(r"$\nu$")
ax1.set_ylabel(r"$\mu$ [meV]")

ax1.legend()

ax2 = plt.subplot(2, 1, 2)
ax2.plot(nu_all, E_all, label=r"$E(\nu)=\int_0^\nu \mu(\nu') d\nu'$")
ax2.set_xlabel(r"$\nu$")
ax2.set_ylabel(r"$E$ [meV]")
#ax2.set_xlim(0.0, 1.0)
ax2.legend()

plt.tight_layout()
plt.savefig("mu_integrated.png", dpi=200)