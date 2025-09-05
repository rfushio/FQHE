import numpy as np
import matplotlib.pyplot as plt

#nu in [1, 2]
mu_MLG_1_2 = np.loadtxt("mu_MLG_1_2.csv", delimiter=",", skiprows=1)
nuL_1_2=mu_MLG_1_2[0, 0]
nuR_1_2=mu_MLG_1_2[-1, 0]
muL_1_2=mu_MLG_1_2[0, 1]
muR_1_2=mu_MLG_1_2[-1, 1]
#Calculate the slope at the left and right end by linear fitting in a small range
nu_left_1_2 = np.linspace(nuL_1_2, nuL_1_2+0.05, 10)
mu_left_1_2 = np.interp(nu_left_1_2, mu_MLG_1_2[:, 0], mu_MLG_1_2[:, 1])
slope_left_1_2 = np.polyfit(nu_left_1_2, mu_left_1_2, 1)[0]
nu_right_1_2 = np.linspace(nuR_1_2-0.05, nuR_1_2, 10)
mu_right_1_2 = np.interp(nu_right_1_2, mu_MLG_1_2[:, 0], mu_MLG_1_2[:, 1])
slope_right_1_2 = np.polyfit(nu_right_1_2, mu_right_1_2, 1)[0]
param_left_1_2 = (-0.5*1.173*(nuL_1_2-1)**(-0.5)*56.1/4.0*np.sqrt(18))/slope_left_1_2
param_right_1_2 = (-0.5*1.173*(1-(nuR_1_2-1))**(-0.5)*56.1/4.0*np.sqrt(18))/slope_right_1_2

#nu in [0, 1]
# Build segment A with finer sampling]
mu_MLG_0_1 = np.loadtxt("mu_MLG_0_1.csv", delimiter=",", skiprows=1)
nuL_0_1=mu_MLG_0_1[0, 0]
nuR_0_1=mu_MLG_0_1[-1, 0]
muL_0_1=mu_MLG_0_1[0, 1]
muR_0_1=mu_MLG_0_1[-1, 1]
#Calculate the slope at the left and right end by linear fitting in a small range
nu_left_0_1 = np.linspace(nuL_0_1, nuL_0_1+0.05, 10)
mu_left_0_1 = np.interp(nu_left_0_1, mu_MLG_0_1[:, 0], mu_MLG_0_1[:, 1])
slope_left_0_1 = np.polyfit(nu_left_0_1, mu_left_0_1, 1)[0]
nu_right_0_1 = np.linspace(nuR_0_1-0.05, nuR_0_1, 10)
mu_right_0_1 = np.interp(nu_right_0_1, mu_MLG_0_1[:, 0], mu_MLG_0_1[:, 1])
slope_right_0_1 = np.polyfit(nu_right_0_1, mu_right_0_1, 1)[0]
param_left_0_1 = (-0.5*1.173*nuL_0_1**(-0.5)*56.1/4.0*np.sqrt(18))/slope_left_0_1
param_right_0_1 = (-0.5*1.173*(1-nuR_0_1)**(-0.5)*56.1/4.0*np.sqrt(18))/slope_right_0_1

#nu in [-1, 0]
mu_MLG_m1_0 = np.loadtxt("mu_MLG_m1_0.csv", delimiter=",", skiprows=1)
nuL_m1_0=mu_MLG_m1_0[0, 0]
nuR_m1_0=mu_MLG_m1_0[-1, 0]
muL_m1_0=mu_MLG_m1_0[0, 1]
muR_m1_0=mu_MLG_m1_0[-1, 1]
#Calculate the slope at the left and right end by linear fitting in a small range
nu_left_m1_0 = np.linspace(nuL_m1_0, nuL_m1_0+0.05, 10)
mu_left_m1_0 = np.interp(nu_left_m1_0, mu_MLG_m1_0[:, 0], mu_MLG_m1_0[:, 1])
slope_left_m1_0 = np.polyfit(nu_left_m1_0, mu_left_m1_0, 1)[0]
nu_right_m1_0 = np.linspace(nuR_m1_0-0.05, nuR_m1_0, 10)
mu_right_m1_0 = np.interp(nu_right_m1_0, mu_MLG_m1_0[:, 0], mu_MLG_m1_0[:, 1])
slope_right_m1_0 = np.polyfit(nu_right_m1_0, mu_right_m1_0, 1)[0]
param_left_m1_0 = (-0.5*1.173*(nuL_m1_0+1)**(-0.5)*56.1/4.0*np.sqrt(18))/slope_left_m1_0
param_right_m1_0 = (-0.5*1.173*(1-(nuR_m1_0+1))**(-0.5)*56.1/4.0*np.sqrt(18))/slope_right_m1_0

#nu in [-2, -1]
mu_MLG_m2_m1 = np.loadtxt("mu_MLG_m2_m1.csv", delimiter=",", skiprows=1)
nuL_m2_m1=mu_MLG_m2_m1[0, 0]
nuR_m2_m1=mu_MLG_m2_m1[-1, 0]
muL_m2_m1=mu_MLG_m2_m1[0, 1]
muR_m2_m1=mu_MLG_m2_m1[-1, 1]
#Calculate the slope at the left and right end by linear fitting in a small range
nu_left_m2_m1 = np.linspace(nuL_m2_m1, nuL_m2_m1+0.05, 10)
mu_left_m2_m1 = np.interp(nu_left_m2_m1, mu_MLG_m2_m1[:, 0], mu_MLG_m2_m1[:, 1])
slope_left_m2_m1 = np.polyfit(nu_left_m2_m1, mu_left_m2_m1, 1)[0]
nu_right_m2_m1 = np.linspace(nuR_m2_m1-0.05, nuR_m2_m1, 10)
mu_right_m2_m1 = np.interp(nu_right_m2_m1, mu_MLG_m2_m1[:, 0], mu_MLG_m2_m1[:, 1])
slope_right_m2_m1 = np.polyfit(nu_right_m2_m1, mu_right_m2_m1, 1)[0]
param_left_m2_m1 = (-0.5*1.173*(nuL_m2_m1+2)**(-0.5)*56.1/4.0*np.sqrt(18))/slope_left_m2_m1
param_right_m2_m1 = (-0.5*1.173*(1-(nuR_m2_m1+2))**(-0.5)*56.1/4.0*np.sqrt(18))/slope_right_m2_m1

print(slope_left_0_1, slope_right_0_1, slope_left_m1_0, slope_right_m1_0)
print(param_left_0_1, param_right_0_1, param_left_m1_0, param_right_m1_0)

#nu in [0, 1]
nu_a = np.linspace(0.0, nuL_0_1, 400)
mu_a = -1.173 * (nu_a**0.5 - nuL_0_1**0.5) * 56.1 / 4.0 * np.sqrt(18) / param_left_0_1 + muL_0_1
mask_b = (mu_MLG_0_1[:, 0] >= nuL_0_1) & (mu_MLG_0_1[:, 0] <= nuR_0_1)
nu_b = mu_MLG_0_1[mask_b, 0]
mu_b = mu_MLG_0_1[mask_b, 1]
nu_c = np.linspace(nuR_0_1, 1.0, 400)
mu_c = 1.173 * ((1 - nu_c)**0.5 - (1-nuR_0_1)**0.5) * 56.1 / 4.0 * np.sqrt(18) / param_right_0_1 + muR_0_1

#nu in [-1, 0]
nu_m1_0_L_ren = np.linspace(0,nuL_m1_0+1, 400)
nu_m1_0_R_ren = np.linspace(nuR_m1_0+1, 1.0, 400)
nu_d = np.linspace(-1.0, nuL_m1_0, 400)
mu_d = -1.173 * (nu_m1_0_L_ren**0.5 - (nuL_m1_0+1)**0.5) * 56.1 / 4.0 * np.sqrt(18) / param_left_m1_0 + muL_m1_0
mask_e = (mu_MLG_m1_0[:, 0] >= nuL_m1_0) & (mu_MLG_m1_0[:, 0] <= nuR_m1_0)
nu_e = mu_MLG_m1_0[mask_e, 0]
mu_e = mu_MLG_m1_0[mask_e, 1]
nu_f = np.linspace(nuR_m1_0, 0.0, 400)
mu_f = 1.173 * ((1 - nu_m1_0_R_ren)**0.5 - (1-(nuR_m1_0+1))**0.5) * 56.1 / 4.0 * np.sqrt(18) / param_right_m1_0 + muR_m1_0

#nu in [-2, -1]
nu_m2_m1_L_ren = np.linspace(0,nuL_m2_m1+2, 400)
nu_m2_m1_R_ren = np.linspace(nuR_m2_m1+2, 1.0, 400)
nu_g = np.linspace(-2.0, nuL_m2_m1, 400)
mu_g = -1.173 * (nu_m2_m1_L_ren**0.5 - (nuL_m2_m1+2)**0.5) * 56.1 / 4.0 * np.sqrt(18) / param_left_m2_m1 + muL_m2_m1
mask_h = (mu_MLG_m2_m1[:, 0] >= nuL_m2_m1) & (mu_MLG_m2_m1[:, 0] <= nuR_m2_m1)
nu_h = mu_MLG_m2_m1[mask_h, 0]
mu_h = mu_MLG_m2_m1[mask_h, 1]
nu_i = np.linspace(nuR_m2_m1, -1.0, 400)
mu_i = 1.173 * ((1 - nu_m2_m1_R_ren)**0.5 - (1-(nuR_m2_m1+2))**0.5) * 56.1 / 4.0 * np.sqrt(18) / param_right_m2_m1 + muR_m2_m1

#nu in [1, 2]
nu_1_2_L_ren = np.linspace(0,nuL_1_2-1, 400)
nu_1_2_R_ren = np.linspace(nuR_1_2-1, 1.0, 400)
nu_j = np.linspace(1.0, nuL_1_2, 400)
mu_j = -1.173 * (nu_1_2_L_ren**0.5 - (nuL_1_2-1)**0.5) * 56.1 / 4.0 * np.sqrt(18) / param_left_1_2 + muL_1_2
mask_k = (mu_MLG_1_2[:, 0] >= nuL_1_2) & (mu_MLG_1_2[:, 0] <= nuR_1_2)
nu_k = mu_MLG_1_2[mask_k, 0]
mu_k = mu_MLG_1_2[mask_k, 1]
nu_l = np.linspace(nuR_1_2, 2.0, 400)
mu_l = 1.173 * ((1 - nu_1_2_R_ren)**0.5 - (1-(nuR_1_2-1))**0.5) * 56.1 / 4.0 * np.sqrt(18) / param_right_1_2 + muR_1_2




# Concatenate and sort to ensure monotonic ν
nu_all = np.concatenate([nu_a, nu_b, nu_c, nu_d, nu_e, nu_f, nu_g, nu_h, nu_i, nu_j, nu_k, nu_l])
mu_all = np.concatenate([mu_a, mu_b, mu_c, mu_d, mu_e, mu_f, mu_g, mu_h, mu_i, mu_j, mu_k, mu_l])
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
np.savetxt("/Users/rikutofushio/00ARIP/Thomas-Fermi-Folder/Thomas-Fermi/data/0-data/Exc_MLG_m2_2.csv", out, delimiter=",", header="nu,mu_meV,E_meV", comments="")

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