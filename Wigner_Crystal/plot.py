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

param=6.0
offset= -(-1.173 * (1-0.746)**0.5* 56.1 /4.0 *np.sqrt(18) /param-1.541)
print(offset)
nus1 = np.linspace(0.0, 0.262, 100)
nus2 = np.linspace(0.746, 1.0, 100)
mu_meV_modified1 = -1.173 * (nus1**0.5-0.262**0.5)* 56.1 /4.0 *np.sqrt(18) /param+2.096+offset
mu_meV_modified2 = 1.173 * ((1-nus2)**0.5-0.254**0.5)* 56.1 /4.0 *np.sqrt(18) /param-1.541+offset
mu_paper = np.loadtxt("mu_paper.csv", delimiter=",", skiprows=1)

nus3 = np.linspace(0.0+1, 0.262+1, 100)
nus4 = np.linspace(0.746+1, 1.0+1, 100)
mu_meV_modified3 = -1.173 * (nus1**0.5-0.262**0.5)* 56.1 /4.0 *np.sqrt(18) /param+2.096+offset
mu_meV_modified4 = 1.173 * ((1-nus2)**0.5-0.254**0.5)* 56.1 /4.0 *np.sqrt(18) /param-1.541+offset

nus5 = np.linspace(0.0+2, 0.262+2, 100)
nus6 = np.linspace(0.746+2, 1.0+2, 100)
mu_meV_modified5 = -1.173 * (nus1**0.5-0.262**0.5)* 56.1 /4.0 *np.sqrt(18) /param+2.096+offset
mu_meV_modified6 = 1.173 * ((1-nus2)**0.5-0.254**0.5)* 56.1 /4.0 *np.sqrt(18) /param-1.541+offset

plt.figure(figsize=(6, 4))
plt.plot(mu_paper[:,0], mu_paper[:,1]+offset)
plt.plot(nus1, mu_meV_modified1, "--", label=r"Ref: $\mu=-1.173\nu^{0.5}E_C/1.5$")
plt.plot(nus2, mu_meV_modified2, "--", label=r"Ref: $\mu=-1.173\nu^{0.5}E_C/1.5$")
mu_paper2=mu_paper[:,0]+1
plt.plot(mu_paper2, mu_paper[:,1]+offset)
plt.plot(nus3, mu_meV_modified3, "--", label=r"Ref: $\mu=-1.173\nu^{0.5}E_C/1.5$")
plt.plot(nus4, mu_meV_modified4, "--", label=r"Ref: $\mu=-1.173\nu^{0.5}E_C/1.5$")
mu_paper3=mu_paper[:,0]+2
plt.plot(mu_paper3, mu_paper[:,1]+offset)
plt.plot(nus5, mu_meV_modified5, "--", label=r"Ref: $\mu=-1.173\nu^{0.5}E_C/1.5$")
plt.plot(nus6, mu_meV_modified6, "--", label=r"Ref: $\mu=-1.173\nu^{0.5}E_C/1.5$")
plt.xlabel(r"$\nu$")
plt.ylabel(r"$\mu$ [meV]")
plt.title("Wigner crystal energy (reference)")
plt.tight_layout()
plt.savefig("wc_ref_mu_paper.png", dpi=200)

# Compose mu(ν) from three connected segments and integrate to E(ν)
# Segments:
#  A: nu ∈ [0, 0.262]  -> mu_meV_modified1
#  B: nu ∈ [0.262, 0.746] from mu_paper (shifted by offset)
#  C: nu ∈ [0.746, 1.0]  -> mu_meV_modified2

# Build segment A with finer sampling
nu_a = np.linspace(0.0, 0.262, 400)
mu_a = -1.173 * (nu_a**0.5 - 0.262**0.5) * 56.1 / 4.0 * np.sqrt(18) / param + 2.096 + offset

# Segment B from paper data within [0.262, 0.746]
mask_b = (mu_paper[:, 0] >= 0.262) & (mu_paper[:, 0] <= 0.746)
nu_b = mu_paper[mask_b, 0]
mu_b = mu_paper[mask_b, 1] + offset

# Segment C with finer sampling
nu_c = np.linspace(0.746, 1.0, 400)
mu_c = 1.173 * ((1 - nu_c)**0.5 - 0.254**0.5) * 56.1 / 4.0 * np.sqrt(18) / param - 1.541 + offset

nu_d = np.linspace(0.0+1, 0.262+1, 400)
mu_d = -1.173 * (nu_a**0.5 - 0.262**0.5) * 56.1 / 4.0 * np.sqrt(18) / param + 2.096 + offset

mask_e = (mu_paper[:, 0] >= 0.262) & (mu_paper[:, 0] <= 0.746)
nu_e = mu_paper[mask_b, 0] + 1
mu_e = mu_paper[mask_b, 1] + offset

nu_f = np.linspace(0.746+1, 1.0+1, 400)
mu_f = 1.173 * ((1 - nu_c)**0.5 - 0.254**0.5) * 56.1 / 4.0 * np.sqrt(18) / param - 1.541 + offset

nu_g = np.linspace(0.0+2, 0.262+2, 400)
mu_g = -1.173 * (nu_a**0.5 - 0.262**0.5) * 56.1 / 4.0 * np.sqrt(18) / param + 2.096 + offset

mask_h = (mu_paper[:, 0] >= 0.262) & (mu_paper[:, 0] <= 0.746)
nu_h = mu_paper[mask_b, 0] + 2
mu_h = mu_paper[mask_b, 1] + offset

nu_i = np.linspace(0.746+2, 1.0+2, 400)
mu_i = 1.173 * ((1 - nu_c)**0.5 - 0.254**0.5) * 56.1 / 4.0 * np.sqrt(18) / param - 1.541 + offset

nu_j = np.linspace(0.0-1, 0.262-1, 400)
mu_j = -1.173 * (nu_a**0.5 - 0.262**0.5) * 56.1 / 4.0 * np.sqrt(18) / param + 2.096 + offset

mask_k = (mu_paper[:, 0] >= 0.262) & (mu_paper[:, 0] <= 0.746)
nu_k = mu_paper[mask_b, 0] - 1
mu_k = mu_paper[mask_b, 1] + offset

nu_l = np.linspace(0.746-1, 1.0-1, 400)
mu_l = 1.173 * ((1 - nu_c)**0.5 - 0.254**0.5) * 56.1 / 4.0 * np.sqrt(18) / param - 1.541 + offset


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
np.savetxt("/Users/rikutofushio/00ARIP/Thomas-Fermi-Folder/Thomas-Fermi/data/0-data/Exc_BLG_crude.csv", out, delimiter=",", header="nu,mu_meV,E_meV", comments="")

# Plot μ and E
plt.figure(figsize=(6, 6))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(nu_all, mu_all, label=r"$\mu(\nu)$")
ax1.set_xlabel(r"$\nu$")
ax1.set_ylabel(r"$\mu$ [meV]")
#ax1.set_xlim(0.0, 1.0)
ax1.legend()

ax2 = plt.subplot(2, 1, 2)
ax2.plot(nu_all, E_all, label=r"$E(\nu)=\int_0^\nu \mu(\nu') d\nu'$")
ax2.set_xlabel(r"$\nu$")
ax2.set_ylabel(r"$E$ [meV]")
#ax2.set_xlim(0.0, 1.0)
ax2.legend()

plt.tight_layout()
plt.savefig("mu_integrated.png", dpi=200)