"""
Plot E (meV) for a Wigner crystal (triangular lattice) using the S.13 reciprocal-space sum
with screened interaction V_eff(q) = V_RPA(q) |F_N(q)|^2, and compare to the user's
reference curve (left unchanged).

Fixes:
- Treat energy_per_electron(...) output as per-electron energy; plot energy per flux E(ν)=ν·ε.
- Adapt the reciprocal cutoff nmax(ν) so that max |G| ≳ target_qmax, improving
  cancellation with the self-term at small ν and avoiding spurious offsets.
- Use graphene-specific N=1 form factor when N=1 is selected.
- Use Dirac-LL RPA (transition-sum with Λ→∞ extrapolation) for ε(q), per Shizuya-like formula.

Conventions:
- Work internally in E_C units with lB = 1, then convert to meV at the end.
- energy_per_electron returns per-electron energy in E_C units when Vq returns E_C units.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from s13 import energy_per_electron, triangular_lattice_parameters
from interactions import E_C_meV, gate_factor_tanh, Veff_over_EC_diracRPA


def estimate_nmax_for_target_qmax(nu: float, target_qmax: float) -> int:
    # lB = 1 unitless inside lattice parameter function
    _, b1, b2 = triangular_lattice_parameters(nu, 1.0)
    bmag = float(np.linalg.norm(b1))
    n_est = int(np.ceil(target_qmax / (bmag * np.sqrt(3.0))))
    return max(n_est, 10)


def main():
    # Physical parameters
    B_T = 18.0
    epsilon_hbn = 4.0

    # Choose LL index (0 for ZLL; 1 for first excited)
    N = 0

    d_nm = 40.0       # gate distance (per side) in nm (approximate)

    # Dimensionless setup
    lB = 1.0
    lB_nm = 25.66 / np.sqrt(B_T)
    d_over_lB = d_nm / lB_nm

    alpha_G = 1.85
    nu0 = 0.0  # ZLL per-isospin occupation away from integers ~0 in Wigner crystal regime
    Lmax_list = (60, 100, 140)

    # ν range
    nus = np.linspace(1e-4, 0.2, 60)

    # Target q-extent for reciprocal sum (dimensionless q = q*lB)
    target_qmax = 12.0

    # Self-term radial integral upper bound (dimensionless q)
    self_qmax = 12.0

    # Energy scale E_C in meV
    EC_meV = E_C_meV(B_T, epsilon_hbn)

    # Gate factor array for plotting/computation convenience
    q_probe = np.linspace(1e-3, self_qmax, 2048)
    gate_arr = gate_factor_tanh(q_probe, d_over_lB) if d_over_lB > 0 else 1.0

    E_meV = []
    for nu in nus:
        nmax = estimate_nmax_for_target_qmax(nu, target_qmax)
        # Build V_eff/E_C using Dirac RPA with extrapolation
        Vq = lambda q: Veff_over_EC_diracRPA(
            q, N=N, d_over_lB=d_over_lB, alpha_G=alpha_G, nu0=nu0, Lmax_list=Lmax_list
        )
        eps = energy_per_electron(
            nu_star=nu,
            lB=lB,
            Vq=Vq,
            nmax=nmax,
            include_self=True,
            self_qmax=self_qmax,
            self_num_points=30000,
            sort_by_shell=True,
            return_series=False,
        )
        E_meV.append((nu * eps) * EC_meV)

    E_meV = np.array(E_meV)

    # Quick sanity print at ν=0.2
    idx = np.argmin(np.abs(nus - 0.2))
    print(f"nu=0.2: E ≈ {E_meV[idx]:.3f} meV")

    # User's reference (energy per flux)
    E_ref = -0.7821 * nus**1.5 * 56.1 / 4.0 * np.sqrt(18)

    plt.figure(figsize=(6, 4))
    plt.plot(nus, E_meV, label=f"Numeric screened (Dirac RPA, N={N}, αG={alpha_G}, d={d_nm}nm)")
    plt.plot(nus, E_ref, "--", label=r"Ref: $E=-0.7821\sqrt{\nu}E_C$")
    plt.xlabel(r"$\nu$")
    plt.xlim(0.0, 0.22)
    plt.ylabel(r"$E$ [meV]")
    plt.title("Wigner crystal energy (screened)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("wc_screened_compare.png", dpi=200)
    # plt.show()


if __name__ == "__main__":
    main()
