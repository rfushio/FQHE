"""
Interaction utilities for Wigner crystal calculations.

Provides:
- lB_nm(B_T): magnetic length (nm)
- E_C_meV(B_T, epsilon_hbn): Coulomb energy scale in meV
- form_factor_F_N(q_tilde, N): Landau level form factor (q_tilde = q*lB)
- gate_factor_tanh(q_tilde, d_over_lB): symmetric two-gate approximation
- epsilon_RPA_constant(alpha_G): simple static RPA dielectric for graphene
- epsilon_RPA_q(q_tilde, alpha_G, model): q-dependent RPA dielectric (exp/const)
- Veff_over_EC(q_tilde, N, d_over_lB, alpha_G, rpa_model): screened V_eff(q)/E_C including |F_N|^2

Notes:
- We work with dimensionless momentum q_tilde = q * lB.
- With this convention, the bare Coulomb in E_C units is V/E_C = 2π / q_tilde.
- Gate screening is approximated as tanh(q_tilde * d_over_lB).
- RPA dielectric: default is a q-dependent model ε_RPA(q) = 1 + (π/2) α_G e^{-q_tilde^2/2},
  which captures strong long-wavelength screening and reduced screening at large q.
- LL form factor: graphene-specific choice for N=1:
  F_0(q) = e^{-x/2},  F_1^gr(q) = e^{-x/2} * (1 - x/2),  where x = (q*lB)^2 / 2.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Iterable

from rpa_graphene import v_rpa_over_EC as v_rpa_over_EC_dirac

Array = np.ndarray


def lB_nm(B_T: float) -> float:
    """Magnetic length in nm: lB[nm] = 25.66 / sqrt(B[T])."""
    if B_T <= 0:
        raise ValueError("B_T must be positive.")
    return 25.66 / math.sqrt(B_T)


def E_C_meV(B_T: float, epsilon_hbn: float) -> float:
    """E_C [meV] = 56.1 * sqrt(B[T]) / epsilon_hbn."""
    if epsilon_hbn <= 0:
        raise ValueError("epsilon_hbn must be positive.")
    return 56.1 * math.sqrt(B_T) / epsilon_hbn


def form_factor_F_N(q_tilde: Array, N: int) -> Array:
    """
    Graphene Landau-level form factor for density-density interaction.
    Define x = (q*lB)^2 / 2. For graphene:
      F_0(q) = e^{-x/2}
      F_1(q) = e^{-x/2} * (1 - x/2)
    """
    q_tilde = np.asarray(q_tilde, dtype=float)
    x = 0.5 * q_tilde ** 2
    if N == 0:
        return np.exp(-0.5 * x)  # e^{-x/2}
    elif N == 1:
        return np.exp(-0.5 * x) * (1.0 - 0.5 * x)
    else:
        raise NotImplementedError("Form factor implemented for graphene N=0 or N=1 only.")


def gate_factor_tanh(q_tilde: Array, d_over_lB: float) -> Array:
    """
    Symmetric two-gate reduction factor approximated by tanh(q d).
    Here q_tilde = q*lB, and d_over_lB = d / lB.
    For a single gate at distance d, the exact factor is (1 - e^{-2 q d}).
    For two symmetric gates, tanh(q d) captures the main suppression at small q.
    """
    if d_over_lB < 0:
        raise ValueError("d_over_lB must be non-negative.")
    q_tilde = np.asarray(q_tilde, dtype=float)
    return np.tanh(q_tilde * d_over_lB)


def epsilon_RPA_constant(alpha_G: float) -> float:
    """Static RPA dielectric approximation: ε_RPA = 1 + (π/2) α_G."""
    if alpha_G < 0:
        raise ValueError("alpha_G must be non-negative.")
    return 1.0 + 0.5 * math.pi * alpha_G


def epsilon_RPA_q(q_tilde: Array, alpha_G: float, model: str = "exp") -> Array:
    """
    q-dependent static RPA dielectric.
    Models:
      - "exp": ε(q) = 1 + (π/2) α_G exp(-q_tilde^2 / 2)
      - "const": ε independent of q (long-wavelength limit)
    """
    q_tilde = np.asarray(q_tilde, dtype=float)
    if model == "const":
        return np.full_like(q_tilde, epsilon_RPA_constant(alpha_G), dtype=float)
    elif model == "exp":
        return 1.0 + 0.5 * math.pi * alpha_G * np.exp(-0.5 * q_tilde ** 2)
    else:
        raise ValueError("Unknown RPA model: %s" % model)


def Veff_over_EC(
    q_tilde: Array,
    *,
    N: int = 0,
    d_over_lB: float = 0.0,
    alpha_G: float = 0.0,
    rpa_model: str = "exp",
) -> Array:
    """
    Screened effective interaction in E_C units:
        V_eff(q)/E_C = [ 2π / q_tilde * gate_factor(q_tilde) / ε_RPA(q) ] * |F_N(q_tilde)|^2

    Args:
        q_tilde: dimensionless momentum q*lB (array)
        N: Landau level index (0 or 1 supported)
        d_over_lB: gate distance in units of lB (use ~ d_nm / lB_nm)
        alpha_G: graphene fine-structure constant (e.g., 1.85)
        rpa_model: "exp" (default) or "const"
    """
    q_tilde = np.asarray(q_tilde, dtype=float)
    out = np.empty_like(q_tilde)

    # Avoid division by zero at q=0; not sampled in reciprocal lattice sum (G≠0)
    mask = q_tilde > 0
    F = form_factor_F_N(q_tilde[mask], N)
    gate = gate_factor_tanh(q_tilde[mask], d_over_lB) if d_over_lB > 0 else 1.0
    eps_rpa = epsilon_RPA_q(q_tilde[mask], alpha_G, model=rpa_model)

    out[mask] = (2.0 * math.pi / q_tilde[mask]) * (gate / eps_rpa) * (np.abs(F) ** 2)

    # At q=0, set a large finite placeholder; not actually used in sums/integrals
    out[~mask] = 2.0 * math.pi * 1e9
    return out


def Veff_over_EC_diracRPA(
    q_tilde: Array,
    *,
    N: int = 0,
    d_over_lB: float = 0.0,
    alpha_G: float = 1.85,
    nu0: float = 0.0,
    Lmax_list: Iterable[int] = (60, 100, 140),
) -> Array:
    """
    Effective interaction in E_C units using Dirac-LL RPA dielectric and gate factor,
    then multiplied by graphene LL form factor |F_N|^2.
    """
    q_tilde = np.asarray(q_tilde, dtype=float)
    F = form_factor_F_N(q_tilde, N)
    gate = gate_factor_tanh(q_tilde, d_over_lB) if d_over_lB > 0 else 1.0
    Vrpa = v_rpa_over_EC_dirac(q_tilde, nu0=nu0, alpha_G=alpha_G, gate_factor=gate, Lmax_list=Lmax_list)
    return Vrpa * (np.abs(F) ** 2)


__all__ = [
    "lB_nm",
    "E_C_meV",
    "form_factor_F_N",
    "gate_factor_tanh",
    "epsilon_RPA_constant",
    "epsilon_RPA_q",
    "Veff_over_EC",
]
__all__.append("Veff_over_EC_diracRPA")
