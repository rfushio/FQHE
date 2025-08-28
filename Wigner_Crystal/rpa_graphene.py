"""
Static RPA for monolayer graphene in B using Dirac-Landau-level transition sums (per Shizuya 2007).

Provides:
- associated_laguerre(n, alpha, x)
- gaas_form_factor_sq(n, m, x)
- graphene_form_factor_sq_dirac(n, m, x)
- epsilon_rpa_dirac(q_tilde, nu0, alpha_G, gate_factor, Lmax_list)
- v_rpa_over_EC(q_tilde, nu0, alpha_G, gate_factor, Lmax_list)

Notes:
- q_tilde = q * lB (dimensionless). x = q_tilde^2/2.
- LL indices n, m ∈ {..., -2, -1, 0, +1, +2, ...}.
- Graphene Dirac LL energies scale as e_n = sgn(n) * sqrt(2|n|); overall ℏ v_F/ℓ_B scale
  cancels in the dimensionless combination entering ε(q) when expressed via α_G = e^2/(ε ℏ v_F).
- Occupations per isospin: ν_m = 1 (m<0), ν_0 = nu0 ∈ [0,1], ν_m = 0 (m>0).
- The per-isospin polarizability is summed over all inter-LL transitions with weight
  (ν_m - ν_n)/(e_n - e_m) * |F_{mn}^{gr}(q)|^2.
- Total ε includes isospin multiplicity factor g_iso (default 4).
- Gate factor multiplies the bare 2D Coulomb potential as V_gate/E_C = (2π/q_tilde) * gate(q).
- For computational stability, we support multiple Λ cutoffs and perform a simple extrapolation
  in Λ^{-1/2} to Λ→∞.

Caveat:
- Exact prefactors and special cases should follow the supplemental (S6–S12). We implement a
  standard Shizuya-like expression; if you provide S6–S12, we can match it one-to-one.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple
import math
import numpy as np

Array = np.ndarray


def associated_laguerre(n: int, alpha: int, x: Array) -> Array:
    """
    L_n^{alpha}(x) for integer n>=0, alpha>=0 via explicit series.
    """
    if n < 0 or alpha < 0:
        raise ValueError("n, alpha must be >= 0")
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    # L_n^{alpha}(x) = sum_{k=0}^n C(n+alpha, n-k) * (-x)^k / k!
    # with C(a,b) binomial.
    out = np.zeros_like(x)
    for k in range(0, n + 1):
        coeff = math.comb(n + alpha, n - k) / math.factorial(k)
        term = coeff * ((-1.0) ** k) * (x ** k)
        out = out + term
    return out


def _gaas_form_factor_sq(n: int, m: int, x: Array) -> Array:
    """
    Isotropic (angle-averaged) GaAs LL density form-factor squared: |f_{n,m}(q)|^2.
    Uses x = q_tilde^2/2, with q_tilde = q lB.
    """
    x = np.asarray(x, dtype=float)
    a = min(n, m)
    b = max(n, m)
    delta = b - a
    if delta == 0:
        # |f_{n,n}|^2 = e^{-x} [L_n(x)]^2
        Ln = associated_laguerre(n, 0, x)
        return np.exp(-x) * (Ln ** 2)
    # general case: e^{-x} (a!/b!) x^{delta} [L_a^{delta}(x)]^2
    ln_ratio = math.lgamma(a + 1) - math.lgamma(b + 1)
    ratio = math.exp(ln_ratio)
    La = associated_laguerre(a, delta, x)
    return np.exp(-x) * ratio * (x ** delta) * (La ** 2)


def graphene_form_factor_sq_dirac(n: int, m: int, x: Array) -> Array:
    """
    Graphene Dirac density form-factor squared between LL n and m (same valley),
    angle-averaged. Based on spinor composition:
      - if n,m >= 1: |F|^2 ≈ (1/4)( |f_{n,m}|^2 + |f_{n-1,m-1}|^2 )
      - if one index is 0 and the other >=1: |F|^2 ≈ (1/2)|f_{|n|,|m|}|^2
      - if n=m=0: |F|^2 = |f_{0,0}|^2 = e^{-x}
    where f_{a,b} are GaAs form factors with nonrelativistic indices a,b >=0.
    """
    x = np.asarray(x, dtype=float)
    an = abs(n)
    am = abs(m)
    if an == 0 and am == 0:
        # f_{0,0}
        return np.exp(-x)
    if an == 0 and am >= 1:
        return 0.5 * _gaas_form_factor_sq(am, 0, x)
    if am == 0 and an >= 1:
        return 0.5 * _gaas_form_factor_sq(an, 0, x)
    # both >=1
    return 0.25 * (_gaas_form_factor_sq(an, am, x) + _gaas_form_factor_sq(an - 1, am - 1, x))


def _dirac_e(n: int) -> float:
    """
    Dimensionless Dirac LL energy e_n = sgn(n) * sqrt(2|n|), with e_0 = 0.
    The overall ℏ v_F / lB scale cancels once expressed via α_G in ε(q).
    """
    if n == 0:
        return 0.0
    return math.copysign(math.sqrt(2.0 * abs(n)), n)


def _nu_occ(n: int, nu0: float) -> float:
    """Per-isospin occupation: 1 for n<0, nu0 for n=0, 0 for n>0."""
    if n < 0:
        return 1.0
    if n == 0:
        return float(nu0)
    return 0.0


def _polarizability_per_isospin(q_tilde: Array, nu0: float, Lmax: int) -> Array:
    """
    Static polarizability Π_ν(q) per isospin in dimensionless form suitable for RPA via α_G.
    We compute S(q) = sum_{m,n} (ν_m - ν_n) |F_{mn}(q)|^2 / (e_n - e_m), excluding n=m.
    Return S(q) as a function of q_tilde (dimensionless). The full dielectric enters as
      ε(q) = 1 - (π/2) α_G * S(q) * gate(q) / (q_tilde)
    up to a known numerical prefactor; see the caveat in the module docstring.
    """
    q_tilde = np.asarray(q_tilde, dtype=float)
    x = 0.5 * q_tilde ** 2
    # Prepare index list
    idxs: List[int] = list(range(-Lmax, Lmax + 1))
    # Exclude non-physical n beyond available bands? Dirac model supports all.

    # Precompute e_n and ν_n
    e = {n: _dirac_e(n) for n in idxs}
    nu = {n: _nu_occ(n, nu0) for n in idxs}

    S = np.zeros_like(q_tilde)
    for m in idxs:
        for n in idxs:
            if n == m:
                continue
            denom = e[n] - e[m]
            if denom == 0.0:
                continue
            w = (nu[m] - nu[n]) / denom
            F2 = graphene_form_factor_sq_dirac(n, m, x)
            S = S + w * F2
    # Normalize by 2π to match density-of-states factor; absorbed later via α_G prefactor.
    return S


def epsilon_rpa_dirac(
    q_tilde: Array,
    *,
    nu0: float,
    alpha_G: float,
    gate_factor: Array | float,
    Lmax_list: Iterable[int] = (60, 100, 140),
    g_isospin: int = 4,
) -> Array:
    """
    RPA dielectric ε(q) via Dirac LL transition sums with simple Λ-extrapolation.

    Args:
        q_tilde: array of q*lB values
        nu0: occupation of the N=0 LL per isospin (0..1); screening weakly depends on it in ZLL
        alpha_G: graphene fine-structure constant
        gate_factor: scalar or array of gate reduction factor multiplying V(q) (e.g., tanh(q d))
        Lmax_list: sequence of cutoffs for extrapolation in Λ^{-1/2}
        g_isospin: number of isospins (default 4)
    """
    q_tilde = np.asarray(q_tilde, dtype=float)
    gate = np.asarray(gate_factor, dtype=float) if isinstance(gate_factor, np.ndarray) else gate_factor

    # Compute S(q) for multiple Λ and extrapolate S(Λ→∞)(q) = a + b Λ^{-1/2} + c Λ^{-1}
    Ss = []
    Xs = []
    for Lmax in Lmax_list:
        S = _polarizability_per_isospin(q_tilde, nu0, Lmax)
        Ss.append(S)
        Xs.append(1.0 / math.sqrt(Lmax))
    Ss = np.stack(Ss, axis=0)  # shape (K, Q)
    X = np.asarray(Xs)[:, None]  # shape (K, 1)

    # Fit quadratic in X: S ≈ a + b X + c X^2; take a as extrapolated limit
    # Solve least squares for each q independently
    K = Ss.shape[0]
    A = np.concatenate([np.ones((K, 1)), X, X ** 2], axis=1)  # (K,3)
    # Normal equation per q
    AtA = A.T @ A  # (3,3)
    AtA_inv = np.linalg.inv(AtA)
    At = A.T  # (3,K)
    coeffs = (AtA_inv @ At @ Ss).T  # (Q,3): columns [a,b,c]
    a = coeffs[:, 0]  # (Q,)

    # Dielectric: ε(q) = 1 - g_iso * C * α_G * gate * S_extrap / q_tilde
    # The constant C is set to π/2 to match long-wavelength behavior in literature.
    C = 0.5 * math.pi
    eps = 1.0 - g_isospin * C * alpha_G * (a / np.maximum(q_tilde, 1e-9)) * gate
    return eps


def v_rpa_over_EC(
    q_tilde: Array,
    *,
    nu0: float,
    alpha_G: float,
    gate_factor: Array | float,
    Lmax_list: Iterable[int] = (60, 100, 140),
) -> Array:
    """
    V_RPA(q)/E_C including gates, for use in V_eff = V_RPA * |F_N|^2.
    Bare V/E_C = 2π / q_tilde. Apply gate factor and divide by ε(q).
    """
    q_tilde = np.asarray(q_tilde, dtype=float)
    eps = epsilon_rpa_dirac(q_tilde, nu0=nu0, alpha_G=alpha_G, gate_factor=gate_factor, Lmax_list=Lmax_list)
    Vbare_over_EC = (2.0 * math.pi / np.maximum(q_tilde, 1e-9))
    if isinstance(gate_factor, np.ndarray):
        Vbare_over_EC = Vbare_over_EC * gate_factor
    else:
        Vbare_over_EC = Vbare_over_EC * float(gate_factor)
    return Vbare_over_EC / eps


__all__ = [
    "associated_laguerre",
    "gaas_form_factor_sq",
    "graphene_form_factor_sq_dirac",
    "epsilon_rpa_dirac",
    "v_rpa_over_EC",
]
