"""
Wigner crystal (triangular lattice) energy per electron per S.13.

Energy per electron (epsilon) is implemented in reciprocal space as:
  epsilon = 1/2 * [ (nu_star / (2*pi)) * sum_{G != 0} V_eff(|G|)  -  V_self ]
where V_self = V_eff(r=0) = (1/(2*pi)) * ∫_0^∞ q V_eff(q) dq,
provided the chosen V_eff(q) decays sufficiently fast at large q
(e.g., due to Landau-level form factors) so the integral converges.

This module does not assume a specific V_eff(q). You must supply V_eff(q)
as a Python callable returning energy units consistent with your problem.

Notation:
- nu_star: partial filling (N_e/N_Φ in the partially filled LL)
- lB: magnetic length (same length unit used for q^{-1})
- G-vectors: reciprocal lattice of a triangular Bravais lattice with density
  ρ = nu_star / (2π lB^2). Lattice constant a and reciprocal basis {b1, b2}
  follow from ρ.

Example usage (with a dummy V_eff):

    import numpy as np
    from s13 import energy_per_electron

    lB = 10.0  # nm (example)
    nu_star = 0.05

    def Vq(q):
        # Example placeholder: Gaussian-decaying potential in q
        return np.exp(-0.5*(q*lB)**2)

    eps = energy_per_electron(nu_star, lB, Vq, nmax=20, include_self=False)
    print("epsilon (arb. units) =", eps)

"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple
import math
import numpy as np

Array = np.ndarray


def triangular_lattice_parameters(nu_star: float, lB: float) -> Tuple[float, Array, Array]:
    """
    Compute triangular lattice constant a and reciprocal basis (b1, b2)
    for a 2D electron crystal at density ρ = nu_star / (2π lB^2).

    Returns:
        a: real-space lattice constant
        b1, b2: 2D reciprocal lattice basis vectors
    """
    if nu_star <= 0:
        raise ValueError("nu_star must be positive for a Wigner crystal.")
    if lB <= 0:
        raise ValueError("lB must be positive.")

    # Area per electron (unit cell area) for density ρ
    rho = nu_star / (2.0 * math.pi * lB * lB)
    acell = 1.0 / rho
    # Triangular lattice: (sqrt(3)/2) a^2 = A_cell
    a = math.sqrt(2.0 * acell / math.sqrt(3.0))

    # Reciprocal basis (one convenient choice)
    # |b1| = |b2| = 4π / (√3 a); angle 60° between them.
    two_pi_over_a = 2.0 * math.pi / a
    b1 = np.array([two_pi_over_a, -two_pi_over_a / math.sqrt(3.0)], dtype=float)
    b2 = np.array([0.0, 2.0 * two_pi_over_a / math.sqrt(3.0)], dtype=float)
    return a, b1, b2


def enumerate_reciprocal_vectors(b1: Array, b2: Array, nmax: int) -> Tuple[Array, Array, Array]:
    """
    Enumerate reciprocal lattice vectors G = m b1 + n b2 for m,n in [-nmax, nmax].

    Returns:
        Gvecs: (N, 2) array of G vectors (excluding G=0)
        Gnorms: (N,) array of |G|
        Qvals: (N,) array of shell indices Q = m^2 + n^2 - m n (triangular metric)
    """
    if nmax < 1:
        raise ValueError("nmax must be >= 1 to enumerate nonzero reciprocal vectors.")

    ms = range(-nmax, nmax + 1)
    ns = range(-nmax, nmax + 1)
    g_list: List[Tuple[float, float]] = []
    q_list: List[int] = []

    for m in ms:
        for n in ns:
            if m == 0 and n == 0:
                continue
            Gx = m * b1[0] + n * b2[0]
            Gy = m * b1[1] + n * b2[1]
            g_list.append((Gx, Gy))
            # Triangular-shell index (integer):
            q_list.append(m * m + n * n - m * n)

    Gvecs = np.asarray(g_list, dtype=float)
    Gnorms = np.linalg.norm(Gvecs, axis=1)
    Qvals = np.asarray(q_list, dtype=int)
    return Gvecs, Gnorms, Qvals


def _self_term_from_radial_integral(
    Vq: Callable[[Array], Array],
    qmax: float,
    num_points: int = 20000,
    qmin: Optional[float] = None,
) -> float:
    """
    Compute V_self = (1/(2π)) * ∫_0^{∞} q V(q) dq by numerical integration up to qmax.

    Args:
        Vq: callable on numpy arrays; returns V(q) in energy units
        qmax: upper limit of radial integral; choose large enough so tail is negligible
        num_points: number of integration samples
        qmin: optional small positive lower bound to avoid q=0 evaluation

    Returns:
        V_self (float)
    """
    if qmax <= 0:
        raise ValueError("qmax must be positive for self-term integral.")
    if num_points < 10:
        raise ValueError("num_points too small for self-term integral.")

    if qmin is None:
        qmin = qmax * 1e-9
    q = np.linspace(qmin, qmax, num_points)
    integrand = q * Vq(q)
    val = (1.0 / (2.0 * math.pi)) * np.trapz(integrand, q)
    return float(val)


def energy_per_electron(
    nu_star: float,
    lB: float,
    Vq: Callable[[Array], Array],
    nmax: int = 20,
    include_self: bool = False,
    self_qmax: Optional[float] = None,
    self_num_points: int = 20000,
    sort_by_shell: bool = True,
    return_series: bool = False,
) -> float | Tuple[float, Array]:
    """
    Evaluate epsilon(nu_star) via S.13 in reciprocal space for a triangular crystal.

    Args:
        nu_star: partial filling (N_e/N_Φ in the active LL). Must be > 0.
        lB: magnetic length (same length unit as used in q^{-1}). Must be > 0.
        Vq: callable returning V_eff(q) for scalar or numpy array q (1/length)
        nmax: reciprocal enumeration bound; uses m,n in [-nmax, nmax] \ {(0,0)}
        include_self: if True, subtract V_self computed via radial integral
        self_qmax: radial integral upper bound; default set from max |G|
        self_num_points: samples for radial integral
        sort_by_shell: if True, sort G by the triangular shell index Q=m^2+n^2-mn
        return_series: if True, also return partial sums after each G-addition

    Returns:
        epsilon (float) or (epsilon, series) where series[k] is epsilon using first k terms
        of the G-sum (useful for convergence diagnostics). When include_self=True,
        the self-term is subtracted only at the end (series excludes the self-term).
    """
    a, b1, b2 = triangular_lattice_parameters(nu_star, lB)
    _, Gnorms, Qvals = enumerate_reciprocal_vectors(b1, b2, nmax)

    # Order G-vectors for convergence (by shell or by magnitude)
    if sort_by_shell:
        order = np.argsort(Qvals, kind="mergesort")
    else:
        order = np.argsort(Gnorms, kind="mergesort")

    Gsorted = Gnorms[order]

    # Evaluate V_eff(q) on the ordered list
    Vvals = Vq(Gsorted)
    if not isinstance(Vvals, np.ndarray):
        # ensure numpy array broadcast behavior if Vq returns scalar for scalar input
        Vvals = np.asarray(Vvals, dtype=float)

    # Partial sums of the reciprocal term
    pref = nu_star / (2.0 * math.pi)
    partial = 0.5 * pref * np.cumsum(Vvals)

    # Self-term (subtract at the end only)
    Vself = 0.0
    if include_self:
        if self_qmax is None:
            # heuristic: integrate well beyond the largest |G| enumerated
            # max |G| for given nmax can be approximated by the last in Gsorted
            if Gsorted.size == 0:
                raise ValueError("No G-vectors enumerated; increase nmax.")
            self_qmax = float(Gsorted.max() * 2.0)
        Vself = _self_term_from_radial_integral(Vq, qmax=self_qmax, num_points=self_num_points)

    epsilon = float(partial[-1] - 0.5 * Vself)

    if return_series:
        return epsilon, partial  # note: series excludes self-term subtraction
    return epsilon


__all__ = [
    "triangular_lattice_parameters",
    "enumerate_reciprocal_vectors",
    "energy_per_electron",
]
