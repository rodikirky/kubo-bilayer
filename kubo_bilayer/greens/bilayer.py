""" bilayer.py
    ----------
    Construction of the full cross-interface retarded Green's function
    for z > 0, z' < 0, by combining the propagation factors F^r_R(z)
    and F̄^r_L(z') with the coincidence value G(0,0) from interface.py.
    #TODO: All in one half-space extension to be added once halspace.py is implemented.

    Physical Background
    -------------------
    For z > 0 and z' < 0, the full interfacial retarded Green's
    function factorizes into three terms (eq. 54 in supervisor's notes):

        G^r(z,z') = F^r_R(z) · G(0,0) · F̄^r_L(z')

    where the propagation factors are defined as (eqs. 52, 53):

        F^r_R(z)  = -G^(0)r_R(z,0) · [G^(0)r_R(0,0)]^{-1}
        F̄^r_L(z') = -[G^(0)r_L(0,0)]^{-1} · G^(0)r_L(0,z')

    and G^(0)r_{R/L}(z,0) are the translational invariant bulk
    Green's functions evaluated via bulk.py.

    Propagation Factors
    -------------------
    Expanding in terms of poles and residues:

        F^r_R(z)  = -[Σ_α e^(i·kαR·z) · ResαR] · [G^r_R(0)]^{-1}
                = -(1/i) · evaluate(z, poles_R, residues_R, 'upper')
                        · [coincidence_value(residues_R, 'upper')]^{-1}

        F̄^r_L(z') = -[G^r_L(0)]^{-1} · [Σ_α e^(i·kαL·z') · ResαL]
                = -(1/i) · [coincidence_value(residues_L, 'lower')]^{-1}
                        · evaluate(z', poles_L, residues_L, 'lower') · (-1)

    Note the sign conventions:
        - F^r_R(z)  requires z  > 0, upper half-plane poles
        - F̄^r_L(z') requires z' < 0, lower half-plane poles
        - Both carry a leading minus sign from eqs. 52, 53

    Functions
    ---------
    compute_F_R(z, poles_R, residues_R)
        Computes the right propagation factor F^r_R(z) for z > 0:

            F^r_R(z) = -G^(0)r_R(z,0) · [G^(0)r_R(0,0)]^{-1}

    compute_F_bar_L(z_prime, poles_L, residues_L)
        Computes the left propagation factor F̄^r_L(z') for z' < 0:

            F̄^r_L(z') = -[G^(0)r_L(0,0)]^{-1} · G^(0)r_L(0,z')

    compute_G_bilayer(z, z_prime, poles_R, residues_R,
                    poles_L, residues_L, G00)
        Assembles the full cross-interface Green's function:

            G^r(z,z') = F^r_R(z) · G(0,0) · F̄^r_L(z')

        for z > 0, z' < 0.

    Notes
    -----
    - z must be strictly positive and z' must be strictly negative.
    These constraints are enforced by evaluate() in bulk.py.
    - The half-space contributions Ḡ^r_L and Ḡ^r_R vanish for
    z > 0, z' < 0 and are therefore not included here. See
    halfspace.py for their implementation (TODO).
    - G00 must have been computed at the same (kx, ky, omega, eta)
    as the poles and residues passed here.

    Dependencies
    ------------
        bulk.py      — provides evaluate() and coincidence_value()
        interface.py — provides assemble_G00()

    Usage
    ------
    - Plot the full spatial Green's function
    - Provide a pedagogical illustration of the matching method

# TODO: Implement compute_F_R, compute_F_bar_L, compute_G_bilayer
"""
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from kubo_bilayer.greens.bulk import *

ArrayC = NDArray[np.complex128]

__all__ = [
    "ArrayC",
    "compute_F_R",
    "compute_F_bar_L",
    "compute_G_bilayer"
]

def compute_F_R(
    z: float,
    poles_R: ArrayC,
    residues_R: list[ArrayC],
) -> ArrayC:
    """
    Compute the right propagation factor F^r_R(z) for z > 0:

        F^r_R(z) = -G^(0)r_R(z,0) · [G^(0)r_R(0,0)]^{-1}

    Parameters
    ----------
    z          : float, must be strictly positive
    poles_R    : ArrayC, shape (k,)
        Upper half-plane poles from compute_poles(..., halfplane='upper')
    residues_R : list[ArrayC], length k
        Residues from compute_residues()

    Returns
    -------
    F_R : ArrayC, shape (n, n)
    """
    G_z = evaluate(z, poles_R, residues_R, halfplane='upper')
    G_0 = coincidence_value(residues_R, halfplane='upper')
    return -G_z @ np.linalg.inv(G_0)


def compute_F_bar_L(
    z_prime: float,
    poles_L: ArrayC,
    residues_L: list[ArrayC],
) -> ArrayC:
    """
    Compute the left propagation factor F̄^r_L(z') for z' < 0:

        F̄^r_L(z') = -[G^(0)r_L(0,0)]^{-1} · G^(0)r_L(0,z')

    Note that G^(0)r_L(0,z') = [G^(0)r_L(z',0)]† for the
    retarded function, so this is computed as the conjugate
    transpose of evaluate(z', ..., halfplane='lower').

    Parameters
    ----------
    z_prime    : float, must be strictly negative
    poles_L    : ArrayC, shape (k,)
        Lower half-plane poles from compute_poles(..., halfplane='lower')
    residues_L : list[ArrayC], length k
        Residues from compute_residues()

    Returns
    -------
    F_bar_L : ArrayC, shape (n, n)
    """
    G_zprime = evaluate(z_prime, poles_L, residues_L, halfplane='lower')
    G_0 = coincidence_value(residues_L, halfplane='lower')
    return -np.linalg.inv(G_0) @ G_zprime


def compute_G_bilayer(
    z: float,
    z_prime: float,
    poles_R: ArrayC,
    residues_R: list[ArrayC],
    poles_L: ArrayC,
    residues_L: list[ArrayC],
    G00: ArrayC,
) -> ArrayC:
    """
    Assemble the full cross-interface retarded Green's function
    for z > 0, z' < 0:

        G^r(z,z') = F^r_R(z) · G(0,0) · F̄^r_L(z')

    Parameters
    ----------
    z          : float, must be strictly positive
    z_prime    : float, must be strictly negative
    poles_R    : ArrayC, shape (k,)
        Upper half-plane poles
    residues_R : list[ArrayC], length k
    poles_L    : ArrayC, shape (k,)
        Lower half-plane poles
    residues_L : list[ArrayC], length k
    G00        : ArrayC, shape (n, n)
        Coincidence value from assemble_G00()

    Returns
    -------
    G : ArrayC, shape (n, n)
    """
    if z <= 0:
        raise ValueError(
            f"z must be strictly positive, got {z}."
        )
    if z_prime >= 0:
        raise ValueError(
            f"z_prime must be strictly negative, got {z_prime}."
        )
    F_R = compute_F_R(z, poles_R, residues_R)
    F_bar_L = compute_F_bar_L(z_prime, poles_L, residues_L)
    return F_R @ G00 @ F_bar_L