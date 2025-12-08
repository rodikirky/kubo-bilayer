from __future__ import annotations

from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray

ArrayC = NDArray[np.complex128]
Greens1D = Callable[[float, float], ArrayC]  # G(z, z') -> (n, n) matrix


def _matrix_inverse(M: ArrayC) -> ArrayC:
    """Small helper to invert an (n, n) complex matrix."""
    M = np.asarray(M, dtype=np.complex128)
    return np.linalg.inv(M)


def _build_F_side(
    G_side: Greens1D,
) -> Tuple[Callable[[float], ArrayC], Callable[[float], ArrayC]]:
    """
    Build F_side(z) and F_side_bar(z') for one half-space:

        F_side(z)      = - G_side(z, 0) @ G_side(0, 0)^(-1)
        F_side_bar(z') = - G_side(0, 0)^(-1) @ G_side(0, z')
    """
    G00 = G_side(0.0, 0.0)
    inv_G00 = _matrix_inverse(G00)

    def F(z: float) -> ArrayC:
        return -G_side(z, 0.0) @ inv_G00

    def F_bar(zp: float) -> ArrayC:
        return -inv_G00 @ G_side(0.0, zp)

    return F, F_bar


def _log_derivatives_at_interface(
    F_L: Callable[[float], ArrayC],
    F_R: Callable[[float], ArrayC],
    m_L: float,
    m_R: float,
    dz: float,
) -> Tuple[ArrayC, ArrayC]:
    """
    Approximate log-derivative matrices L_L and L_R at z=0 from the left/right.

    Continuum formula (from your note):
        L_side = F_side'(0_side) / (2 m_side)

    We approximate derivatives with simple one-sided finite differences:

        F_L'(0-) ≈ (F_L(-dz) - F_L(-2 dz)) / dz
        F_R'(0+) ≈ (F_R(+2 dz) - F_R(+dz)) / dz
    """
    F_L_minus1 = F_L(-dz)
    F_L_minus2 = F_L(-2.0 * dz)
    F_L_prime = (F_L_minus1 - F_L_minus2) / dz

    F_R_plus1 = F_R(+dz)
    F_R_plus2 = F_R(+2.0 * dz)
    F_R_prime = (F_R_plus2 - F_R_plus1) / dz

    L_L = F_L_prime / (2.0 * m_L)
    L_R = F_R_prime / (2.0 * m_R)

    return L_L, L_R


def construct_glued_greens_function(
    G_L: Greens1D,
    G_R: Greens1D,
    H_int: ArrayC,
    m_L: float,
    m_R: float,
    dz: float,
) -> Greens1D:
    """
    Construct the fully glued interfacial retarded Green's function G(z, z').

    This is the numerical analogue of the full interfacial Green's function::

        G(z,z') = G_L_bar(z,z') + G_R_bar(z,z')
                  + F(z) @ G00 @ F_bar(z')

    with:
        - F(z)      = θ(-z) F_L(z) + θ(+z) F_R(z)
        - F_bar(z') = θ(-z') F_bar_L(z') + θ(+z') F_bar_R(z')
        - G00       = [ -H_int + L_R - L_L ]^(-1) # log-derivative matrices
        - L_side    = F_side'(0_side) / (2 m_side) # inner log-derivative approaching zero from the side

    Assumptions
    -----------
    - G_L(z, z') is defined (and physically non-zero) for z, z' ≤ 0 (left half-space).
    - G_R(z, z') is defined (and physically non-zero) for z, z' ≥ 0 (right half-space).
    - H_int is the n×n interface Hamiltonian / boundary potential at the interface
      for the given (k_parallel, omega).
    - m_L, m_R are the effective masses on each side (entering the log-derivative).
    - dz is a small positive step used for the finite-difference derivative at z=0.

    Returns
    -------
    G_full : Callable
        A function G_full(z, z') -> (n, n) complex ndarray giving the glued
        interfacial retarded Green's function for the fixed (k_parallel, omega)
        encoded in G_L, G_R and H_int.
    """
    H_int = np.asarray(H_int, dtype=np.complex128)
    n = H_int.shape[0]
    if H_int.shape != (n, n):
        raise ValueError(f"H_int must be (n, n). Got {H_int.shape}.")

    # Build F_L, F_R and their barred partners from the side bulk GFs:
    F_L, F_bar_L = _build_F_side(G_L)
    F_R, F_bar_R = _build_F_side(G_R)

    # Compute L_L, L_R via finite differences at z=0:
    L_L, L_R = _log_derivatives_at_interface(F_L, F_R, m_L=m_L, m_R=m_R, dz=dz)

    # Interface matrix G(0,0):
    G00_inv = -H_int + L_R - L_L
    G00 = _matrix_inverse(G00_inv)

    # Helper zeros:
    zero_mat = np.zeros_like(H_int, dtype=np.complex128)

    def G_full(z: float, zp: float) -> ArrayC:
        """
        Glued interfacial Green's function at (z, z').

        For z,z' on the left/right, this reproduces the continuum structure:
            G = G_L_bar + G_R_bar + F G00 F_bar
        with half-space contributions "barred" by their domain.
        """

        # Half-space contributions (barred Green's functions):
        if (z <= 0.0) and (zp <= 0.0):
            G_L_bar = G_L(z, zp)
        else:
            G_L_bar = zero_mat

        if (z >= 0.0) and (zp >= 0.0):
            G_R_bar = G_R(z, zp)
        else:
            G_R_bar = zero_mat

        # F(z):
        if z < 0.0:
            F_z = F_L(z)
        elif z > 0.0:
            F_z = F_R(z)
        else:
            # z == 0: average one-sided limits as a simple, symmetric choice
            F_z = 0.5 * (F_L(-dz) + F_R(+dz))

        # F_bar(z'):
        if zp < 0.0:
            F_bar_zp = F_bar_L(zp)
        elif zp > 0.0:
            F_bar_zp = F_bar_R(zp)
        else:
            F_bar_zp = 0.5 * (F_bar_L(-dz) + F_bar_R(+dz))

        return G_L_bar + G_R_bar + F_z @ G00 @ F_bar_zp
    # at z=zp=0, this should return G00. Must check this later.
    return G_full # Callable function G(z, z') -> (n, n) complex ndarray
