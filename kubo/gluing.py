from __future__ import annotations

from typing import Callable, Tuple
import numpy as np
import warnings
from numpy.typing import NDArray

ArrayC = NDArray[np.complex128]
Greens1D = Callable[[float, float], ArrayC]  # G(z, z') -> (n, n) matrix


def _matrix_inverse(M: ArrayC) -> NDArray:
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
    G00 = np.asarray(G00, dtype=np.complex128)

    if G00.ndim != 2 or G00.shape[0] != G00.shape[1]:
        raise ValueError(f"G_side(0,0) must be a square matrix, got shape {G00.shape}.")

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

    Continuum formula:
        L_side = F_side'(0_side) / (2 m_side)

    We use simple one-sided finite differences anchored at z=0:

        F_L'(0-) ≈ (F_L(0.0) - F_L(-dz)) / dz
        F_R'(0+) ≈ (F_R(+dz) - F_R(0.0)) / dz
    """
    if dz <= 0:
        raise ValueError(f"dz must be positive, got {dz}.")
    if m_L == 0 or m_R == 0:
        raise ValueError("m_L and m_R must be non-zero for log-derivatives (generally true for realistic physics).")

    # Left side: approach 0 from negative z
    F_L_0 = F_L(0.0)
    F_L_m1 = F_L(-dz)
    F_L_prime = (F_L_0 - F_L_m1) / dz

    # Right side: approach 0 from positive z
    F_R_0 = F_R(0.0)
    F_R_p1 = F_R(+dz)
    F_R_prime = (F_R_p1 - F_R_0) / dz

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
    - G_L(z, z') is defined for z, z' ≤ 0 (left half-space) and used in G_full
    for z, z' < 0.
    - G_R(z, z') is defined for z, z' ≥ 0 (right half-space) and used in G_full
    for z, z' > 0.
    - The interface plane z = z' = 0 is represented explicitly via G00 (the interface
    Green's function in the reduced space).
    - H_int is the n×n interface Hamiltonian / boundary potential at the interface
      for the given (k_parallel, omega).
    - m_L, m_R are the effective masses on each side (entering the log-derivative).
    - dz is a small positive step used for the finite-difference derivative of F(z) at z=0.

    Returns
    -------
    G_full : Callable
        A function G_full(z, z') -> (n, n) complex ndarray giving the glued
        interfacial retarded Green's function for the fixed (k_parallel, omega)
        encoded in G_L, G_R and H_int.
    """
    H_int = np.asarray(H_int, dtype=np.complex128)
    n = H_int.shape[0]
    I = np.eye(n, dtype=np.complex128)
    if H_int.shape != (n, n):
        raise ValueError(f"H_int must be (n, n). Got {H_int.shape}.")

    # Quick check that G_L and G_R are compatible with H_int at the interface:
    G_L_00 = np.asarray(G_L(0.0, 0.0), dtype=np.complex128)
    G_R_00 = np.asarray(G_R(0.0, 0.0), dtype=np.complex128)

    if G_L_00.shape != H_int.shape or G_R_00.shape != H_int.shape:
        raise ValueError(
            "G_L(0,0) and G_R(0,0) must have the same shape as H_int. "
            f"Got G_L(0,0)={G_L_00.shape}, G_R(0,0)={G_R_00.shape}, H_int={H_int.shape}."
        )
    
    # Build F_L, F_R and their barred partners from the side bulk GFs:
    F_L, F_bar_L = _build_F_side(G_L)
    F_R, F_bar_R = _build_F_side(G_R)

    # Compute L_L, L_R via finite differences at z=0:
    L_L, L_R = _log_derivatives_at_interface(F_L, F_R, m_L=m_L, m_R=m_R, dz=dz)

    # Interface matrix G(0,0):
    G00_inv = -H_int + L_R - L_L # different G00 than in _build_F_side, this is the full interface one
    G00 = _matrix_inverse(G00_inv)

    if not np.all(np.isfinite(G00)):
        raise FloatingPointError(
            "Interface Green's function G00 contains NaN or inf. "
            "Check H_int, m_L/m_R, dz, and the side Green's functions."
        )

    # Helper zeros:
    zero_mat = np.zeros_like(H_int, dtype=np.complex128)

    def G_full(z: float, zp: float) -> ArrayC:
        """
        Glued interfacial Green's function at (z, z').

        For z,z' on the left/right, this reproduces the continuum structure:
            G = G_L_bar + G_R_bar + F G00 F_bar
        with half-space contributions "barred" by their domain.
        """
        if z == 0.0 and zp == 0.0:
            # at the interface, return G00 directly
            # no half-space contributions, since G00 contains all interface physics
            # analytically, G(0,0) = G00 is true since F(0) = F_bar(0) = -I
            return G00
        eps = 1e-12
        if (0 < abs(z) < eps) or (0 < abs(zp) < eps):
            warnings.warn(
                "G_full called very close to the interface (z~0 or z'~0). "
                "If you meant to have z = z' = 0: check your grid "
                "or use G_full(0,0) = G00 directly, since the physics "
                "heavily depends on whether z and z' are on the left or right of the interface. "
                "\nElse be aware that the values of G around zero are sensitive to the derivative resolution dz "
                "and the numerical resolution of the continuity condition at the interface.",
                RuntimeWarning,
            )

        # Half-space contributions (barred Green's functions):
        # strict inequalities possible due to continuity at (0,0) with G00
        if (z < 0.0) and (zp < 0.0): 
            G_L_bar = G_L(z, zp)
        else:
            G_L_bar = zero_mat

        if (z > 0.0) and (zp > 0.0):
            G_R_bar = G_R(z, zp)
        else:
            G_R_bar = zero_mat

        # F(z):
        if z < 0.0:
            F_z = F_L(z)
        elif z > 0.0:
            F_z = F_R(z)
        else:
            # analytically, F(0) = -I
            F_z = -I

        # F_bar(z'):
        if zp < 0.0:
            F_bar_zp = F_bar_L(zp)
        elif zp > 0.0:
            F_bar_zp = F_bar_R(zp)
        else:
            # analytically, F_bar(0) = -I
            F_bar_zp = -I 
            
        # At z = z' = 0 we return G00 directly (see special case above), so
        # G_full(0, 0) is exactly the interface Green's function in the reduced space of (k_parallel; omega).
        # G_full is continuous everywhere, its derivative has a jump at z = 0 and z' = 0 due to the interface.
        return G_L_bar + G_R_bar + F_z @ G00 @ F_bar_zp
    
    return G_full # Callable function G(z, z') -> (n, n) complex ndarray
