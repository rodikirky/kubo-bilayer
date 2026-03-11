"""
halfspace.py
------------
Construction of the half-space Green's functions Ḡ^r_L and Ḡ^r_R,
describing propagation within a single half-space in the presence
of a hard-wall boundary condition at z = 0.

Physical Background
-------------------
The half-space Green's function is obtained from the translationally
invariant bulk Green's function by subtracting the reflection term
that enforces G_bar(z=0, z') = 0 (eq. 47 in supervisor's notes):

    Ḡ^r(z,z') = G^(0)r(z,z') - G^(0)r(z,0)·[G^(0)r(0,0)]^{-1}·G^(0)r(0,z')

For z and z' on the same side this is nonzero. For z and z' on
opposite sides it vanishes identically (eq. 49).

For the right half-space (z, z' > 0), using upper half-plane poles:

    G^(0)r_R(z,z') = evaluate(z-z', poles_R, residues_R, 'upper')  if z > z'
                   = evaluate(z-z', poles_R, residues_R, 'upper')  if z < z'
                   (evaluate handles both signs of delta_z)

    Ḡ^r_R(z,z') = G^(0)r_R(z,z') - G^(0)r_R(z,0)·[G^(0)r_R(0,0)]^{-1}·G^(0)r_R(0,z')

For the left half-space (z, z' < 0), using lower half-plane poles:

    Ḡ^r_L(z,z') = G^(0)r_L(z,z') - G^(0)r_L(z,0)·[G^(0)r_L(0,0)]^{-1}·G^(0)r_L(0,z')

Note on evaluate() arguments
-----------------------------
evaluate(delta_z, ...) requires delta_z != 0 and that its sign
matches the halfplane convention. For the right half-space:

    G^(0)r_R(z, 0)  = evaluate(z,    ..., 'upper')   z > 0  ✓
    G^(0)r_R(0, z') = evaluate(-z',  ..., 'upper')   z' > 0, -z' < 0  ✗

The second call is problematic: evaluate(..., 'upper') requires
delta_z > 0. Instead we use the symmetry of the bulk Green's
function:

    G^(0)r(0, z') = [G^(0)r(z', 0)]†   (retarded GF symmetry)

So G^(0)r_R(0, z') = evaluate(z', ..., 'upper').conj().T for z' > 0.
Analogously for the left side.

Functions
---------
compute_G_bar_R(z, z_prime, poles_R, residues_R)
    Half-space Green's function for z, z' > 0.

compute_G_bar_L(z, z_prime, poles_L, residues_L)
    Half-space Green's function for z, z' < 0.

Dependencies
------------
    bulk.py — provides evaluate() and coincidence_value()
"""
import numpy as np
from numpy.typing import NDArray

from kubo_bilayer.greens.bulk import evaluate, coincidence_value

ArrayC = NDArray[np.complex128]

__all__ = [
    "compute_G_bar_R",
    "compute_G_bar_L",
]

def compute_G_bar_R(
    z: float,
    z_prime: float,
    poles_R: ArrayC,
    residues_R: list[ArrayC],
    poles_L: ArrayC,
    residues_L: list[ArrayC],
) -> ArrayC:
    """
    Half-space retarded Green's function for z, z' > 0.

        Ḡ^r_R(z,z') = G^(0)r_R(z,z') - G^(0)r_R(z,0)·[G^(0)r_R(0,0)]^{-1}·G^(0)r_R(0,z')

    Returns a zero matrix if z <= 0 or z' <= 0 (Theta function condition).

    G^(0)r_R(z, z') = G^(0)r_R(z - z', 0), evaluated with upper half-plane
    poles for delta_z > 0 and lower half-plane poles for delta_z < 0.
    G^(0)r_R(z, 0) always uses upper half-plane poles (z > 0).
    G^(0)r_R(0, z') always uses lower half-plane poles (-z' < 0).

    Parameters
    ----------
    z, z_prime  : float
    poles_R     : ArrayC — upper half-plane poles
    residues_R  : list[ArrayC]
    poles_L     : ArrayC — lower half-plane poles (needed for delta_z < 0
                  and for G^(0)r(0, z') = evaluate(-z', ..., 'lower'))
    residues_L  : list[ArrayC]

    Returns
    -------
    G_bar_R : ArrayC, shape (n, n) — zero matrix if z <= 0 or z' <= 0
    """
    n = residues_R[0].shape[0]
    if z <= 0 or z_prime <= 0:
        return np.zeros((n, n), dtype=np.complex128)

    delta_z = z - z_prime

    if delta_z > 0:
        G_zz = evaluate(delta_z, poles_R, residues_R, halfplane='upper')
    elif delta_z < 0:
        G_zz = evaluate(delta_z, poles_L, residues_L, halfplane='lower')
    else:
        G_zz = coincidence_value(residues_R, halfplane='upper')

    G_z0  = evaluate(z,        poles_R, residues_R, halfplane='upper')
    G_0zp = evaluate(-z_prime, poles_L, residues_L, halfplane='lower')
    G_00  = coincidence_value(residues_R, halfplane='upper')

    return G_zz - G_z0 @ np.linalg.inv(G_00) @ G_0zp


def compute_G_bar_L(
    z: float,
    z_prime: float,
    poles_R: ArrayC,
    residues_R: list[ArrayC],
    poles_L: ArrayC,
    residues_L: list[ArrayC],
) -> ArrayC:
    """
    Half-space retarded Green's function for z, z' < 0.

        Ḡ^r_L(z,z') = G^(0)r_L(z,z') - G^(0)r_L(z,0)·[G^(0)r_L(0,0)]^{-1}·G^(0)r_L(0,z')

    Returns a zero matrix if z >= 0 or z' >= 0 (Theta function condition).

    G^(0)r_L(z, z') = G^(0)r_L(z - z', 0), evaluated with lower half-plane
    poles for delta_z < 0 and upper half-plane poles for delta_z > 0.
    G^(0)r_L(z, 0) always uses lower half-plane poles (z < 0).
    G^(0)r_L(0, z') always uses upper half-plane poles (-z' > 0).

    Parameters
    ----------
    z, z_prime  : float
    poles_R     : ArrayC — upper half-plane poles (needed for delta_z > 0
                  and for G^(0)r(0, z') = evaluate(-z', ..., 'upper'))
    residues_R  : list[ArrayC]
    poles_L     : ArrayC — lower half-plane poles
    residues_L  : list[ArrayC]

    Returns
    -------
    G_bar_L : ArrayC, shape (n, n) — zero matrix if z >= 0 or z' >= 0
    """
    n = residues_L[0].shape[0]
    if z >= 0 or z_prime >= 0:
        return np.zeros((n, n), dtype=np.complex128)

    delta_z = z - z_prime

    if delta_z < 0:
        G_zz = evaluate(delta_z, poles_L, residues_L, halfplane='lower')
    elif delta_z > 0:
        G_zz = evaluate(delta_z, poles_R, residues_R, halfplane='upper')
    else:
        G_zz = coincidence_value(residues_L, halfplane='lower')

    G_z0  = evaluate(z,        poles_L, residues_L, halfplane='lower')
    G_0zp = evaluate(-z_prime, poles_R, residues_R, halfplane='upper')
    G_00  = coincidence_value(residues_L, halfplane='lower')

    return G_zz - G_z0 @ np.linalg.inv(G_00) @ G_0zp