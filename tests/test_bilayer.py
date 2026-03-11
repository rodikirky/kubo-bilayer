from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import pytest
from kubo_bilayer.numerics.poles import compute_poles
from kubo_bilayer.numerics.residues import compute_residues
from kubo_bilayer.greens.bulk import *
from kubo_bilayer.greens.interface import *
from kubo_bilayer.greens.bilayer import *
from conftest import ATOL_APPROX, ATOL_STRICT, ETA, OMEGA, OMEGA_DEGENERATE

#------------------------------------------------
# region compute_F_R
#------------------------------------------------

def test_F_R_shape(scalar_hamiltonian):
    """F^r_R(z) should be n×n."""
    poles_R, orders_R = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    residues_R = compute_residues(
        scalar_hamiltonian, poles_R, orders_R,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    F_R = compute_F_R(1., poles_R, residues_R)
    assert F_R.shape == (scalar_hamiltonian.matrix_dim,
                         scalar_hamiltonian.matrix_dim)

def test_F_R_decays(scalar_hamiltonian):
    """F^r_R(z) should decay exponentially for increasing z."""
    poles_R, orders_R = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    residues_R = compute_residues(
        scalar_hamiltonian, poles_R, orders_R,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    F_R_1 = compute_F_R(1., poles_R, residues_R)
    F_R_2 = compute_F_R(2., poles_R, residues_R)
    assert np.abs(F_R_2[0,0]) < np.abs(F_R_1[0,0])

def test_F_R_at_zero_is_minus_identity(scalar_hamiltonian):
    """
    F^r_R(z→0) = -G^r_R(0) · [G^r_R(0)]^{-1} = -I
    """
    poles_R, orders_R = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    residues_R = compute_residues(
        scalar_hamiltonian, poles_R, orders_R,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    F_R = compute_F_R(1e-10, poles_R, residues_R)
    n = scalar_hamiltonian.matrix_dim
    assert np.allclose(F_R, -np.eye(n, dtype=np.complex128), atol=ATOL_APPROX)

# endregion

#------------------------------------------------
# region compute_F_bar_L
#------------------------------------------------

def test_F_bar_L_shape(scalar_hamiltonian):
    """F̄^r_L(z') should be n×n."""
    poles_L, orders_L = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='lower',
    )
    residues_L = compute_residues(
        scalar_hamiltonian, poles_L, orders_L,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    F_bar_L = compute_F_bar_L(-1., poles_L, residues_L)
    assert F_bar_L.shape == (scalar_hamiltonian.matrix_dim,
                              scalar_hamiltonian.matrix_dim)

def test_F_bar_L_decays(scalar_hamiltonian):
    """F̄^r_L(z') should decay exponentially for decreasing z'."""
    poles_L, orders_L = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='lower',
    )
    residues_L = compute_residues(
        scalar_hamiltonian, poles_L, orders_L,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    F_bar_L_1 = compute_F_bar_L(-1., poles_L, residues_L)
    F_bar_L_2 = compute_F_bar_L(-2., poles_L, residues_L)
    assert np.abs(F_bar_L_2[0,0]) < np.abs(F_bar_L_1[0,0])

def test_F_bar_L_at_zero_is_minus_identity(scalar_hamiltonian):
    """
    F̄^r_L(z'→0) = -[G^r_L(0)]^{-1} · G^r_L(0) = -I
    """
    poles_L, orders_L = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='lower',
    )
    residues_L = compute_residues(
        scalar_hamiltonian, poles_L, orders_L,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    F_bar_L = compute_F_bar_L(-1e-10, poles_L, residues_L)
    n = scalar_hamiltonian.matrix_dim
    assert np.allclose(F_bar_L, -np.eye(n, dtype=np.complex128), atol=ATOL_APPROX)

# endregion

#------------------------------------------------
# region compute_G_bilayer
#------------------------------------------------

def test_G_bilayer_shape(scalar_hamiltonian):
    """G^r(z,z') should be n×n."""
    poles_R, orders_R = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    poles_L, orders_L = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='lower',
    )
    residues_R = compute_residues(
        scalar_hamiltonian, poles_R, orders_R,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    residues_L = compute_residues(
        scalar_hamiltonian, poles_L, orders_L,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    H0, H1, H2 = scalar_hamiltonian.hamiltonian_kz_polynomial(0., 0.)
    Az = 2 * H2
    G0_R = coincidence_value(residues_R, halfplane='upper')
    dG0_R = coincidence_derivative(poles_R, residues_R, halfplane='upper')
    G0_L = coincidence_value(residues_L, halfplane='lower')
    dG0_L = coincidence_derivative(poles_L, residues_L, halfplane='lower')
    L_R = boundary_derivative(Az, H1, dG0_R, G0_R)
    L_L = boundary_derivative(Az, H1, dG0_L, G0_L)
    n = scalar_hamiltonian.matrix_dim
    h_int = np.zeros((n, n), dtype=np.complex128)
    G00 = assemble_G00(h_int, L_R, L_L)
    G = compute_G_bilayer(1., -1., poles_R, residues_R,
                          poles_L, residues_L, G00)
    assert G.shape == (n, n)

def test_G_bilayer_rejects_invalid_z(scalar_hamiltonian):
    """z <= 0 or z' >= 0 should raise ValueError."""
    poles_R, orders_R = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    poles_L, orders_L = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='lower',
    )
    residues_R = compute_residues(
        scalar_hamiltonian, poles_R, orders_R,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    residues_L = compute_residues(
        scalar_hamiltonian, poles_L, orders_L,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    n = scalar_hamiltonian.matrix_dim
    G00 = np.eye(n, dtype=np.complex128)
    with pytest.raises(ValueError):
        compute_G_bilayer(-1., -1., poles_R, residues_R,
                          poles_L, residues_L, G00)
    with pytest.raises(ValueError):
        compute_G_bilayer(1., 1., poles_R, residues_R,
                          poles_L, residues_L, G00)

def test_G_bilayer_decays_in_z(scalar_hamiltonian):
    """G^r(z,z') should decay as z increases."""
    poles_R, orders_R = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    poles_L, orders_L = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='lower',
    )
    residues_R = compute_residues(
        scalar_hamiltonian, poles_R, orders_R,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    residues_L = compute_residues(
        scalar_hamiltonian, poles_L, orders_L,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    H0, H1, H2 = scalar_hamiltonian.hamiltonian_kz_polynomial(0., 0.)
    Az = 2 * H2
    G0_R = coincidence_value(residues_R, halfplane='upper')
    dG0_R = coincidence_derivative(poles_R, residues_R, halfplane='upper')
    G0_L = coincidence_value(residues_L, halfplane='lower')
    dG0_L = coincidence_derivative(poles_L, residues_L, halfplane='lower')
    L_R = boundary_derivative(Az, H1, dG0_R, G0_R)
    L_L = boundary_derivative(Az, H1, dG0_L, G0_L)
    n = scalar_hamiltonian.matrix_dim
    h_int = np.zeros((n, n), dtype=np.complex128)
    G00 = assemble_G00(h_int, L_R, L_L)
    G1 = compute_G_bilayer(1., -1., poles_R, residues_R,
                           poles_L, residues_L, G00)
    G2 = compute_G_bilayer(2., -1., poles_R, residues_R,
                           poles_L, residues_L, G00)
    assert np.abs(G2[0,0]) < np.abs(G1[0,0])

def test_G_bilayer_hermitian_conjugate(scalar_hamiltonian):
    """
    For identical L and R Hamiltonians with h_int=0,
    G^r(z,z') should satisfy [G^r(z,z')]† ≈ G^r(z',z)†
    approximately at small eta.
    """
    #TODO: Needs to be refined. Trivial at this point.
    poles_R, orders_R = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    poles_L, orders_L = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='lower',
    )
    residues_R = compute_residues(
        scalar_hamiltonian, poles_R, orders_R,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    residues_L = compute_residues(
        scalar_hamiltonian, poles_L, orders_L,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    H0, H1, H2 = scalar_hamiltonian.hamiltonian_kz_polynomial(0., 0.)
    Az = 2 * H2
    G0_R = coincidence_value(residues_R, halfplane='upper')
    dG0_R = coincidence_derivative(poles_R, residues_R, halfplane='upper')
    G0_L = coincidence_value(residues_L, halfplane='lower')
    dG0_L = coincidence_derivative(poles_L, residues_L, halfplane='lower')
    L_R = boundary_derivative(Az, H1, dG0_R, G0_R)
    L_L = boundary_derivative(Az, H1, dG0_L, G0_L)
    n = scalar_hamiltonian.matrix_dim
    h_int = np.zeros((n, n), dtype=np.complex128)
    G00 = assemble_G00(h_int, L_R, L_L)
    G_zz = compute_G_bilayer(1., -1., poles_R, residues_R,
                              poles_L, residues_L, G00)
    G_zpz = compute_G_bilayer(1., -1., poles_R, residues_R,
                               poles_L, residues_L, G00)
    assert np.allclose(G_zz, G_zpz.conj().T, atol=ATOL_APPROX)


def test_semigroup_property(scalar_hamiltonian):
    """
    Verify the semigroup property (eq. 74 in supervisor's notes):
        G^(0)r(z1+z2, 0) = G^(0)r(z1, 0) · [G^(0)r(0,0)]^{-1} · G^(0)r(z2, 0)
    which is equivalent to:
        -F_R(z1) · G0_R · (-F_R(z2)) = G_bulk(z1+z2)
    """
    poles_R, orders_R = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    residues_R = compute_residues(
        scalar_hamiltonian, poles_R, orders_R,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    G0_R = coincidence_value(residues_R, halfplane='upper')
    F_R_1 = compute_F_R(1., poles_R, residues_R)
    F_R_2 = compute_F_R(1., poles_R, residues_R)
    G_bulk_2 = evaluate(2., poles_R, residues_R, halfplane='upper')
    assert np.allclose(F_R_1 @ G0_R @ F_R_2, G_bulk_2, atol=ATOL_APPROX)