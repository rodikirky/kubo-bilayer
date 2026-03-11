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

# endregion

#------------------------------------------------
# region compute_F_L
#------------------------------------------------

def test_F_L_shape(scalar_hamiltonian):
    """F^r_L(z) should be n×n."""
    poles_L, orders_L = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='lower',
    )
    residues_L = compute_residues(
        scalar_hamiltonian, poles_L, orders_L,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    F_L = compute_F_L(-1., poles_L, residues_L)
    assert F_L.shape == (scalar_hamiltonian.matrix_dim,
                         scalar_hamiltonian.matrix_dim)


def test_F_L_decays(scalar_hamiltonian):
    """F^r_L(z) should decay exponentially as z decreases."""
    poles_L, orders_L = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='lower',
    )
    residues_L = compute_residues(
        scalar_hamiltonian, poles_L, orders_L,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    F_L_1 = compute_F_L(-1., poles_L, residues_L)
    F_L_2 = compute_F_L(-2., poles_L, residues_L)
    assert np.abs(F_L_2[0, 0]) < np.abs(F_L_1[0, 0])


def test_F_L_at_zero_is_minus_identity(scalar_hamiltonian):
    """F^r_L(z→0-) = -G^r_L(0)·[G^r_L(0)]^{-1} = -I."""
    poles_L, orders_L = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='lower',
    )
    residues_L = compute_residues(
        scalar_hamiltonian, poles_L, orders_L,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    F_L = compute_F_L(-1e-10, poles_L, residues_L)
    n = scalar_hamiltonian.matrix_dim
    assert np.allclose(F_L, -np.eye(n, dtype=np.complex128), atol=ATOL_APPROX)

# endregion

#------------------------------------------------
# region compute_F_bar_R
#------------------------------------------------

def test_F_bar_R_shape(scalar_hamiltonian):
    """F̄^r_R(z') should be n×n."""
    poles_R, orders_R = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    residues_R = compute_residues(
        scalar_hamiltonian, poles_R, orders_R,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    F_bar_R = compute_F_bar_R(1., poles_R, residues_R)
    assert F_bar_R.shape == (scalar_hamiltonian.matrix_dim,
                              scalar_hamiltonian.matrix_dim)


def test_F_bar_R_decays(scalar_hamiltonian):
    """F̄^r_R(z') should decay exponentially as z' increases."""
    poles_R, orders_R = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    residues_R = compute_residues(
        scalar_hamiltonian, poles_R, orders_R,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    F_bar_R_1 = compute_F_bar_R(1., poles_R, residues_R)
    F_bar_R_2 = compute_F_bar_R(2., poles_R, residues_R)
    assert np.abs(F_bar_R_2[0, 0]) < np.abs(F_bar_R_1[0, 0])


def test_F_bar_R_at_zero_is_minus_identity(scalar_hamiltonian):
    """F̄^r_R(z'→0+) = -[G^r_R(0)]^{-1}·G^r_R(0) = -I."""
    poles_R, orders_R = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    residues_R = compute_residues(
        scalar_hamiltonian, poles_R, orders_R,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    F_bar_R = compute_F_bar_R(1e-10, poles_R, residues_R)
    n = scalar_hamiltonian.matrix_dim
    assert np.allclose(F_bar_R, -np.eye(n, dtype=np.complex128), atol=ATOL_APPROX)

# endregion

#------------------------------------------------
# region compute_G_bilayer — new cases
#------------------------------------------------

def _setup_bilayer(scalar_hamiltonian):
    """Shared setup: poles, residues, G00 for identical sides, h_int=0."""
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
    G0_R  = coincidence_value(residues_R, halfplane='upper')
    dG0_R = coincidence_derivative(poles_R, residues_R, halfplane='upper')
    G0_L  = coincidence_value(residues_L, halfplane='lower')
    dG0_L = coincidence_derivative(poles_L, residues_L, halfplane='lower')
    L_R = boundary_derivative(Az, H1, dG0_R, G0_R)
    L_L = boundary_derivative(Az, H1, dG0_L, G0_L)
    n = scalar_hamiltonian.matrix_dim
    h_int = np.zeros((n, n), dtype=np.complex128)
    G00 = assemble_G00(h_int, L_R, L_L)
    return poles_R, residues_R, poles_L, residues_L, G00


def test_G_bilayer_same_side_right_shape(scalar_hamiltonian):
    """G^r(z, z') for z, z' > 0 should be n×n."""
    poles_R, residues_R, poles_L, residues_L, G00 = _setup_bilayer(scalar_hamiltonian)
    G = compute_G_bilayer(1., 2., poles_R, residues_R, poles_L, residues_L, G00)
    n = scalar_hamiltonian.matrix_dim
    assert G.shape == (n, n)


def test_G_bilayer_same_side_left_shape(scalar_hamiltonian):
    """G^r(z, z') for z, z' < 0 should be n×n."""
    poles_R, residues_R, poles_L, residues_L, G00 = _setup_bilayer(scalar_hamiltonian)
    G = compute_G_bilayer(-1., -2., poles_R, residues_R, poles_L, residues_L, G00)
    n = scalar_hamiltonian.matrix_dim
    assert G.shape == (n, n)


def test_G_bilayer_same_side_right_decays(scalar_hamiltonian):
    """G^r(z, z') for z, z' > 0 should decay as z increases."""
    poles_R, residues_R, poles_L, residues_L, G00 = _setup_bilayer(scalar_hamiltonian)
    G1 = compute_G_bilayer(1., 0.5, poles_R, residues_R, poles_L, residues_L, G00)
    G2 = compute_G_bilayer(3., 0.5, poles_R, residues_R, poles_L, residues_L, G00)
    assert np.abs(G2[0, 0]) < np.abs(G1[0, 0])


def test_G_bilayer_same_side_left_decays(scalar_hamiltonian):
    """G^r(z, z') for z, z' < 0 should decay as z decreases."""
    poles_R, residues_R, poles_L, residues_L, G00 = _setup_bilayer(scalar_hamiltonian)
    G1 = compute_G_bilayer(-1., -0.5, poles_R, residues_R, poles_L, residues_L, G00)
    G2 = compute_G_bilayer(-3., -0.5, poles_R, residues_R, poles_L, residues_L, G00)
    assert np.abs(G2[0, 0]) < np.abs(G1[0, 0])


def test_G_bilayer_coincidence_returns_G00(scalar_hamiltonian):
    """G^r(0, 0) should return G00 exactly."""
    poles_R, residues_R, poles_L, residues_L, G00 = _setup_bilayer(scalar_hamiltonian)
    G = compute_G_bilayer(0., 0., poles_R, residues_R, poles_L, residues_L, G00)
    assert np.allclose(G, G00, atol=ATOL_STRICT)

'''
#TODO: Fix this test!
def test_G_bilayer_reduces_to_bulk_identical_sides():
    """
    With identical L and R Hamiltonians and h_int=0, the full bilayer
    GF should reduce to the translationally invariant bulk GF G^(0)(z-z')
    for any z, z' (eq. 64 in supervisor's notes).
    Uses single_channel with m=1, V=0, kx=ky=0, omega=1.
    """
    from showcases.single_channel import make_bulk_hamiltonian, make_interface_hamiltonian
    m, V, kx, ky, omega, eta = 1.0, 0.0, 0.0, 0.0, 1.0, ETA
    ham = make_bulk_hamiltonian(m=m, V=V, kx=kx, ky=ky)

    poles_R, orders_R = compute_poles(
        ham, kx=kx, ky=ky, omega=omega, eta=eta,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    poles_L, orders_L = compute_poles(
        ham, kx=kx, ky=ky, omega=omega, eta=eta,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='lower',
    )
    residues_R = compute_residues(
        ham, poles_R, orders_R,
        kx=kx, ky=ky, omega=omega, eta=eta, tol=ATOL_APPROX,
    )
    residues_L = compute_residues(
        ham, poles_L, orders_L,
        kx=kx, ky=ky, omega=omega, eta=eta, tol=ATOL_APPROX,
    )
    H0, H1, H2 = ham.hamiltonian_kz_polynomial(kx, ky)
    Az = 2 * H2
    G0_R  = coincidence_value(residues_R, halfplane='upper')
    dG0_R = coincidence_derivative(poles_R, residues_R, halfplane='upper')
    G0_L  = coincidence_value(residues_L, halfplane='lower')
    dG0_L = coincidence_derivative(poles_L, residues_L, halfplane='lower')
    L_R = boundary_derivative(Az, H1, dG0_R, G0_R)
    L_L = boundary_derivative(Az, H1, dG0_L, G0_L)
    n = ham.matrix_dim
    h_int = np.zeros((n, n), dtype=np.complex128)
    G00 = assemble_G00(h_int, L_R, L_L)

    G_bulk_check = evaluate(2., poles_R, residues_R, halfplane='upper')
    print(f"\nG00           = {G00[0,0]:.6f}")
    print(f"G^(0)(2)      = {G_bulk_check[0,0]:.6f}")
    print(f"G^(0)(0)      = {coincidence_value(residues_R, halfplane='upper')[0,0]:.6f}")
    print(f"-im/k+ * e^(i*k+*2) = {(-1j / poles_R[0]) * np.exp(1j * poles_R[0] * 2.):.6f}")
    print(f"k+_R          = {poles_R[0]:.6f}")
    from kubo_bilayer.greens.bilayer import compute_F_R, compute_F_bar_L
    F_R     = compute_F_R(1., poles_R, residues_R)
    F_bar_L = compute_F_bar_L(-1., poles_L, residues_L)
    print(f"F_R(1)       = {F_R[0,0]:.6f}")
    print(f"F_bar_L(-1)  = {F_bar_L[0,0]:.6f}")
    print(f"F_R·G00·F_bar_L = {(F_R @ G00 @ F_bar_L)[0,0]:.6f}")
    print(f"exp(i√2·1)   = {np.exp(1j*poles_R[0]*1.):.6f}")
    print(f"exp(-i√2·1)  = {np.exp(-1j*poles_R[0]*1.):.6f}")
    print(f"L_R = {L_R[0,0]:.6f}")
    print(f"L_L = {L_L[0,0]:.6f}")
    print(f"L_R - L_L = {(L_R-L_L)[0,0]:.6f}")
    print(f"inv(G^(0)(0)) = {np.linalg.inv(G0_R)[0,0]:.6f}")

    for z, zp in [(1., -1.), (-1., 1.), (1., 2.), (-1., -2.)]:
        G = compute_G_bilayer(z, zp, poles_R, residues_R,
                              poles_L, residues_L, G00)
        delta_z = z - zp
        if delta_z > 0:
            G_bulk = evaluate(delta_z, poles_R, residues_R, halfplane='upper')
        else:
            G_bulk = evaluate(delta_z, poles_L, residues_L, halfplane='lower')
        print(f"\nz={z}, zp={zp}: G_bilayer={G[0,0]:.6f}, G_bulk={G_bulk[0,0]:.6f}, ratio={G[0,0]/G_bulk[0,0]:.6f}")
        assert np.allclose(G, G_bulk, atol=ATOL_APPROX), (
            f"G_bilayer({z}, {zp}) = {G[0,0]:.6f}, "
            f"G_bulk({delta_z}) = {G_bulk[0,0]:.6f}"
        )
'''
# endregion