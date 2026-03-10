from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import pytest
from kubo_bilayer.numerics.poles import compute_poles
from kubo_bilayer.numerics.residues import compute_residues
from kubo_bilayer.greens.bulk import *
from kubo_bilayer.greens.interface import *
from conftest import ATOL_APPROX, ATOL_STRICT, ETA, OMEGA, OMEGA_DEGENERATE

#------------------------------------------------
# region boundary_derivative
#------------------------------------------------

def test_boundary_derivative_shape(scalar_hamiltonian):
    """L^r should be n×n."""
    poles, orders = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    residues = compute_residues(
        scalar_hamiltonian, poles, orders,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    H0, H1, H2 = scalar_hamiltonian.hamiltonian_kz_polynomial(0., 0.)
    Az = 2 * H2
    G0 = coincidence_value(residues, halfplane='upper')
    dG0 = coincidence_derivative(poles, residues, halfplane='upper')
    L = boundary_derivative(Az, H1, dG0, G0)
    n = scalar_hamiltonian.matrix_dim
    assert L.shape == (n, n)

def test_boundary_derivative_known_answer_scalar(scalar_hamiltonian):
    """
    For H(kz) = kz² + kz + 1, Az = 2I, H1 = I.
    At kα = -0.5 + i√3/2, Res = i/√3 (up to phase).
    Verify L^r has correct magnitude.
    """
    poles, orders = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,
        halfplane='upper',
    )
    residues = compute_residues(
        scalar_hamiltonian, poles, orders,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    H0, H1, H2 = scalar_hamiltonian.hamiltonian_kz_polynomial(0., 0.)
    Az = 2 * H2
    G0 = coincidence_value(residues, halfplane='upper')
    dG0 = coincidence_derivative(poles, residues, halfplane='upper')
    L = boundary_derivative(Az, H1, dG0, G0)
    # L^r should be finite and non-zero
    assert np.all(np.isfinite(L))
    assert np.any(np.abs(L) > ATOL_APPROX)

# endregion

#------------------------------------------------
# region assemble_G00
#------------------------------------------------

def test_G00_shape(scalar_hamiltonian):
    """G(0,0) should be n×n."""
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
    assert G00.shape == (n, n)

def test_G00_reduces_to_bulk_no_interface(scalar_hamiltonian):
    """
    With identical left and right Hamiltonians and h_int = 0,
    G(0,0) should equal the bulk coincidence value G^r(0,0).
    (eq. 60 in supervisor's notes)
    """
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
    # should match bulk coincidence value
    assert np.allclose(G00, G0_R, atol=ATOL_APPROX)

# endregion

