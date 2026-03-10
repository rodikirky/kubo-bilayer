from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import pytest
from kubo_bilayer.numerics.poles import *
from conftest import ATOL_APPROX, ATOL_STRICT, ETA, OMEGA, OMEGA_DEGENERATE

#------------------------------------------------
# region build_companion_matrices 
#------------------------------------------------
def test_companion_shape(scalar_hamiltonian):
    # for an n×n Hamiltonian, L and M should be 2n×2n
    L, M = build_companion_matrices(scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA)
    n = scalar_hamiltonian.matrix_dim
    assert L.shape == (2*n, 2*n)
    assert M.shape == (2*n, 2*n)

def test_companion_block_structure(scalar_hamiltonian):
    """Zero blocks of L and M should be exactly zero."""
    L, M = build_companion_matrices(scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA)
    n = scalar_hamiltonian.matrix_dim
    assert np.allclose(L[n:, n:], 0.)   # lower right of L
    assert np.allclose(M[:n, n:], 0.)   # upper right of M
    assert np.allclose(M[n:, :n], 0.)   # lower left of M

def test_companion_rejects_complex_kx(scalar_hamiltonian):
    """Complex kx should raise ValueError."""
    with pytest.raises(ValueError):
        build_companion_matrices(scalar_hamiltonian, kx=1.+1j, ky=0., omega=OMEGA, eta=ETA)

def test_companion_rejects_complex_ky(scalar_hamiltonian):
    """Complex ky should raise ValueError."""
    with pytest.raises(ValueError):
        build_companion_matrices(scalar_hamiltonian, kx=0., ky=1.+1j, omega=OMEGA, eta=ETA)

def test_companion_rejects_complex_omega(scalar_hamiltonian):
    """Complex omega should raise ValueError."""
    with pytest.raises(ValueError):
        build_companion_matrices(scalar_hamiltonian, kx=0., ky=0., omega=1.+1j, eta=ETA)

def test_companion_rejects_zero_eta(scalar_hamiltonian):
    """eta=0 should raise ValueError."""
    with pytest.raises(ValueError):
        build_companion_matrices(scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=0.)

def test_companion_rejects_negative_eta(scalar_hamiltonian):
    """Negative eta should raise ValueError."""
    with pytest.raises(ValueError):
        build_companion_matrices(scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=-0.1)
    
def test_companion_known(scalar_hamiltonian):
    """
    For H(kz) = kz² + kz + 1 at kx=ky=omega=0, eta=0.1
    the companion matrices should be exactly:
        L = [[ 1+0j,  -1+0.1j],   M = [[-1+0j,  0+0j],
             [ 1+0j,   0+0j  ]]        [ 0+0j,  1+0j]]
    """
    L, M = build_companion_matrices(scalar_hamiltonian, kx=0., ky=0., omega=0, eta=0.1)
    expected_L = np.array([[ 1.+0.j, 1.-0.1j],
                           [ 1.+0.j, 0.+0.j ]], dtype=np.complex128)
    expected_M = np.array([[-1.+0.j,  0.+0.j],
                           [ 0.+0.j,  1.+0.j]], dtype=np.complex128)
    assert np.allclose(L, expected_L)
    assert np.allclose(M, expected_M)

# endregion

#------------------------------------------------
# region solve_companion_evp 
#------------------------------------------------
def test_evp_returns_two_poles(scalar_hamiltonian):
    """Scalar 1x1 Hamiltonian should yield exactly 2 eigenvalues."""
    L, M = build_companion_matrices(scalar_hamiltonian, kx=0., ky=0., omega=0., eta=ETA)
    kz_all = solve_companion_evp(L, M)
    assert len(kz_all) == 2

def test_evp_known_poles(scalar_hamiltonian, expected_upper_pole, expected_lower_pole):
    """Both roots of kz² + kz + 1 = 0 should appear in the eigenvalues."""
    L, M = build_companion_matrices(scalar_hamiltonian, kx=0., ky=0., omega=0., eta=ETA)
    kz_all = solve_companion_evp(L, M)
    assert any(np.isclose(kz_all, expected_upper_pole, atol=ATOL_STRICT))
    assert any(np.isclose(kz_all, expected_lower_pole, atol=ATOL_STRICT))

# endregion


#------------------------------------------------
# region filter_upper_halfplane 
#------------------------------------------------
def test_filter_returns_upper_halfplane(scalar_hamiltonian):
    """Should return exactly one pole in the upper half-plane."""
    L, M = build_companion_matrices(scalar_hamiltonian, kx=0., ky=0., omega=0., eta=ETA)
    kz_all = solve_companion_evp(L, M)
    poles = filter_upper_halfplane(kz_all, tol=ATOL_STRICT)
    assert len(poles) == 1
    assert np.imag(poles[0]) > 0

def test_filter_correct_pole_location(scalar_hamiltonian, expected_upper_pole):
    """Upper half-plane pole should match analytic result at eta→0."""
    L, M = build_companion_matrices(scalar_hamiltonian, kx=0., ky=0., omega=0., eta=ETA)
    kz_all = solve_companion_evp(L, M)
    poles = filter_upper_halfplane(kz_all, tol=ATOL_STRICT)
    assert np.isclose(poles[0], expected_upper_pole, atol=ATOL_STRICT)

# endregion

#------------------------------------------------
# region cluster_poles 
#------------------------------------------------
def test_cluster_no_clustering_scalar(scalar_hamiltonian):
    """Two well-separated poles should not be clustered."""
    L, M = build_companion_matrices(scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA)
    kz_all = solve_companion_evp(L, M)
    poles = filter_upper_halfplane(kz_all, tol=ATOL_STRICT)
    unique_poles, orders = cluster_poles(poles, tol=ATOL_STRICT)
    assert len(unique_poles) == 1
    assert orders[0] == 1

def test_cluster_detects_degenerate_pole(degenerate_hamiltonian):
    """Two near-coincident poles should be clustered into one candidate order-2 pole."""
    L, M = build_companion_matrices(degenerate_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA)
    kz_all = solve_companion_evp(L, M)
    poles = filter_upper_halfplane(kz_all, tol=ATOL_STRICT)
    unique_poles, orders = cluster_poles(poles, tol=ATOL_STRICT)
    assert len(unique_poles) == 1
    assert orders[0] == 2

def test_cluster_known_degenerate_pole_location(degenerate_hamiltonian, expected_degenerate_pole):
    """Cluster centre should be close to the analytic double pole at kz = i."""
    L, M = build_companion_matrices(degenerate_hamiltonian, kx=0., ky=0., omega=OMEGA_DEGENERATE, eta=ETA)
    kz_all = solve_companion_evp(L, M)
    poles = filter_upper_halfplane(kz_all, tol=ATOL_STRICT)
    unique_poles, orders = cluster_poles(poles, tol=ATOL_STRICT)
    assert np.isclose(unique_poles[0], expected_degenerate_pole, atol=ATOL_STRICT)
# endregion

#------------------------------------------------
# region compute_poles 
#------------------------------------------------
def test_compute_poles_scalar(scalar_hamiltonian, expected_upper_pole):
    """Full pipeline should return one pole matching the analytic result."""
    unique_poles, orders = compute_poles(scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA, tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX)
    assert len(unique_poles) == 1
    assert orders[0] == 1
    assert np.isclose(unique_poles[0], expected_upper_pole, atol=ATOL_APPROX)

def test_compute_poles_degenerate(degenerate_hamiltonian, expected_degenerate_pole):
    """Full pipeline should return one clustered pole of candidate order 2."""
    unique_poles, orders = compute_poles(degenerate_hamiltonian, kx=0., ky=0., omega=OMEGA_DEGENERATE, eta=ETA, tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX)
    assert len(unique_poles) == 1
    assert orders[0] == 2
    assert np.isclose(unique_poles[0], expected_degenerate_pole, atol=ATOL_APPROX)