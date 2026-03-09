from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import pytest
from showcases.toy_trivial import make_scalar_hamiltonian
from showcases.toy_degenerate import make_degenerate_pole_hamiltonian
from kubo_bilayer.numerics.residues import *
from conftest import ATOL_APPROX, ATOL_STRICT, ETA, OMEGA, OMEGA_DEGENERATE

#------------------------------------------------
# region compute_null_vectors 
#------------------------------------------------
def test_null_vectors_scalar(scalar_hamiltonian, expected_upper_pole):
    """P(kα) should have exactly one null vector at the known pole."""
    U0, V0 = compute_null_vectors(
        scalar_hamiltonian, expected_upper_pole,
        kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol=ATOL_APPROX,
    )
    assert U0.shape == (1, 1)
    assert V0.shape == (1, 1)

def test_null_vectors_degenerate(degenerate_hamiltonian, expected_degenerate_pole):
    """P(kα) should have a two-dimensional nullspace at a degenerate pole."""
    U0, V0 = compute_null_vectors(
        degenerate_hamiltonian, expected_degenerate_pole,
        kx=0., ky=0., omega=OMEGA_DEGENERATE, eta=ETA,
        tol=ATOL_APPROX,
    )
    assert U0.shape == (2, 2)
    assert V0.shape == (2, 2)

def test_null_vectors_orthonormal_scalar(scalar_hamiltonian, expected_upper_pole):
    """SVD null vectors should be orthonormal."""
    U0, V0 = compute_null_vectors(
        scalar_hamiltonian, expected_upper_pole,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    assert np.allclose(U0.conj().T @ U0, np.eye(U0.shape[1]), atol=ATOL_STRICT)
    assert np.allclose(V0.conj().T @ V0, np.eye(V0.shape[1]), atol=ATOL_STRICT)

def test_null_vectors_annihilate_P_scalar(scalar_hamiltonian, expected_upper_pole):
    """P(kα) applied to null vectors should give zero."""
    H_at_pole = scalar_hamiltonian.evaluate(0., 0., expected_upper_pole)
    P = (OMEGA + 1j * ETA) * scalar_hamiltonian.identity - H_at_pole
    U0, V0 = compute_null_vectors(
        scalar_hamiltonian, expected_upper_pole,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    assert np.allclose(P @ V0, 0., atol=ATOL_APPROX)
    assert np.allclose(U0.conj().T @ P, 0., atol=ATOL_APPROX)

def test_null_vector_U0_scalar(scalar_hamiltonian, expected_upper_pole):
    """U0 should be a unit vector spanning the left nullspace of P(kα)."""
    U0, V0 = compute_null_vectors(
        scalar_hamiltonian, expected_upper_pole,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    # U0 should be a unit vector
    assert np.isclose(np.linalg.norm(U0), 1., atol=ATOL_APPROX)
    # U0 should span the nullspace: P(kα) @ U0 ≈ 0 (left null vector)
    H_at_pole = scalar_hamiltonian.evaluate(0., 0., expected_upper_pole)
    P = (OMEGA + 1j * ETA) * scalar_hamiltonian.identity - H_at_pole
    assert np.allclose(U0.conj().T @ P, 0., atol=ATOL_APPROX)

def test_null_vector_V0_scalar(scalar_hamiltonian, expected_upper_pole):
    """V0 should be a unit vector spanning the right nullspace of P(kα)."""
    U0, V0 = compute_null_vectors(
        scalar_hamiltonian, expected_upper_pole,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    # V0 should be a unit vector
    assert np.isclose(np.linalg.norm(V0), 1., atol=ATOL_APPROX)
    # V0 should span the nullspace: P(kα) @ V0 ≈ 0 (right null vector)
    H_at_pole = scalar_hamiltonian.evaluate(0., 0., expected_upper_pole)
    P = (OMEGA + 1j * ETA) * scalar_hamiltonian.identity - H_at_pole
    assert np.allclose(P @ V0, 0., atol=ATOL_APPROX)

def test_null_vectors_consistent_scalar(scalar_hamiltonian, expected_upper_pole):
    """For a Hermitian matrix P, left and right null vectors should span
    the same space, i.e. |U0† @ V0| = 1."""
    U0, V0 = compute_null_vectors(
        scalar_hamiltonian, expected_upper_pole,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    # the overlap should have magnitude 1 for a 1D nullspace
    assert np.isclose(np.abs(U0.conj().T @ V0), 1., atol=ATOL_APPROX)

# endregion