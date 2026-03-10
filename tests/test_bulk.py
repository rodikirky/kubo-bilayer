from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import pytest
from kubo_bilayer.numerics.poles import compute_poles
from kubo_bilayer.numerics.residues import compute_residues
from kubo_bilayer.analytics.bulk import *
from conftest import ATOL_APPROX, ATOL_STRICT, ETA, OMEGA, OMEGA_DEGENERATE

# -----------------------
# Test evaluate()
# -----------------------
def test_evaluate_scalar(scalar_hamiltonian, expected_upper_pole):
    """G^r(Δz) should decay exponentially for Δz > 0."""
    poles, orders = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,halfplane='upper'
    )
    residues = compute_residues(
        scalar_hamiltonian, poles, orders,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    G1 = evaluate(1., poles, residues,halfplane='upper')
    G2 = evaluate(2., poles, residues,halfplane='upper')
    # G^r should decay — |G(2)| < |G(1)|
    assert np.abs(G2[0,0]) < np.abs(G1[0,0])

def test_evaluate_rejects_nonpositive_delta_z(scalar_hamiltonian):
    """delta_z <= 0 should raise ValueError."""
    poles, orders = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,halfplane='upper'
    )
    residues = compute_residues(
        scalar_hamiltonian, poles, orders,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    with pytest.raises(ValueError):
        evaluate(0., poles, residues,halfplane='upper')
    with pytest.raises(ValueError):
        evaluate(-1., poles, residues,halfplane='upper')

# -------------------------
# Test coincidence_value()
# -------------------------
def test_coincidence_value_scalar(scalar_hamiltonian):
    """G^r(0) should equal the limit of evaluate() as Δz → 0."""
    poles, orders = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,halfplane='upper'
    )
    residues = compute_residues(
        scalar_hamiltonian, poles, orders,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    G0 = coincidence_value(residues,halfplane='upper')
    G_limit = evaluate(1e-10, poles, residues,halfplane='upper')
    assert np.allclose(G0, G_limit, atol=ATOL_APPROX)

def test_coincidence_value_consistent_with_evaluate(scalar_hamiltonian):
    """coincidence_value() should match evaluate() at Δz → 0."""
    poles, orders = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,halfplane='upper'
    )
    residues = compute_residues(
        scalar_hamiltonian, poles, orders,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    G0_direct = coincidence_value(residues,halfplane='upper')
    G0_evaluate = evaluate(1e-10, poles, residues,halfplane='upper')
    assert np.allclose(G0_direct, G0_evaluate, atol=ATOL_APPROX)

# -------------------------
# Test coincidence_derivative()
# -------------------------
def test_coincidence_derivative_scalar(scalar_hamiltonian):
    """∂_Δz G^r(0) should match numerical derivative of evaluate()."""
    poles, orders = compute_poles(
        scalar_hamiltonian, kx=0., ky=0., omega=OMEGA, eta=ETA,
        tol_filter=ATOL_STRICT, tol_cluster=ATOL_APPROX,halfplane='upper'
    )
    residues = compute_residues(
        scalar_hamiltonian, poles, orders,
        kx=0., ky=0., omega=OMEGA, eta=ETA, tol=ATOL_APPROX,
    )
    dG0 = coincidence_derivative(poles, residues,halfplane='upper')
    # numerical derivative via finite difference
    h = 1e-6
    dG0_numerical = (evaluate(h, poles, residues,halfplane='upper') - evaluate(2*h, poles, residues,halfplane='upper')) / (-h)
    assert np.allclose(dG0, dG0_numerical, atol=ATOL_APPROX)