"""
toy_trivial.py
--------------
A trivial, non-physical scalar Hamiltonian for unit testing the
pole computation machinery in kubo_bilayer.numerics.poles.

System
------
The Hamiltonian is a 1x1 matrix polynomial in kz with integer
coefficients:

    H(kz) = kz² + kz + 1

This is a purely mathematical test case with no physical
interpretation. The coefficient matrices in BulkCoeffs are:

    Az = 2I   →   H2 = 1/2 * Az = I     (kz² coefficient)
    Bz = I    →   H1 = Bz = I           (kz  coefficient)
    D  = I    →   H0 = D  = I           (constant term)

all other coefficients are zero.

Analytic Solution
-----------------
At kx = ky = omega = eta = 0, the poles of the Green's function

    G̃^r(kz) = [(omega + i*eta)·I - H(kz)]⁻¹

are the roots of the characteristic equation

    kz² + kz + 1 = 0

which are exactly:

    kz = -1/2 ± i*sqrt(3)/2

Of these, only

    kz_upper = -1/2 + i*sqrt(3)/2

lies in the upper half of the complex plane and contributes to
the retarded Green's function. This value serves as the reference
for all pole computation tests.

Usage
-----
The factory function make_scalar_hamiltonian() returns a
BulkHamiltonian instance ready to be passed directly to
build_companion_matrices() and compute_poles().

    from showcase.toy_trivial import make_scalar_hamiltonian
    hamiltonian = make_scalar_hamiltonian()

Notes
-----
- All in-plane momenta kx, ky should be set to 0. when using
  this Hamiltonian, as all in-plane coefficients are zero and
  nonzero values have no physical meaning here.
- To test matrix dimension handling without changing the pole
  structure, the scalar case can be trivially extended to n_orb
  dimensions by replacing I with n_orb x n_orb identity matrices.
  The poles remain identical but the companion matrices become
  2*n_orb x 2*n_orb.

Dependencies
------------
    BulkCoeffs, BulkHamiltonian  — from kubo_bilayer.setup.hamiltonians
"""
import numpy as np
from kubo_bilayer.setup.hamiltonians import BulkCoeffs, BulkHamiltonian

def make_scalar_hamiltonian() -> BulkHamiltonian:
    """
    Trivial 1x1 Hamiltonian: H(kz) = kz² + kz + 1
    Analytic upper half-plane pole: kz = -0.5 + i*sqrt(3)/2
    No physics — pure numerical test case.
    """
    I = np.array([[1.]], dtype=np.complex128)
    Z = np.array([[0.]], dtype=np.complex128)

    coeffs = BulkCoeffs(
        Ax=Z, Ay=Z, Az=2*I,   # H2 = 1/2 * Az = I  → kz² coefficient
        Bx=Z, By=Z, Bz=I,     # H1 = Bz = I        → kz coefficient
        Cxy=Z, Cyz=Z, Czx=Z,
        D=I,                   # H0 = D = I         → constant
    )
    return BulkHamiltonian.from_coeffs(coeffs)