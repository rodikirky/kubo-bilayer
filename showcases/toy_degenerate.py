"""
toy_degenerate.py
-----------------
A trivial, non-physical 2x2 matrix Hamiltonian with a known
degenerate pole for unit testing the pole clustering machinery
in kubo_bilayer.numerics.poles and the residue construction
in kubo_bilayer.numerics.residues.

System
------
The Hamiltonian is a 2x2 matrix polynomial in kz with a
degenerate pole structure:

    H(kz) = kz²·I₂

This is a purely mathematical test case with no physical
interpretation. The coefficient matrices in BulkCoeffs are:

    Az = 2·I₂  →   H2 = 1/2 * Az = I₂    (kz² coefficient)

all other coefficients are zero.

Analytic Solution
-----------------
At kx = ky = 0, omega = 1, eta → 0, the poles of the Green's
function

    G̃^r(kz) = [(omega + i*eta)·I - H(kz)]⁻¹
             = [(1 + i*eta)·I - kz²·I]⁻¹

are the roots of the characteristic equation

    kz² = 1 + i*eta

which gives:

    kz = ±sqrt(1 + i*eta) ≈ ±1  (for small eta)

Of these, only

    kz_upper ≈ +1

lies in the upper half of the complex plane for eta > 0.

Degenerate Structure
--------------------
Since H(kz) = kz²·I₂ is proportional to the identity, both
rows of the 2x2 matrix (H(kα) - ω·I) vanish simultaneously
at the pole kα ≈ +1. This gives a two-dimensional nullspace
with two linearly independent null vectors.

The SVD of H(kα) - ω·I therefore yields two small singular
values, confirming this as a degenerate first-order pole rather
than a genuine second-order pole. Concretely:

    S = W0† M'(kα) V0

is invertible (first-order), but the residue decomposes into
a sum of two independent rank-1 terms rather than a single one.

This distinguishes it from a genuine second-order pole, where
the nullspace is one-dimensional and S is singular, requiring
Jordan chain construction.

Expected Clustering Behaviour
------------------------------
After solve_companion_evp() at omega=1, eta=1e-6, the four
eigenvalues of the 4x4 companion system should contain two
near-coincident values close to +1 in the upper half-plane:

    kz_1 ≈ +1 + ε₁
    kz_2 ≈ +1 + ε₂

cluster_poles() should group these into a single cluster with:

    unique_pole  ≈ +1
    order        = 2  (candidate — confirmed as degenerate
                       first-order in residues.py via
                       nullspace dimension of H(kα) - ω·I)

Usage
-----
The factory function make_degenerate_hamiltonian() returns a
BulkHamiltonian instance ready to be passed directly to
build_companion_matrices() and compute_poles().

    from showcases.toy_degenerate import make_degenerate_hamiltonian
    hamiltonian = make_degenerate_hamiltonian()

Notes
-----
- Use omega=1. and a small eta when constructing companion
  matrices for this Hamiltonian, as omega=0 places the pole
  on the real axis.
- This toy model is specifically designed to test the boundary
  between cluster_poles() and residues.py — cluster_poles()
  flags it as a candidate order-2 pole, and residues.py
  confirms it as degenerate first-order via SVD.
- For a genuine second-order pole test case, see
  toy_second_order.py (TODO).

Dependencies
------------
    BulkCoeffs, BulkHamiltonian  — from kubo_bilayer.setup.hamiltonians
"""
import numpy as np
from kubo_bilayer.setup.hamiltonians import BulkCoeffs, BulkHamiltonian

def make_degenerate_pole_hamiltonian() -> BulkHamiltonian:
    """
    2x2 Hamiltonian: H(kz) = kz²·I₂
    At omega=1, eta→0 both eigenvalues give kz = ±1,
    yielding a degenerate (not higher-order) pole at kz = +1.
    """
    I2 = np.eye(2, dtype=np.complex128)
    Z2 = np.zeros((2, 2), dtype=np.complex128)

    coeffs = BulkCoeffs(
        Ax=Z2, Ay=Z2, Az=2*I2,
        Bx=Z2, By=Z2, Bz=Z2,
        Cxy=Z2, Cyz=Z2, Czx=Z2,
        D=Z2,
    )
    return BulkHamiltonian.from_coeffs(coeffs)