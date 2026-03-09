"""
poles.py
--------
Computation of the poles of the retarded bulk Green's function

    G̃^r(kz) = [(ω+i*eta)·I-H(kx, ky, kz)]⁻¹

in the upper half of the complex kz-plane, for fixed (kx, ky, ω).

Procedure
---------
The bulk Hamiltonian H(kx, ky, kz) is a quadratic polynomial in kz:

    H(kz) = H0 + H1 kz + H2 kz²

where H0, H1, H2 are nxn coefficient matrices at fixed (kx, ky),
obtained from BulkHamiltonian.hamiltonian_kz_polynomial(). The poles
of G̃^r(kz) are the values of kz solving the quadratic PEP:

    (H2 kz² + H1 kz + (H0 - (ω+i*eta)·I)) v = 0.

Step 1 — Companion linearization
    The quadratic PEP is recast as a 2nx2n generalized linear
    eigenvalue problem

        L w = kz M w,

    with companion matrices

        L = [ H1   (H0 - (ω+i*eta)·I) ]      M = [ -H2   0 ]
            [ I            0          ]          [  0    I ]

    The 2n eigenvalues of this problem are exactly the 2n poles of
    G̃^r(kz). The ω shift is introduced here and nowhere else.

Step 2 — Eigenvalue solve
    The generalized eigenvalue problem is solved numerically via
    numpy.linalg.eig(L, M), yielding all 2n poles in the complex
    kz-plane.

Step 3 — Upper half-plane filter
    Only poles with Im(kz) > tol contribute to the retarded Green's
    function, which decays for z > 0. Poles below the real axis
    are discarded. Poles on the real axis raise an error.

Step 4 — Clustering and order detection
    Numerically, a genuine higher-order pole appears as a cluster of
    near-coincident eigenvalues. Poles within a distance tol of each
    other in the complex plane are grouped, and each cluster is
    replaced by its mean. The maximal order of a pole is 2, but the
    eigenvalues my be further degenerate. 
    Higher cluster size is flagged as a candidate for a 2nd order
    pole and passed to residues.py, where the definitive order
    determination is made.

Notes
-----
- The maximum pole order for a quadratic matrix polynomial of
  dimension n is 2.
- Second-order poles occur only at exceptional points in (kx, ky, ω)
  space and are not expected at generic grid points. A runtime warning
  is issued when a candidate higher-order cluster is detected.
- The eigenvectors of the companion problem are not returned, as the
  residue computation in residues.py derives the nullspace vectors
  independently and more stably via SVD.

Dependencies
------------
    BulkHamiltonian  — provides hamiltonian_kz_polynomial()
    residues.py      — consumes the output (poles, candidate_orders)
"""
#TODO: Make sure that poles on the real axis raise an error.
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from setup.hamiltonians import BulkHamiltonian

ArrayC = NDArray[np.complex128]

def build_companion_matrices(
    Hamiltonian: BulkHamiltonian,
    kx: float,
    ky: float,
    omega: float,
    eta: float
) -> Tuple[ArrayC, ArrayC]:
    """
    Build the 2n x 2n companion matrices L, M for the
    generalized linear eigenvalue problem
        L w = kz M w
    equivalent to the quadratic PEP
        M v = 0, 
    where 
        M = (ω+i*eta)·I-H(kx, ky, kz) and v not 0.

    Parameters
    ----------
    Hamiltonian: BulkHamiltonian, dataclass
        Provides coefficient matrices H0, H1, H2 from hamiltonian_kz_polynomial()
        and identity matrix I as property of BulkHamiltonian
    kx, ky: float
        In-plane momenta
    omega : complex
        Frequency shift
    eta : float, real, positive
        Broadening

    Returns
    -------
    L, M : ArrayC, shape (2n, 2n)
    """
    # validate real inputs
    for name, val in [("kx", kx), ("ky", ky), ("omega", omega), ("eta", eta)]:
        if not np.isreal(val):
            raise ValueError(f"{name} must be real, got {val}.")
    if eta <= 0:
        raise ValueError(f"eta must be strictly positive, got {eta}.")
    H0, H1, H2 = Hamiltonian.hamiltonian_kz_polynomial(kx,ky)
    n = Hamiltonian.matrix_dim
    I = Hamiltonian.identity
    Z = np.zeros((n, n), dtype=np.complex128)

    R = np.array(H0 - (omega+1j*eta) * I, dtype=np.complex128)

    L = np.block([[H1,  R],
                  [ I,  Z]])
    M = np.block([[-H2, Z],
                  [ Z,  I]])

    return L, M