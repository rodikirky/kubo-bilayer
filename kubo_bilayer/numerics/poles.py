""" poles.py
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
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig

from kubo_bilayer.setup.hamiltonians import BulkHamiltonian

ArrayC = NDArray[np.complex128]

__all__ = [
    "ArrayC",
    "build_companion_matrices",
    "solve_companion_evp",
    "filter_upper_halfplane",
    "cluster_poles",
    "compute_poles"
]

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

def solve_companion_evp(
    L: ArrayC,
    M: ArrayC,
    ) -> ArrayC:
    """
    Solve the generalized linear eigenvalue problem
        L w = kz M w
    returning all 2n eigenvalues kz in the complex plane.

    Parameters
    ----------
    L, M : ArrayC, shape (2n, 2n)
        Companion matrices from build_companion_matrices()

    Returns
    -------
    kz_all : ArrayC, shape (2n,)
        All eigenvalues in the complex plane, unfiltered.
    """
    kz_all, _ = eig(L, M)
    return kz_all

def filter_upper_halfplane(
    kz_all: ArrayC,
    tol: float = 1e-10,
) -> ArrayC:
    """
    Filter poles to keep only those in the upper half of the
    complex kz-plane, i.e. Im(kz) > tol.

    Poles with Im(kz) < -tol are discarded — these contribute
    to the advanced Green's function, not the retarded one.

    Poles with |Im(kz)| <= tol lie on the real axis and
    correspond to propagating modes. These raise a warning
    as they require special treatment.

    Parameters
    ----------
    kz_all : ArrayC, shape (2n,)
        All eigenvalues from solve_companion_evp()
    tol : float
        Threshold for Im(kz) > 0 selection

    Returns
    -------
    poles : ArrayC, shape (m,)
        Poles in the upper half-plane, m <= 2n
    """
    real_axis = np.abs(np.imag(kz_all)) <= tol
    if np.any(real_axis):
        import warnings
        warnings.warn(
            f"{np.sum(real_axis)} pole(s) found on the real axis. "
            "These may require special treatment.",
            RuntimeWarning,
        )

    upper = np.imag(kz_all) > tol
    return kz_all[upper]

def cluster_poles(
    poles: ArrayC,
    tol: float,
) -> Tuple[ArrayC, NDArray[np.intp]]:
    """
    Cluster near-coincident poles and return unique poles with
    candidate orders.

    Numerically, a genuine higher-order pole appears as a cluster
    of near-coincident eigenvalues rather than an exact duplicate.
    Poles within a distance tol of each other in the complex plane
    are grouped, and each cluster is replaced by its mean.

    The cluster size is returned as a candidate pole order and
    passed to residues.py, where the definitive order determination
    is made via singular-value decomposition (SVD).

    Parameters
    ----------
    poles : ArrayC, shape (m,)
        Poles in the upper half-plane from filter_upper_halfplane()
    tol : float
        Distance threshold below which two poles are considered
        a cluster candidate.

    Returns
    -------
    unique_poles : ArrayC, shape (k,)
        Cluster centres, computed as the mean of each cluster.
    candidate_orders : NDArray[np.intp], shape (k,)
        Cluster size for each unique pole. Values > 1 indicate
        candidates for higher-order poles to be confirmed in  residues.py.
    """
    if len(poles) == 0:
        return poles, np.array([], dtype=np.intp)

    # pairwise distances in the complex plane
    distances = np.abs(poles[:, None] - poles[None, :])

    visited = np.zeros(len(poles), dtype=bool)
    unique_poles = []
    candidate_orders = []

    for i in range(len(poles)):
        if visited[i]:
            continue
        cluster = np.where(distances[i] < tol)[0]
        visited[cluster] = True
        unique_poles.append(np.mean(poles[cluster]))
        candidate_orders.append(len(cluster))

    return (
        np.array(unique_poles, dtype=np.complex128),
        np.array(candidate_orders, dtype=np.intp),
    )

def compute_poles(
    hamiltonian: BulkHamiltonian,
    kx: float,
    ky: float,
    omega: float,
    eta: float,
    tol_filter: float,
    tol_cluster: float,
) -> Tuple[ArrayC, NDArray[np.intp]]:
    """
    Compute the poles of the retarded bulk Green's function

        G̃^r(kz) = [(omega + i*eta)·I - H(kx, ky, kz)]⁻¹

    in the upper half of the complex kz-plane, for fixed (kx, ky, omega).

    This function chains the four steps of the pole computation:
        1. build_companion_matrices  — companion linearization of the PEP
        2. solve_companion_evp       — generalized eigenvalue solve
        3. filter_upper_halfplane    — keep only Im(kz) > tol_filter
        4. cluster_poles             — group near-coincident eigenvalues

    Parameters
    ----------
    hamiltonian  : BulkHamiltonian
    kx, ky       : float, in-plane momenta
    omega        : float, frequency
    eta          : float, broadening, must be strictly positive
    tol_filter   : float, threshold for upper half-plane filter
    tol_cluster  : float, threshold for pole clustering

    Returns
    -------
    unique_poles      : ArrayC, shape (k,)
        Unique poles in the upper half-plane.
    candidate_orders  : NDArray[np.intp], shape (k,)
        Cluster size for each unique pole. Values > 1 are
        candidate higher-order poles for residues.py to confirm.
    """
    L, M = build_companion_matrices(hamiltonian, kx, ky, omega, eta)
    kz_all = solve_companion_evp(L, M)
    poles = filter_upper_halfplane(kz_all, tol=tol_filter)
    return cluster_poles(poles, tol=tol_cluster)