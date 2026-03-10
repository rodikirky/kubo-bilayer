""" residues.py
    -----------
    Construction of the residues of the retarded bulk Green's function

        G̃^r(kz) = [(ω + i*eta)·I - H(kx, ky, kz)]⁻¹

    at each pole kα in the upper half of the complex kz-plane,
    for fixed (kx, ky, ω).

    Physical Background
    -------------------
    The real-space retarded bulk Green's function is obtained from
    G̃^r(kz) via contour integration, closing in the upper half-plane.
    By the residue theorem, it decomposes as a sum over poles:

        G^r(Δz) = i Σ_α e^(i kα Δz) Res(G̃^r; kα)

    where Δz>0 and the residues Res(G̃^r; kα) are n×n matrices. Their
    construction is the purpose of this module.

    Procedure
    ---------
    At each pole kα, the matrix

        P(kα) = (ω + i*eta)·I - H(kx, ky, kα)

    is singular by definition. The residue structure depends on the
    order of the pole, determined via the following steps:

    Step 1 — SVD of P(kα)
        The singular value decomposition

            P(kα) = U · Σ · V†

        is computed. The null vectors are extracted from the columns
        of U and V† corresponding to singular values below a
        threshold:

            V0 : right null vector(s) of P(kα)
            U0 : left null vector(s)  of P(kα)

        The number of singular values below threshold gives the
        nullspace dimension, which distinguishes degenerate
        first-order poles (dim > 1) from candidate second-order
        poles (dim = 1, flagged by cluster_poles).

    Step 2 — Order determination via S matrix
        For candidate higher-order poles (cluster size > 1,
        nullspace dimension = 1), the matrix

            S = U0† · P'(kα) · V0

        is computed, where P'(kα) = -H1 - 2·kα·H2 is the derivative
        of the PEP matrix at kα, with H1 and H2 obtained from
        BulkHamiltonian.hamiltonian_kz_polynomial().

        - S invertible  → genuine first-order pole, despite clustering
        - S singular    → genuine second-order pole, Jordan chain needed

        A RuntimeWarning is issued in both cases to flag that a
        candidate higher-order pole was encountered and resolved.

    Step 3 — Residue construction
        Three cases are handled:

        Case 1 — Simple first-order pole (nullspace dim = 1, S invertible):
            The residue is rank-1:

                Res(G̃^r; kα) = V0 · S⁻¹ · U0†

        Case 2 — Degenerate first-order pole (nullspace dim > 1):
            The residue decomposes into a sum of rank-1 terms,
            one per null vector pair:

                Res(G̃^r; kα) = Σ_j V0_j · S_j⁻¹ · U0_j†

            where S_j = U0_j† · P'(kα) · V0_j for each j.

        Case 3 — Genuine second-order pole (nullspace dim = 1, S singular):
            The Laurent expansion has two singular terms:

                G̃^r(kz) = R₋₂/(kz - kα)² + R₋₁/(kz - kα) + regular

            R₋₂ and R₋₁ are constructed via the Jordan chain:

                P(kα) v₀ = 0
                P(kα) v₁ = -P'(kα) v₀

            A RuntimeWarning is issued as genuine second-order poles
            are not expected at generic grid points.

    Notes
    -----
    - The ω and eta values passed here must match those used in
    poles.py to compute kα — P(kα) is only singular for the
    correct (ω, eta) pair.
    - All SVD thresholds are controlled by tol, which should match
    the cluster tolerance used in cluster_poles().
    - No default values are provided for numerical tolerances. All
    thresholds must be passed explicitly by the caller, as reasonable
    values depend on the energy scale of the physical system.
    - The genuine second-order pole case (Case 3) is implemented
    for completeness but is not expected at generic (kx, ky, ω)
    grid points. Even thinking of a test case like
    toy_double_pole.py (TODO) is difficult.

    Note on second-order poles
    --------------------------
    Genuine second-order poles are not expected at generic (kx, ky, ω)
    grid points and have not been observed in the systems studied here.
    However, their existence for Hermitian Hamiltonians has not been
    ruled out in general. The Jordan chain construction in residues.py
    is included (TODO) to handle this case if encountered.

    Dependencies
    ------------
        BulkHamiltonian  — provides hamiltonian_kz_polynomial()
                        for P'(kα) = H1 + 2·kα·H2
        poles.py         — provides unique_poles, candidate_orders
                        consumed as inputs here
"""
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from kubo_bilayer.setup.hamiltonians import BulkHamiltonian

ArrayC = NDArray[np.complex128]

__all__ = [
    "ArrayC",
    "compute_null_vectors",
    "compute_S_matrix",
    "compute_residue_from_S",
    "compute_residues"
]

def compute_null_vectors(
    hamiltonian: BulkHamiltonian,
    pole: complex,
    kx: float,
    ky: float,
    omega: float,
    eta: float,
    tol: float,
) -> Tuple[ArrayC, ArrayC]:
    """
    Compute the left and right null vectors of

        P(kα) = (ω + i*eta)·I - H(kx, ky, kα)

    via SVD. The null vectors correspond to singular values
    below tol.

    Parameters
    ----------
    hamiltonian : BulkHamiltonian
    pole        : complex, kα from compute_poles()
    kx, ky      : float, in-plane momenta
    omega, eta  : float, must match values used in compute_poles()
    tol         : float, threshold for null space detection

    Returns
    -------
    U0 : ArrayC, shape (n, d)
        Left null vectors, columns of U corresponding to
        singular values below tol.
    V0 : ArrayC, shape (n, d)
        Right null vectors, columns of V† corresponding to
        singular values below tol.
    """
    H_at_pole = hamiltonian.evaluate(kx, ky, pole)
    P = (omega + 1j * eta) * hamiltonian.identity - H_at_pole
    U, S, Vh = np.linalg.svd(P)
    null_mask = S < tol
    U0 = U[:, null_mask]
    V0 = Vh.conj().T[:, null_mask]
    return U0, V0

def compute_S_matrix(
    hamiltonian: BulkHamiltonian,
    pole: complex,
    kx: float,
    ky: float,
    U0: ArrayC,
    V0: ArrayC,
) -> ArrayC:
    """
    Compute the S matrix

        S = U0† · P'(kα) · V0

    where P'(kα) = H1 + 2·kα·H2 is the derivative of the PEP
    matrix at the pole kα.

    S is used to determine the pole order:
        - S invertible → first-order pole (simple or degenerate)
        - S singular   → genuine second-order pole

    And to construct the residue in the first-order case:
        - Res(G̃^r; kα) = V0 · S⁻¹ · U0†  (first-order pole kα)

    Parameters
    ----------
    hamiltonian : BulkHamiltonian
    pole        : complex, kα from compute_poles()
    kx, ky      : float, in-plane momenta
    U0          : ArrayC, shape (n, d), left null vectors
    V0          : ArrayC, shape (n, d), right null vectors

    Returns
    -------
    S : ArrayC, shape (d, d)
    """
    H0, H1, H2 = hamiltonian.hamiltonian_kz_polynomial(kx, ky)
    P_prime = -H1 - 2 * pole * H2
    return U0.conj().T @ P_prime @ V0

def compute_residue_from_S(
    U0: ArrayC,
    V0: ArrayC,
    S: ArrayC,
) -> ArrayC:
    """
    Construct the residue matrix of the retarded bulk Green's function

        Res(G̃^r; kα) = V0 · S⁻¹ · U0†

    at a pole kα, from the null vectors and S matrix.

    Handles two cases:

    Case 1 — Simple first-order pole (nullspace dim = 1):
        S is 1×1 and invertible. The residue is rank-1:
            Res = V0 · S⁻¹ · U0†

    Case 2 — Degenerate first-order pole (nullspace dim > 1):
        S is d×d and invertible. The residue is a sum of
        rank-1 terms:
            Res = V0 · S⁻¹ · U0†
        Note this is the same formula — the matrix dimensions
        handle the summation automatically.

    Parameters
    ----------
    U0 : ArrayC, shape (n, d)
        Left null vectors from compute_null_vectors()
    V0 : ArrayC, shape (n, d)
        Right null vectors from compute_null_vectors()
    S  : ArrayC, shape (d, d)
        S matrix from compute_S_matrix()

    Returns
    -------
    residue : ArrayC, shape (n, n)
    """
    S_inv = np.linalg.inv(S)
    return V0 @ S_inv @ U0.conj().T

def compute_residues(
    hamiltonian: BulkHamiltonian,
    poles: ArrayC,
    candidate_orders: NDArray[np.intp],
    kx: float,
    ky: float,
    omega: float,
    eta: float,
    tol: float,
) -> list[ArrayC]:
    """
    Compute the residues of the retarded bulk Green's function

        G̃^r(kz) = [(ω + i·eta)·I - H(kx, ky, kz)]⁻¹

    at each pole in the upper half-plane.

    This function chains the residue computation steps:
        1. compute_null_vectors  — SVD of P(kα)
        2. compute_S_matrix      — S = U0† · P'(kα) · V0
        3. compute_residue_from_S — Res = V0 · S⁻¹ · U0†

    For each pole, the candidate order from cluster_poles() is
    checked against the nullspace dimension to confirm whether
    the pole is simple, degenerate, or genuine second-order.
    A RuntimeWarning is issued for any candidate higher-order
    pole encountered.

    Parameters
    ----------
    hamiltonian       : BulkHamiltonian
    poles             : ArrayC, shape (k,)
        Unique poles from compute_poles()
    candidate_orders  : NDArray[np.intp], shape (k,)
        Candidate orders from compute_poles()
    kx, ky            : float, in-plane momenta
    omega, eta        : float, must match values used in compute_poles()
    tol               : float, SVD threshold for null vector detection

    Returns
    -------
    residues : list[ArrayC], length k
        Residue matrix for each unique pole, shape (n, n) each.
    """
    residues = []

    for pole, order in zip(poles, candidate_orders):
        U0, V0 = compute_null_vectors(
            hamiltonian, pole, kx, ky, omega, eta, tol=tol
        )
        nullspace_dim = U0.shape[1]

        if order > 1 and nullspace_dim == 1:
            raise NotImplementedError(
                f"Genuine second-order pole detected at kz ≈ {pole:.6f}. "
                f"Jordan chain construction is not yet implemented. "
                f"This is not expected at generic (kx, ky, omega) grid points."
            )

        S = compute_S_matrix(hamiltonian, pole, kx, ky, U0, V0)
        residues.append(compute_residue_from_S(U0, V0, S))

    return residues