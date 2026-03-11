"""
bulk.py
-------
Construction of the bulk retarded Green's function in real space
from the poles and residues computed in kubo_bilayer.numerics.

Physical Background
-------------------
In a translation invariant bulk system, the retarded Green's
function in k-space is:

    G̃^r(kz) = [(ω + i·eta)·I - H(kx, ky, kz)]⁻¹

Its real-space form is obtained via contour integration over kz,
closing in the upper half-plane for Δz > 0. By the residue
theorem, this yields a sum over the poles {kα} in the upper
half-plane:

    G^r(Δz) = i · Σ_α e^(i·kα·Δz) · Res_α

where Res_α = Res(G̃^r; kα) are the n×n residue matrices
computed in kubo_bilayer.numerics.residues.

Functions
---------
evaluate(Δz, poles, residues)
    Evaluates G^r(Δz) at a given Δz > 0 as the sum:

        G^r(Δz) = i · Σ_α e^(i·kα·Δz) · Res_α

    This is the building block for both the half-space Green's
    function in halfspace.py and the interface coincidence value
    in interface.py.

coincidence_value(poles, residues)
    Returns the coincidence value G^r(Δz=0):

        G^r(0) = i · Σ_α Res_α

    This is a special case of evaluate() at Δz=0, but is used
    frequently enough to warrant its own function. It appears
    directly in the gluing formula for G(0,0) in interface.py.

coincidence_derivative(poles, residues)
    Returns the derivative of G^r with respect to Δz, evaluated
    at Δz=0:

        ∂_Δz G^r(Δz)|_{Δz=0} = i · Σ_α i·kα · Res_α
                               = -Σ_α kα · Res_α

    This quantity is consumed by interface.py where it is
    combined with the appropriate coefficient matrices to
    form the boundary derivative L^r.

Notes
-----
- Δz must be strictly positive for the sum to converge, since
  Im(kα) > 0 for all poles in the upper half-plane ensures
  exponential decay.
- The poles and residues passed here must have been computed at
  the same (kx, ky, ω, eta) as will be used in the response
  calculation.
- This module is agnostic to whether it is computing the left
  or right bulk Green's function — the distinction is carried
  entirely by which poles and residues are passed in.

Dependencies
------------
    kubo_bilayer.numerics.poles    — provides poles {kα}
    kubo_bilayer.numerics.residues — provides residues {Res_α}
"""
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from kubo_bilayer.setup.hamiltonians import BulkHamiltonian
from kubo_bilayer.numerics.poles import compute_poles

ArrayC = NDArray[np.complex128]

__all__ = [
    "ArrayC",
    "evaluate",
    "coincidence_value",
    "coincidence_derivative"
]

def coincidence_value(
    residues: list[ArrayC],
    halfplane: str,    # 'upper' or 'lower'
) -> ArrayC:
    """
    Evaluate the coincidence value of the bulk retarded Green's
    function at Δz = 0:

        G^r(0) = i · Σ_α Res_α

    This is the limit of evaluate() as Δz → 0 from above, but
    computed directly without the exponential factor.

    Parameters
    ----------
    residues : list[ArrayC], length k
        Residue matrices from compute_residues(), each shape (n, n).
    halfplane : str
        Which half-plane the contour closes in.
        Must be 'upper' (right bulk Green's function, Δz > 0)
        or 'lower' (left bulk Green's function, Δz < 0).

    Returns
    -------
    G0 : ArrayC, shape (n, n)
        Coincidence value G^r(0).

    Note
    -----
    - Poles are not needed in the computation.
    """
    if halfplane not in ('upper', 'lower'):
        raise ValueError(
            f"halfplane must be 'upper' or 'lower', got '{halfplane}'."
        )
    sign = 1j if halfplane == 'upper' else -1j
    return sign * sum(residues)

def evaluate(
    delta_z: float,
    poles: ArrayC,
    residues: list[ArrayC],
    halfplane: str,    # 'upper' or 'lower'
) -> ArrayC:
    """
    Evaluate the bulk retarded Green's function at a given Δz > 0:

        G^r(Δz) = i · Σ_α e^(i·kα·Δz) · Res_α

    Parameters
    ----------
    delta_z  : float
    poles    : ArrayC, shape (k,)
        Poles in the upper half-plane from compute_poles().
    residues : list[ArrayC], length k
        Residue matrices from compute_residues(), each shape (n, n).
    halfplane : str
        Which half-plane the contour closes in.
        Must be 'upper' (right bulk Green's function, Δz > 0)
        or 'lower' (left bulk Green's function, Δz < 0).

    Returns
    -------
    G : ArrayC, shape (n, n)
        Bulk retarded Green's function at Δz.
    """
    if delta_z==0:
        return coincidence_value(residues, halfplane)

    if halfplane not in ('upper', 'lower'):
        raise ValueError(
            f"halfplane must be 'upper' or 'lower', got '{halfplane}'."
        )
    sign = 1j if halfplane == 'upper' else -1j
    if halfplane == 'upper' and delta_z <= 0:
        raise ValueError(f"halfplane must be 'lower' for delta_z<=0. Got {halfplane}.")
    if halfplane == 'lower' and delta_z >= 0:
        raise ValueError(f"halfplane must be 'upper' for delta_z>=0. Got {halfplane}.")
    return sign * sum(
        np.exp(1j * pole * delta_z) * residue
        for pole, residue in zip(poles, residues)
    )

def coincidence_derivative(
    poles: ArrayC,
    residues: list[ArrayC],
    halfplane: str,    # 'upper' or 'lower'
) -> ArrayC:
    """
    Evaluate the derivative of the bulk retarded Green's function
    with respect to Δz, at Δz = 0:

        lim_{Δz→0±} ∂_Δz G^r(Δz) = ±i · Σ_α i·kα · Res_α
                               = -(±1)Σ_α kα · Res_α

    This quantity is consumed by interface.py where it is combined
    with the appropriate coefficient matrices to form the boundary
    derivative L^r.

    Parameters
    ----------
    poles    : ArrayC, shape (k,)
        Poles in the upper half-plane from compute_poles().
    residues : list[ArrayC], length k
        Residue matrices from compute_residues(), each shape (n, n).
    halfplane : str
        Which half-plane the contour closes in.
        Must be 'upper' (right bulk Green's function, Δz > 0)
        or 'lower' (left bulk Green's function, Δz < 0).

    Returns
    -------
    dG0 : ArrayC, shape (n, n)
        Derivative of G^r at Δz = 0.
    """
    if halfplane not in ('upper', 'lower'):
        raise ValueError(
            f"halfplane must be 'upper' or 'lower', got '{halfplane}'."
        )
    sign = -1 if halfplane == 'upper' else 1
    return sign*sum(
        pole * residue
        for pole, residue in zip(poles, residues)
    )