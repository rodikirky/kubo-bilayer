""" interface.py
    ------------
    Construction of the coincidence value G(0,0) of the interfacial
    retarded Green's function, by matching the two half-space Green's
    functions at the interface.

    Physical Background
    -------------------
    At the interface z = 0, the two bulk Green's functions are joined
    by the interface Hamiltonian h_int. The coincidence value G(0,0)
    encodes all information about the interface and appears as the
    central factor in the full interfacial Green's function for
    z > 0, z' < 0:

        G^r(z,z') = F^r_R(z) · G(0,0) · F̄^r_L(z')

    where F^r_R(z) and F̄^r_L(z') are the propagation factors defined
    in bilayer.py.

    Gluing Formula
    --------------
    The coincidence value is given by (eq. 55 in supervisor's notes):

        G(0,0) = 1 / (-h_int + L^r_R - L^r_L)
            = -[-h_int + L^r_R - L^r_L]^{-1}

    where the boundary derivatives are (eqs. 56, 57):

        L^r_R = -(A_R/2) · [lim_{z→0+} d/dz F^r_R(z)] + i·B_R/2
        L^r_L = -(A_L/2) · [lim_{z→0-} d/dz F^r_L(z)] + i·B_L/2

    with:
        A_λ = 2·H2_λ = Az_λ
            the kz² coefficient matrix of the bulk Hamiltonian,
            related to the inverse effective mass tensor.

        B_λ = H1_λ(kx, ky) = kx·Czx_λ + ky·Cyz_λ + Bz_λ
            the kz¹ coefficient matrix at fixed (kx, ky),
            present in any system with mixed kz·k_∥ terms in the
            Hamiltonian, including the OAM showcase (γ_λ·k_∥·Σx).

        lim_{z→0±} d/dz F^r_{R/L}(z)
            the boundary limit of the spatial derivative of the
            propagation factor, expressed via coincidence_derivative()
            and coincidence_value() from bulk.py as:

            lim_{z→0+} d/dz F^r_R(z)
                = -∂_Δz G^r_R(0) · [G^r_R(0)]^{-1}
            lim_{z→0-} d/dz F^r_L(z)
                = ∂_Δz G^r_L(0) · [G^r_L(0)]^{-1}

    Functions
    ---------
    boundary_derivative(Az, H1, coincidence_deriv, coincidence_val)
        Computes the boundary derivative for one side:

            L^r = -(Az/2) · ∂_Δz G^r(0) · [G^r(0)]^{-1} + i·H1/2

        Parameters
        ----------
        Az               : ArrayC, shape (n, n)
            kz² coefficient matrix Az = 2·H2 of the bulk Hamiltonian.
        H1               : ArrayC, shape (n, n)
            kz¹ coefficient matrix H1(kx, ky) of the bulk Hamiltonian,
            evaluated at the current (kx, ky) grid point.
        coincidence_deriv : ArrayC, shape (n, n)
            Output of coincidence_derivative() from bulk.py.
        coincidence_val  : ArrayC, shape (n, n)
            Output of coincidence_value() from bulk.py.

        Returns
        -------
        L : ArrayC, shape (n, n)

    compute_G00(h_int, L_R, L_L)
        Assembles the coincidence value:

            G(0,0) = -[-h_int + L^r_R - L^r_L]^{-1}

        Parameters
        ----------
        h_int : ArrayC, shape (n, n)
            Interface Hamiltonian evaluated at (kx, ky).
        L_R   : ArrayC, shape (n, n)
            Right boundary derivative from boundary_derivative().
        L_L   : ArrayC, shape (n, n)
            Left boundary derivative from boundary_derivative().

        Returns
        -------
        G00 : ArrayC, shape (n, n)

    Notes
    -----
    - h_int must be evaluated at the same (kx, ky) as the poles
    and residues used to compute L^r_L and L^r_R.
    - The matrix [-h_int + L^r_R - L^r_L] must be invertible. A
    singular matrix here indicates a bound state at the interface
    and will raise a LinAlgError.
    - The sign convention follows eq. 55 of the supervisor's notes:
    G(0,0) = 1/(-h_int + L^r_R - L^r_L), note the minus sign
    on h_int.
    - Az is obtained from BulkHamiltonian as 2·H2, i.e.
    Az = 2 * hamiltonian.hamiltonian_kz_polynomial(kx, ky)[2]
    - H1 is obtained directly from
    hamiltonian.hamiltonian_kz_polynomial(kx, ky)[1]

    Dependencies
    ------------
        bulk.py                         — provides coincidence_value()
                                        and coincidence_derivative()
        kubo_bilayer.setup.hamiltonians — provides InterfaceHamiltonian.evaluate()
                                        and BulkHamiltonian.hamiltonian_kz_polynomial()
"""
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from kubo_bilayer.setup.hamiltonians import BulkHamiltonian

ArrayC = NDArray[np.complex128]

__all__ = [
    "ArrayC",
    "boundary_derivative",
    "assemble_G00",
    #"coincidence_derivative"
]


def boundary_derivative(
    Az: ArrayC,
    H1: ArrayC,
    coincidence_deriv: ArrayC,
    coincidence_val: ArrayC,
) -> ArrayC:
    """
    Compute the boundary derivative for one side of the interface:

        L^r = -(Az/2) · ∂_Δz G^r(0) · [G^r(0)]^{-1} + i·H1/2

    Parameters
    ----------
    Az               : ArrayC, shape (n, n)
        kz² coefficient matrix Az = 2·H2 of the bulk Hamiltonian.
    H1               : ArrayC, shape (n, n)
        kz¹ coefficient matrix H1(kx, ky) of the bulk Hamiltonian,
        evaluated at the current (kx, ky) grid point.
    coincidence_deriv : ArrayC, shape (n, n)
        Output of coincidence_derivative() from bulk.py.
    coincidence_val  : ArrayC, shape (n, n)
        Output of coincidence_value() from bulk.py.

    Returns
    -------
    L : ArrayC, shape (n, n)
    """
    return (
        -0.5 * Az @ coincidence_deriv @ np.linalg.inv(coincidence_val)
        + 0.5j * H1
    )

def assemble_G00(
    h_int: ArrayC,
    L_R: ArrayC,
    L_L: ArrayC,
) -> ArrayC:
    """
    Assemble the coincidence value of the interfacial Green's function:

        G(0,0) = -[-h_int + L^r_R - L^r_L]^{-1}

    Parameters
    ----------
    h_int : ArrayC, shape (n, n)
        Interface Hamiltonian evaluated at (kx, ky).
    L_R   : ArrayC, shape (n, n)
        Right boundary derivative from boundary_derivative().
    L_L   : ArrayC, shape (n, n)
        Left boundary derivative from boundary_derivative().

    Returns
    -------
    G00 : ArrayC, shape (n, n)
    """
    G00 = -h_int + L_R - L_L
    return -np.linalg.inv(G00)
