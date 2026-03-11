""" zp_chains.py
    ------------
    Construction of the matrix chain shared by all ten terms (I)-(Vb)
    in the Fermi-surface response.

    The chain for a pair of pole indices (αR, αL) is:

        Chain(αR, αL) = ResαR · G_R(0)⁻¹ · G(0,0) · G_L(0)⁻¹ · ResαL

    The middle factor

        middle = G_R(0)⁻¹ · G(0,0) · G_L(0)⁻¹

    is independent of the pole indices and is precomputed once per
    (kx, ky) point via build_middle_factor(). Each chain is then
    assembled by two matrix multiplications via build_chain().

    The conjugate transpose Chain(α'R, α'L)† needed for the advanced
    GF side is just build_chain(res_a_primeR, middle, res_a_primeL).conj().T
    and is not wrapped in a separate function.

    Functions
    ---------
    build_middle_factor(GR0_inv, G00, GL0_inv)
        Computes G_R(0)⁻¹ · G(0,0) · G_L(0)⁻¹.

    build_chain(res_aR, middle, res_aL)
        Computes ResαR · middle · ResαL.

    Dependencies
    ------------
        kubo_bilayer.greens.bulk      — provides G_L(0), G_R(0)
        kubo_bilayer.greens.interface — provides G(0,0)
        kubo_bilayer.numerics.residues — provides ResαL, ResαR
"""
import numpy as np
from numpy.typing import NDArray

ArrayC = NDArray[np.complex128]

__all__ = [
    "build_middle_factor",
    "build_chain",
]


def build_middle_factor(
    GR0_inv: ArrayC,
    G00: ArrayC,
    GL0_inv: ArrayC,
) -> ArrayC:
    """
    Precompute the pole-independent middle factor:

        middle = G_R(0)⁻¹ · G(0,0) · G_L(0)⁻¹

    Called once per (kx, ky) point before the quadruple sum.

    Parameters
    ----------
    GR0_inv : ArrayC, shape (n, n)
        Inverse of the right bulk GF coincidence value G_R(0).
    G00     : ArrayC, shape (n, n)
        Interface coincidence value from greens/interface.py.
    GL0_inv : ArrayC, shape (n, n)
        Inverse of the left bulk GF coincidence value G_L(0).

    Returns
    -------
    middle : ArrayC, shape (n, n)
    """
    return GR0_inv @ G00 @ GL0_inv


def build_chain(
    res_aR: ArrayC,
    middle: ArrayC,
    res_aL: ArrayC,
) -> ArrayC:
    """
    Assemble the matrix chain for a pair of pole indices (αR, αL):

        Chain(αR, αL) = ResαR · middle · ResαL

    where middle = G_R(0)⁻¹ · G(0,0) · G_L(0)⁻¹ is precomputed
    via build_middle_factor().

    For the advanced GF side, the conjugate transpose is obtained as:
        build_chain(res_a_primeR, middle, res_a_primeL).conj().T

    Parameters
    ----------
    res_aR  : ArrayC, shape (n, n)
        Residue matrix for right pole αR.
    middle  : ArrayC, shape (n, n)
        Precomputed middle factor from build_middle_factor().
    res_aL  : ArrayC, shape (n, n)
        Residue matrix for left pole αL.

    Returns
    -------
    chain : ArrayC, shape (n, n)
    """
    return res_aR @ middle @ res_aL