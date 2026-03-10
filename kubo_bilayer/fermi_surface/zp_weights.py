""" zp_weights.py
    -------------
    Scalar weight factors arising from the analytical evaluation of the
    z'-integral in the Fermi-surface contribution to the spatially
    resolved linear response, as derived in Appendix A of the thesis.

    Physical Background
    -------------------
    In the cross-interface configuration z > 0, z' < 0, the Fermi-surface
    integrand contains a z'-integral of the form

        ∫_{-∞}^{0} dz' e^{i(k*_{α'L} - k_{αL}) z'} f(k_{αR}, k*_{α'R})

    where f encodes optional pole factors from derivatives of G^r or G^a
    with respect to z (not z'). The z'-integral itself always evaluates to

        ∫_{-∞}^{0} dz' e^{i(k*_{α'L} - k_{αL}) z'}
            = i / (k_{αL} - k*_{α'L})

    which converges because Im(k_{αL}) > 0 and Im(k*_{α'L}) < 0 for all
    upper-half-plane poles.

    It is convenient to decompose every scalar weight into two shared
    building blocks:

        d      = k_{αL} - k*_{α'L}         (complex, the denominator)
        |d|²   = |k_{αL} - k*_{α'L}|²      (real, positive)

    and two numerator building blocks that appear across multiple terms:

        nL  = k*_{αL} - k_{α'L}            (appears in I, IIIa, IIIb)
        pL  = |k_{αL}|² - k_{αL} k_{α'L}   (appears in IIa, IVb, Va)
        qL  = k*_{α'L} k*_{αL} - |k_{α'L}|² (appears in IIb, IVa, Vb)

    Note that pL and qL are related by complex conjugation:
        qL = (pL)*  when αL = α'L,
    but are independent in general.

    The Ten Scalar Weights
    ----------------------
    Each weight is a scalar function of the relevant subset of
    (k_{αL}, k_{α'L}, k_{αR}, k_{α'R}). Terms I, IIa, IIb depend only
    on the left poles; terms IIIa–Vb additionally involve one right pole.

    Term  | Expression
    ------|--------------------------------------------------------------
    I     |  i · nL / |d|²
    IIa   |  pL / |d|²
    IIb   | -qL / |d|²
    IIIa  |  k*_{α'R} · i · nL / |d|²
    IIIb  | -k_{αR}   · i · nL / |d|²
    IVa   | -i · k_{αR}   · (-qL) / |d|²    =  i · k_{αR}   · qL / |d|²
    IVb   | -i · k*_{α'R} · pL    / |d|²
    Va    |  i · k_{αR}   · pL    / |d|²
    Vb    |  i · k*_{α'R} · qL    / |d|²

    Functions
    ---------
    _zp_building_blocks(k_aL, k_a_primeL)
        Computes the shared building blocks nL, pL, qL, and |d|²
        for a given pair of left poles (k_{αL}, k_{α'L}).
        All ten weight functions call this internally.

    weight_I(k_aL, k_a_primeL)
    weight_IIa(k_aL, k_a_primeL)
    weight_IIb(k_aL, k_a_primeL)
    weight_IIIa(k_aL, k_a_primeL, k_a_primeR)
    weight_IIIb(k_aL, k_a_primeL, k_aR)
    weight_IVa(k_aL, k_a_primeL, k_aR)
    weight_IVb(k_aL, k_a_primeL, k_a_primeR)
    weight_Va(k_aL, k_a_primeL, k_aR)
    weight_Vb(k_aL, k_a_primeL, k_a_primeR)
        Each returns a single complex scalar weight for the corresponding
        term in the z'-integral.

    Notes
    -----
    - All functions operate on scalar complex pole values, not arrays.
      The quadruple sum over (αR, α'R, αL, α'L) is handled in
      zp_assembly.py, which calls these functions at each index tuple.
    - No matrix content appears here. The matrix chains
      ResαR · G_R(0)⁻¹ · G(0,0) · G_L(0)⁻¹ · ResαL
      are constructed in zp_chains.py.

    Dependencies
    ------------
        None. This module is fully self-contained.
"""
__all__ = [
    "_zp_building_blocks",
    "weight_I",
    "weight_IIa",
    "weight_IIb",
    "weight_IIIa",
    "weight_IIIb",
    "weight_IVa",
    "weight_IVb",
    "weight_Va",
    "weight_Vb",
]
def _zp_building_blocks(
    k_aL: complex,
    k_a_primeL: complex,
) -> tuple[complex, complex, complex, float]:
    """
    Compute the shared scalar building blocks for the z'-integral weights.

    Parameters
    ----------
    k_aL       : complex, k_{αL}  — a left pole
    k_a_primeL : complex, k_{α'L} — another left pole

    Returns
    -------
    nL   : complex,  k*_{αL} - k_{α'L}
    pL   : complex,  |k_{αL}|² - k_{αL} k_{α'L}
    qL   : complex,  k*_{α'L} k*_{αL} - |k_{α'L}|²
    d_sq : float,    |k_{αL} - k*_{α'L}|²
    """
    k_aL_conj       = k_aL.conjugate()
    k_a_primeL_conj = k_a_primeL.conjugate()

    nL   = k_aL_conj - k_a_primeL
    pL   = k_aL_conj * k_aL - k_aL * k_a_primeL
    qL   = k_a_primeL_conj * k_aL_conj - k_a_primeL_conj * k_a_primeL
    d_sq = abs(k_aL - k_a_primeL_conj) ** 2

    return nL, pL, qL, d_sq

#region weights I-Vb
def weight_I(
    k_aL: complex,
    k_a_primeL: complex,
) -> complex:
    """
    Scalar weight for term I:

        w_I = i · (k*_{αL} - k_{α'L}) / |k_{αL} - k*_{α'L}|²

    Parameters
    ----------
    k_aL       : complex, k_{αL}  — a left pole
    k_a_primeL : complex, k_{α'L} — another left pole

    Returns
    -------
    complex scalar weight
    """
    nL, _, _, d_sq = _zp_building_blocks(k_aL, k_a_primeL)
    return 1j * nL / d_sq

def weight_IIa(
    k_aL: complex,
    k_a_primeL: complex,
) -> complex:
    """
    Scalar weight for term IIa:

        w_IIa = (|k_{αL}|² - k_{αL} k_{α'L}) / |k_{αL} - k*_{α'L}|²
              = pL / d_sq
    """
    _, pL, _, d_sq = _zp_building_blocks(k_aL, k_a_primeL)
    return pL / d_sq


def weight_IIb(
    k_aL: complex,
    k_a_primeL: complex,
) -> complex:
    """
    Scalar weight for term IIb:

        w_IIb = -(k*_{α'L} k*_{αL} - |k_{α'L}|²) / |k_{αL} - k*_{α'L}|²
              = -qL / d_sq
    """
    _, _, qL, d_sq = _zp_building_blocks(k_aL, k_a_primeL)
    return -qL / d_sq

def weight_IIIa(
    k_aL: complex,
    k_a_primeL: complex,
    k_a_primeR: complex,
) -> complex:
    """
    Scalar weight for term IIIa:

        w_IIIa = k*_{α'R} · i · nL / d_sq
    """
    nL, _, _, d_sq = _zp_building_blocks(k_aL, k_a_primeL)
    return k_a_primeR.conjugate() * 1j * nL / d_sq


def weight_IIIb(
    k_aL: complex,
    k_a_primeL: complex,
    k_aR: complex,
) -> complex:
    """
    Scalar weight for term IIIb:

        w_IIIb = -k_{αR} · i · nL / d_sq
    """
    nL, _, _, d_sq = _zp_building_blocks(k_aL, k_a_primeL)
    return -k_aR * 1j * nL / d_sq

def weight_IVa(
    k_aL: complex,
    k_a_primeL: complex,
    k_aR: complex,
) -> complex:
    """
    Scalar weight for term IVa:

        w_IVa = i · k_{αR} · qL / d_sq
    """
    _, _, qL, d_sq = _zp_building_blocks(k_aL, k_a_primeL)
    return 1j * k_aR * qL / d_sq

def weight_IVb(
    k_aL: complex,
    k_a_primeL: complex,
    k_a_primeR: complex,
) -> complex:
    """
    Scalar weight for term IVb:

        w_IVb = -i · k*_{α'R} · pL / d_sq
    """
    _, pL, _, d_sq = _zp_building_blocks(k_aL, k_a_primeL)
    return -1j * k_a_primeR.conjugate() * pL / d_sq

def weight_Va(
    k_aL: complex,
    k_a_primeL: complex,
    k_aR: complex,
) -> complex:
    """
    Scalar weight for term Va:

        w_Va = i · k_{αR} · pL / d_sq
    """
    _, pL, _, d_sq = _zp_building_blocks(k_aL, k_a_primeL)
    return 1j * k_aR * pL / d_sq


def weight_Vb(
    k_aL: complex,
    k_a_primeL: complex,
    k_a_primeR: complex,
) -> complex:
    """
    Scalar weight for term Vb:

        w_Vb = i · k*_{α'R} · qL / d_sq
    """
    _, _, qL, d_sq = _zp_building_blocks(k_aL, k_a_primeL)
    return 1j * k_a_primeR.conjugate() * qL / d_sq

# endregion
