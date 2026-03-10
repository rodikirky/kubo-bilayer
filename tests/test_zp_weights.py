import numpy as np
import pytest
from kubo_bilayer.fermi_surface.zp_weights import *
from conftest import ATOL_APPROX, ATOL_STRICT, ETA, OMEGA, OMEGA_DEGENERATE


#------------------------------------------------
# region _zp_building_blocks 
#------------------------------------------------
def test_building_blocks_analytic_case():
    """
    Check all four outputs against hand-computed values for
    k_aL = 1+2j, k_a_primeL = 3+4j.

    By hand:
        k_aL_conj       = 1-2j
        k_a_primeL_conj = 3-4j

        nL   = k_aL_conj - k_a_primeL
                = (1-2j) - (3+4j) = -2-6j

        pL   = |k_aL|² - k_aL * k_a_primeL
                = (1+4) - (1+2j)(3+4j)
                = 5 - (3+4j+6j+8j²)
                = 5 - (3+10j-8)
                = 5 - (-5+10j)
                = 10-10j

        qL   = k_a_primeL_conj * k_aL_conj - |k_a_primeL|²
                = (3-4j)(1-2j) - (9+16)
                = (3-6j-4j+8j²) - 25
                = (3-10j-8) - 25
                = -5-10j - 25
                = -30-10j

        d_sq = |k_aL - k_a_primeL_conj|²
                = |(1+2j) - (3-4j)|²
                = |(-2+6j)|²
                = 4+36 = 40
    """
    k_aL       = 1 + 2j
    k_a_primeL = 3 + 4j

    nL, pL, qL, d_sq = _zp_building_blocks(k_aL, k_a_primeL)

    assert np.isclose(nL,   -2 - 6j)
    assert np.isclose(pL,   10 - 10j)
    assert np.isclose(qL,  -30 - 10j)
    assert np.isclose(d_sq, 40.0)

def test_building_blocks_diagonal_case():
    """
    When k_aL == k_a_primeL = a + bj, the outputs simplify to:

        nL   = k_aL_conj - k_aL = (a-bj) - (a+bj) = -2bj
        pL   = |k_aL|² - k_aL²
                = (a²+b²) - (a+bj)²
                = (a²+b²) - (a²-b²+2abj)
                = 2b² - 2abj
                Hmm, this is not zero in general...

        Wait — let's recheck. pL = |k_aL|² - k_aL * k_a_primeL.
        With k_a_primeL = k_aL:
            pL = |k_aL|² - k_aL * k_aL = |k_aL|² - k_aL²

        For k_aL = a+bj:
            |k_aL|² = a²+b²
            k_aL²   = a²-b²+2abj
            pL      = 2b²-2abj  ← not zero

        Similarly qL = k_a_primeL_conj * k_aL_conj - |k_a_primeL|²
                        = |k_aL|²* - |k_aL|²  ... wait:
                        = k_aL_conj * k_aL_conj - |k_aL|²
                        = (k_aL_conj)² - |k_aL|²
                        = (a-bj)² - (a²+b²)
                        = a²-b²-2abj - a²-b²
                        = -2b²-2abj  ← not zero

        And d_sq = |k_aL - k_aL_conj|²
                    = |2bj|² = 4b²

    Use k_aL = k_a_primeL = 2+3j (a=2, b=3):
        nL   = -6j
        pL   = 2*9 - 2*2*3j = 18-12j
        qL   = -18-12j
        d_sq = 4*9 = 36
    """
    k_aL       = 2 + 3j
    k_a_primeL = 2 + 3j

    nL, pL, qL, d_sq = _zp_building_blocks(k_aL, k_a_primeL)

    assert np.isclose(nL,   -6j)
    assert np.isclose(pL,   18 - 12j)
    assert np.isclose(qL,  -18 - 12j)
    assert np.isclose(d_sq, 36.0)

def test_building_blocks_d_sq_real_positive():
    """
    d_sq = |k_aL - k*_{α'L}|² must be real and positive for any
    pair of upper-half-plane poles (Im > 0).
    """
    rng = np.random.default_rng(42)
    for _ in range(20):
        # generate poles with strictly positive imaginary parts
        k_aL       = rng.uniform(-3, 3) + 1j * rng.uniform(0.1, 3)
        k_a_primeL = rng.uniform(-3, 3) + 1j * rng.uniform(0.1, 3)

        _, _, _, d_sq = _zp_building_blocks(k_aL, k_a_primeL)

        assert abs(d_sq.imag) < 1e-14, (
            f"d_sq has nonzero imaginary part: {d_sq}"
        )
        assert d_sq.real > 0, (
            f"d_sq is not positive: {d_sq}"
        )


# --------------------------------------------------------
# region weight_I
# --------------------------------------------------------

def test_weight_I_analytic_case():
    """
    w_I = i * (-2-6j) / 40 = (3-j) / 20
    """
    w = weight_I(1+2j, 3+4j)
    assert np.isclose(w, (3-1j)/20, atol=ATOL_STRICT)

# TODO: test that weight_I is real and positive on the diagonal,
# i.e. weight_I(k, k) == 1 / (2 * k.imag) for any upper-half-plane pole k.

# endregion

# --------------------------------------------------------
# region weight_IIa / weight_IIb
# --------------------------------------------------------

def test_weight_IIa_analytic_case():
    """
    w_IIa = pL / d_sq = (10-10j) / 40 = (1-j) / 4
    """
    w = weight_IIa(1+2j, 3+4j)
    assert np.isclose(w, (1-1j)/4, atol=ATOL_STRICT)


def test_weight_IIb_analytic_case():
    """
    w_IIb = -qL / d_sq = (30+10j) / 40 = (3+j) / 4
    """
    w = weight_IIb(1+2j, 3+4j)
    assert np.isclose(w, (3+1j)/4, atol=ATOL_STRICT)

# endregion

# --------------------------------------------------------
# region weight_IIIa / weight_IIIb
# --------------------------------------------------------

def test_weight_IIIa_analytic_case():
    """
    w_IIIa = k*_{α'R} · i · nL / d_sq
           = (1-3j) · (3-j) / 20
           = -j / 2
    """
    w = weight_IIIa(1+2j, 3+4j, k_a_primeR=1+3j)
    assert np.isclose(w, -1j/2, atol=ATOL_STRICT)


def test_weight_IIIb_analytic_case():
    """
    w_IIIb = -k_{αR} · i · nL / d_sq
           = -(2+1j) · (3-j) / 20
           = -(7+j) / 20
    """
    w = weight_IIIb(1+2j, 3+4j, k_aR=2+1j)
    assert np.isclose(w, -(7+1j)/20, atol=ATOL_STRICT)

# endregion

# --------------------------------------------------------
# region weight_IVa / weight_IVb
# --------------------------------------------------------

def test_weight_IVa_analytic_case():
    """
    w_IVa = i · (2+1j) · (-30-10j) / 40
          = i · (-50-50j) / 40
          = (5-5j) / 4
    """
    w = weight_IVa(1+2j, 3+4j, k_aR=2+1j)
    assert np.isclose(w, (5-5j)/4, atol=ATOL_STRICT)


def test_weight_IVb_analytic_case():
    """
    w_IVb = -i · (1-3j) · (10-10j) / 40
          = -i · (-20-40j) / 40
          = (-40+20j) / 40
          = (-2+j) / 2
    """
    w = weight_IVb(1+2j, 3+4j, k_a_primeR=1+3j)
    assert np.isclose(w, (-2+1j)/2, atol=ATOL_STRICT)

# endregion

# --------------------------------------------------------
# region weight_Va / weight_Vb
# --------------------------------------------------------

def test_weight_Va_analytic_case():
    """
    w_Va = i · (2+1j) · (10-10j) / 40
         = i · (30-10j) / 40
         = (1+3j) / 4
    """
    w = weight_Va(1+2j, 3+4j, k_aR=2+1j)
    assert np.isclose(w, (1+3j)/4, atol=ATOL_STRICT)


def test_weight_Vb_analytic_case():
    """
    w_Vb = i · (1-3j) · (-30-10j) / 40
         = i · (-60+80j) / 40
         = (-4-3j) / 2
    """
    w = weight_Vb(1+2j, 3+4j, k_a_primeR=1+3j)
    assert np.isclose(w, (-4-3j)/2, atol=ATOL_STRICT)

# endregion