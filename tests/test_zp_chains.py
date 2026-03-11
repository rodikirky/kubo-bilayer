""" test_zp_chains.py
    ------------------
    Tests for the matrix chain functions in
    kubo_bilayer.fermi_surface.zp_chains.

    Currently tests:
        build_middle_factor
        build_chain
"""
import numpy as np
from conftest import ATOL_STRICT
from kubo_bilayer.fermi_surface.zp_chains import *


def test_middle_factor_analytic_case():
    """
    For diagonal matrices the result is just elementwise multiplication.
    GR0_inv = diag(2, 3), G00 = diag(4, 5), GL0_inv = diag(6, 7):
        middle = diag(2*4*6, 3*5*7) = diag(48, 105)
    """
    GR0_inv = np.diag([2., 3.]).astype(np.complex128)
    G00     = np.diag([4., 5.]).astype(np.complex128)
    GL0_inv = np.diag([6., 7.]).astype(np.complex128)

    middle = build_middle_factor(GR0_inv, G00, GL0_inv)

    assert np.allclose(middle, np.diag([48., 105.]), atol=ATOL_STRICT)


def test_middle_factor_order_matters():
    """
    Verify that the multiplication order GR0_inv @ G00 @ GL0_inv is
    respected, not e.g. GL0_inv @ G00 @ GR0_inv.
    Use non-symmetric matrices so that order is detectable.
    """
    GR0_inv = np.array([[1, 2], [0, 1]], dtype=np.complex128)
    G00     = np.array([[1, 0], [3, 1]], dtype=np.complex128)
    GL0_inv = np.array([[1, 4], [0, 1]], dtype=np.complex128)

    middle   = build_middle_factor(GR0_inv, G00, GL0_inv)
    reversed = build_middle_factor(GL0_inv, G00, GR0_inv)

    assert not np.allclose(middle, reversed, atol=ATOL_STRICT)


def test_build_chain_analytic_case():
    """
    For diagonal matrices:
    res_aR = diag(2, 3), middle = diag(4, 5), res_aL = diag(6, 7):
        chain = diag(2*4*6, 3*5*7) = diag(48, 105)
    """
    res_aR  = np.diag([2., 3.]).astype(np.complex128)
    middle  = np.diag([4., 5.]).astype(np.complex128)
    res_aL  = np.diag([6., 7.]).astype(np.complex128)

    chain = build_chain(res_aR, middle, res_aL)

    assert np.allclose(chain, np.diag([48., 105.]), atol=ATOL_STRICT)


def test_build_chain_conj_transpose():
    """
    The advanced GF chain Chain(α'R, α'L)† is just
    build_chain(res_a_primeR, middle, res_a_primeL).conj().T.
    Verify this is not equal to the original chain for generic inputs.
    """
    rng    = np.random.default_rng(42)
    middle = rng.random((3, 3)) + 1j * rng.random((3, 3))
    res_aR = rng.random((3, 3)) + 1j * rng.random((3, 3))
    res_aL = rng.random((3, 3)) + 1j * rng.random((3, 3))

    chain      = build_chain(res_aR, middle, res_aL)
    chain_conj = build_chain(res_aR, middle, res_aL).conj().T

    assert not np.allclose(chain, chain_conj, atol=ATOL_STRICT)