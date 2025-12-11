# ------------------------------------------------------------------------
# Imports & Setup
# ------------------------------------------------------------------------

import numpy as np
import pytest

from kubo.gluing import (
    _matrix_inverse,
    _build_F_side,
    _log_derivatives_at_interface,
    construct_glued_greens_function,
)

ArrayC = np.ndarray


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------

def _constant_halfspace_G(a: complex, n: int):
    """
    Return a Greens1D-like function G(z,z') = a * I_n for any (z,z').
    This is a purely algebraic toy model to test the gluing machinery.
    """
    I = np.eye(n, dtype=np.complex128) * a

    def G(z: float, zp: float) -> ArrayC:
        return I

    return G


# ------------------------------------------------------------------------
# Unit tests for helpers
# ------------------------------------------------------------------------

def test_matrix_inverse_basic():
    """_matrix_inverse should return a proper inverse for a complex matrix."""
    M = np.array([[1 + 1j, 2 - 1j],
                  [0.5j,     3     ]], dtype=np.complex128)
    Minv = _matrix_inverse(M)
    I = M @ Minv

    assert I.shape == (2, 2)
    assert np.allclose(I, np.eye(2, dtype=np.complex128))


def test_build_F_side_constant_case():
    """
    For a constant G(z,z') = a I, _build_F_side should give F(z) = F_bar(z) = -I.
    """
    n = 2
    a = 1.0 + 0.3j
    G = _constant_halfspace_G(a, n)

    F, F_bar = _build_F_side(G)

    for z in [0.0, -1.0, 2.0]:
        assert np.allclose(F(z),      -np.eye(n, dtype=np.complex128))
        assert np.allclose(F_bar(z),  -np.eye(n, dtype=np.complex128))


def test_log_derivatives_exponential():
    """
    _log_derivatives_at_interface should correctly approximate log derivatives
    at z=0 for simple exponential test functions.
    """
    lam = 0.7 + 0.2j
    m_L = 1.3
    m_R = 0.8
    dz = 1e-4

    # 1x1 "matrices"
    def F_L(z: float) -> ArrayC:
        return np.array([[np.exp(lam * z)]], dtype=np.complex128)

    def F_R(z: float) -> ArrayC:
        return np.array([[np.exp(-lam * z)]], dtype=np.complex128)

    L_L, L_R = _log_derivatives_at_interface(F_L, F_R, m_L, m_R, dz)

    L_L_expected = lam / (2 * m_L)
    L_R_expected = -lam / (2 * m_R)

    assert L_L.shape == (1, 1)
    assert L_R.shape == (1, 1)
    assert np.allclose(L_L[0, 0], L_L_expected, rtol=1e-4, atol=1e-6)
    assert np.allclose(L_R[0, 0], L_R_expected, rtol=1e-4, atol=1e-6)


# ------------------------------------------------------------------------
# Tests for construct_glued_greens_function
# ------------------------------------------------------------------------

def test_construct_glued_greens_constant_case():
    """
    In a constant toy model with
      G_L(z,z') = a_L I,
      G_R(z,z') = a_R I,
      H_int     = v I,

    we have:
      F_L = F_R = F_bar_L = F_bar_R = -I,
      L_L = L_R = 0,
      G00 = (-H_int)^(-1).

    Then the glued Green's function should satisfy:
      - for z,z'<0 : G = a_L I + G00
      - for z,z'>0 : G = a_R I + G00
      - for z<0<z' or z'>0>z : G = G00
      - for z=z'=0 : G = G00  (by construction)
    """
    n = 2
    a_L = 1.0 + 0.5j
    a_R = 2.0 - 0.3j
    v = 1.5  # interface strength

    H_int = v * np.eye(n, dtype=np.complex128)

    G_L = _constant_halfspace_G(a_L, n)
    G_R = _constant_halfspace_G(a_R, n)

    m_L = 1.0
    m_R = 1.0
    dz = 0.1  # arbitrary; irrelevant for constant case

    G_full = construct_glued_greens_function(G_L, G_R, H_int, m_L, m_R, dz)

    # For constant G, _build_F_side gives F=-I, so L_L=L_R=0 and
    # G00 = (-H_int)^(-1)
    G00_expected = _matrix_inverse(-H_int)

    # Left-left region
    G_ll = G_full(-1.0, -2.0)
    assert G_ll.shape == (n, n)
    assert np.allclose(G_ll, a_L * np.eye(n) + G00_expected)

    # Right-right region
    G_rr = G_full(+1.0, +2.0)
    assert np.allclose(G_rr, a_R * np.eye(n) + G00_expected)

    # Cross terms: left -> right
    G_lr = G_full(-1.0, +1.0)
    assert np.allclose(G_lr, G00_expected)

    # Cross terms: right -> left
    G_rl = G_full(+1.0, -1.0)
    assert np.allclose(G_rl, G00_expected)

    # Interface point
    G_00 = G_full(0.0, 0.0)
    assert np.allclose(G_00, G00_expected)

