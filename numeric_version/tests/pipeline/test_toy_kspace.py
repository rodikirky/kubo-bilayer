import numpy as np
import pytest

from kubo.config import PhysicsConfig
from kubo.greens import (
    kspace_greens_retarded_matrix,
    kspace_greens_retarded,
)

# --------------------------------------------------------------------------------------
# region Core correctness
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "omega,kx,ky,kz,eta",
    [
        (0.2,  0.1,  0.2,  0.3, 1e-3),
        (1.7, -0.4,  0.0,  0.9, 1e-4),
        (-0.5, 0.0, -0.6,  0.2, 1e-3),
    ],
)
def test_kspace_greens_retarded_matrix_solves_inverse_identity(omega, kx, ky, kz, eta, toy_H_spec):
    """
    Fundamental property:
        [(ω + iη)I - H(k)] G^R(ω,k) = I
    """
    Hk = toy_H_spec(kx, ky, kz)
    GR = kspace_greens_retarded_matrix(omega=omega, h_k=Hk, eta=eta)

    dim = Hk.shape[0]
    mat = (omega + 1j * eta) * np.eye(dim, dtype=complex) - Hk
    I = np.eye(dim, dtype=complex)

    assert np.allclose(mat @ GR, I, rtol=1e-10, atol=1e-10)
    assert np.allclose(GR @ mat, I, rtol=1e-10, atol=1e-10)


def test_kspace_greens_retarded_wrapper_matches_matrix(toy_H_spec):
    """
    The convenience wrapper should match the direct matrix version.
    """
    H = toy_H_spec

    omega = 0.35
    kx, ky, kz = 0.2, -0.1, 0.4
    eta = 1e-3

    GR_wrap = kspace_greens_retarded(
        omega=omega, kx=kx, ky=ky, kz=kz, H=H, physics=PhysicsConfig(eta=eta)
    )

    GR_mat = kspace_greens_retarded_matrix(
        omega=omega, h_k=H(kx, ky, kz), eta=eta
    )

    assert np.allclose(GR_wrap, GR_mat, rtol=1e-12, atol=1e-12)


def _advanced_matrix(omega: float, h_k: np.ndarray, eta: float) -> np.ndarray:
    """
    G^A(ω, k) = [ (ω - iη) I - H(k) ]^{-1}
    """
    h_k = np.asarray(h_k, dtype=complex)
    dim = h_k.shape[0]
    I = np.eye(dim, dtype=complex)
    mat = (omega - 1j * eta) * I - h_k
    return np.linalg.solve(mat, np.eye(dim, dtype=complex))

def test_ga_is_conjugate_transpose_of_gr(toy_bulk_spec):
    omega = 0.35
    eta = 1e-3
    kx, ky, kz = 0.2, -0.1, 0.4
    Hk = toy_bulk_spec.hamiltonian((kx, ky, kz))

    GR = kspace_greens_retarded_matrix(omega, Hk, eta)
    GA = _advanced_matrix(omega, Hk, eta)

    assert np.allclose(GA, GR.conj().T, rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize("omega,kx,ky,kz", [
    (0.3, 0.1, 0.2, 0.3),
    (1.1, -0.4, 0.0, 0.9),
    (-0.7, 0.0, -0.6, 0.2),
])
def test_kspace_greens_no_nan_inf(toy_H_spec, omega, kx, ky, kz):
    GR = kspace_greens_retarded(
        omega=omega, kx=kx, ky=ky, kz=kz,
        H=toy_H_spec, physics=PhysicsConfig(eta=1e-3)
    )
    assert np.isfinite(GR.real).all()
    assert np.isfinite(GR.imag).all()


# endregion
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# region Toy model specific structure
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "omega,kx,ky,kz,eta",
    [
        (0.1, 0.3, 0.0, 0.2, 1e-3),
        (2.0, 0.0, 0.4, 0.5, 1e-4),
        (-1.2, -0.2, 0.1, 0.0, 1e-3),
    ],
)
def test_toy_kspace_greens_is_diagonal_and_matches_closed_form(omega, kx, ky, kz, eta, toy_bulk_spec):
    """
    For the toy model:
        H(k) = e(k) I + Δ σ_z  =>  diag(e+Δ, e-Δ)
    Therefore:
        G^R = diag( 1/(ω+iη-(e+Δ)), 1/(ω+iη-(e-Δ)) )
    """
    bulk = toy_bulk_spec
    m, Delta = bulk.mass, bulk.gap

    k2 = kx * kx + ky * ky + kz * kz
    e = k2 / (2.0 * m)

    GR = kspace_greens_retarded_matrix(omega, bulk.hamiltonian((kx, ky, kz)), eta)

    # Off-diagonals should be ~ 0
    assert abs(GR[0, 1]) < 1e-12
    assert abs(GR[1, 0]) < 1e-12

    g_plus = 1.0 / (omega + 1j * eta - (e + Delta))
    g_minus = 1.0 / (omega + 1j * eta - (e - Delta))

    assert np.allclose(GR[0, 0], g_plus, rtol=1e-12, atol=1e-12)
    assert np.allclose(GR[1, 1], g_minus, rtol=1e-12, atol=1e-12)


def test_retarded_gf_has_negative_imag_diagonal_for_toy(toy_bulk_spec):
    """
    For η>0 and real ω, a retarded GF should have Im G_ii(ω) < 0
    (away from numerical pathologies).
    """
    bulk = toy_bulk_spec
    omega = 0.8
    eta = 1e-3
    kx, ky, kz = 0.2, 0.1, 0.4

    GR = kspace_greens_retarded_matrix(omega, bulk.hamiltonian((kx, ky, kz)), eta)

    assert GR[0, 0].imag < 0.0
    assert GR[1, 1].imag < 0.0

def test_toy_greens_even_in_k(toy_bulk_spec):
    omega = 0.6
    eta = 1e-3
    k = (0.2, -0.1, 0.4)
    km = tuple(-x for x in k)

    Hk  = toy_bulk_spec.hamiltonian(k)
    Hkm = toy_bulk_spec.hamiltonian(km)

    GR  = kspace_greens_retarded_matrix(omega, Hk, eta)
    GRm = kspace_greens_retarded_matrix(omega, Hkm, eta)

    assert np.allclose(GR, GRm, rtol=1e-12, atol=1e-12)

# endregion
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# region Input validation tests
# --------------------------------------------------------------------------------------

def test_kspace_greens_retarded_matrix_rejects_non_square():
    omega = 0.1
    eta = 1e-3

    with pytest.raises(ValueError):
        kspace_greens_retarded_matrix(omega, np.array([1.0, 2.0, 3.0]), eta)

    with pytest.raises(ValueError):
        kspace_greens_retarded_matrix(omega, np.ones((2, 3)), eta)

# endregion
# --------------------------------------------------------------------------------------