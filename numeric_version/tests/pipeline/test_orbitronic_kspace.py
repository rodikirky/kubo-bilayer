from __future__ import annotations

import numpy as np
import pytest

from kubo.config import PhysicsConfig
from kubo.greens import (
    kspace_greens_retarded_matrix,
    kspace_greens_retarded,
)

from kubo.models.orbitronic import OrbitronicBulk, OrbitronicBulkParams,_as_real3


# ----------------------------
# Helpers
# ----------------------------
def _advanced_matrix(omega: float, h_k: np.ndarray, eta: float) -> np.ndarray:
    """G^A(ω,k) = [(ω - iη)I - H(k)]^{-1}"""
    h_k = np.asarray(h_k, dtype=complex)
    dim = h_k.shape[0]
    I = np.eye(dim, dtype=complex)
    mat = (omega - 1j * eta) * I - h_k
    return np.linalg.solve(mat, I)


def _random_unitary(dim: int, seed: int = 0) -> np.ndarray:
    """Generate a random unitary via QR (det not enforced)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(A)
    # Fix phases so diag(R) has positive real part (standard trick)
    ph = np.exp(-1j * np.angle(np.diag(R)))
    return Q @ np.diag(ph)


@pytest.fixture
def orbitronic_bulk_default() -> OrbitronicBulk:
    # Choose "reasonable" non-zero params to exercise all terms
    return OrbitronicBulk(
        mass=1.3,
        gamma=0.7,
        J=0.4,
        magnetisation=_as_real3([0.2, -0.1, 0.97]),
    )


# --------------------------------------------------------------------------------------
# region Core smoke / invariants
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", [(0.2, -0.1, 0.35), (0.0, 0.4, -0.2), (-0.3, -0.25, 0.1)])
def test_orbitronic_hamiltonian_is_hermitian(orbitronic_bulk_default, k):
    """
    Orbitronic bulk Hamiltonian should be Hermitian for real k and real params.
    This catches basis/unitarity mistakes and accidental dtype issues early.
    """
    Hk = orbitronic_bulk_default.hamiltonian(k)
    assert Hk.shape == (3, 3)
    assert np.allclose(Hk, Hk.conj().T, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "omega,k,eta",
    [
        (0.2, (0.1, 0.2, 0.3), 1e-3),
        (1.1, (-0.4, 0.0, 0.9), 1e-4),
        (-0.7, (0.0, -0.6, 0.2), 1e-3),
    ],
)
def test_orbitronic_kspace_gr_solves_inverse_identity(orbitronic_bulk_default, omega, k, eta):
    """
    Fundamental property:
        [(ω + iη)I - H(k)] G^R(ω,k) = I
    """
    kx, ky, kz = k
    Hk = orbitronic_bulk_default.hamiltonian(k)

    GR = kspace_greens_retarded_matrix(omega=omega, h_k=Hk, eta=eta)

    I = np.eye(Hk.shape[0], dtype=complex)
    mat = (omega + 1j * eta) * I - Hk

    assert np.allclose(mat @ GR, I, rtol=1e-10, atol=1e-10)
    assert np.allclose(GR @ mat, I, rtol=1e-10, atol=1e-10)


def test_orbitronic_ga_is_conjugate_transpose_of_gr(orbitronic_bulk_default):
    """
    For Hermitian H and real ω,η>0:
        G^A(ω,k) = [G^R(ω,k)]†
    """
    omega = 0.35
    eta = 1e-3
    k = (0.2, -0.1, 0.4)

    Hk = orbitronic_bulk_default.hamiltonian(k)
    GR = kspace_greens_retarded_matrix(omega, Hk, eta)
    GA = _advanced_matrix(omega, Hk, eta)

    assert np.allclose(GA, GR.conj().T, rtol=1e-12, atol=1e-12)


def test_orbitronic_retarded_causality_sign_via_antihermitian_part(orbitronic_bulk_default):
    """
    Retarded GF should have negative semi-definite anti-Hermitian part:
        A = (G - G†) / (2i)  has eigenvalues <= 0
    (Numerically: allow tiny positive noise ~ 1e-10)
    """
    omega = 0.8
    eta = 1e-3
    k = (0.2, 0.1, 0.4)

    Hk = orbitronic_bulk_default.hamiltonian(k)
    GR = kspace_greens_retarded_matrix(omega, Hk, eta)

    A = (GR - GR.conj().T) / (2j)  # Hermitian
    evals = np.linalg.eigvalsh(A)
    assert evals.max() <= 1e-10


def test_orbitronic_kspace_greens_wrapper_matches_matrix(orbitronic_bulk_default):
    """
    The model-agnostic wrapper should match direct matrix version.
    """
    omega = 0.35
    eta = 1e-3
    kx, ky, kz = 0.2, -0.1, 0.4

    def H(kx: float, ky: float, kz: float) -> np.ndarray:
        return orbitronic_bulk_default.hamiltonian((kx, ky, kz))

    GR_wrap = kspace_greens_retarded(
        omega=omega, kx=kx, ky=ky, kz=kz, H=H, physics=PhysicsConfig(eta=eta)
    )
    GR_mat = kspace_greens_retarded_matrix(omega, H(kx, ky, kz), eta)

    assert np.allclose(GR_wrap, GR_mat, rtol=1e-12, atol=1e-12)


def test_orbitronic_kspace_greens_no_nan_inf(orbitronic_bulk_default):
    omega = 0.3
    eta = 1e-3
    k = (0.1, 0.2, 0.3)
    Hk = orbitronic_bulk_default.hamiltonian(k)
    GR = kspace_greens_retarded_matrix(omega, Hk, eta)

    assert np.isfinite(GR.real).all()
    assert np.isfinite(GR.imag).all()


# endregion
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# region Orbitronic-specific structure / symmetries
# --------------------------------------------------------------------------------------

def test_orbitronic_k_even_when_exchange_off():
    """
    With J=0 and M=0, H(k) depends on k only via k^2 and (k·L)^2,
    so H(k) = H(-k) and therefore G^R(k) = G^R(-k).
    """
    bulk = OrbitronicBulk(
        mass=1.2,
        gamma=0.9,
        J=0.0,
        magnetisation=_as_real3([0.0, 0.0, 0.0]),
    )

    omega = 0.6
    eta = 1e-3
    k = (0.2, -0.1, 0.4)
    km = tuple(-x for x in k)

    GR = kspace_greens_retarded_matrix(omega, bulk.hamiltonian(k), eta)
    GRm = kspace_greens_retarded_matrix(omega, bulk.hamiltonian(km), eta)

    assert np.allclose(GR, GRm, rtol=1e-12, atol=1e-12)


def test_orbitronic_high_frequency_limit_scaling(orbitronic_bulk_default):
    """
    For |ω| >> ||H||,  G^R(ω,k) ≈ 1/(ω+iη) I, with relative error ~ O(||H||/|ω|).
    """
    omega = 1e4
    eta = 1e-3
    k = (0.2, -0.1, 0.4)

    Hk = orbitronic_bulk_default.hamiltonian(k)
    GR = kspace_greens_retarded_matrix(omega, Hk, eta)

    z = omega + 1j * eta
    I = np.eye(3, dtype=complex)
    approx = I / z

    # Relative error should scale like ||H||/|ω|
    rel_err = np.linalg.norm(GR - approx, ord="fro") / np.linalg.norm(approx, ord="fro")
    scale = np.linalg.norm(Hk, ord="fro") / abs(omega)

    # Generous constant to avoid flakiness across params/platforms
    assert rel_err <= 5.0 * scale + 1e-12



def test_orbitronic_unitary_covariance_under_basis_change():
    """
    OrbitronicBulk supports a unitary basis transform U such that:
        H' = U† H U
    The GF should transform covariantly:
        G' = U† G U
    This is a *powerful* test for subtle basis handling issues.
    """
    U = _random_unitary(3, seed=123)

    M = np.asarray([0.2, -0.1, 0.97], dtype=float)
    params = OrbitronicBulkParams(
        mass=1.3,
        gamma=0.7,
        J=0.4,
        magnetisation=M,
    )
    bulk = OrbitronicBulk.from_params(params, basis=None)
    bulk_rot = OrbitronicBulk.from_params(params, basis=U)

    omega = 0.45
    eta = 1e-3
    k = (0.23, -0.17, 0.31)

    H = bulk.hamiltonian(k)
    H_rot = bulk_rot.hamiltonian(k)

    # Hamiltonian covariance check
    assert np.allclose(H_rot, U.conj().T @ H @ U, rtol=1e-12, atol=1e-12)

    # Green's function covariance check
    G = kspace_greens_retarded_matrix(omega, H, eta)
    G_rot = kspace_greens_retarded_matrix(omega, H_rot, eta)
    assert np.allclose(G_rot, U.conj().T @ G @ U, rtol=1e-10, atol=1e-12)


# endregion
# --------------------------------------------------------------------------------------
