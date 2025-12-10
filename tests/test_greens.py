# ---------------------------------------------
# Imports
# ---------------------------------------------
import numpy as np
import pytest

from kubo.config import PhysicsConfig
from kubo.greens import (
    BulkHamiltonian,
    kspace_greens_retarded_matrix,
    kspace_greens_retarded,
)

# ---------------------------------------------
# Fixtures
# ---------------------------------------------
@pytest.fixture
def physics() -> PhysicsConfig:
    return PhysicsConfig() # default parameters

@pytest.fixture
def scalar_h() -> BulkHamiltonian:
    # Simple 1×1 free-particle-like dispersion: ε(k) = kx^2 + ky^2 + kz^2
    def H(kx: float, ky: float, kz: float) -> np.ndarray:
        eps = kx**2 + ky**2 + kz**2
        return np.array([[eps]], dtype=float)
    return H

@pytest.fixture
def two_by_two_h() -> BulkHamiltonian:
    # Simple 2×2 Hermitian Hamiltonian: H = [[m, Δ], [Δ, -m]]
    def H(kx: float, ky: float, kz: float) -> np.ndarray:
        m = kx  # just to make it k-dependent
        delta = 0.5
        return np.array([[m, delta], [delta, -m]], dtype=float)
    return H


# ---------------------------------------------
# region k-space GF
# ---------------------------------------------
def test_kspace_greens_retarded_matrix_scalar(physics: PhysicsConfig, scalar_h):
    omega = 1.0
    kx, ky, kz = 0.5, 0.0, 0.0

    h_k = scalar_h(kx, ky, kz)   # shape (1,1)
    G = kspace_greens_retarded_matrix(omega, h_k, physics.eta)

    # Shape and dtype
    assert G.shape == (1, 1)
    assert np.iscomplexobj(G)

    eps = kx**2 + ky**2 + kz**2
    denom = (omega + 1j * physics.eta) - eps
    G_exact = 1.0 / denom

    assert G[0, 0] == pytest.approx(G_exact)

    # For retarded GF, Im G^R(ω) should be negative for real ω
    assert G[0, 0].imag < 0.0

def test_kspace_greens_retarded_matrix_two_by_two(physics: PhysicsConfig, two_by_two_h):
    omega = 0.3
    kx, ky, kz = 0.1, -0.2, 0.0

    h_k = two_by_two_h(kx, ky, kz)
    G = kspace_greens_retarded_matrix(omega, h_k, physics.eta)

    # Shape and dtype
    assert G.shape == (2, 2)
    assert np.iscomplexobj(G)

    # Check that ( (ω + iη)I - H ) * G ≈ I
    dim = h_k.shape[0]
    mat = (omega + 1j * physics.eta) * np.eye(dim, dtype=complex) - h_k.astype(complex)
    I_rec = mat @ G
    assert np.allclose(I_rec, np.eye(dim))

def test_kspace_greens_retarded_matrix_rejects_nonsquare(physics: PhysicsConfig):
    omega = 1.0
    eta = physics.eta
    h_bad = np.zeros((2, 3), dtype=float)

    with pytest.raises(ValueError):
        kspace_greens_retarded_matrix(omega, h_bad, eta)

def test_kspace_greens_retarded_calls_H_and_uses_eta(physics: PhysicsConfig):
    calls = []

    def H_mock(kx: float, ky: float, kz: float) -> np.ndarray:
        # Record call for inspection
        calls.append((kx, ky, kz))
        # 1×1 Hamiltonian with fixed value
        return np.array([[2.0]], dtype=float)

    omega = 0.5
    kx, ky, kz = 0.1, 0.2, -0.3

    G = kspace_greens_retarded(omega, kx, ky, kz, H_mock, physics)

    # H should have been called exactly once with the given k
    assert calls == [(kx, ky, kz)]

    # For H = 2.0, the scalar GF is 1/(ω + iη - 2)
    denom = (omega + 1j * physics.eta) - 2.0
    G_exact = 1.0 / denom

    assert G.shape == (1, 1)
    assert G[0, 0] == pytest.approx(G_exact)


# endregion
# ---------------------------------------------

# ---------------------------------------------
# region real-space GF
# ---------------------------------------------
# To be implemented later
# endregion
# ---------------------------------------------