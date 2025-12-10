# ---------------------------------------------
# Imports
# ---------------------------------------------
import numpy as np
import pytest

from kubo.config import PhysicsConfig, GridConfig
from kubo.greens import (
    BulkHamiltonian,
    kspace_greens_retarded_matrix,
    kspace_greens_retarded,
    _fourier_kz_to_z,
    realspace_greens_retarded,
)
from kubo.grids import build_kz_grid_fft

# ---------------------------------------------
# Fixtures
# ---------------------------------------------
@pytest.fixture
def physics() -> PhysicsConfig:
    return PhysicsConfig() # default parameters

@pytest.fixture
def grid() -> GridConfig:
    return GridConfig()  # default nz, z_max etc.

@pytest.fixture
def constant_h() -> BulkHamiltonian:
    # H(k) = 0, so G^R is kz-independent
    def H(kx: float, ky: float, kz: float) -> np.ndarray:
        return np.array([[0.0]], dtype=float)
    return H

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
def test_fourier_kz_to_z_constant_yields_delta_like_peak(grid: GridConfig):
    """
    For G(kz) = 1, the real-space G(z) should be a discrete delta peak
    at z=0 (up to scaling), i.e. one big value and ~0 elsewhere.
    """
    z, kz = build_kz_grid_fft(grid)
    N = grid.nz
    L = 2.0 * grid.z_max

    G_kz = np.ones(N, dtype=complex)  # constant in kz

    G_z = _fourier_kz_to_z(G_kz, grid, axis=0)

    assert G_z.shape == (N,)
    assert np.iscomplexobj(G_z)

    mid = N // 2

    # Large central peak ~ N/L
    assert G_z[mid] == pytest.approx(N / L)

    # All other points should be ~0
    mask = np.ones(N, dtype=bool)
    mask[mid] = False
    assert np.allclose(G_z[mask], 0.0, atol=1e-12)

def test_fourier_kz_to_z_raises_for_size_mismatch(grid: GridConfig):
    """
    If the kz axis length does not match grid.nz, we should error out.
    """
    N = grid.nz
    G_kz = np.ones(N + 1, dtype=complex)  # wrong length

    with pytest.raises(ValueError):
        _fourier_kz_to_z(G_kz, grid, axis=0)

def test_realspace_greens_retarded_constant_H(grid: GridConfig,
                                              physics: PhysicsConfig,
                                              constant_h: BulkHamiltonian):
    """
    For a kz-independent 1x1 Hamiltonian H=0, G^R(ω,kz) is constant in kz,
    so G^R(ω,z) should be a delta-like peak at z=0 with known amplitude.
    """
    omega = 1.0
    kx = 0.0
    ky = 0.0

    z, G_z = realspace_greens_retarded(
        omega=omega,
        kx=kx,
        ky=ky,
        H=constant_h,
        physics=physics,
        grid=grid,
    )

    N = grid.nz
    L = 2.0 * grid.z_max
    mid = N // 2

    # Shapes
    assert z.shape == (N,)
    assert G_z.shape == (N, 1, 1)
    assert np.iscomplexobj(G_z)

    # G^R(ω, kz) = 1 / (ω + iη - 0) = const in kz
    G_const = 1.0 / (omega + 1j * physics.eta)

    # Central peak amplitude ~ G_const * N/L
    assert G_z[mid, 0, 0] == pytest.approx(G_const * N / L)

    # All other z should be ~0
    mask = np.ones(N, dtype=bool)
    mask[mid] = False
    assert np.allclose(G_z[mask, 0, 0], 0.0, atol=1e-10)

    # Retarded GF: Im part at the peak should be negative
    assert G_z[mid, 0, 0].imag < 0.0

    
# endregion
# ---------------------------------------------