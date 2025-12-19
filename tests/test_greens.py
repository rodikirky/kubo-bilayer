# ---------------------------------------------
# Imports
# ---------------------------------------------
import numpy as np
import pytest

from kubo.config import PhysicsConfig, GridConfig
from kubo.greens import (
    BulkHamiltonian,
    kspace_greens_retarded_matrix,
    kspace_greens_retarded_matrix_batched,
    kspace_greens_retarded,
    kspace_greens_retarded_on_kz_grid,
    _fourier_kz_to_z,
    fourier_kz_to_z,
    realspace_greens_retarded,
    realspace_greens_retarded_with_kz,
    delta_z_zero_index,
    realspace_kernel_retarded_with_meta,
)

from kubo.grids import build_delta_z_kz_grids_fft

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
    z, kz = build_delta_z_kz_grids_fft(grid)
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

# ---------------------------------------------
# region NEW: batched k-space GF
# ---------------------------------------------
def test_kspace_greens_retarded_matrix_batched_matches_loop(physics: PhysicsConfig):
    rng = np.random.default_rng(0)
    omega = 0.7
    eta = physics.eta

    # batch of 2x2 matrices, make them Hermitian-ish to be well-behaved
    A = rng.normal(size=(5, 2, 2)) + 1j * rng.normal(size=(5, 2, 2))
    h_batch = A + np.swapaxes(A.conj(), -1, -2)

    G_batched = kspace_greens_retarded_matrix_batched(omega, h_batch, eta)
    assert G_batched.shape == h_batch.shape
    assert np.iscomplexobj(G_batched)

    G_loop = np.stack([kspace_greens_retarded_matrix(omega, h_batch[i], eta) for i in range(h_batch.shape[0])], axis=0)
    assert np.allclose(G_batched, G_loop, atol=1e-12)


def test_kspace_greens_retarded_matrix_batched_accepts_single_matrix(physics: PhysicsConfig):
    omega = 1.2
    eta = physics.eta
    h = np.array([[0.3, 0.1], [0.1, -0.2]], dtype=float)

    G1 = kspace_greens_retarded_matrix(omega, h, eta)
    G2 = kspace_greens_retarded_matrix_batched(omega, h, eta)

    assert G2.shape == (2, 2)
    assert np.allclose(G2, G1, atol=1e-12)


def test_kspace_greens_retarded_matrix_batched_rejects_bad_shape(physics: PhysicsConfig):
    omega = 1.0
    eta = physics.eta

    with pytest.raises(ValueError):
        kspace_greens_retarded_matrix_batched(omega, np.array([1.0, 2.0, 3.0]), eta)  # ndim < 2

    with pytest.raises(ValueError):
        kspace_greens_retarded_matrix_batched(omega, np.zeros((4, 2, 3)), eta)  # last two axes not square


# endregion
# ---------------------------------------------


# ---------------------------------------------
# region NEW: kz-grid helpers + public wrappers
# ---------------------------------------------
def test_kspace_greens_retarded_on_kz_grid_matches_pointwise(physics: PhysicsConfig, two_by_two_h: BulkHamiltonian):
    omega = 0.4
    kx, ky = 0.2, -0.1
    kz = np.linspace(-1.0, 1.0, 9)

    G_kz = kspace_greens_retarded_on_kz_grid(omega, kx, ky, kz, two_by_two_h, physics)
    assert G_kz.shape == (kz.size, 2, 2)
    assert np.iscomplexobj(G_kz)

    for i in range(kz.size):
        G_i = kspace_greens_retarded(omega, kx, ky, float(kz[i]), two_by_two_h, physics)
        assert np.allclose(G_kz[i], G_i, atol=1e-12)


def test_kspace_greens_retarded_on_kz_grid_rejects_non_1d_kz(physics: PhysicsConfig, constant_h: BulkHamiltonian):
    kz_bad = np.zeros((3, 3), dtype=float)
    with pytest.raises(ValueError):
        kspace_greens_retarded_on_kz_grid(1.0, 0.0, 0.0, kz_bad, constant_h, physics)


def test_fourier_kz_to_z_wrapper_is_identical(grid: GridConfig):
    z, kz = build_delta_z_kz_grids_fft(grid)
    N = grid.nz
    G_kz = (np.arange(N) + 1j * np.arange(N)[::-1]).astype(complex)

    a = _fourier_kz_to_z(G_kz, grid, axis=0)
    b = fourier_kz_to_z(G_kz, grid, axis=0)
    assert np.allclose(a, b, atol=0.0)


def test_delta_z_zero_index_matches_convention(grid: GridConfig):
    z, _ = build_delta_z_kz_grids_fft(grid)
    mid = delta_z_zero_index(grid)

    assert mid == grid.nz // 2
    assert z[mid] == pytest.approx(0.0)


# endregion
# ---------------------------------------------


# ---------------------------------------------
# region NEW: real-space with kz + kernel meta object
# ---------------------------------------------
def test_realspace_greens_retarded_with_kz_consistency(grid: GridConfig, physics: PhysicsConfig, constant_h: BulkHamiltonian):
    omega = 0.9
    kx = 0.0
    ky = 0.0

    delta_z, kz, G_z, G_kz = realspace_greens_retarded_with_kz(
        omega=omega, kx=kx, ky=ky, H=constant_h, physics=physics, grid=grid
    )

    N = grid.nz
    assert delta_z.shape == (N,)
    assert kz.shape == (N,)
    assert G_z.shape == (N, 1, 1)
    assert G_kz.shape == (N, 1, 1)

    # Definition: G_z should be the FFT of G_kz with the module’s convention.
    G_z_re = _fourier_kz_to_z(G_kz, grid, axis=0)
    assert np.allclose(G_z, G_z_re, atol=1e-12)


def test_realspace_kernel_retarded_with_meta_carry_k_info_false(grid: GridConfig, physics: PhysicsConfig, constant_h: BulkHamiltonian):
    rk = realspace_kernel_retarded_with_meta(
        omega=1.0, kx=0.0, ky=0.0, H=constant_h, physics=physics, grid=grid,
        carry_k_info=False, edge_m=7, edge_action="none"
    )

    assert rk.kz is None
    assert rk.G_kz is None
    assert rk.delta_z.shape == (grid.nz,)
    assert rk.G_dz.shape[-2:] == (1, 1)
    assert rk.diag.center_index == grid.nz // 2
    assert rk.diag.edge_m == 7
    assert rk.diag.edge_action == "none"

    # constant-H case -> delta-like center peak => edges ~0 => tiny leak
    assert rk.diag.edge_leak_ratio <= 1e-8


def test_realspace_kernel_retarded_with_meta_carry_k_info_true(grid: GridConfig, physics: PhysicsConfig, constant_h: BulkHamiltonian):
    rk = realspace_kernel_retarded_with_meta(
        omega=1.0, kx=0.0, ky=0.0, H=constant_h, physics=physics, grid=grid,
        carry_k_info=True, edge_m=5, edge_action="warn"
    )

    assert rk.kz is not None
    assert rk.G_kz is not None
    assert rk.kz.shape == (grid.nz,)
    assert rk.G_kz.shape == (grid.nz, 1, 1)


def test_realspace_kernel_retarded_with_meta_invalid_edge_action_warns(grid: GridConfig, physics: PhysicsConfig, constant_h: BulkHamiltonian):
    with pytest.warns(UserWarning):
        _ = realspace_kernel_retarded_with_meta(
            omega=1.0, kx=0.0, ky=0.0, H=constant_h, physics=physics, grid=grid,
            carry_k_info=False, edge_action="definitely_not_valid"
        )

@pytest.mark.parametrize("dim", [1, 2])
def test_kspace_greens_retarded_matrix_batched_matches_loop_dims(physics: PhysicsConfig, dim: int):
    rng = np.random.default_rng(0)
    omega = 0.7
    eta = physics.eta
    B = 5

    A = rng.normal(size=(B, dim, dim)) + 1j * rng.normal(size=(B, dim, dim))
    h_batch = A + np.swapaxes(A.conj(), -1, -2)

    G_batched = kspace_greens_retarded_matrix_batched(omega, h_batch, eta)
    G_loop = np.stack([kspace_greens_retarded_matrix(omega, h_batch[i], eta) for i in range(B)], axis=0)

    assert G_batched.shape == (B, dim, dim)
    assert np.allclose(G_batched, G_loop, atol=1e-12)


# endregion
# ---------------------------------------------
