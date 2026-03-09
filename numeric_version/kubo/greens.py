from __future__ import annotations

import numpy as np
import warnings
from numpy.typing import NDArray 
from typing import Callable
from dataclasses import dataclass

from .plotting import profile_amplitude_over_first_axis, edge_leak_ratio
from .config import GridConfig, PhysicsConfig
from .grids import build_delta_z_kz_grids_fft

# Type alias: any callable that takes (kx, ky, kz) and returns a matrix
BulkHamiltonian = Callable[[float, float, float], np.ndarray] 
# Type alias: matrix of complex numbers
ArrayC = NDArray[np.complex128]

# ----------------------------------------------
# region k-space GF
# ----------------------------------------------
def kspace_greens_retarded_matrix(
    omega: float,
    h_k: np.ndarray,
    eta: float,
) -> np.ndarray:
    """
    G^R(ω, k) = [ (ω + iη) I - H(k) ]^{-1}
    """
    # TODO: use batched version for single matrix too and avoid code duplication
    h_k = np.asarray(h_k, dtype=complex)

    # Ensure H(k) is an N×N matrix (not e.g. a vector or rectangular array or higher rank tensor).
    if h_k.ndim != 2 or h_k.shape[0] != h_k.shape[1]:
        raise ValueError("h_k must be a square matrix.")

    dim = h_k.shape[0]
    mat = (omega + 1j * eta) * np.eye(dim, dtype=complex) - h_k
    return np.linalg.solve(mat, np.eye(dim, dtype=complex)) # More stable than direct inversion


def kspace_greens_retarded_matrix_batched(
    omega: float,
    h_k: np.ndarray,
    eta: float,
) -> np.ndarray:
    """
    Batched version of the retarded GF:

        G^R(ω, k) = [ (ω + iη) I - H(k) ]^{-1}

    Accepts h_k with shape (..., dim, dim) and returns same shape.
    Works for scalar case too (dim, dim).
    """
    # TODO: test this function properly
    h_k = np.asarray(h_k, dtype=complex)

    if h_k.ndim < 2 or h_k.shape[-1] != h_k.shape[-2]:
        raise ValueError("h_k must have shape (..., dim, dim) with square last two axes.")

    dim = h_k.shape[-1]
    I = np.eye(dim, dtype=complex)

    # Broadcast I to the leading batch shape
    mat = (omega + 1j * eta) * I - h_k
    rhs = np.broadcast_to(I, mat.shape)

    return np.linalg.solve(mat, rhs)


def kspace_greens_retarded(
    omega: float,
    kx: float,
    ky: float,
    kz: float,
    H: BulkHamiltonian,
    physics: PhysicsConfig,
) -> np.ndarray:
    """
    Convenience wrapper: build H(k) and compute G^R for one (ω, k).

    This is model-agnostic: H can be toy, orbitronic, etc.
    """
    h_k = H(kx, ky, kz)
    return kspace_greens_retarded_matrix(omega, h_k, physics.eta)

def kspace_greens_retarded_on_kz_grid(
    omega: float,
    kx: float,
    ky: float,
    kz: np.ndarray,
    H: BulkHamiltonian,
    physics: PhysicsConfig,
) -> np.ndarray:
    """
    Public convenience: compute G^R(ω,kx,ky,kz_j) for a whole kz grid.

    Returns
    -------
    G_kz : np.ndarray
        Shape (nkz, dim, dim)
    """
    # TODO: test this function properly
    kz = np.asarray(kz, dtype=float)
    if kz.ndim != 1:
        raise ValueError("kz must be a 1D array.")

    # Determine matrix dimension from a single evaluation
    dim = np.asarray(H(kx, ky, float(kz[0]))).shape[0]
    G_kz = np.empty((kz.size, dim, dim), dtype=complex)

    # NOTE: Loop for now. This removes np.stack from scripts.
    # TODO: Later, when you a batched Hamiltonian H_vec, this is the one place to upgrade.
    for i, kz_i in enumerate(kz):
        h_k = H(kx, ky, float(kz_i))
        G_kz[i] = kspace_greens_retarded_matrix(omega, h_k, physics.eta)

    return G_kz

# endregion
# ----------------------------------------------

# ----------------------------------------------
# region real space GF
# ----------------------------------------------
def _build_fft_Gkz_input_for_fixed_omega_kpar(
    omega: float,
    kx: float,
    ky: float,
    kz: np.ndarray,
    H: BulkHamiltonian,
    physics: PhysicsConfig,
) -> np.ndarray:
    """
    Compute G^R(ω, kx, ky, kz_j) on a 1D kz grid.

    Currently implemented as a Python loop over kz. This is the place to
    optimize / vectorize later (e.g. batched Hamiltonian evaluation or numba).

    Returns
    -------
    G_kz : np.ndarray
        Array of shape (nkz, dim, dim) with G^R(kz_j) along axis 0.
    """
    # TODO: Replace with kspace_greens_retarded_on_kz_grid later, when you have time for re-testing
    # just do _build_fft_Gkz_input_for_fixed_omega_kpar = kspace_greens_retarded_on_kz_grid instead
    dim = H(kx, ky, kz[0]).shape[0]
    G_kz = np.empty((kz.size, dim, dim), dtype=complex)

    # TODO: vectorize over kz (and possibly ω, k_parallel) once H and k-space computation supports batched input
    for i, kz_i in enumerate(kz):
        G_kz[i] = kspace_greens_retarded(omega, kx, ky, kz_i, H, physics)

    return G_kz

def _fourier_kz_to_z(
    G_kz: np.ndarray,
    cfg: GridConfig,
    axis: int = 0, # kz grid usually the first argument of G_kz
) -> np.ndarray:
    """
    Transform G^R(kz, ...) -> G^R(z, ...) along the kz axis using an FFT.

    We use the convention
        G^R(z) ≈ (1/L) Σ_n e^{i kz_n z} G^R(kz_n),
    with L = 2*z_max and kz_n from 2π * fftfreq(N, d=dz).

    Parameters
    ----------
    G_kz : np.ndarray
        Array of G^R sampled on the kz grid returned by build_kz_grid_fft,
        with kz as the axis `axis`.
    cfg : GridConfig
        Grid configuration. Uses cfg.nz and cfg.z_max.
    axis : int
        Axis/argument of G_kz corresponding to the kz grid., i.e. G_kz.shape[axis] == cfg.nz.

    Returns
    -------
    G_z : np.ndarray
        G^R(z, ...) on the same grid, same shape as G_kz but with the
        kz-axis replaced by z.
    """
    # Get the z-grid (and ensure we’re using the same N, L)
    N = cfg.nz
    if N != G_kz.shape[axis]:
        raise ValueError(f"Size of G_kz along axis {axis} must match grid.nz = {N}, got {G_kz.shape[axis]}.")
    L = 2.0 * cfg.z_max

    # Inverse DFT along kz axis -> periodic function on [0, L)
    G_z_unshifted = np.fft.ifft(G_kz, axis=axis) * (N / L)

    # Shift to match our centered z grid [-z_max + dz/2, z_max - dz/2]
    G_z = np.fft.fftshift(G_z_unshifted, axes=axis)

    return G_z

def fourier_kz_to_z(G_kz: np.ndarray, cfg: GridConfig, axis: int = 0) -> np.ndarray:
    # Public wrapper for _fourier_kz_to_z without having to change the underscore everywhere
    # TODO: test this function properly
    return _fourier_kz_to_z(G_kz, cfg, axis=axis)


def realspace_greens_retarded(
    omega: float,
    kx: float,
    ky: float,
    H: BulkHamiltonian,
    physics: PhysicsConfig,
    grid: GridConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute G^R(ω, kx, ky, z) by Fourier transforming G^R(ω, kx, ky, kz).

    Parameters
    ----------
    omega : float
        Frequency ω.
    kx : float
        kx component of momentum.
    ky : float
        ky component of momentum.
    H : BulkHamiltonian
        Function that returns H(kx, ky, kz).
    physics : PhysicsConfig
        Physics configuration (for η).
    grid : GridConfig
        Grid configuration (for kz and z grids).

    Returns
    -------
    delta_z : np.ndarray
        Real-space z grid (cell-centered).
    G_z : np.ndarray
        G^R(ω, kx, ky, z) on the z grid.
    """
    # TODO: Replace with realspace_greens_retarded_with_kz later, when you have time for re-testing
    # just do delta_z, _, G_z, _ = realspace_greens_retarded_with_kz(omega=omega, kx=kx, ky=ky, H=H, physics=physics, grid=grid)

    # Build kz grid for FFT
    delta_z, kz = build_delta_z_kz_grids_fft(grid)
    N = grid.nz
    assert delta_z.size == N
    assert kz.size == N 
    assert delta_z[N//2] == 0.0 

    # Compute G^R(ω, kx, ky, kz) on the kz grid
    G_kz = _build_fft_Gkz_input_for_fixed_omega_kpar(
        omega,
        kx,
        ky,
        kz,
        H,
        physics,
    )

    # Fourier transform to real space z
    G_z = _fourier_kz_to_z(
        G_kz,
        grid,
        axis=0,
    )

    return delta_z, G_z

# Helper: index of delta_z == 0
# G(0,0) will be at this index in the returned G_z array
def delta_z_zero_index(cfg: GridConfig) -> int:
    return cfg.nz // 2

def realspace_greens_retarded_with_kz(
    omega: float,
    kx: float,
    ky: float,
    H: BulkHamiltonian,
    physics: PhysicsConfig,
    grid: GridConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Like realspace_greens_retarded, but also returns kz and G_kz for diagnostics.

    Returns
    -------
    delta_z : (N,)
    kz      : (N,)
    G_z     : (N, dim, dim)
    G_kz    : (N, dim, dim)
    """
    # TODO: test this function properly
    delta_z, kz = build_delta_z_kz_grids_fft(grid)  # :contentReference[oaicite:6]{index=6}
    G_kz = kspace_greens_retarded_on_kz_grid(omega, kx, ky, kz, H, physics)
    G_z = _fourier_kz_to_z(G_kz, grid, axis=0)      # :contentReference[oaicite:7]{index=7}
    return delta_z, kz, G_z, G_kz

# endregion
# ----------------------------------------------

# ----------------------------------------------
# region Diagnostics
# ----------------------------------------------
@dataclass(frozen=True)
class KernelDiagnostics:
    edge_leak_ratio: float
    center_index: int
    edge_warn: float = 1e-6
    edge_error: float = 1e-3
    edge_action: str = "warn"  # "none"|"warn"|"error"
    edge_m: int = 10

@dataclass(frozen=True)
class RealSpaceKernel:
    delta_z: NDArray[np.float64]    # (N,)
    kz: NDArray[np.float64] | None  # (N,) FFT kz grid (optional but nice for diagnostic plots)
    G_dz: ArrayC                    # (N, dim, dim)
    G_kz: ArrayC | None             # (N, dim, dim) (optional)
    diag: KernelDiagnostics

def realspace_kernel_retarded_with_meta(
        omega: float,
        kx: float,
        ky: float,
        H: BulkHamiltonian,
        physics: PhysicsConfig,
        grid: GridConfig,
        carry_k_info: bool = False,     # Set carry_k_info = True for kz, G_kz metadata
        edge_m: int = 10,
        edge_action: str = "warn"       # "none"|"warn"|"error"
        ) -> RealSpaceKernel:
    if carry_k_info:
        delta_z, kz, G_z, G_kz = realspace_greens_retarded_with_kz(omega,kx,ky,H,physics,grid)
    else:
        delta_z, G_z = realspace_greens_retarded(omega,kx,ky,H,physics,grid)
        kz = None
        G_kz = None
    N = delta_z.size
    mid = N // 2

    amp = profile_amplitude_over_first_axis(G_z, mode="fro")
    leak = edge_leak_ratio(amp, m=edge_m, center_index=mid)
    if edge_action not in {"warn", "error", "none"}:
        warnings.warn(f"Choose edge_action from 'warn', 'error', 'none'. Got {edge_action}, will be treated like 'none'.")
    return RealSpaceKernel(
        delta_z=delta_z,
        kz=kz,
        G_dz=G_z,
        G_kz=G_kz,
        diag=KernelDiagnostics(edge_leak_ratio=leak, center_index=mid, edge_action=edge_action, edge_m=edge_m),
    )