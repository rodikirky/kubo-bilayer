from __future__ import annotations

import numpy as np
from typing import Callable

from .config import GridConfig, PhysicsConfig
from .grids import build_delta_z_kz_grids_fft

# Type alias: any callable that takes (kx, ky, kz) and returns a matrix
BulkHamiltonian = Callable[[float, float, float], np.ndarray] 

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
    h_k = np.asarray(h_k, dtype=complex)

    # Ensure H(k) is an N×N matrix (not e.g. a vector or rectangular array or higher rank tensor).
    if h_k.ndim != 2 or h_k.shape[0] != h_k.shape[1]:
        raise ValueError("h_k must be a square matrix.")

    dim = h_k.shape[0]
    mat = (omega + 1j * eta) * np.eye(dim, dtype=complex) - h_k
    return np.linalg.solve(mat, np.eye(dim, dtype=complex)) # More stable than direct inversion


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
    z : np.ndarray
        Real-space z grid (cell-centered).
    G_z : np.ndarray
        G^R(ω, kx, ky, z) on the z grid.
    """
    # Build kz grid for FFT
    z, kz = build_delta_z_kz_grids_fft(grid)

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

    return z, G_z

# endregion
# ----------------------------------------------