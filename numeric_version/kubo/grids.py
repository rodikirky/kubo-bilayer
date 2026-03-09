from __future__ import annotations

import numpy as np

from .config import GridConfig

# Helper to enforce that certain grid sizes are odd, so that 0 lies on the grid.
def _require_odd(name: str, n: int) -> None:
    if n % 2 == 0:
        raise ValueError(f"{name} must be odd so that 0 lies on the grid, got {n}.")

# For Fermi-sea integral
# TODO: might not need this, z' could be sourced from the FFT grid as a differnet index than z.
def build_zp_grid(cfg: GridConfig) -> np.ndarray:
    # not a linspace to match the FFT grid below
    N = cfg.nz
    _require_odd("nz", N)
    L = 2.0 * cfg.z_max
    dz_step = L / N
    j = np.arange(N)
    return (j - N // 2) * dz_step

def build_kz_grid_diagnostic(
    cfg: GridConfig,
    *,
    nkz: int = 2001,
    kz_max: float | None = None,
) -> np.ndarray:
    """
    Diagnostic / plotting kz grid (NOT FFT-coupled).

    Use this when you want to inspect G(ω,kx,ky,kz) over a wide kz range
    to locate on-shell peaks / near-poles independent of the FFT box size.

    Parameters
    ----------
    cfg : GridConfig
        Used only for a sensible default of kz_max if not provided.
    nkz : int
        Number of kz samples. Must be odd so kz=0 is included.
    kz_max : float | None
        Max |kz| for the scan. If None, defaults to cfg.k_max.

    Returns
    -------
    kz : np.ndarray
        1D array of shape (nkz,) spanning [-kz_max, kz_max] including 0.
    """
    _require_odd("nkz", nkz)

    if kz_max is None:
        kz_max = float(cfg.k_max)

    if kz_max <= 0.0:
        raise ValueError(f"kz_max must be positive, got {kz_max}.")

    return np.linspace(-kz_max, kz_max, nkz, endpoint=True, dtype=float)


# For the functional trace
def build_omega_grid(cfg: GridConfig) -> np.ndarray:
    _require_odd("nomega", cfg.nomega)
    return np.linspace(-cfg.omega_max, cfg.omega_max, cfg.nomega, endpoint=True)

def build_k_parallel_grid_polar(cfg: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays (k_parallel, phi)."""
    k_par = np.linspace(0.0, cfg.k_max, cfg.nk_parallel, endpoint=True)
    phi = np.linspace(0.0, 2.0 * np.pi, cfg.nphi, endpoint=False)
    return k_par, phi

def build_k_parallel_grid_cartesian(cfg: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays (k_x, k_y)."""
    _require_odd("nk_parallel", cfg.nk_parallel)
    k_x = np.linspace(-cfg.k_max, cfg.k_max, cfg.nk_parallel, endpoint=True)
    k_y = np.linspace(-cfg.k_max, cfg.k_max, cfg.nk_parallel, endpoint=True)
    return k_x, k_y


# For the functional trace and FFT along kz <-> z
def build_delta_z_kz_grids_fft(cfg: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    FFT-compatible pair (delta_z, kz), where delta_z = z-z':

    - delta_z: real-space grid sampled from [-z_max, z_max) with uniform spacing around 0
    - kz: angular wave numbers from 2π * fftfreq, same length as z
    
    This uses cfg.nz as the FFT size and cfg.z_max to set the box size.
    Centered FFT real-space grid corresponding to a half-open periodic domain:
    The periodic box has length L = 2*z_max and is interpreted as [-z_max, z_max).
    The delta_z-grid contains N equidistant points with spacing
    dz_step = L / N, running from -z_max + dz_step/2 to z_max - dz_step/2, so the box edges
    ±z_max are not sampled as grid points but are the periodic cell boundaries.
    """
    N = cfg.nz
    _require_odd("nz", N)
    L = 2.0 * cfg.z_max          # total length in z
    dz_step = L / N                   # grid spacing

    # z in [-z_max, z_max), centered around 0
    j = np.arange(N)
    delta_z = (j - N // 2) * dz_step

    # k in angular units, consistent with exp(i * kz * z)
    kz = 2.0 * np.pi * np.fft.fftfreq(N, d=dz_step)

    return delta_z, kz