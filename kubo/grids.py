from __future__ import annotations

import numpy as np

from .config import GridConfig

# Helper to enforce that certain grid sizes are odd, so that 0 lies on the grid.
def _require_odd(name: str, n: int) -> None:
    if n % 2 == 0:
        raise ValueError(f"{name} must be odd so that 0 lies on the grid, got {n}.")

# For Fermi-sea integral
def build_omega_grid(cfg: GridConfig) -> np.ndarray:
    _require_odd("nomega", cfg.nomega)
    return np.linspace(-cfg.omega_max, cfg.omega_max, cfg.nomega, endpoint=True)

# For the functional trace
def build_k_parallel_grid_polar(cfg: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays (k_parallel, phi)."""
    _require_odd("nk_parallel", cfg.nk_parallel)
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
def build_kz_grid_fft(cfg: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    FFT-compatible pair (z, kz):

    - z: real-space grid sampled from [-z_max, z_max) with uniform spacing around 0
    - kz: angular wave numbers from 2π * fftfreq, same length as z
    
    This uses cfg.nz as the FFT size and cfg.z_max to set the box size.
    The periodic box has length L = 2*z_max and is interpreted as [-z_max, z_max).
    We use a cell-centred grid: z contains N equidistant points with spacing
    dz = L / N, running from -z_max + dz/2 to z_max - dz/2, so the box edges
    ±z_max are not sampled as grid points but are the periodic cell boundaries.
    """
    N = cfg.nz
    _require_odd("nz", N)
    L = 2.0 * cfg.z_max          # total length in z
    dz = L / N                   # grid spacing

    # z in [-z_max, z_max), centered around 0
    j = np.arange(N)
    z = (j - N // 2) * dz

    # k in angular units, consistent with exp(i * kz * z)
    kz = 2.0 * np.pi * np.fft.fftfreq(N, d=dz)

    return z, kz