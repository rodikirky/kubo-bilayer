from __future__ import annotations

import numpy as np

from .config import GridConfig


def build_kz_grid(cfg: GridConfig) -> np.ndarray:
    return np.linspace(-cfg.k_max, cfg.k_max, cfg.nkz, endpoint=True)


def build_k_parallel_grid(cfg: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays (k_parallel, phi)."""
    k_par = np.linspace(0.0, cfg.k_max, cfg.nk_parallel, endpoint=True)
    phi = np.linspace(0.0, 2.0 * np.pi, cfg.nphi, endpoint=False)
    return k_par, phi


def build_z_grid(cfg: GridConfig) -> np.ndarray:
    return np.linspace(-cfg.z_max, cfg.z_max, cfg.nz, endpoint=True)


def build_omega_grid(cfg: GridConfig) -> np.ndarray:
    return np.linspace(-cfg.omega_max, cfg.omega_max, cfg.nomega, endpoint=True)
