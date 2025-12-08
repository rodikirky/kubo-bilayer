from __future__ import annotations

from typing import Callable

import numpy as np

from .config import PhysicsConfig


BulkHamiltonian = Callable[[float, float, float], np.ndarray] # Type alias: any callable that takes (kx, ky, kz) and returns a matrix


def kspace_greens_retarded_matrix(
    omega: float,
    h_k: np.ndarray,
    eta: float,
) -> np.ndarray:
    """
    G^R(ω, k) = [ (ω + iη) I - H(k) ]^{-1}
    """
    dim = h_k.shape[0]
    mat = (omega + 1j * eta) * np.eye(dim, dtype=complex) - h_k
    return np.linalg.inv(mat)


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

# Real space computation to follow here