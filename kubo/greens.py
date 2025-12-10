from __future__ import annotations

from typing import Callable

import numpy as np

from .config import PhysicsConfig


# Type alias: any callable that takes (kx, ky, kz) and returns a matrix
BulkHamiltonian = Callable[[float, float, float], np.ndarray] 


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

# Real space computation to follow here