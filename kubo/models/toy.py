from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

ArrayC = NDArray[np.complex128]


@dataclass
class ToyBulkParams:
    """Parameters for a very simple 2x2 toy bulk model."""
    mass: float = 1.0
    gap: float = 1.0  # Δ


@dataclass
class ToyInterfaceParams:
    """Parameters for a trivial 2x2 interface Hamiltonian."""
    strength: float = 0.0  # scalar shift on the interface


@dataclass
class ToyBulk:
    """
    Very simple 2x2 bulk model:

        H(k) = (k^2 / 2m) I + Δ sigma_z,

    with sigma_z = diag(1, -1).
    """
    mass: float
    gap: float

    def __post_init__(self):
        EPS_MASS = 1e-12
        if abs(self.mass) < EPS_MASS:
            raise ValueError(
                f"mass is too small (|mass| < {EPS_MASS}); got mass={self.mass}"
            )
        
    @classmethod
    def from_params(cls, params: ToyBulkParams) -> "ToyBulk":
        return cls(mass=params.mass, gap=params.gap)

    @property
    def identity(self) -> ArrayC:
        return np.eye(2, dtype=np.complex128)

    @property
    def sigma_z(self) -> ArrayC:
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    def hamiltonian(self, k: Sequence[float]) -> ArrayC:
        """
        H(k) = (k^2 / 2m) I + Δ sigma_z  for k = (kx, ky, kz).
        
        TODO: add a batched version that accepts arrays of k with shape (..., 3)
        and returns (..., dim, dim) to enable vectorized Green's function evaluation.
        """
        kx, ky, kz = map(float, k)
        k2 = kx * kx + ky * ky + kz * kz
        ek = k2 / (2.0 * self.mass)
        return ek * self.identity + self.gap * self.sigma_z


@dataclass
class ToyInterface:
    """
    Trivial 2×2 interface Hamiltonian:

        H_int = strength * I.
    """
    strength: float

    @classmethod
    def from_params(cls, params: ToyInterfaceParams) -> "ToyInterface":
        return cls(strength=params.strength)

    def hamiltonian(self, kx: float, ky: float) -> ArrayC:
        _ = float(kx), float(ky)  # unused, kept for API compatibility
        return self.strength * np.eye(2, dtype=np.complex128)
