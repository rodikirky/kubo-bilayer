from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

ArrayC = NDArray[np.complex128]


@dataclass
class ToyBulkParams:
    """Parameters for a very simple 2×2 toy bulk model."""
    mass: float = 1.0
    gap: float = 1.0  # Δ


@dataclass
class ToyInterfaceParams:
    """Parameters for a trivial 2×2 interface Hamiltonian."""
    strength: float = 0.0  # scalar shift on the interface


@dataclass
class ToyBulk:
    """
    Very simple 2×2 bulk model:

        H(k) = (k^2 / 2m) I + Δ σ_z,

    with σ_z = diag(1, -1).
    """
    mass: float
    gap: float

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
        H(k) = (k^2 / 2m) I + Δ σ_z  for k = (kx, ky, kz).
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
