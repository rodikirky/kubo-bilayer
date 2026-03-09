'''
This is a 2x2 decoupled toy model with the single channel orbitronic hamiltonian on the diagonal.
The bulk Hamiltonians are then:
H = [[h,0][0,h]]
where h = k^2/2m + V
and we are assuming an interface potential v_imp ("impurity"). 
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

ArrayC = NDArray[np.complex128]

#region Params
@dataclass
class SingleBulkParams:
    """Parameters for decoupled 2x2 toy bulk model."""
    mass: float = 1.0       # m
    potential: float = 1.0  # V


@dataclass
class SingleInterfaceParams:
    """Parameters for a trivial 2x2 interface Hamiltonian."""
    interface_potential: float = 0.0  # v_imp
#endregion

#region ToyBulk
@dataclass
class SingleBulk:
    """
    Analytically solved decoupled 2x2 bulk model:

        H(k) = (k^2 / 2m + V) I.
    """
    mass: float
    potential: float

    def __post_init__(self):
        EPS_MASS = 1e-12
        if abs(self.mass) < EPS_MASS:
            raise ValueError(
                f"mass is too small (|mass| < {EPS_MASS}); got mass={self.mass}"
            )
        
    @classmethod
    def from_params(cls, params: SingleBulkParams) -> "SingleBulk":
        return cls(mass=params.mass, potential=params.potential)

    @property
    def identity(self) -> ArrayC:
        return np.eye(2, dtype=np.complex128)

    #@property
    #def sigma_z(self) -> ArrayC:
    #    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    def hamiltonian(self, k: Sequence[float]) -> ArrayC:
        """
        H(k) = (k^2 / 2m + V) I for k = (kx, ky, kz).
        
        TODO: add a batched version that accepts arrays of k with shape (..., 3)
        and returns (..., dim, dim) to enable vectorized Green's function evaluation.
        """
        kx, ky, kz = map(float, k)
        k2 = kx * kx + ky * ky + kz * kz
        ek = k2 / (2.0 * self.mass)
        return (ek * + self.potential) * self.identity
#endregion

#region ToyInterface
@dataclass
class SingleInterface:
    """
    Trivial 2×2 interface Hamiltonian:

        H_int = v_imp * I.
    """
    interface_potential: float # v_imp

    @classmethod
    def from_params(cls, params: SingleInterfaceParams) -> "SingleInterface":
        return cls(interface_potential=params.interface_potential)

    def hamiltonian(self, kx: float, ky: float) -> ArrayC:
        _ = float(kx), float(ky)  # unused, kept for API compatibility
        return self.interface_potential * np.eye(2, dtype=np.complex128)
#endregion

#region SingleObservable

#endregion