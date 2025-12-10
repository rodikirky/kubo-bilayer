from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Callable, Any


ModelName = Literal["toy", "orbitronic", "user_defined"]


@dataclass
class GridConfig:
    # we want odd nz, nk_parallel and nomega so that 0 is exactly on the grid
    nomega: int = 255
    nz: int = 255          # real-space z grid after FFT of auxilliary GF
    nk_parallel: int = 255 # in-plane momentum grid
    nphi: int = 64         # starts at 0, goes to 2pi
    k_max: float = 2.0
    z_max: float = 20.0
    omega_max: float = 5.0


@dataclass
class ModelConfig:
    """
    Model selection + model-specific parameters.

    bulk_left_params / bulk_right_params / interface_params are expected to be instances of
    model-specific dataclasses, e.g.:

    - ToyBulkParams, ToyInterfaceParams
    - OrbitronicBulkParams, OrbitronicInterfaceParams
    - (UserDefinedParams) ...
    """
    name: ModelName = "toy"

    bulk_left_params: Any | None = None
    bulk_right_params: Any | None = None
    interface_params: Any | None = None

    # Optional: direct override for advanced / user-defined models.
    hamiltonian_factory: Callable | None = None


@dataclass
class PhysicsConfig:
    eta: float = 1e-6       # broadening (might need to be modified for numerics)
    mu: float = 0.0         # chemical potential
    temperature: float = 0.0  # Fermi smearing, in whatever units you choose


@dataclass
class KuboConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
