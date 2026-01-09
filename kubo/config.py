# TODO: Tests need to be altered to reflect the recent changes.
# TODO: Add docstrings to the new classes and methods where needed and the whole file.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Callable, Any, Optional
from .models.toy import ToyBulkParams, ToyInterfaceParams
from .models.orbitronic import OrbitronicBulkParams, OrbitronicInterfaceParams
import numpy as np


ModelName = Literal["toy", "orbitronic", "user_defined"]
HamiltonianFunction = Callable[[float, float, float], np.ndarray]


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
    Defaults to toy model.

    bulk_left_params / bulk_right_params / interface_params are expected to be instances of
    model-specific dataclasses, e.g. ToyBulkParams, OrbitronicBulkParams, etc.
    """
    name: ModelName = "toy"

    bulk_left_params: Optional[Any] = None
    bulk_right_params: Optional[Any] = None
    interface_params: Optional[Any] = None

    # Optional override (esp. for user_defined)
    # TODO: Think this over. Do I want to keep this?
    # Originally intended to allow custom Hamiltonians outside the registry,
    # but now this seems incompatible with gluing two sides together.
    # Really, this would need to be a "Hamiltonian factory" that can distinguish sides.
    # But then I might as well just define a new model in the registry.
    hamiltonian_function: Optional[HamiltonianFunction] = None

    def __post_init__(self) -> None:
        # helper: allow None or instance of cls
        def _is_none_or_instance(x: Any, cls: type) -> bool:
            return x is None or isinstance(x, cls)

        if self.name == "toy":
            # Allow None so presets can omit params and your factory fills defaults.
            if not _is_none_or_instance(self.bulk_left_params, ToyBulkParams):
                raise TypeError(f"toy: bulk_left_params must be ToyBulkParams or None, got {type(self.bulk_left_params)}")
            if not _is_none_or_instance(self.bulk_right_params, ToyBulkParams):
                raise TypeError(f"toy: bulk_right_params must be ToyBulkParams or None, got {type(self.bulk_right_params)}")
            if not _is_none_or_instance(self.interface_params, ToyInterfaceParams):
                raise TypeError(f"toy: interface_params must be ToyInterfaceParams or None, got {type(self.interface_params)}")
            return

        if self.name == "orbitronic":
            if not _is_none_or_instance(self.bulk_left_params, OrbitronicBulkParams):
                raise TypeError(
                    f"orbitronic: bulk_left_params must be OrbitronicBulkParams or None, got {type(self.bulk_left_params)}"
                )
            if not _is_none_or_instance(self.bulk_right_params, OrbitronicBulkParams):
                raise TypeError(
                    f"orbitronic: bulk_right_params must be OrbitronicBulkParams or None, got {type(self.bulk_right_params)}"
                )
            if not _is_none_or_instance(self.interface_params, OrbitronicInterfaceParams):
                raise TypeError(
                    f"orbitronic: interface_params must be OrbitronicInterfaceParams or None, got {type(self.interface_params)}"
                )
            return

        if self.name == "user_defined":
            if self.hamiltonian_function is None:
                raise ValueError("model.name='user_defined' requires model.hamiltonian_factory to be set.")
            # For user_defined we generally don't care about param types.
            return

        # If you add more models later, you’ll land here if you forget to update.
        raise ValueError(f"Unknown model.name={self.name!r}. Expected 'toy', 'orbitronic', or 'user_defined'.")


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
