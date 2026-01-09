from __future__ import annotations

from typing import Literal
import warnings
from numpy.typing import NDArray

from kubo.config import ModelConfig, HamiltonianFunction

SideChoice = Literal["left", "right"]

def _model_specific_dataclasses(model_name: str) -> tuple[type, type, type, type]:
    if model_name == "toy":
        from .toy import ToyBulk, ToyBulkParams, ToyInterface, ToyInterfaceParams
        bulk_dc = ToyBulk
        bulk_params_dc = ToyBulkParams
        interface_dc = ToyInterface
        interface_params_dc = ToyInterfaceParams

    elif model_name == "orbitronic":
        from .orbitronic import OrbitronicBulk, OrbitronicBulkParams, OrbitronicInterface, OrbitronicInterfaceParams
        bulk_dc = OrbitronicBulk
        bulk_params_dc = OrbitronicBulkParams
        interface_dc = OrbitronicInterface
        interface_params_dc = OrbitronicInterfaceParams

    elif model_name == "user_defined":
        raise ValueError(
            "model.name='user_defined' requires ModelConfig.hamiltonian_function to be set."
            "This is not supported for gluing, since distinctive sides are required."
        )


    else:
        raise ValueError(f"Unknown model.name={model_name!r}")
    
    return bulk_dc, bulk_params_dc, interface_dc, interface_params_dc

def _model_specific_parameters(model: ModelConfig) -> tuple[object, object, object]:
    _, bulk_params_dc, _, interface_params_dc = _model_specific_dataclasses(model.name)

    left_bulk_params = model.bulk_left_params if model.bulk_left_params is not None else bulk_params_dc()
    right_bulk_params = model.bulk_right_params if model.bulk_right_params is not None else bulk_params_dc()
    interface_params = model.interface_params if model.interface_params is not None else interface_params_dc()
    return left_bulk_params, right_bulk_params, interface_params

def build_bulk_hamiltonian(model: ModelConfig, side: SideChoice) -> HamiltonianFunction:
    """
    Return a bulk Hamiltonian callable H(kx, ky, kz) for the selected model and side.

    - If model.hamiltonian_function is provided, it is returned directly (side is ignored).
    - Otherwise dispatches by model.name and binds the selected side's bulk params once via Bulk.from_params(...).
    """
    # Override path: side becomes informational only.
    if model.hamiltonian_function is not None:
        warnings.warn(
            f"Using ModelConfig.hamiltonian_function override: bulk side choice {side} is ignored.",
            category=UserWarning,
            stacklevel=2,
        )
        return model.hamiltonian_function

    bulk_dc, _, _, _ = _model_specific_dataclasses(model.name)
    left_bulk_params, right_bulk_params, _ = _model_specific_parameters(model)

    if side == "left":
        if model.bulk_left_params is None:
            warnings.warn(
                f"No left bulk parameters provided for model={model.name!r}. Falling back to model defaults.",
                category=UserWarning,
                stacklevel=2,
            )
        params = left_bulk_params

    elif side == "right":
        if model.bulk_right_params is None:
            warnings.warn(
                f"No right bulk parameters provided for model={model.name!r}. Falling back to model defaults.",
                category=UserWarning,
                stacklevel=2,
            )
        params = right_bulk_params

    else:
        raise ValueError(f"side must be 'left' or 'right'. Got {side!r}.")

    model_bulk = bulk_dc.from_params(params) 
    return lambda kx, ky, kz: model_bulk.hamiltonian((kx, ky, kz))    

def build_gluing_components(model: ModelConfig, no_defaults: bool = True
                            ) -> tuple[HamiltonianFunction, HamiltonianFunction, float, float, HamiltonianFunction]:
    """
    Return interface gluing Hamiltonians and related data for the selected model.

    Computing interface Hamiltonians of the full bilayer problem is done via a gluing formula. 
    This requires the left and right bulk Hamiltonians as well as the 2D interface Hamiltonian treated as 
    a 3D potential and the effective single-particle masses of the bulk materials.
    
    Parameters
    -----------
    model: ModelConfig
        Model configuration, including bulk and interface parameters.
    no_defaults: bool, default=True
        If True, raises an error if any of bulk_left_params, bulk_right_params, or interface_params are None.
        If False, falls back to model-specific defaults when parameters are missing.
        This makes defaulting a conscious choice.

    Returns
    -------
    left_hamiltonian: HamiltonianFunction (= Callable[[float, float, float], np.ndarray])
        Callable H_left(kx, ky, kz) for the left bulk Hamiltonian.
    right_hamiltonian: HamiltonianFunction
        Callable H_right(kx, ky, kz) for the right bulk Hamiltonian.
    left_mass: float
        Effective mass of the left bulk (for boundary-condition/log-derivative terms in continuous gluing).
    right_mass: float
        Effective mass of the right bulk (for boundary-condition/log-derivative terms in continuous gluing).    
    interface_potential: HamiltonianFunction
        Callable V_interface(kx, ky, kz) for the interface Hamiltonian treated as a potential.
        The kz argument is ignored, since the interface Hamiltonian is a 2D mapping: (kx, ky)-> H_int(kx,ky).
    
    Raises
    ------
    ValueError
        If model.hamiltonian_function is set (incompatible with gluing).
        If no_defaults=True and any of bulk_left_params, bulk_right_params, or interface_params are None.
    
    Warnings
    --------
    Warns if no_defaults=False and any of bulk_left_params, bulk_right_params, or interface_params are None.
    This is to make the user aware of implicit defaulting, which may be unintended, even if generally allowed by no_defaults=False.
    """
    if model.hamiltonian_function is not None:
        raise ValueError(
            "ModelConfig.hamiltonian_function override is incompatible with interface gluing Hamiltonians."
        )
        # TODO: enable a custom choice of Hamiltonians later. But side specifications must be added too. Or delete custom Hamiltonian function in ModelConfig entirely.
    if (model.bulk_left_params is None or model.bulk_right_params is None or model.interface_params is None) and no_defaults:
        raise ValueError(
            "no_defaults=True requires all of bulk_left_params, bulk_right_params, and interface_params to be explicitly provided."
        )
    if model.bulk_left_params is None:
        warnings.warn(
            f"No left bulk parameters provided for model={model.name!r}. Falling back to model defaults.",
            category=UserWarning,
            stacklevel=2,
        )
    if model.bulk_right_params is None:
        warnings.warn(
            f"No right bulk parameters provided for model={model.name!r}. Falling back to model defaults.",
            category=UserWarning,
            stacklevel=2,
        )
    if model.interface_params is None:
        warnings.warn(
            f"No interface parameters provided for model={model.name!r}. Falling back to model defaults.",
            category=UserWarning,
            stacklevel=2,
        )
    left_bulk_params, right_bulk_params, interface_params = _model_specific_parameters(model)
    bulk_dc, _, interface_dc, _ = _model_specific_dataclasses(model.name)
    bulk_left = bulk_dc.from_params(left_bulk_params)  
    bulk_right = bulk_dc.from_params(right_bulk_params) 
    interface = interface_dc.from_params(interface_params) 

    # Left/right bulk Hamiltonians are 3D mappings: (kx, ky, kz) -> H_bulk(kx, ky, kz)
    left_hamiltonian = lambda kx, ky, kz: bulk_left.hamiltonian((kx, ky, kz))
    right_hamiltonian = lambda kx, ky, kz: bulk_right.hamiltonian((kx, ky, kz))
    # Effective masses for boundary condition terms
    left_mass = bulk_left.mass
    right_mass = bulk_right.mass

    # The interface Hamiltonian is a 2D mapping originally: (kx, ky) -> H_int(kx, ky).
    # We transform it into a potential that accepts (kx, ky, kz) but ignores kz
    interface_potential = lambda kx, ky, kz: interface.hamiltonian((kx, ky)) 

    return left_hamiltonian, right_hamiltonian, left_mass, right_mass, interface_potential



