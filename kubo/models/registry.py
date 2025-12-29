from __future__ import annotations

from typing import Literal
import warnings

from kubo.config import ModelConfig, HamiltonianFunction

SideChoice = Literal["left", "right"]

def build_bulk_hamiltonian(model: ModelConfig, side: SideChoice) -> HamiltonianFunction:
    """
    Return a bulk Hamiltonian callable H(kx, ky, kz) for the selected model and side.

    - If model.hamiltonian_function is provided, it is returned directly (side is ignored).
    - Otherwise dispatches by model.name and binds the selected side's bulk params once via Bulk.from_params(...).
    """
    # Override path: side becomes informational only.
    if model.hamiltonian_function is not None:
        if side in ("left", "right"):
            warnings.warn(
                f"Using ModelConfig.hamiltonian_function override: bulk side choice {side} is ignored.",
                category=UserWarning,
                stacklevel=2,
            )
        return model.hamiltonian_function

    if model.name == "toy":
        from .toy import ToyBulk, ToyBulkParams
        bulk_dc = ToyBulk
        bulk_params_dc = ToyBulkParams

    elif model.name == "orbitronic":
        from .orbitronic import OrbitronicBulk, OrbitronicBulkParams
        bulk_dc = OrbitronicBulk
        bulk_params_dc = OrbitronicBulkParams

    elif model.name == "user_defined":
        raise ValueError(
            "model.name='user_defined' requires ModelConfig.hamiltonian_function to be set."
        )

    else:
        raise ValueError(f"Unknown model.name={model.name!r}")

    if side == "left":
        if model.bulk_left_params is None:
            warnings.warn(
                f"No left bulk parameters provided for model={model.name!r}. Falling back to model defaults.",
                category=UserWarning,
                stacklevel=2,
            )
        params = model.bulk_left_params if model.bulk_left_params is not None else bulk_params_dc()

    elif side == "right":
        if model.bulk_right_params is None:
            warnings.warn(
                f"No right bulk parameters provided for model={model.name!r}. Falling back to model defaults.",
                category=UserWarning,
                stacklevel=2,
            )
        params = model.bulk_right_params if model.bulk_right_params is not None else bulk_params_dc()

    else:
        raise ValueError(f"side must be 'left' or 'right'. Got {side!r}.")

    model_bulk = bulk_dc.from_params(params)  # type: ignore[attr-defined]
    return lambda kx, ky, kz: model_bulk.hamiltonian((kx, ky, kz))    

    
