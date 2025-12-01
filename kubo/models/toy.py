from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ToyBulkParams:
    """Parameters for a simple 2x2 toy bulk model."""
    mass: float = 1.0
    gap: float = 1.0
    v: float = 1.0  # velocity scale for kx, ky


@dataclass
class ToyInterfaceParams:
    """Parameters for a simple 2x2 toy interface Hamiltonian."""
    strength: float = 0.0  # interface onsite shift / coupling


def toy_bulk_hamiltonian(
    kx: float,
    ky: float,
    kz: float,
    params: ToyBulkParams | None = None,
) -> np.ndarray:
    """
    Simple 2x2 Dirac-like toy Hamiltonian:

        H = (k^2 / 2m) I + d · sigma_Pauli ,

    with d = (v kx, v ky, gap).
    """
    if params is None:
        params = ToyBulkParams()

    k2 = kx * kx + ky * ky + kz * kz
    ek = k2 / (2.0 * params.mass)

    d_x = params.v * kx
    d_y = params.v * ky
    d_z = params.gap

    # Pauli-like structure in 2×2
    h = np.empty((2, 2), dtype=complex)
    h[0, 0] = ek + d_z
    h[1, 1] = ek - d_z
    h[0, 1] = d_x - 1j * d_y
    h[1, 0] = d_x + 1j * d_y
    return h


def toy_interface_hamiltonian(
    kx: float,
    ky: float,
    params: ToyInterfaceParams | None = None,
) -> np.ndarray:
    """
    Very simple 2x2 interface Hamiltonian.

    For now just an onsite term proportional to the identity; extend later
    if you want Rashba-like or texture terms at the interface.
    """
    if params is None:
        params = ToyInterfaceParams()

    h_int = params.strength * np.eye(2, dtype=complex)
    return h_int
