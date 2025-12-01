from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# --- Angular momentum L=1 matrices ------------------------------------


def orbital_L_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (Lx, Ly, Lz) for l = 1 in the standard |1, m> basis (m = +1, 0, -1).
    """
    # Note: units with ħ = 1
    sqrt2 = np.sqrt(2.0)

    Lx = np.array(
        [
            [0.0,        1.0 / sqrt2, 0.0],
            [1.0 / sqrt2, 0.0,        1.0 / sqrt2],
            [0.0,        1.0 / sqrt2, 0.0],
        ],
        dtype=complex,
    )

    Ly = np.array(
        [
            [0.0,        -1j / sqrt2, 0.0],
            [1j / sqrt2,  0.0,        -1j / sqrt2],
            [0.0,         1j / sqrt2, 0.0],
        ],
        dtype=complex,
    )

    Lz = np.array(
        [
            [1.0, 0.0,  0.0],
            [0.0, 0.0,  0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=complex,
    )

    return Lx, Ly, Lz


# --- Bulk orbitronic model --------------------------------------------


@dataclass
class OrbitronicBulkParams:
    """Minimal parameter set for a bulk orbitronic Hamiltonian."""
    mass: float = 1.0          # kinetic term mass
    gamma: float = 1.0         # strength of (k·L)^2 term
    J: float = 0.0             # exchange coupling
    Mx: float = 0.0            # magnetization components
    My: float = 0.0
    Mz: float = 1.0


def orbitronic_bulk_hamiltonian(
    kx: float,
    ky: float,
    kz: float,
    params: OrbitronicBulkParams | None = None,
) -> np.ndarray:
    """
    3×3 orbitronic bulk Hamiltonian (numeric version):

        H = (k^2 / 2m) I + γ (k·L)^2 + J (M·L)

    where L = (Lx, Ly, Lz) are the l=1 angular momentum matrices.
    """
    if params is None:
        params = OrbitronicBulkParams()

    Lx, Ly, Lz = orbital_L_matrices()

    k2 = kx * kx + ky * ky + kz * kz
    ek = k2 / (2.0 * params.mass)

    # k · L and (k · L)^2
    L_k = kx * Lx + ky * Ly + kz * Lz
    L_k_sq = L_k @ L_k

    # M · L
    M_dot_L = params.Mx * Lx + params.My * Ly + params.Mz * Lz

    H = (
        ek * np.eye(3, dtype=complex)
        + params.gamma * L_k_sq
        + params.J * M_dot_L
    )
    return H


# --- Interface orbitronic model ---------------------------------------


@dataclass
class OrbitronicInterfaceParams:
    """
    Minimal interface parameter set.

    Extend this with the actual texture / Rashba / crystal field terms when
    you translate your full symbolic interface Hamiltonian.
    """
    L0: float = 0.0  # simple onsite strength for now


def orbitronic_interface_hamiltonian(
    kx: float,
    ky: float,
    params: OrbitronicInterfaceParams | None = None,
) -> np.ndarray:
    """
    Placeholder 3×3 interface Hamiltonian for the orbitronic model.

    For now, just L0 * I. Later, plug in the full k_parallel-dependent
    interface structure from your symbolic code (texture, Rashba, etc.).
    """
    if params is None:
        params = OrbitronicInterfaceParams()

    dim = 3
    return params.L0 * np.eye(dim, dtype=complex)
