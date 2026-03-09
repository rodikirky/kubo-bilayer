from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

ArrayC = NDArray[np.complex128]


# ---------------------------------------------------------------------------
# Coefficient containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BulkCoeffs:
    """
    Coefficient matrices for a quadratic bulk Hamiltonian:
    H = 1/2 k_x^2 Ax + 1/2 k_y^2 Ay + 1/2 k_z^2 Az
      + kx Bx + ky By + kz Bz
      + kx ky Cxy + ky kz Cyz + kz kx Czx
      + D
    All matrices must be square, share the same shape, and be Hermitian.
    """
    # Second-order terms
    Ax: ArrayC
    Ay: ArrayC
    Az: ArrayC
    # First-order terms
    Bx: ArrayC
    By: ArrayC
    Bz: ArrayC
    # Cross terms
    Cxy: ArrayC
    Cyz: ArrayC
    Czx: ArrayC
    # Constant
    D: ArrayC

    def __post_init__(self) -> None:
        matrices = {
            "Ax": self.Ax, "Ay": self.Ay, "Az": self.Az,
            "Bx": self.Bx, "By": self.By, "Bz": self.Bz,
            "Cxy": self.Cxy, "Cyz": self.Cyz, "Czx": self.Czx,
            "D": self.D,
        }
        ref_shape = self.D.shape
        if ref_shape[0] != ref_shape[1]:
            raise ValueError("Coefficient matrices must be square.")
        for name, mat in matrices.items():
            if mat.shape != ref_shape:
                raise ValueError(
                    f"{name} has shape {mat.shape}, expected {ref_shape}."
                )
            if not np.allclose(mat, mat.conj().T):
                raise ValueError(f"{name} is not Hermitian.")


@dataclass(frozen=True)
class InterfaceCoeffs:
    """
    Coefficient matrices for a quadratic 2D interface Hamiltonian:
    H_int = 1/2 kx^2 Ax + 1/2 ky^2 Ay
          + kx Bx + ky By
          + kx ky Cxy
          + D
    All matrices must be square, share the same shape, and be Hermitian.
    """
    # Second-order terms
    Ax: ArrayC
    Ay: ArrayC
    # First-order terms
    Bx: ArrayC
    By: ArrayC
    # Cross term
    Cxy: ArrayC
    # Constant
    D: ArrayC

    def __post_init__(self) -> None:
        matrices = {
            "Ax": self.Ax, "Ay": self.Ay,
            "Bx": self.Bx, "By": self.By,
            "Cxy": self.Cxy,
            "D": self.D,
        }
        ref_shape = self.D.shape
        if ref_shape[0] != ref_shape[1]:
            raise ValueError("Coefficient matrices must be square.")
        for name, mat in matrices.items():
            if mat.shape != ref_shape:
                raise ValueError(
                    f"{name} has shape {mat.shape}, expected {ref_shape}."
                )
            if not np.allclose(mat, mat.conj().T):
                raise ValueError(f"{name} is not Hermitian.")


# ---------------------------------------------------------------------------
# Hamiltonian classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BulkHamiltonian:
    """
    Quadratic bulk Hamiltonian, constructed from a BulkCoeffs instance.

    Internally stores coefficients as named tuples for each polynomial order:
        quadratic_coeff : (Ax, Ay, Az)
        linear_coeff    : (Bx, By, Bz)
        mixed_coeff     : (Cxy, Cyz, Czx)
        constant        : D
    """
    quadratic_coeff: Tuple[ArrayC, ArrayC, ArrayC]
    linear_coeff:    Tuple[ArrayC, ArrayC, ArrayC]
    mixed_coeff:     Tuple[ArrayC, ArrayC, ArrayC]
    constant:        ArrayC

    @classmethod
    def from_coeffs(cls, params: BulkCoeffs) -> "BulkHamiltonian":
        """Construct a BulkHamiltonian from a BulkCoeffs dataclass."""
        return cls(
            quadratic_coeff=(params.Ax, params.Ay, params.Az),
            linear_coeff=(params.Bx, params.By, params.Bz),
            mixed_coeff=(params.Cxy, params.Cyz, params.Czx),
            constant=params.D,
        )

    @property
    def matrix_dim(self) -> int:
        return self.constant.shape[0]

    @property
    def identity(self) -> ArrayC:
        return np.eye(self.matrix_dim, dtype=np.complex128)

    def evaluate(self, kx: complex, ky: complex, kz: complex) -> ArrayC:
        """
        Evaluate the Hamiltonian matrix at a given (kx, ky, kz).
        H(k) = 1/2 kx^2 Ax + 1/2 ky^2 Ay + 1/2 kz^2 Az
             + kx Bx + ky By + kz Bz
             + kx ky Cxy + ky kz Cyz + kz kx Czx
             + D
        """
        A = self.quadratic_coeff
        B = self.linear_coeff
        C = self.mixed_coeff
        return (
            0.5 * kx**2 * A[0] + 0.5 * ky**2 * A[1] + 0.5 * kz**2 * A[2]
            + kx * B[0] + ky * B[1] + kz * B[2]
            + kx * ky * C[0] + ky * kz * C[1] + kz * kx * C[2]
            + self.constant
        )

    def hamiltonian_kz_polynomial(
        self,
        kx: complex,
        ky: complex,
    ) -> Tuple[ArrayC, ArrayC, ArrayC]:
        """
        At fixed (kx, ky), return the H0, H1 and H2 matrices such that
            H(kz) = H0 + H1 * kz + H2 * kz**2.

        Parameters
        ----------
        kx, ky    : in-plane momenta

        Returns
        -------
        H0 : ArrayC   — kz-independent part of the Hamiltonian
        H1 : ArrayC   — coefficient of the kz-linear term
        H2 : ArrayC   — coefficient of the kz-quadratic term
        """
        A = self.quadratic_coeff
        B = self.linear_coeff
        C = self.mixed_coeff
        D = self.constant

        H0 = np.array(
            0.5 * kx**2 * A[0] + 0.5 * ky**2 * A[1]
            + kx * B[0] + ky * B[1] + kx * ky * C[0]
            + D,
            dtype=np.complex128,
        )
        H1 = np.array(B[2] + C[1] * ky + C[2] * kx, dtype=np.complex128)
        H2 = np.array(A[2], dtype=np.complex128)

        return H0, H1, H2

    @property
    def velocity_coeffs(
        self,
    ) -> Tuple[
        Tuple[ArrayC, ArrayC, ArrayC],
        Tuple[ArrayC, ArrayC, ArrayC],
        Tuple[ArrayC, ArrayC, ArrayC],
        Tuple[ArrayC, ArrayC, ArrayC],
    ]:
        """
        Coefficient matrices for the velocity operators
        v_i(kx, ky, kz) = V0_i + Vx_i kx + Vy_i ky + Vz_i kz,
        derived from v_i = ∂H/∂k_i:

            v_x = Bx  +  Ax kx  + Cxy ky + Czx kz
            v_y = By  + Cxy kx  +  Ay ky + Cyz kz
            v_z = Bz  + Czx kx  + Cyz ky +  Az kz

        Returns
        -------
        V0_tuple : (Bx, By, Bz)             — constant term for (vx, vy, vz)
        Vx_tuple : (Ax, Cxy, Czx)           — kx coefficient
        Vy_tuple : (Cxy, Ay, Cyz)           — ky coefficient
        Vz_tuple : (Czx, Cyz, Az)           — kz coefficient
        """
        A = self.quadratic_coeff   # (Ax, Ay, Az)
        B = self.linear_coeff      # (Bx, By, Bz)
        C = self.mixed_coeff       # (Cxy, Cyz, Czx)

        V0_tuple = B
        Vx_tuple = (A[0],  C[0],  C[2])
        Vy_tuple = (C[0],  A[1],  C[1])
        Vz_tuple = (C[2],  C[1],  A[2])

        return V0_tuple, Vx_tuple, Vy_tuple, Vz_tuple

    def velocity_kz_polynomial(
        self,
        direction: int,
        kx: complex,
        ky: complex,
    ) -> Tuple[ArrayC, ArrayC]:
        """
        At fixed (kx, ky), return the W and V matrices such that
            v_i(kz) = W_i + V_i * kz.

        Parameters
        ----------
        direction : 0=x, 1=y, 2=z
        kx, ky    : in-plane momenta

        Returns
        -------
        W_i : ArrayC   — kz-independent part of v_i
        V_i : ArrayC   — coefficient of the kz-linear term
        """
        i = direction
        V0_tuple, Vx_tuple, Vy_tuple, Vz_tuple = self.velocity_coeffs

        W_i = np.array(
            V0_tuple[i] + Vx_tuple[i] * kx + Vy_tuple[i] * ky,
            dtype=np.complex128,
        )
        V_i = np.array(Vz_tuple[i], dtype=np.complex128)

        return W_i, V_i


@dataclass(frozen=True)
class InterfaceHamiltonian:
    """
    Quadratic 2D interface Hamiltonian, constructed from an InterfaceCoeffs instance.

    Internally stores:
        quadratic_coeff : (Ax, Ay)
        linear_coeff    : (Bx, By)
        mixed_coeff     : Cxy
        constant        : D
    """
    quadratic_coeff: Tuple[ArrayC, ArrayC]
    linear_coeff:    Tuple[ArrayC, ArrayC]
    mixed_coeff:     ArrayC
    constant:        ArrayC

    @classmethod
    def from_coeffs(cls, params: InterfaceCoeffs) -> "InterfaceHamiltonian":
        """Construct an InterfaceHamiltonian from an InterfaceCoeffs dataclass."""
        return cls(
            quadratic_coeff=(params.Ax, params.Ay),
            linear_coeff=(params.Bx, params.By),
            mixed_coeff=params.Cxy,
            constant=params.D,
        )

    @property
    def matrix_dim(self) -> int:
        return self.constant.shape[0]

    def evaluate(self, kx: complex, ky: complex) -> ArrayC:
        """
        Evaluate the interface Hamiltonian matrix at a given (kx, ky).
        H_int(k∥) = 1/2 kx^2 Ax + 1/2 ky^2 Ay
                  + kx Bx + ky By
                  + kx ky Cxy
                  + D
        """
        A = self.quadratic_coeff
        B = self.linear_coeff
        return (
            0.5 * kx**2 * A[0] + 0.5 * ky**2 * A[1]
            + kx * B[0] + ky * B[1]
            + kx * ky * self.mixed_coeff
            + self.constant
        )