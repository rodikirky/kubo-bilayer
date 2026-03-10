""" operators.py
    ------------
    Dataclasses for the observable current density operator in a planar
    interface bilayer system.

    Physical Background
    -------------------
    The current density of an observable A in direction
    β at fixed (kx, ky) is given by:

        σ̂^β_A(z) = (1/2) { A, v̂β(kx, ky) }

    where v̂β(kx, ky) = Wβ(kx, ky) + Vβ(kx, ky) k̂z is the velocity operator in direction β,
    expressed as a kz-polynomial at fixed (kx, ky):

        Wβ : kz-independent part of vβ   (from velocity_kz_polynomial)
        Vβ : coefficient of kz           (from velocity_kz_polynomial)

    In the z'-integral, this operator enters through two anticommutators:

        {A, Wβ} : the kz-independent part of σ̂^β_A
        {A, Vβ} : the kz-linear part of σ̂^β_A

    These are the two objects returned by current_density_coeffs().

    The velocity coefficients Wβ and Vβ depend on (kx, ky) and are
    obtained externally from BulkHamiltonian.velocity_kz_polynomial()
    before being passed in here. This keeps the observable decoupled
    from any specific Hamiltonian instance.

    Structure
    ---------
    ObservableCoeffs
        User-facing input container. Holds the observable matrix A and
        the current direction β. Validated at construction time.

    ObservableOperator
        Constructed from ObservableCoeffs via from_coeffs(). Exposes
        current_density_coeffs(W_beta, V_beta) which computes the two
        anticommutators needed in the z'-integral.

    Notes
    -----
    - A must be a Hermitian matrix of the same dimension as the
      Hamiltonian coefficient matrices.
    - beta must be 0 (x), 1 (y), or 2 (z). A different current
      direction means a different ObservableOperator instance.
    - The perturbation operator B̂^λ(z') = Θ(-z') v̂λ(z') is not
      represented as a dataclass here. Its velocity coefficients Wλ
      and Vλ are fetched directly from the left-side BulkHamiltonian
      via velocity_kz_polynomial(lambda, kx, ky) at the call site.

    Dependencies
    ------------
        kubo_bilayer.setup.hamiltonians — provides velocity_kz_polynomial()
                                         whose output is passed into
                                         current_density_coeffs()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

ArrayC = NDArray[np.complex128]

__all__ = [
    "ArrayC",
    "ObservableCoeffs",
    "ObservableOperator",
]


@dataclass(frozen=True)
class ObservableCoeffs:
    """
    Input container for the observable current density operator.

    Parameters
    ----------
    A    : ArrayC, shape (n, n)
        The observable matrix (e.g. the orbital angular momentum
        operator Lz). Must be Hermitian.
    beta : int
        Current direction: 0 = x, 1 = y, 2 = z.
        A different direction means a different instance.
    """
    A:    ArrayC
    beta: int

    def __post_init__(self) -> None:
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError(
                f"A must be a square matrix, got shape {self.A.shape}."
            )
        if not np.allclose(self.A, self.A.conj().T):
            raise ValueError("A must be Hermitian.")
        if self.beta not in (0, 1, 2):
            raise ValueError(
                f"beta must be 0 (x), 1 (y), or 2 (z), got {self.beta}."
            )


@dataclass(frozen=True)
class ObservableOperator:
    """
    Observable current density operator, constructed from ObservableCoeffs.

    Holds the observable matrix A and the current direction beta.
    The kz-polynomial coefficients of the current density operator
    are computed via current_density_coeffs(), which takes the velocity
    coefficients Wβ and Vβ as external inputs.

    Construct via ObservableOperator.from_coeffs(params).
    Obtain Wβ and Vβ from BulkHamiltonian.velocity_kz_polynomial()
    """
    A:    ArrayC
    beta: int

    @classmethod
    def from_coeffs(cls, params: ObservableCoeffs) -> "ObservableOperator":
        """Construct an ObservableOperator from an ObservableCoeffs instance."""
        return cls(A=params.A, beta=params.beta)

    def current_density_coeffs(
        self,
        W_beta: ArrayC,
        V_beta: ArrayC,
    ) -> Tuple[ArrayC, ArrayC]:
        """
        Compute the kz-polynomial coefficients of the observable current
        density operator at fixed (kx, ky):

            σ̂^β_A  ↔  (1/2){ A, Wβ }  +  (1/2){ A, Vβ } k̂z

        Returns the two anticommutator matrices that enter the z'-integral
        as {A, Wβ} and {A, Vβ} (the factor of 1/2 is absorbed into the
        prefactor of the full response formula).

        Parameters
        ----------
        W_beta : ArrayC, shape (n, n)
            kz-independent part of vβ at fixed (kx, ky).
            Obtained from BulkHamiltonian.velocity_kz_polynomial(beta, kx, ky)[0].
        V_beta : ArrayC, shape (n, n)
            kz-linear coefficient of vβ at fixed (kx, ky).
            Obtained from BulkHamiltonian.velocity_kz_polynomial(beta, kx, ky)[1].

        Returns
        -------
        AW : ArrayC, shape (n, n)
            {A, Wβ} = A @ Wβ + Wβ @ A
        AV : ArrayC, shape (n, n)
            {A, Vβ} = A @ Vβ + Vβ @ A
        """
        AW = self.A @ W_beta + W_beta @ self.A
        AV = self.A @ V_beta + V_beta @ self.A
        return AW, AV