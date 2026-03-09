# TODO: Tests need to be altered/added for the recent changes made in this file.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Sequence, Optional, Literal, Union, Callable

import numpy as np
from numpy.typing import NDArray

ArrayC = NDArray[np.complex128]


# ----------------------------------------------------------------------
# region Helpers
# ----------------------------------------------------------------------


def _as_real3(vec: Union[Sequence[float],NDArray[np.float64]]) -> NDArray[np.float64]:
    """Convert input to a real 3-vector of shape (3,)."""
    arr = np.asarray(vec, dtype=float).reshape(3)
    return arr


def _check_unitary(matrix: ArrayC, tol: float = 1e-10) -> None:
    """Raise ValueError if U is not unitary within tolerance."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.shape != (3, 3):
        raise ValueError(f"Basis matrix must be (3,3). Got {matrix.shape}.")
    I = np.eye(3, dtype=np.complex128)
    UU = matrix.conj().T @ matrix
    if not np.allclose(UU, I, atol=tol):
        raise ValueError("Basis matrix is not unitary within tolerance.")


def canonical_L_matrices(basis: Optional[ArrayC] = None) -> Tuple[ArrayC, ArrayC, ArrayC]:
    """
    Return Lx, Ly, Lz for L=1 in the px, py, pz basis, optionally transformed
    by a unitary basis matrix U (3x3):

        L_i' = U^† L_i U.
    """
    # Canonical L=1 matrices in px, py, pz basis
    Lx = np.array(
        [[0, 0, 0],
         [0, 0, -1j],
         [0, 1j, 0]],
        dtype=np.complex128,
    )
    Ly = np.array(
        [[0, 0, 1j],
         [0, 0, 0],
         [-1j, 0, 0]],
        dtype=np.complex128,
    )
    Lz = np.array(
        [[0, -1j, 0],
         [1j, 0, 0],
         [0, 0, 0]],
        dtype=np.complex128,
    )

    if basis is None:
        return Lx, Ly, Lz

    U = np.asarray(basis, dtype=np.complex128)
    _check_unitary(U)
    U_dag = U.conj().T

    Lx_p = U_dag @ Lx @ U
    Ly_p = U_dag @ Ly @ U
    Lz_p = U_dag @ Lz @ U
    return Lx_p, Ly_p, Lz_p
#endregion

# ----------------------------------------------------------------------
# region Bulk orbitronic Hamiltonian (N or F)
# ----------------------------------------------------------------------
@dataclass
class OrbitronicBulkParams:
    mass: float = 1.0
    gamma: float = 0.5
    J: float = 1.0
    magnetisation: Union[Sequence[float],NDArray[np.float64]] = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )

@dataclass
class OrbitronicBulk:
    """
    Orbitronic bulk Hamiltonian

        H(k) = (k^2 / 2m) * I
               + gamma (k·L)^2
               + J (M·L),

    where k, M are real 3-vectors and L = (Lx, Ly, Lz) are 3x3 matrices.

    This class is agnostic to N vs F: you simply use different parameters.
    """
    mass: float
    gamma: float
    J: float
    magnetisation: NDArray[np.float64]
    L: Tuple[ArrayC, ArrayC, ArrayC] = canonical_L_matrices()

    @classmethod
    def from_params(
        cls,
        params: OrbitronicBulkParams,
        basis: Optional[ArrayC] = None,
    ) -> "OrbitronicBulk":
        """Generates OrbitronicBulk from parameters using the canonical L matrices."""
        mass = params.mass
        if mass <= 0:
            raise ValueError("mass must be positive.")
        gamma = params.gamma
        J = params.J
        magnetisation = params.magnetisation
        M = _as_real3(magnetisation)
        L = canonical_L_matrices(basis)
        return cls(mass=mass, gamma=gamma, J=J, magnetisation=M, L=L) # type: ignore

    @property
    def identity(self) -> ArrayC:
        return np.eye(3, dtype=np.complex128)

    def potential(self, k: Union[Sequence[float],NDArray[np.float64]]) -> ArrayC:
        """
        Orbitronic potential:

            V(k) = γ (k·L)^2 + J (M·L).
        """
        k_vec = _as_real3(k).astype(float)
        Lx, Ly, Lz = self.L
        M = self.magnetisation

        # k·L and M·L
        kL = k_vec[0] * Lx + k_vec[1] * Ly + k_vec[2] * Lz
        ML = M[0] * Lx + M[1] * Ly + M[2] * Lz

        orbital_term = self.gamma * (kL @ kL)
        exchange_term = self.J * ML
        return orbital_term + exchange_term

    def hamiltonian(self, k: Union[Sequence[float],NDArray[np.float64]]) -> ArrayC:
        """
        Full bulk (single-particle) Hamiltonian at momentum k = (kx, ky, kz):

            H(k) = (k^2 / 2m) I + V(k).
        
        TODO: add a batched version that accepts arrays of k with shape (..., 3) 
        and returns (..., dim, dim) to enable vectorized Green's function evaluation.
        """
        k_vec = _as_real3(k).astype(float)
        k2 = float(np.dot(k_vec, k_vec))
        kinetic = (k2 / (2.0 * self.mass)) * self.identity
        V = self.potential(k)
        return kinetic + V
    
    def greens_retarded(self, omega: float, eta: float, k: Union[Sequence[float],NDArray[np.float64]]):
        h_k = self.hamiltonian(k)
        mat = (omega + 1j * eta) * self.identity - h_k
        # TODO: Install linalg error safeguard here and in greens.py
        return np.linalg.solve(mat, self.identity) # More stable than direct inversion

    def hamiltonian_kz_poly_coeffs(self, kx: float, ky: float) -> tuple[ArrayC, ArrayC, ArrayC]:
        """
        Return (H0, H1, H2) such that for any (possibly complex) kz:
            H(kz) = H0 + kz*H1 + kz**2 * H2
        with kx,ky held fixed.

        Useful for root computation.
        """
        kx = float(kx)
        ky = float(ky)

        Lx, Ly, Lz = self.L
        I = self.identity

        # A_parallel = kx*Lx + ky*Ly
        A_par = kx * Lx + ky * Ly

        # Exchange term J (M·L) is kz-independent
        M = self.magnetisation
        ML = M[0] * Lx + M[1] * Ly + M[2] * Lz

        # Build coefficients
        H2 = (1.0 / (2.0 * self.mass)) * I + self.gamma * (Lz @ Lz)
        H1 = self.gamma * (A_par @ Lz + Lz @ A_par)
        H0 = ((kx * kx + ky * ky) / (2.0 * self.mass)) * I + self.gamma * (A_par @ A_par) + self.J * ML

        return H0, H1, H2

    def hamiltonian_from_kz_poly(self, kz: complex, H0: ArrayC, H1: ArrayC, H2: ArrayC) -> ArrayC:
        """Evaluate H(kz) from precomputed coefficients using Horner's rule."""
        kz = np.complex128(kz)
        return (H2 * kz + H1) * kz + H0  # H0 + kz*H1 + kz^2*H2

    def dH_dkz_from_kz_poly(self, kz: complex, H1: ArrayC, H2: ArrayC) -> ArrayC:
        """
        Evaluate ∂H/∂kz = H1 + 2*kz*H2 from coefficients.
        Needed for the residue theorem.
        """
        kz = np.complex128(kz)
        return H1 + (2.0 * kz) * H2

    def M_and_Mprime_from_kz_poly(
        self, kz: complex, omega: float, eta: float, H0: ArrayC, H1: ArrayC, H2: ArrayC
    ) -> tuple[ArrayC, ArrayC]:
        """
        Convenient for residue theorem. M is the inverse of the k-space retarded Green's function.
        
        Returns M(kz) and M'(kz) where
            M(kz) = (omega+i*eta)I - H(kz) 
            M'(kz) = dM/dkz = - dH/dkz
        """
        I = self.identity
        Hk = self.hamiltonian_from_kz_poly(kz, H0, H1, H2)
        dH = self.dH_dkz_from_kz_poly(kz, H1, H2)
        M = (omega + 1j * eta) * I - Hk
        Mp = -dH
        return M, Mp
    
    def velocity_components(self, k: Union[Sequence[float],NDArray[np.float64]], hbar: float = 1.0
                            ) -> tuple[ArrayC, ArrayC, ArrayC]:
        """
        Return (v_x, v_y, v_z) at momentum k, where

            v_i(k) = (1/ħ) ∂H/∂k_i.

        For H(k) = k^2/(2m) I + gamma (k·L)^2 + J (M·L),

            ∂H/∂k_j = (k_j / m) I + gamma (L_j (k·L) + (k·L) L_j).

        Note:
        Beware that this identity does not hold when translation invariance is broken,
        e.g. in the presence of impurities or interfaces.
        In such cases, the k-operators need to be carefully defined!
        """
        k_vec = _as_real3(k).astype(float)
        kx, ky, kz = k_vec
        Lx, Ly, Lz = self.L
        I = self.identity

        # k·L
        kL = kx * Lx + ky * Ly + kz * Lz

        # Derivatives:
        dH_dkx = (kx / self.mass) * I + self.gamma * (Lx @ kL + kL @ Lx)
        dH_dky = (ky / self.mass) * I + self.gamma * (Ly @ kL + kL @ Ly)
        dH_dkz = (kz / self.mass) * I + self.gamma * (Lz @ kL + kL @ Lz)

        v_x = dH_dkx / hbar
        v_y = dH_dky / hbar
        v_z = dH_dkz / hbar
        return v_x, v_y, v_z

    def orbital_current_operator(
        self,
        k: Sequence[float],
        flow_dir: Literal["x", "y", "z"],
        L_comp: Literal["x", "y", "z"],
        hbar: float = 1.0,
    ) -> ArrayC:
        """
        Orbital-angular-momentum current operator J_i^{L_comp}(k):

            J_i^{Lα}(k) = 1/2 { v_i(k), L_α }.

        Parameters
        ----------
        k : (kx, ky, kz)
            Momentum at which to evaluate the operator.
        flow_dir : "x" | "y" | "z"
            Spatial direction of the current (i).
        L_comp : "x" | "y" | "z"
            Which component of orbital angular momentum is transported (α).
        hbar : float, optional
            Planck constant; use 1.0 for natural units.

        Note:
        Beware that this identity does not hold when translation invariance is broken,
        e.g. in the presence of impurities or interfaces.
        In such cases, the k-operators need to be carefully defined!
        """
        if L_comp not in ("x", "y", "z"):
            raise ValueError(f"L_comp must be 'x', 'y', or 'z'. Got {L_comp}.")
        if flow_dir not in ("x", "y", "z"):
            raise ValueError(f"flow_dir must be 'x', 'y', or 'z'. Got {flow_dir}.")
        
        v_x, v_y, v_z = self.velocity_components(k, hbar=hbar)
        Lx, Ly, Lz = self.L

        v_map = {"x": v_x, "y": v_y, "z": v_z}
        L_map = {"x": Lx, "y": Ly, "z": Lz}

        v_i = v_map[flow_dir]
        L_alpha = L_map[L_comp]

        # Symmetrized product:
        J_i_alpha = 0.5 * (v_i @ L_alpha + L_alpha @ v_i)
        return J_i_alpha
#endregion

# ----------------------------------------------------------------------
# region Interface Hamiltonian
# ----------------------------------------------------------------------

@dataclass
class OrbitronicInterfaceParams:
    m_int: float = 1.0
    gamma_int: float = 0.5
    alpha: float = 1.0
    beta: float = 0.2
    delta_CF: float = 1.0
    L0: float = 1.0

@dataclass
class OrbitronicInterface:
    """
    Orbitronic interface Hamiltonian (δ(z) term).

    The model we implement is:

        H_int(k_parallel) = L0 * [
            k_parallel^2 / (2 m_int)
            + γ_int (k_parallel · L)^2
            + α (k_x L_y - k_y L_x)
            + (Δ_CF + β k_parallel^2) (1 - L_z^2)
        ],

    where k_parallel = (k_x, k_y) and L = (Lx, Ly, Lz) are 3x3 matrices
    (usually the same L-matrices as used on one side of the interface).
    """

    m_int: float
    gamma_int: float
    alpha: float
    beta: float
    delta_CF: float
    L0: float
    L: Tuple[ArrayC, ArrayC, ArrayC]

    @classmethod
    def from_params(
        cls,
        params: OrbitronicInterfaceParams,
        basis: Optional[ArrayC] = None,
    ) -> "OrbitronicInterface":
        """
        Generates OrbitronicInterface object from parameters given as OrbitronicInterfaceParams object 
        using the canonical L matrices.
        """
        # Extract and validate params
        m_int = params.m_int
        if m_int <= 0:
            raise ValueError("m_int must be positive.")
        gamma_int= params.gamma_int
        alpha = params.alpha
        beta = params.beta
        delta_CF = params.delta_CF
        L0 = params.L0

        L = canonical_L_matrices(basis)
        return cls(
            m_int=m_int,
            gamma_int=gamma_int,
            alpha=alpha,
            beta=beta,
            delta_CF=delta_CF,
            L0=L0,
            L=L,
        )

    def _orbital_texture_potential(self, kx: float, ky: float) -> ArrayC:
        Lx, Ly, _ = self.L
        dotkL = kx * Lx + ky * Ly
        return self.gamma_int * (dotkL @ dotkL)

    def _orbital_rashba_potential(self, kx: float, ky: float) -> ArrayC:
        Lx, Ly, _ = self.L
        return self.alpha * (kx * Ly - ky * Lx)

    def hamiltonian(self, kx: float, ky: float) -> ArrayC:
        """
        Interface Hamiltonian H_int(kx, ky) as a 3x3 complex matrix.
        """
        kx = float(kx)
        ky = float(ky)
        k_par_sq = kx * kx + ky * ky

        _, _, Lz = self.L
        Lz2 = Lz @ Lz

        kinetic = k_par_sq / (2.0 * self.m_int) * np.eye(3, dtype=np.complex128)
        V_tex = self._orbital_texture_potential(kx, ky)
        V_OR = self._orbital_rashba_potential(kx, ky)
        CF_term = (self.delta_CF + self.beta * k_par_sq) * (np.eye(3, dtype=np.complex128) - Lz2)

        H_int = self.L0 * (kinetic + V_tex + V_OR + CF_term)
        return H_int
#endregion
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#region Observable & perturbation operators
# in real space, not k-space
# ----------------------------------------------------------------------
'''
def orbital_density_operator(
        system: OrbitronicBulk,
        component: Literal["x", "y", "z"],
        ) -> ArrayC:
    comp_index = {"x": 0, "y": 1, "z": 2}[component]
    L_operator = system.L[comp_index]
    return L_operator

def velocity_operator(
        system: OrbitronicBulk,
        flow_dir: Literal["x", "y", "z"],
        hbar: float = 1.0,

) -> Callable:
    """
    The formula for the velocity operator is 
    v_i(k) = (1/ħ) ∂H/∂k_i for i = x, y, z.
    
    In real space, the k-operators become derivatives.
    """
    flow_index = {"x": 0, "y": 1, "z": 2}[flow_dir]
    L = system.L # (3,3,3,)

    # Derivative functions in real space
    def deriv_func_x(
            operating_on: Callable, 
            evaluate_at: NDArray[np.float64],
            ) -> ArrayC:
        x, y, z = evaluate_at
        # kx -> -i ∂/∂x (maybe separate two-sided derivative)
        # placeholder for derivative operation
        return operating_on(evaluate_at) * 1j # not yet differentiated
    
    def deriv_func_y(
            operating_on: Callable, 
            evaluate_at: NDArray[np.float64],
            ) -> ArrayC:
        x, y, z = evaluate_at
        # ky -> -i ∂/∂y
        # placeholder for derivative operation
        return operating_on(evaluate_at) * 1j # not yet differentiated
    
    def deriv_func_z(
            operating_on: Callable, 
            evaluate_at: NDArray[np.float64],
            ) -> ArrayC:
        x, y, z = evaluate_at
        # kz -> -i ∂/∂z
        # placeholder for derivative operation
        return operating_on(evaluate_at) * 1j # not yet differentiated
    
    deriv_funcs = (deriv_func_x, deriv_func_y, deriv_func_z)

    def k_vector_realspace(
            operating_on: Callable, 
            evaluate_at: NDArray[np.float64],
            ) -> ArrayC:
        dx, dy, dz = deriv_funcs
        result = [dx(operating_on,evaluate_at),dy(operating_on,evaluate_at),dz(operating_on,evaluate_at)]
        result_vec = np.array(result, dtype=np.complex128)
        return result_vec

    # k·L becomes a product of derivative function in real space
    def kL_operator(
            operating_on: Callable, 
            evaluate_at: NDArray[np.float64],
            k_vec_operator: Callable | None = None, # enables custom choice
            ) -> ArrayC:
        
        if k_vec_operator is None:
            k_operation_result = k_vector_realspace(operating_on,evaluate_at)
        else: 
            k_operation_result = k_vec_operator(operating_on,evaluate_at)

        kL_result = np.zeros(L[0].shape)
        for i in range(3):
            term_i = L[i] @ k_operation_result[i]
            kL_result += term_i
        kL_result = np.array(kL_result, dtype=np.complex128)
        return kL_result
    
    # Piece together the velocity component
    def velocity(
            operating_on: Callable,
            evaluate_at: NDArray[np.float64],
            k_vec_operator: Callable | None = None, # enables custom choice
            ) -> ArrayC:
        if k_vec_operator is None:
            derivative_in_flow_dir = k_vector_realspace(operating_on,evaluate_at)[flow_index]
        else: 
            derivative_in_flow_dir = k_vec_operator(operating_on,evaluate_at)[flow_index]
        kL = kL_operator(operating_on,evaluate_at,k_vec_operator=k_vec_operator)
        L_dir = L[flow_index]

        # (Li @ kL + kL @ Li) component
        Li_kL = L_dir @ kL
        helper_operating_on = lambda point: L_dir @ operating_on(point)
        kL_L =  
        dH_flow_dir = derivative_in_flow_dir/system.mass + system.gamma *  + 

    dH_dkx = (kx / system.mass) * I + system.gamma * (Lx @ kL + kL @ Lx)
    dH_dky = (ky / system.mass) * I + system.gamma * (Ly @ kL + kL @ Ly)
    dH_dkz = (kz / system.mass) * I + system.gamma * (Lz @ kL + kL @ Lz)

    v_x = dH_dkx / hbar
    v_y = dH_dky / hbar
    v_z = dH_dkz / hbar

def orbital_current_density_operator(
        system: OrbitronicBulk,
        flow_dir: Literal["x", "y", "z"],
        L_comp: Literal["x", "y", "z"],
):
    pass
'''

#endregion
