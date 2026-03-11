""" orbitronics.py
    --------------
    The orbitronic benchmark system for kubo_bilayer.

    Physical System
    ---------------
    A p-orbital particle (L=1, three orbitals px, py, pz) with piecewise
    constant parameters across an interface at z=0:

        H = p̂ (1/2m(z)) p̂ + γ(z) (k·L)² + J(z) (M(z)·L) + H_int(kx,ky) δ(z)

    with

        m(z)  = Θ(-z) mL  + Θ(z) mR
        γ(z)  = Θ(-z) γL  + Θ(z) γR
        J(z)  = Θ(-z) JL  + Θ(z) JR
        M(z)  = Θ(-z) ML  + Θ(z) MR

    where L = (Lx, Ly, Lz) are the L=1 angular momentum matrices in the
    px, py, pz basis.

    In the kubo_bilayer Hamiltonian notation, with in-plane momenta
    (kx, ky) fixed:

        H2 = 1/(2m) I + γ Lz²
        H1 = γ (A_par Lz + Lz A_par),    A_par = kx Lx + ky Ly
        H0 = (kp²/2m) I + γ A_par² + J (M·L)

        h_int(kx, ky) = L0 * [
            kp²/(2 m_int) I
            + γ_int (A_par · L)²
            + α (kx Ly - ky Lx)
            + (Δ_CF + β kp²)(I - Lz²)
        ]

    where kp² = kx² + ky².

    Coefficient mapping to BulkCoeffs
    ----------------------------------
    Expanding H(kz) = H0 + H1·kz + H2·kz²:

        Az  = 2*H2 = 1/m I + 2γ Lz²
        Cyz = γ (Ly Lz + Lz Ly)          (ky·kz coefficient)
        Czx = γ (Lz Lx + Lx Lz)          (kz·kx coefficient)
        Bz  = 0                           (no kz term independent of kx, ky)
        Ax  = 1/m I + 2γ Lx²             (kx² coefficient, ×2 for convention)
        Ay  = 1/m I + 2γ Ly²             (ky² coefficient, ×2 for convention)
        Bx  = 0                           (no odd kx term)
        By  = 0                           (no odd ky term)
        Cxy = 2γ (Lx Ly + Ly Lx) / 2     (kx·ky coefficient)
        D   = J (M·L)                     (constant term)

    Note: BulkCoeffs stores the kx² coefficient as Ax (appearing as
    1/2 kx² Ax in the Hamiltonian), so Ax = 2*H2_x = 1/m I + 2γ Lx².

    Coefficient mapping to InterfaceCoeffs
    ----------------------------------------
    Expanding h_int(kx, ky) = L0 * [...]:

        Let A_par = kx Lx + ky Ly. Then:
        γ_int (A_par·L)² expands as:
            kx² γ_int Lx²  +  ky² γ_int Ly²  +  kx ky γ_int (Lx Ly + Ly Lx)
        (cross terms with Lz vanish since [A_par, Lz] contributes only
        to the kx·ky cross term)

        Ax  = L0 * (1/m_int I + 2γ_int Lx²)
        Ay  = L0 * (1/m_int I + 2γ_int Ly²)
        Cxy = L0 * γ_int (Lx Ly + Ly Lx)
        Bx  = L0 * α Ly
        By  = L0 * (-α Lx)
        D   = L0 * Δ_CF (I - Lz²)

    Note: the β kp² correction to the crystal field contributes to
    Ax and Ay as +2 L0 β (I - Lz²) and to Cxy as zero (isotropic).

    Parameters
    ----------
    Bulk (per side):
        m     : float — effective mass (> 0)
        gamma : float — orbital texture coupling
        J     : float — exchange coupling
        M     : array-like, shape (3,) — magnetisation direction

    Interface:
        m_int    : float — interface effective mass (> 0)
        gamma_int: float — interface orbital texture coupling
        alpha    : float — orbital Rashba coupling
        beta     : float — momentum-dependent crystal-field correction
        delta_CF : float — crystal-field splitting
        L0       : float — overall interface energy scale

    Usage
    -----
        from showcases.orbitronics import (
            make_bulk_hamiltonian,
            make_interface_hamiltonian,
        )

        ham_L = make_bulk_hamiltonian(m=mL, gamma=gL, J=JL, M=ML, kx=kx, ky=ky)
        ham_R = make_bulk_hamiltonian(m=mR, gamma=gR, J=JR, M=MR, kx=kx, ky=ky)
        h_int = make_interface_hamiltonian(
            m_int=m_int, gamma_int=g_int, alpha=alpha,
            beta=beta, delta_CF=dCF, L0=L0,
        )

    Dependencies
    ------------
        BulkCoeffs, BulkHamiltonian, InterfaceCoeffs, InterfaceHamiltonian
            — from kubo_bilayer.setup.hamiltonians
"""
import numpy as np
from numpy.typing import NDArray
from typing import Sequence, Union
from kubo_bilayer.setup.hamiltonians import (
    BulkCoeffs, BulkHamiltonian,
    InterfaceCoeffs, InterfaceHamiltonian,
)
from kubo_bilayer.setup.operators import ObservableCoeffs, ObservableOperator

ArrayC = NDArray[np.complex128]


# ---------------------------------------------------------------------------
# L=1 angular momentum matrices in the px, py, pz basis
# ---------------------------------------------------------------------------

def _canonical_L_matrices() -> tuple[ArrayC, ArrayC, ArrayC]:
    """Return (Lx, Ly, Lz) for L=1 in the px, py, pz basis."""
    Lx = np.array(
        [[0,   0,   0  ],
         [0,   0,  -1j ],
         [0,  1j,   0  ]],
        dtype=np.complex128,
    )
    Ly = np.array(
        [[0,   0,  1j ],
         [0,   0,   0  ],
         [-1j, 0,   0  ]],
        dtype=np.complex128,
    )
    Lz = np.array(
        [[0,  -1j,  0 ],
         [1j,  0,   0 ],
         [0,   0,   0 ]],
        dtype=np.complex128,
    )
    return Lx, Ly, Lz


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_bulk_hamiltonian(
    m: float,
    gamma: float,
    J: float,
    M: Union[Sequence[float], NDArray],
    kx: float,
    ky: float,
) -> BulkHamiltonian:
    """
    Orbitronic bulk Hamiltonian for one side of the interface.

        H(kz) = H0 + H1·kz + H2·kz²

    with

        H2 = 1/(2m) I + γ Lz²
        H1 = γ (A_par Lz + Lz A_par),   A_par = kx Lx + ky Ly
        H0 = (kp²/2m) I + γ A_par² + J (M·L)

    Parameters
    ----------
    m     : float — effective mass (> 0)
    gamma : float — orbital texture coupling
    J     : float — exchange coupling
    M     : array-like, shape (3,) — magnetisation vector
    kx    : float — in-plane momentum x-component
    ky    : float — in-plane momentum y-component
    """
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}.")

    Lx, Ly, Lz = _canonical_L_matrices()
    I = np.eye(3, dtype=np.complex128)
    Z = np.zeros((3, 3), dtype=np.complex128)

    M = np.asarray(M, dtype=float).reshape(3)
    A_par = kx * Lx + ky * Ly
    ML    = M[0] * Lx + M[1] * Ly + M[2] * Lz
    kp2   = kx**2 + ky**2

    # BulkCoeffs convention: H = 1/2 kx² Ax + ... so Ax = 2 * (kx² coefficient)
    Ax  = (1.0 / m) * I + 2.0 * gamma * (Lx @ Lx)
    Ay  = (1.0 / m) * I + 2.0 * gamma * (Ly @ Ly)
    Az  = (1.0 / m) * I + 2.0 * gamma * (Lz @ Lz)
    Bx  = Z
    By  = Z
    Bz  = Z
    Cxy = gamma * (Lx @ Ly + Ly @ Lx)
    Cyz = gamma * (Ly @ Lz + Lz @ Ly)
    Czx = gamma * (Lz @ Lx + Lx @ Lz)
    D   = (kp2 / (2.0 * m)) * I + gamma * (A_par @ A_par) + J * ML

    coeffs = BulkCoeffs(
        Ax=Ax, Ay=Ay, Az=Az,
        Bx=Bx, By=By, Bz=Bz,
        Cxy=Cxy, Cyz=Cyz, Czx=Czx,
        D=D,
    )
    return BulkHamiltonian.from_coeffs(coeffs)


def make_interface_hamiltonian(
    m_int: float,
    gamma_int: float,
    alpha: float,
    beta: float,
    delta_CF: float,
    L0: float,
) -> InterfaceHamiltonian:
    """
    Orbitronic interface Hamiltonian (δ(z) term):

        h_int(kx, ky) = L0 * [
            kp²/(2 m_int) I
            + γ_int (A_par·L)²
            + α (kx Ly - ky Lx)
            + (Δ_CF + β kp²)(I - Lz²)
        ]

    Expanding in powers of kx, ky and mapping to InterfaceCoeffs
    (convention: H_int = 1/2 kx² Ax + 1/2 ky² Ay + kx Bx + ky By
                       + kx ky Cxy + D):

        Ax  = L0 * (1/m_int I + 2γ_int Lx² + 2β (I - Lz²))
        Ay  = L0 * (1/m_int I + 2γ_int Ly² + 2β (I - Lz²))
        Bx  = L0 * α Ly
        By  = L0 * (-α Lx)
        Cxy = L0 * γ_int (Lx Ly + Ly Lx)
        D   = L0 * Δ_CF (I - Lz²)

    Parameters
    ----------
    m_int    : float — interface effective mass (> 0)
    gamma_int: float — interface orbital texture coupling
    alpha    : float — orbital Rashba coupling
    beta     : float — momentum-dependent crystal-field correction
    delta_CF : float — crystal-field splitting
    L0       : float — overall interface energy scale
    """
    if m_int <= 0:
        raise ValueError(f"m_int must be positive, got {m_int}.")

    Lx, Ly, Lz = _canonical_L_matrices()
    I   = np.eye(3, dtype=np.complex128)
    Lz2 = Lz @ Lz
    ILz2 = I - Lz2

    Ax  = L0 * ((1.0 / m_int) * I + 2.0 * gamma_int * (Lx @ Lx) + 2.0 * beta * ILz2)
    Ay  = L0 * ((1.0 / m_int) * I + 2.0 * gamma_int * (Ly @ Ly) + 2.0 * beta * ILz2)
    Bx  = L0 * alpha * Ly
    By  = L0 * (-alpha * Lx)
    Cxy = L0 * gamma_int * (Lx @ Ly + Ly @ Lx)
    D   = L0 * delta_CF * ILz2

    coeffs = InterfaceCoeffs(
        Ax=Ax, Ay=Ay,
        Bx=Bx, By=By,
        Cxy=Cxy,
        D=D,
    )
    return InterfaceHamiltonian.from_coeffs(coeffs)

def make_orbital_current_observable(
    alpha: str,
    beta: int,
) -> ObservableOperator:
    """
    Observable current density operator for the orbital angular momentum
    component L_alpha in current direction beta:

        σ̂^β_{L_α}(z) = (1/2) { L_α, v̂_β(kx, ky) }

    The velocity coefficients W_β and V_β are not included here — they
    are fetched at the call site via
        BulkHamiltonian.velocity_kz_polynomial(beta, kx, ky)
    and passed into ObservableOperator.current_density_coeffs().

    Parameters
    ----------
    alpha : str
        Which L component to use as the observable: 'x', 'y', or 'z'.
    beta  : int
        Current flow direction: 0=x, 1=y, 2=z.

    Returns
    -------
    ObservableOperator with A = L_alpha in the canonical px, py, pz basis.
    """
    if alpha not in ('x', 'y', 'z'):
        raise ValueError(f"alpha must be 'x', 'y', or 'z', got '{alpha}'.")
    if beta not in (0, 1, 2):
        raise ValueError(f"beta must be 0, 1, or 2, got {beta}.")

    Lx, Ly, Lz = _canonical_L_matrices()
    L_map = {'x': Lx, 'y': Ly, 'z': Lz}
    A = L_map[alpha]

    coeffs = ObservableCoeffs(A=A, beta=beta)
    return ObservableOperator.from_coeffs(coeffs)