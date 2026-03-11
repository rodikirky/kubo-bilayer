""" single_channel.py
    ------------------
    A single-channel benchmark system for kubo_bilayer.

    Physical System
    ---------------
    A scalar particle with piecewise constant mass and potential:

        H = p̂ (1/2m(z)) p̂ + V(z) + v_int * δ(z)

    with

        m(z) = Θ(-z) * mL + Θ(z) * mR
        V(z) = Θ(-z) * VL + Θ(z) * VR

    In the kubo_bilayer Hamiltonian notation, with in-plane momenta
    (kx, ky) included in H0 via the kinetic energy k_parallel²/(2m):

        H2 = 1/(2m) * I         (kz² coefficient, via Az = 2*H2)
        H1 = 0
        H0 = (V + kp²/(2m)) * I (constant term, kp² = kx²+ky²)

        h_int = v_int * I        (scalar interface potential)

    Usage
    -----
        from showcases.single_channel import (
            make_bulk_hamiltonian,
            make_interface_hamiltonian,
        )

        ham_L = make_bulk_hamiltonian(m=mL, V=VL, kx=kx, ky=ky)
        ham_R = make_bulk_hamiltonian(m=mR, V=VR, kx=kx, ky=ky)
        h_int = make_interface_hamiltonian(v_int=v_int)

    Dependencies
    ------------
        BulkCoeffs, BulkHamiltonian, InterfaceHamiltonian
            — from kubo_bilayer.setup.hamiltonians
"""
import numpy as np
from kubo_bilayer.setup.hamiltonians import BulkCoeffs, BulkHamiltonian, InterfaceCoeffs, InterfaceHamiltonian


def make_bulk_hamiltonian(m: float, V: float, kx: float, ky: float) -> BulkHamiltonian:
    """
    Single-channel bulk Hamiltonian: H(kz) = kz²/(2m) + (V + kp²/(2m))

    In kubo_bilayer notation:
        Az = 2*H2 = 1/m * I    (kz² coefficient)
        H1 = 0
        H0 = (V + (kx²+ky²)/(2m)) * I

    Parameters
    ----------
    m  : effective mass (> 0)
    V  : on-site potential
    kx : in-plane momentum x-component
    ky : in-plane momentum y-component
    """
    I = np.array([[1.]], dtype=np.complex128)
    Z = np.array([[0.]], dtype=np.complex128)

    kp2 = kx**2 + ky**2
    H0_val = V + kp2 / (2.0 * m)

    coeffs = BulkCoeffs(
        Ax=Z, Ay=Z, Az=1.0 / m * I,   # Az = 2*H2 = 1/m * I
        Bx=Z, By=Z, Bz=Z,             # H1 = 0
        Cxy=Z, Cyz=Z, Czx=Z,
        D=H0_val * I,
    )
    return BulkHamiltonian.from_coeffs(coeffs)


def make_interface_hamiltonian(v_int: float) -> InterfaceHamiltonian:
    """
    Scalar interface potential: h_int = v_int * I (1×1 matrix).

    Parameters
    ----------
    v_int : interface potential strength
    """
    I = np.array([[1.]], dtype=np.complex128)
    Z = np.array([[0.]], dtype=np.complex128)

    coeffs = InterfaceCoeffs(
        Ax=Z, Ay=Z,
        Bx=Z, By=Z,
        Cxy=Z,
        D=v_int * I,
    )
    return InterfaceHamiltonian.from_coeffs(coeffs)