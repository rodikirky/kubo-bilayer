""" benchmark_single_channel.py
    ----------------------------
    Numeric vs. analytic comparison of the cross-interface retarded
    Green's function G(z, z') for the single-channel benchmark system
    defined in showcases/single_channel.py.

    Physical system
    ---------------
    Scalar particle with piecewise constant mass and potential:

        H = p̂ (1/2m(z)) p̂ + V(z) + v_int * δ(z)

    Analytic solution
    -----------------
    From the pedagogical note (GF_pedagogy.pdf), the cross-interface
    Green's function for z > 0, z' < 0 is (eq. 24, half-space terms
    vanish for z and z' on opposite sides):

        G(z, z') = F_R(z) · G(0,0) · F-bar_L(z')
                 = exp(i·k+_R·z) · G(0,0) · exp(-i·k+_L·z')

    with upper-half-plane poles (eq. 7):

        k+_s = sqrt(2·m_s·(Omega_s + i·eta))
        Omega_s = omega - V_s - (kx²+ky²)/(2·m_s)

    and interface Green's function (eqs. 27, 31, 32):

        G(0,0) = 1 / (-v_int + i·k+_R/(2·m_R) + i·k+_L/(2·m_L))

    Test parameters
    ---------------
    (mL, VL, mR, VR, v_int, kx, ky, omega, eta, z, z')
    = (1, 1, 2, 2, 3, 1, 1, 1, 1e-5, 1, -1)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from showcases.single_channel import make_bulk_hamiltonian, make_interface_hamiltonian
from kubo_bilayer.numerics.poles import compute_poles
from kubo_bilayer.numerics.residues import compute_residues
from kubo_bilayer.greens.bulk import coincidence_value, coincidence_derivative
from kubo_bilayer.greens.interface import boundary_derivative, assemble_G00
from kubo_bilayer.greens.bilayer import compute_G_bilayer

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
mL, VL = 1.0, 1.0
mR, VR = 2.0, 2.0
v_int  = 3.0
kx, ky = 1.0, 1.0
omega  = 1.0
eta    = 1e-5
z      = 1.0
z_prime = -1.0

# Numerical tolerances (calibrated for this system's energy scale)
TOL_FILTER  = 1e-8
TOL_CLUSTER = 1e-6
TOL_SVD     = 1e-6

# ---------------------------------------------------------------------------
# Analytic solution
# ---------------------------------------------------------------------------

def _upper_pole(m, V):
    kp2 = kx**2 + ky**2
    Omega = omega - V - kp2 / (2.0 * m)
    return np.sqrt(2.0 * m * (Omega + 1j * eta))

kpL = _upper_pole(mL, VL)
kpR = _upper_pole(mR, VR)

L_L_analytic = +1j * kpL / (2.0 * mL)
L_R_analytic = -1j * kpR / (2.0 * mR)

G00_analytic = 1.0 / (-v_int + L_R_analytic - L_L_analytic)
G_analytic   = np.exp(1j * kpR * z) * G00_analytic * np.exp(-1j * kpL * z_prime)

# ---------------------------------------------------------------------------
# Numeric pipeline
# ---------------------------------------------------------------------------

ham_L = make_bulk_hamiltonian(m=mL, V=VL, kx=kx, ky=ky)
ham_R = make_bulk_hamiltonian(m=mR, V=VR, kx=kx, ky=ky)
h_int = make_interface_hamiltonian(v_int=v_int).evaluate(kx, ky)

# Poles and residues
poles_R, orders_R = compute_poles(
    ham_R, kx, ky, omega, eta,
    tol_filter=TOL_FILTER, tol_cluster=TOL_CLUSTER, halfplane='upper'
)
poles_L, orders_L = compute_poles(
    ham_L, kx, ky, omega, eta,
    tol_filter=TOL_FILTER, tol_cluster=TOL_CLUSTER, halfplane='lower'
)
residues_R = compute_residues(
    ham_R, poles_R, orders_R, kx, ky, omega, eta, tol=TOL_SVD
)
residues_L = compute_residues(
    ham_L, poles_L, orders_L, kx, ky, omega, eta, tol=TOL_SVD
)

# Boundary derivatives
_, H1_R, H2_R = ham_R.hamiltonian_kz_polynomial(kx, ky)
_, H1_L, H2_L = ham_L.hamiltonian_kz_polynomial(kx, ky)
Az_R = 2.0 * H2_R
Az_L = 2.0 * H2_L

G0_R  = coincidence_value(residues_R, halfplane='upper')
dG0_R = coincidence_derivative(poles_R, residues_R, halfplane='upper')
G0_L  = coincidence_value(residues_L, halfplane='lower')
dG0_L = coincidence_derivative(poles_L, residues_L, halfplane='lower')

L_R = boundary_derivative(Az_R, H1_R, dG0_R, G0_R)
L_L = boundary_derivative(Az_L, H1_L, dG0_L, G0_L)

# Interface and bilayer Green's functions
G00_numeric = assemble_G00(h_int, L_R, L_L)
G_numeric   = compute_G_bilayer(
    z, z_prime, poles_R, residues_R, poles_L, residues_L, G00_numeric
)

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
# Diagnostic: bulk poles
print("=== Diagnostics ===")
print(f"  k+_L analytic : {kpL:.6f}")
print(f"  k+_R analytic : {kpR:.6f}")
print(f"  poles_L numeric: {poles_L}")
print(f"  poles_R numeric: {poles_R}")
print()

# Diagnostic: bulk coincidence values
print(f"  G0_L analytic : {-1j * mL / kpL:.6f}")
print(f"  G0_R analytic : {-1j * mR / kpR:.6f}")
print(f"  G0_L numeric  : {G0_L[0,0]:.6f}")
print(f"  G0_R numeric  : {G0_R[0,0]:.6f}")
print()

# Diagnostic: boundary derivatives
print(f"  L_L analytic : {L_L_analytic:.6f}")
print(f"  L_R analytic : {L_R_analytic:.6f}")
print(f"  L_L numeric  : {L_L[0,0]:.6f}")
print(f"  L_R numeric  : {L_R[0,0]:.6f}")

print("=== Single-channel benchmark ===")
print(f"  Parameters: mL={mL}, VL={VL}, mR={mR}, VR={VR}, v_int={v_int}")
print(f"              kx={kx}, ky={ky}, omega={omega}, eta={eta}")
print(f"              z={z}, z'={z_prime}")
print()
print(f"  G(0,0) analytic : {G00_analytic:.6f}")
print(f"  G(0,0) numeric  : {G00_numeric[0,0]:.6f}")
print(f"  G(0,0) error    : {abs(G00_numeric[0,0] - G00_analytic):.2e}")
print()
print(f"  G(z,z') analytic : {G_analytic:.6f}")
print(f"  G(z,z') numeric  : {G_numeric[0,0]:.6f}")
print(f"  G(z,z') error    : {abs(G_numeric[0,0] - G_analytic):.2e}")