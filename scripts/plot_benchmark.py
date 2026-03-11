""" plot_benchmark.py
    ------------------
    Visual benchmark of the single-channel cross-interface Green's
    function G(z, z') for fixed z' < 0 over a range of z in (0, z_max].

    Usage
    -----
        python scripts/plot_benchmark.py <z_prime> [z_max]

    Examples
    --------
        python scripts/plot_benchmark.py -1 1.0
        python scripts/plot_benchmark.py -1 3.0 10
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from showcases.single_channel import make_bulk_hamiltonian, make_interface_hamiltonian
from kubo_bilayer.numerics.poles import compute_poles
from kubo_bilayer.numerics.residues import compute_residues
from kubo_bilayer.greens.bulk import coincidence_value, coincidence_derivative
from kubo_bilayer.greens.interface import boundary_derivative, assemble_G00
from kubo_bilayer.greens.bilayer import compute_G_bilayer

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
if len(sys.argv) < 3:
    print("Usage: python scripts/plot_benchmark.py <z_prime> <omega> [z_max]")
    print("  z_prime must be < 0")
    sys.exit(1)

z_prime = float(sys.argv[1])
omega   = float(sys.argv[2])
z_max   = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

if z_prime >= 0:
    raise ValueError(f"z_prime must be strictly negative, got {z_prime}.")

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
mL, VL = 1.0, 1.0
mR, VR = 2.0, 2.0
v_int  = 3.0
kx, ky = 1.0, 1.0
eta    = 1e-5
n_points = 300

TOL_FILTER  = 1e-8
TOL_CLUSTER = 1e-6
TOL_SVD     = 1e-6

# ---------------------------------------------------------------------------
# Numeric pipeline (z-independent parts)
# ---------------------------------------------------------------------------
ham_L = make_bulk_hamiltonian(m=mL, V=VL, kx=kx, ky=ky)
ham_R = make_bulk_hamiltonian(m=mR, V=VR, kx=kx, ky=ky)
h_int = make_interface_hamiltonian(v_int=v_int).evaluate(kx, ky)

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

_, H1_R, H2_R = ham_R.hamiltonian_kz_polynomial(kx, ky)
_, H1_L, H2_L = ham_L.hamiltonian_kz_polynomial(kx, ky)

L_R = boundary_derivative(2.0 * H2_R, H1_R,
      coincidence_derivative(poles_R, residues_R, halfplane='upper'),
      coincidence_value(residues_R, halfplane='upper'))
L_L = boundary_derivative(2.0 * H2_L, H1_L,
      coincidence_derivative(poles_L, residues_L, halfplane='lower'),
      coincidence_value(residues_L, halfplane='lower'))

G00_numeric = assemble_G00(h_int, L_R, L_L)

# ---------------------------------------------------------------------------
# Analytic solution
# ---------------------------------------------------------------------------
kp2 = kx**2 + ky**2
kpL = np.sqrt(2.0 * mL * (omega - VL - kp2 / (2.0 * mL) + 1j * eta))
kpR = np.sqrt(2.0 * mR * (omega - VR - kp2 / (2.0 * mR) + 1j * eta))

L_L_analytic = +1j * kpL / (2.0 * mL)
L_R_analytic = -1j * kpR / (2.0 * mR)

G00_analytic = 1.0 / (-v_int + L_R_analytic - L_L_analytic)

# ---------------------------------------------------------------------------
# Sweep over z
# ---------------------------------------------------------------------------
z_values = np.linspace(1e-6, z_max, n_points)

G_numeric = np.array([
    compute_G_bilayer(z, z_prime, poles_R, residues_R,
                      poles_L, residues_L, G00_numeric)[0, 0]
    for z in z_values
])
G_analytic = np.array([
    np.exp(1j * kpR * z) * G00_analytic * np.exp(-1j * kpL * z_prime)
    for z in z_values
])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
fig.suptitle(
    f"Single-channel benchmark: $G(z, z'={z_prime})$\n"
    f"$m_L={mL},\\ V_L={VL},\\ m_R={mR},\\ V_R={VR},"
    f"\\ v_{{\\rm int}}={v_int},\\ k_x={kx},\\ k_y={ky},"
    f"\\ \\omega={omega},\\ \\eta={eta}$",
    fontsize=10
)

plot_data = [
    (np.real(G_analytic), np.real(G_numeric), r"$\mathrm{Re}\, G(z,z')$"),
    (np.imag(G_analytic), np.imag(G_numeric), r"$\mathrm{Im}\, G(z,z')$"),
    (np.abs(G_analytic),  np.abs(G_numeric),  r"$|G(z,z')|$"),
]

for ax, (analytic, numeric, ylabel) in zip(axes, plot_data):
    ax.plot(z_values, analytic, lw=2,         label="analytic")
    ax.plot(z_values, numeric,  lw=1.5, ls="--", label="numeric")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel(r"$z$")
plt.tight_layout()

outfile = f"benchmark_single_channel_zp{z_prime}_om{omega}.png"
plt.savefig(outfile, dpi=150)
plt.show()
print(f"Saved: {outfile}")