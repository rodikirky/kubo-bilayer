""" plot_orbitronics.py
    -------------------
    Visualise the full bilayer retarded Green's function G^r(z, z') for the
    orbitronic model defined in showcases/orbitronics.py.

    Fixes z' on the left side (z' < 0) and sweeps z across both sides
    (or right side only by default), plotting selected matrix elements and
    the full Frobenius norm.

    Default physical parameters
    ---------------------------
    Bulk (identical left and right sides):
        mL = mR = 1.0
        gammaL = gammaR = 0.5
        JL = JR = 1.0
        ML = MR = [0, 0, 1]   (out-of-plane magnetisation)

    Interface:
        m_int    = 1.0
        gamma_int = 0.5
        alpha    = 1.0
        beta     = 0.2
        delta_CF = 1.0
        L0       = 1.0

    Evaluation:
        kx = 0.2,  ky = 0.0
        omega = 0.0,  eta = 0.05
        z' = -5.0,  z_max = 20.0

    Usage
    -----
        python scripts/plot_orbitronics.py
        python scripts/plot_orbitronics.py --omega 0.5 --zp -3.0 --z-max 15.0
        python scripts/plot_orbitronics.py --ij1 0 0 --ij2 1 1 --all-z
        python scripts/plot_orbitronics.py --downsample 2 --eta 0.01
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from showcases.orbitronics import make_bulk_hamiltonian, make_interface_hamiltonian
from kubo_bilayer.numerics.poles import compute_poles
from kubo_bilayer.numerics.residues import compute_residues
from kubo_bilayer.greens.bulk import coincidence_value, coincidence_derivative
from kubo_bilayer.greens.interface import boundary_derivative, assemble_G00
from kubo_bilayer.greens.bilayer import compute_G_bilayer

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # Bulk — left side
    mL=1.0, gammaL=0.5, JL=1.0, ML=[0.0, 0.0, 1.0],
    # Bulk — right side (same as left by default)
    mR=1.0, gammaR=0.5, JR=1.0, MR=[0.0, 0.0, 1.0],
    # Interface
    m_int=1.0, gamma_int=0.5, alpha=1.0, beta=0.2, delta_CF=1.0, L0=1.0,
    # Evaluation
    kx=0.2, ky=0.0, omega=0.0, eta=0.05,
    # Plot
    z_prime=-5.0, z_max=20.0, n_points=300,
)

# Numerical tolerances
TOL_FILTER  = 1e-8
TOL_CLUSTER = 1e-6
TOL_SVD     = 1e-6


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot orbitronic bilayer GF G^r(z, z') vs z at fixed z'."
    )
    p.add_argument("--omega",      type=float, default=None, help="Frequency (default 0.0)")
    p.add_argument("--kx",         type=float, default=None, help="In-plane kx (default 0.2)")
    p.add_argument("--ky",         type=float, default=None, help="In-plane ky (default 0.0)")
    p.add_argument("--eta",        type=float, default=None, help="Broadening (default 0.05)")
    p.add_argument("--zp",         type=float, default=None, help="Fixed z' < 0 (default -5.0)")
    p.add_argument("--z-max",      type=float, default=None, help="Max |z| for plot (default 20.0)")
    p.add_argument("--n-points",   type=int,   default=None, help="Number of z points (default 300)")
    p.add_argument(
        "--ij1", nargs=2, type=int, default=None, metavar=("I", "J"),
        help="First matrix element to plot, e.g. --ij1 0 0 (default (0,0))",
    )
    p.add_argument(
        "--ij2", nargs=2, type=int, default=None, metavar=("I", "J"),
        help="Second matrix element to plot, e.g. --ij2 1 1 (default (1,1))",
    )
    p.add_argument(
        "--all-z", action="store_true",
        help="Plot both sides (z < 0 and z > 0). Default: right side only (z > 0).",
    )
    p.add_argument(
        "--downsample", type=int, default=1,
        help="Downsample factor for plotting (default 1, no downsampling).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frobenius_norm(G_line: np.ndarray) -> np.ndarray:
    """G_line: (Nz, n, n) → (Nz,) Frobenius norm."""
    return np.sqrt(np.sum(np.abs(G_line) ** 2, axis=(-2, -1)))


def _plot_channel(
    z: np.ndarray,
    g: np.ndarray,
    i: int,
    j: int,
    z_prime: float,
    ax_re: plt.Axes,
    ax_im: plt.Axes,
    ax_abs: plt.Axes,
) -> None:
    """Plot Re, Im, |G| for one matrix element on provided axes."""
    label = f"G[{i},{j}]"
    ax_re.plot(z, g.real, lw=1.2)
    ax_re.set_ylabel(f"Re {label}")
    ax_re.axvline(0, color="k", lw=0.7, ls="--")
    ax_re.axvline(z_prime, color="gray", lw=0.7, ls=":")

    ax_im.plot(z, g.imag, lw=1.2, color="C1")
    ax_im.set_ylabel(f"Im {label}")
    ax_im.axvline(0, color="k", lw=0.7, ls="--")
    ax_im.axvline(z_prime, color="gray", lw=0.7, ls=":")

    ax_abs.plot(z, np.abs(g), lw=1.2, color="C2")
    ax_abs.set_ylabel(f"|{label}|")
    ax_abs.set_yscale("log")
    ax_abs.axvline(0, color="k", lw=0.7, ls="--")
    ax_abs.axvline(z_prime, color="gray", lw=0.7, ls=":")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    D = DEFAULTS

    # Resolve parameters
    omega   = args.omega    if args.omega    is not None else D["omega"]
    kx      = args.kx      if args.kx       is not None else D["kx"]
    ky      = args.ky      if args.ky       is not None else D["ky"]
    eta     = args.eta     if args.eta      is not None else D["eta"]
    z_prime = args.zp      if args.zp       is not None else D["z_prime"]
    z_max   = args.z_max   if args.z_max    is not None else D["z_max"]
    n_pts   = args.n_points if args.n_points is not None else D["n_points"]
    i1, j1  = tuple(args.ij1) if args.ij1 is not None else (0, 0)
    i2, j2  = tuple(args.ij2) if args.ij2 is not None else (1, 1)
    ds      = max(int(args.downsample), 1)

    if z_prime >= 0:
        raise ValueError(f"z' must be < 0 (left side). Got z'={z_prime}.")

    print("=== Orbitronic bilayer GF plot ===")
    print(f"  omega={omega}, kx={kx}, ky={ky}, eta={eta}")
    print(f"  z'={z_prime}, z_max={z_max}, n_points={n_pts}")
    print(f"  channels: ({i1},{j1}), ({i2},{j2})")

    # -----------------------------------------------------------------------
    # Build Hamiltonians
    # -----------------------------------------------------------------------
    ham_L = make_bulk_hamiltonian(
        m=D["mL"], gamma=D["gammaL"], J=D["JL"], M=D["ML"], kx=kx, ky=ky,
    )
    ham_R = make_bulk_hamiltonian(
        m=D["mR"], gamma=D["gammaR"], J=D["JR"], M=D["MR"], kx=kx, ky=ky,
    )
    h_int_obj = make_interface_hamiltonian(
        m_int=D["m_int"], gamma_int=D["gamma_int"], alpha=D["alpha"],
        beta=D["beta"], delta_CF=D["delta_CF"], L0=D["L0"],
    )
    h_int = h_int_obj.evaluate(kx, ky)

    # -----------------------------------------------------------------------
    # Poles and residues
    # -----------------------------------------------------------------------
    poles_R, orders_R = compute_poles(
        ham_R, kx, ky, omega, eta,
        tol_filter=TOL_FILTER, tol_cluster=TOL_CLUSTER, halfplane='upper',
    )
    poles_L, orders_L = compute_poles(
        ham_L, kx, ky, omega, eta,
        tol_filter=TOL_FILTER, tol_cluster=TOL_CLUSTER, halfplane='lower',
    )
    residues_R = compute_residues(
        ham_R, poles_R, orders_R, kx, ky, omega, eta, tol=TOL_SVD,
    )
    residues_L = compute_residues(
        ham_L, poles_L, orders_L, kx, ky, omega, eta, tol=TOL_SVD,
    )

    print(f"  poles_R: {len(poles_R)} upper half-plane poles")
    print(f"  poles_L: {len(poles_L)} lower half-plane poles")

    # -----------------------------------------------------------------------
    # Boundary derivatives and G00
    # -----------------------------------------------------------------------
    _, H1_R, H2_R = ham_R.hamiltonian_kz_polynomial(kx, ky)
    _, H1_L, H2_L = ham_L.hamiltonian_kz_polynomial(kx, ky)

    G0_R  = coincidence_value(residues_R, halfplane='upper')
    dG0_R = coincidence_derivative(poles_R, residues_R, halfplane='upper')
    G0_L  = coincidence_value(residues_L, halfplane='lower')
    dG0_L = coincidence_derivative(poles_L, residues_L, halfplane='lower')

    L_R = boundary_derivative(2.0 * H2_R, H1_R, dG0_R, G0_R)
    L_L = boundary_derivative(2.0 * H2_L, H1_L, dG0_L, G0_L)
    G00 = assemble_G00(h_int, L_R, L_L)

    print(f"  G(0,0) Frobenius norm: {np.linalg.norm(G00):.4f}")

    # -----------------------------------------------------------------------
    # z grid
    # -----------------------------------------------------------------------
    if args.all_z:
        z_left  = np.linspace(-z_max, -1e-6, n_pts // 2)
        z_right = np.linspace( 1e-6,  z_max, n_pts // 2)
        z_grid  = np.concatenate([z_left, z_right])
    else:
        z_grid = np.linspace(1e-6, z_max, n_pts)

    # -----------------------------------------------------------------------
    # Evaluate G^r(z, z') for each z
    # -----------------------------------------------------------------------
    print(f"  Evaluating G^r(z, z'={z_prime}) for {len(z_grid)} z points...")
    G_line = np.zeros((len(z_grid), 3, 3), dtype=np.complex128)
    for idx, z in enumerate(z_grid):
        G_line[idx] = compute_G_bilayer(
            z, z_prime, poles_R, residues_R, poles_L, residues_L, G00,
        )

    # Validate channel indices
    n = G_line.shape[-1]
    for (ii, jj), name in [((i1, j1), "ij1"), ((i2, j2), "ij2")]:
        if not (0 <= ii < n and 0 <= jj < n):
            raise ValueError(
                f"--{name} ({ii},{jj}) out of range for {n}x{n} matrix."
            )

    g1   = G_line[:, i1, j1]
    g2   = G_line[:, i2, j2]
    norm = _frobenius_norm(G_line)

    # Downsample
    z_plot    = z_grid[::ds]
    g1_plot   = g1[::ds]
    g2_plot   = g2[::ds]
    norm_plot = norm[::ds]

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    same_channel = (i1 == i2 and j1 == j2)
    n_channel_panels = 1 if same_channel else 2

    fig = plt.figure(figsize=(10, 3 * (3 * n_channel_panels + 1)))
    fig.suptitle(
        f"Orbitronic G^r(z, z'={z_prime})  "
        f"[omega={omega}, kx={kx}, ky={ky}, eta={eta}]",
        fontsize=11,
    )

    total_rows = 3 * n_channel_panels + 1
    gs = gridspec.GridSpec(total_rows, 1, figure=fig, hspace=0.45)

    # Channel 1
    ax_re1  = fig.add_subplot(gs[0])
    ax_im1  = fig.add_subplot(gs[1])
    ax_abs1 = fig.add_subplot(gs[2])
    _plot_channel(z_plot, g1_plot, i1, j1, z_prime, ax_re1, ax_im1, ax_abs1)
    ax_re1.set_title(f"Channel ({i1},{j1})", fontsize=9)

    # Channel 2 (if different)
    if not same_channel:
        ax_re2  = fig.add_subplot(gs[3])
        ax_im2  = fig.add_subplot(gs[4])
        ax_abs2 = fig.add_subplot(gs[5])
        _plot_channel(z_plot, g2_plot, i2, j2, z_prime, ax_re2, ax_im2, ax_abs2)
        ax_re2.set_title(f"Channel ({i2},{j2})", fontsize=9)

    # Frobenius norm
    ax_norm = fig.add_subplot(gs[-1])
    ax_norm.plot(z_plot, norm_plot, lw=1.2, color="C3")
    ax_norm.set_yscale("log")
    ax_norm.set_ylabel("||G(z,z')||_F")
    ax_norm.set_xlabel("z")
    ax_norm.set_title("Frobenius norm", fontsize=9)
    ax_norm.axvline(0, color="k", lw=0.7, ls="--", label="interface (z=0)")
    ax_norm.axvline(z_prime, color="gray", lw=0.7, ls=":", label=f"z'={z_prime}")
    ax_norm.legend(fontsize=8)

    # x-label on bottom channel panel
    (ax_abs2 if not same_channel else ax_abs1).set_xlabel("z")

    # Output filename
    side = "allz" if args.all_z else "rightz"
    fname = (
        f"orbitronics_GF_zp{z_prime}_om{omega}_kx{kx}_ky{ky}_{side}.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.show()


if __name__ == "__main__":
    main()