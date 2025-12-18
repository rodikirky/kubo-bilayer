from __future__ import annotations

import argparse
import numpy as np

from kubo.config import GridConfig, PhysicsConfig
from kubo.models.toy import ToyBulk, ToyBulkParams
from kubo.greens import realspace_greens_retarded_with_kz

from kubo.plotting import (
    profile_amplitude_over_first_axis,
    edge_leak_ratio,
    plot_profile,
    plot_complex_components,
    show,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--omega", type=float, default=0.0)
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--kx", type=float, default=0.0)
    p.add_argument("--ky", type=float, default=0.0)

    p.add_argument("--nz", type=int, default=5013)
    p.add_argument("--z-max", type=float, default=1000.0)

    # The rest are required by your GridConfig even if unused here.
    p.add_argument("--k-max", type=float, default=5.0)
    p.add_argument("--nk-parallel", type=int, default=1)
    p.add_argument("--nphi", type=int, default=1)
    p.add_argument("--omega-max", type=float, default=1.0)
    p.add_argument("--nomega", type=int, default=1)
    return p.parse_args()

def main() -> None:
    args = parse_args()

    grid = GridConfig(
        nz=args.nz,
        z_max=args.z_max,
        k_max=args.k_max,
        nk_parallel=args.nk_parallel,
        nphi=args.nphi,
        omega_max=args.omega_max,
        nomega=args.nomega,
    )
    physics = PhysicsConfig(eta=args.eta)

    toy = ToyBulk.from_params(ToyBulkParams())

    print("[run] grid   =", grid)
    print("[run] physics=", physics)
    print(f"[run] omega={args.omega}, kx={args.kx}, ky={args.ky}")

    L = 2 * grid.z_max
    dz = 2 * grid.z_max / grid.nz
    dk = 2 * np.pi / L
    print(f"[run] derived: L={L}, dz={dz}, dk≈{dk}")

    def H(kx: float, ky: float, kz: float) -> np.ndarray:
        return toy.hamiltonian((kx, ky, kz))

    delta_z, kz, G_dz, G_kz = realspace_greens_retarded_with_kz(
        omega=args.omega,
        kx=args.kx,
        ky=args.ky,
        H=H,
        physics=physics,
        grid=grid,
    )

    mid = grid.nz // 2  # after fftshift, this is the Δz≈0 index

    # ---- Diagnostics ----
    amp_dz = profile_amplitude_over_first_axis(G_dz, mode="fro")
    amp_dz_max = profile_amplitude_over_first_axis(G_dz, mode="max")
    leak = edge_leak_ratio(amp_dz, m=10, center_index=mid)
    print(f"[diag] edge_leak_ratio (m=10): {leak:.3e}  (smaller => safer vs wrap-around)")
    if leak > 1e-5:
        print("[WARN] Edge leakage not negligible; wrap-around may affect Δz lookups/gluing.")

    # Toy model should be diagonal (off-diagonal ~ 0)
    offdiag_max = np.max(np.abs(G_dz[:, 0, 1]))
    diag_max = np.max(np.abs(np.diagonal(G_dz, axis1=1, axis2=2)))
    print(f"[diag] max |G_dz[0,1]|: {offdiag_max:.3e}")
    print(f"[diag] max |diag(G_dz)|: {diag_max:.3e}")

    # ---- Plots ----

    # 1) Core: decay of Frobenius norm in real space (wrap-around check)
    plot_profile(
        delta_z,
        amp_dz,
        title=f"Toy: ||G^R(ω={args.omega}; Δz)||  (η={args.eta})",
        xlabel="Δz",
        ylabel="Frobenius norm",
        logy=True,
    )
    
    # 2) Core: decay of max norm in real space 
    #plot_profile(
    #    delta_z,
    #    amp_dz_max,
    #    title=f"Toy: ||G^R(ω={args.omega}; Δz)||  (η={args.eta})",
    #    xlabel="Δz",
    #    ylabel="Max norm",
    #    logy=True,
    #)

    # 3) Real/imag of a representative real-space element
    plot_complex_components(
        delta_z,
        G_dz[:, 0, 0],
        title="Toy: G^R_00(Δz) components",
        xlabel="Δz",
        ylabel="G_00",
    )
    plot_complex_components(
        delta_z,
        G_dz[:, 1, 1],
        title="Toy: G^R_11(Δz) components",
        xlabel="Δz",
        ylabel="G_11",
    )

    # 4) kz diagnostic: choose a representative scalar element
    amp_kz_00 = np.abs(G_kz[:, 1, 1])
    plot_profile(
        kz,
        amp_kz_00,
        title="Toy: |G^R_11(kz)| vs kz (kz sampling diagnostic)",
        xlabel="kz",
        ylabel="|G_11|",
        logy=True,
    )

    

    show()


if __name__ == "__main__":
    main()
