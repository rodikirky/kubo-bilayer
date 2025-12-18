from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

from kubo.models.toy import ToyBulk, ToyBulkParams
from kubo.presets import PRESETS
from kubo.grids import build_kz_grid_diagnostic
from kubo.greens import (
    realspace_greens_retarded_with_kz,
    kspace_greens_retarded_on_kz_grid,
)
from kubo.plotting import (
    profile_amplitude_over_first_axis,
    edge_leak_ratio,
    plot_profile,
    plot_complex_components,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--preset",
        type=str,
        default="toy_fft_near_shell_mid",
        choices=sorted(PRESETS.keys()),
        help="Name of a DevPreset from kubo/presets.py",
    )

    # Optional overrides (if you want to deviate from the preset)
    p.add_argument("--omega", type=float, default=None)
    p.add_argument("--kx", type=float, default=None)
    p.add_argument("--ky", type=float, default=None)
    p.add_argument("--eta", type=float, default=None)

    # Diagnostic kz scan (independent of FFT box)
    p.add_argument("--kz-diag-max", type=float, default=8.0)
    p.add_argument("--nkz-diag", type=int, default=4001)

    # Quick visual clarity options
    p.add_argument("--dz-zoom", type=float, default=None, help="If set, plot only |Δz|<dz_zoom for component plots.")
    p.add_argument("--downsample", type=int, default=1, help="Downsample factor for component plots (>=1).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    preset = PRESETS[args.preset]
    grid = preset.grid
    physics = preset.physics

    omega = preset.omega if args.omega is None else float(args.omega)
    kx = preset.kx if args.kx is None else float(args.kx)
    ky = preset.ky if args.ky is None else float(args.ky)

    if args.eta is not None:
        # If PhysicsConfig is frozen in your code, replace this with "make a new PhysicsConfig".
        physics = type(physics)(**{**asdict(physics), "eta": float(args.eta)})

    print(f"[run] preset={preset.name}")
    print("[run] grid   =", grid)
    print("[run] physics=", physics)
    print(f"[run] omega={omega}, kx={kx}, ky={ky}")

    # Derived FFT quantities (based on the preset grid)
    L = 2.0 * grid.z_max
    dz = L / grid.nz
    dk = 2.0 * np.pi / L
    print(f"[run] derived: L={L}, dz={dz}, dk≈{dk}")

    # Toy model (defaults)
    toy = ToyBulk.from_params(ToyBulkParams())

    def H(kx_: float, ky_: float, kz_: float) -> np.ndarray:
        return toy.hamiltonian((kx_, ky_, kz_))

    # ---- FFT-based real-space kernel + its FFT kz grid ----
    delta_z, kz_fft, G_dz, G_kz_fft = realspace_greens_retarded_with_kz(
        omega=omega,
        kx=kx,
        ky=ky,
        H=H,
        physics=physics,
        grid=grid,
    )

    mid = grid.nz // 2
    amp_dz = profile_amplitude_over_first_axis(G_dz, mode="fro")
    leak = edge_leak_ratio(amp_dz, m=10, center_index=mid)

    print(f"[run] kz_fft range: [{kz_fft.min():.3f}, {kz_fft.max():.3f}]")
    print(f"[diag] edge_leak_ratio (m=10): {leak:.3e}  (smaller => safer vs wrap-around)")

    # ---- Wide diagnostic kz scan (independent of FFT box) ----
    kz_diag = build_kz_grid_diagnostic(grid, nkz=args.nkz_diag, kz_max=args.kz_diag_max)
    G_kz_diag = kspace_greens_retarded_on_kz_grid(
        omega=omega,
        kx=kx,
        ky=ky,
        kz=kz_diag,
        H=H,
        physics=physics,
    )
    print(f"[run] kz_diag range: [{kz_diag.min():.3f}, {kz_diag.max():.3f}] (nkz={kz_diag.size})")

    # ---- Optional zoom/downsample for component plots ----
    ds = max(1, int(args.downsample))
    if args.dz_zoom is None:
        idx = slice(None, None, ds)
        mask = False
    else:
        mask = np.abs(delta_z) < float(args.dz_zoom)
        dz_sel = delta_z[mask]
        idx = np.arange(dz_sel.size)[::ds]  # indices in the masked array

    # Helper to apply selection
    def sel(x: np.ndarray) -> np.ndarray:
        if args.dz_zoom is None:
            return x[::ds]
        else:
            return x[mask][idx]

    # ---- Plots ----

    # (1) G00(Δz) components
    plot_complex_components(
        sel(delta_z),
        sel(G_dz[:, 0, 0]),
        title="Toy: G^R_00(Δz) components",
        xlabel="Δz",
        ylabel="G_00",
    )

    # (2) G11(Δz) components (this is the near/on-shell channel at ω=0 for the toy)
    plot_complex_components(
        sel(delta_z),
        sel(G_dz[:, 1, 1]),
        title="Toy: G^R_11(Δz) components",
        xlabel="Δz",
        ylabel="G_11",
    )

    # (3) Frobenius norm decay ||G(Δz)||_F (wrap-around check)
    plot_profile(
        delta_z,
        amp_dz,
        title=f"Toy: ||G^R(ω={omega}; Δz)||  (η={physics.eta})",
        xlabel="Δz",
        ylabel="Frobenius norm",
        logy=True,
    )

    # (4) Wide kz diagnostic: |G11(kz)| (this should show peaks near ±sqrt(2) for ω=0, Δ=1, m=1)
    amp_kz_diag_11 = np.abs(G_kz_diag[:, 1, 1])
    plot_profile(
        kz_diag,
        amp_kz_diag_11,
        title="Toy: |G^R_11(kz)| vs kz (wide diagnostic kz grid)",
        xlabel="kz",
        ylabel="|G_11|",
        logy=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
