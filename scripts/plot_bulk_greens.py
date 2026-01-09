"""
Plot bulk retarded Green's function diagnostics (model-agnostic).

This script is a lightweight development/diagnostics tool for the *bulk* part of the
bilayer Kubo pipeline. It uses a DevPreset from `kubo.presets` plus the model registry
to construct a bulk Hamiltonian H(kx, ky, kz), then:

1) Computes the real-space bulk Green's function G^R(Δz) via an FFT in kz
   (using the preset's FFT box settings).
2) Computes a wide kz diagnostic scan (independent of the FFT box) to visualize
   k-space structure and compare against FFT kz coverage.
3) Produces quick plots:
   - Complex components of selected matrix entries G^R_{ij}(Δz)
   - Frobenius norm profile ||G^R(Δz)||_F (wrap-around sanity check)
   - |G^R_{ij}(kz)| on wide diagnostic kz grid and on the FFT kz grid

Presets & philosophy
--------------------
Presets are defined in `kubo/presets.py` as `DevPreset` objects and are intentionally
pure configuration: grid + physics + model + default bulk side + default plot channels.
The script stays model-agnostic by building the Hamiltonian through the registry:

    H = build_bulk_hamiltonian(preset.model, side=preset.bulk_side)

Typical usage
-------------
Run as a module (recommended from repo root / installed editable package):

    python -m scripts.plot_bulk_greens --preset toy_fft_near_shell_mid

Toy presets example:

    python -m scripts.plot_bulk_greens --preset toy_fft_debug_evanescent
    python -m scripts.plot_bulk_greens --preset toy_fft_near_shell_wrap_safe

Orbitronic presets example (if defined):

    python -m scripts.plot_bulk_greens --preset orbitronic_fft_mid

Useful overrides (without editing presets)
------------------------------------------
Change frequency / momentum point:

    python -m scripts.plot_bulk_greens --preset toy_fft_near_shell_mid --omega 0.2
    python -m scripts.plot_bulk_greens --preset toy_fft_near_shell_mid --kx 0.5 --ky 0.25

Change broadening η (PhysicsConfig.eta):

    python -m scripts.plot_bulk_greens --preset orbitronic_fft_mid --eta 0.05

Choose bulk side (left/right) for plotting:

    python -m scripts.plot_bulk_greens --preset orbitronic_fft_mid --side right

Choose which matrix entries to plot (two channels):

    python -m scripts.plot_bulk_greens --preset orbitronic_fft_mid --ij1 0 0 --ij2 2 2

Diagnostic kz scan controls:

    python -m scripts.plot_bulk_greens --preset toy_fft_near_shell_mid --kz-diag-max 10 --nkz-diag 8001

Plot clarity controls (zoom around Δz=0 and/or downsample point density):

    python -m scripts.plot_bulk_greens --preset toy_fft_near_shell_mid --dz-zoom 50 --downsample 2

Notes
-----
- This script is for *bulk-only* diagnostics. Interface/gluing diagnostics belong in
  separate scripts (e.g. later stages of the pipeline).
- The wrap-around / aliasing risk of FFT-based real-space kernels can be quickly
  assessed using the Frobenius profile and the edge-leak ratio.
"""
# region Imports
from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np

from kubo.presets import PRESETS
from kubo.models.registry import build_bulk_hamiltonian
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
    show,
    plot_kz_diagnostic_with_fft_coverage
)
from kubo.diagnostics.kz_coverage import kz_coverage_metrics
# endregion

# region Argument parsing
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--preset",
        type=str,
        default="toy_fft_near_shell_mid",
        choices=sorted(PRESETS.keys()),
        help="Name of a DevPreset from kubo/presets.py",
    )

    # Optional overrides (deviate from preset)
    p.add_argument("--omega", type=float, default=None)
    p.add_argument("--kx", type=float, default=None)
    p.add_argument("--ky", type=float, default=None)
    p.add_argument("--eta", type=float, default=None)

    # Optional overrides for bulk side + channels
    p.add_argument("--side", choices=["left", "right"], default=None)
    p.add_argument("--ij1", nargs=2, type=int, default=None, metavar=("I", "J"))
    p.add_argument("--ij2", nargs=2, type=int, default=None, metavar=("I", "J"))

    # Diagnostic kz scan (independent of FFT box)
    p.add_argument("--kz-diag-max", type=float, default=8.0)
    p.add_argument("--nkz-diag", type=int, default=4001)

    # Quick visual clarity options
    p.add_argument("--dz-zoom", type=float, default=None, help="If set, plot only |Δz|<dz_zoom for component plots.")
    p.add_argument("--downsample", type=int, default=1, help="Downsample factor for component plots (>=1).")

    return p.parse_args()
# endregion

# region Main script
def main() -> None:
    args = parse_args()

    # region Preset and overrides
    preset = PRESETS[args.preset]
    grid = preset.grid
    physics = preset.physics
    model = preset.model

    omega = preset.omega if args.omega is None else float(args.omega)
    kx = preset.kx if args.kx is None else float(args.kx)
    ky = preset.ky if args.ky is None else float(args.ky)

    # side + channels default to preset, but can be overridden by CLI
    side = preset.bulk_side if args.side is None else str(args.side)

    (i1, j1) = preset.plot_channels[0] if args.ij1 is None else (int(args.ij1[0]), int(args.ij1[1]))
    (i2, j2) = preset.plot_channels[1] if args.ij2 is None else (int(args.ij2[0]), int(args.ij2[1]))

    if args.eta is not None:
        # If PhysicsConfig is frozen, reconstruct via dataclasses.asdict
        physics = type(physics)(**{**asdict(physics), "eta": float(args.eta)})

    print(f"[run] preset={preset.name}")
    print(f"[run] model={model.name}, side={side}")
    print("[run] grid   =", grid)
    print("[run] physics=", physics)
    print(f"[run] omega={omega}, kx={kx}, ky={ky}")
    if getattr(preset, "note", ""):
        print(f"[run] note: {preset.note}")
    # endregion

    # region Grids & kernels
    # Derived FFT quantities (based on the preset grid)
    L = 2.0 * grid.z_max
    dz = L / grid.nz
    dk = 2.0 * np.pi / L
    print(f"[run] derived: L={L}, dz={dz}, dk≈{dk}")

    # ---- Model-agnostic bulk Hamiltonian ----
    # uses registry + params
    H = build_bulk_hamiltonian(model, side=side) # type: ignore

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
    amp_kz_diag = np.abs(G_kz_diag[:, i1, j1])
    amp_kz_fft = np.abs(G_kz_fft[:, i1, j1])
    
    # region Diagnostic metrics
    m = kz_coverage_metrics(kz_diag, amp_kz_diag, kz_fft, p=2.0, q_levels=(0.95, 0.99))
    print(
        "[diag] kz coverage: "
        f"mass_in_fft={m['mass_fraction_inside_fft']:.4f}, "
        f"K95={m['K_95']:.3f}, ratio95={m['coverage_ratio_95']:.3f}, "
        f"K99={m['K_99']:.3f}, ratio99={m['coverage_ratio_99']:.3f}, "
        f"peak_kz={m['kz_peak']:.3f}, peak_inside_fft={bool(m['peak_inside_fft'])}"
    )
    # endregion

    # region Plotting
    # ---- Optional zoom/downsample for component plots ----
    ds = max(1, int(args.downsample))
    if args.dz_zoom is None:
        mask = None
        idx = slice(None, None, ds)
    else:
        mask = np.abs(delta_z) < float(args.dz_zoom)
        idx = np.arange(np.count_nonzero(mask))[::ds]

    # zoom selection wrapper
    def _sel(x: np.ndarray) -> np.ndarray: 
        if mask is None:
            return x[::ds]
        return x[mask][idx]

    # (1) G(i1,j1)(Δz) components with optional zoom
    plot_complex_components(
        _sel(delta_z),
        _sel(G_dz[:, i1, j1]),
        title=f"{model.name} ({side}): G^R_{i1}{j1}(Δz) components",
        xlabel="Δz",
        ylabel=f"G_{i1}{j1}",
    )

    # (2) G(i2,j2)(Δz) components with optional zoom
    plot_complex_components(
        _sel(delta_z),
        _sel(G_dz[:, i2, j2]),
        title=f"{model.name} ({side}): G^R_{i2}{j2}(Δz) components",
        xlabel="Δz",
        ylabel=f"G_{i2}{j2}",
    )

    # (3) Frobenius norm decay ||G(Δz)||_F (wrap-around check)
    plot_profile(
        delta_z,
        amp_dz,
        title=f"{model.name} ({side}): ||G^R(ω={omega}; Δz)||  (η={physics.eta})",
        xlabel="Δz",
        ylabel="Frobenius norm",
        logy=True,
    )

    # (4) Wide kz diagnostic and FFT kz diagnostic for (i1,j1)
    plot_kz_diagnostic_with_fft_coverage(
        kz_diag,
        amp_kz_diag,
        kz_fft,
        amp_kz_fft,
        title=f"{model.name} ({side}): |G^R_{i1}{j1}(kz)| with FFT kz coverage overlay",
        xlabel="kz",
        ylabel=f"|G_{i1}{j1}|",
        logy=True,
        show_fft_points=True,
        shade_fft_range=True,
    )

    show()
    # endregion
# endregion

if __name__ == "__main__":
    main()
