'''
Plotting script to visualize the fully glued interfacial Green's function in real space,
using FFT-based kernels, which should align on both bulk sides.

This script is a diagnostic plotting utility analogous to
`scripts.plot_bulk_greens`, but for the full bilayer/interface system
constructed via a real-space gluing formula.

What it computes
----------------
Given a gluing preset (see `kubo/presets.py`, `GLUING_PRESETS`) and input parameters, the script:

1) Builds model-specific gluing ingredients using the registry:
   - left/right bulk Hamiltonians H_L(kx,ky,kz), H_R(kx,ky,kz)
   - effective masses m_L, m_R used in the gluing boundary condition
   - interface Hamiltonian H_int(kx,ky) treated as a 3D potential V_int(kx,ky,kz)
     (kz is ignored by construction)

2) Computes auxiliary bulk real-space kernels g_L(Δz), g_R(Δz) using an FFT-based
   kz grid (see `kubo.greens.realspace_kernel_retarded_with_meta`).

3) Precomputes gluing objects (interface matrix G00 and kernel factors).

4) Evaluates the glued retarded Green's function:
      G^R_glued(z, z')
   on the absolute z-grid provided by the FFT box.

What it plots
--------------
This script fixes a single point z' on the *left* side of the interface (z'<0)
and plots the glued GF as a function of z:

- Re/Im parts of one or two selected matrix elements:
    G^R_glued(z, z')_{i1,j1} and optionally {i2,j2}
- Frobenius norm over the internal matrix at fixed z':
    ||G^R_glued(z, z')||_F
- "Selected-channel" Frobenius norm over the chosen entries (treated as a 1x2 vector):
    sqrt(|G_{i1,j1}|^2 + |G_{i2,j2}|^2)

Coordinate conventions
----------------------
- z and z' are *absolute* coordinates on the centered FFT box grid.
- Δz = z - z' is computed internally for diagnostics/printing, but the x-axis
  of the current plots is the absolute z (restricted to the right side).
        
Important arguments
-------------------
--preset
    Selects a gluing development preset from GLUING_PRESETS.

--zp
    Desired absolute z' on the left (must be < 0). The script snaps to the nearest
    available grid point with z'<0. If omitted, chooses the left grid point closest to 0.

--z-max
    Optional absolute cutoff for the plotted z values (0<|z|<=z_max).

--ij1 / --ij2
    Matrix indices (i, j) of entries to plot.

--all-z
    If set, plots for all z on both sides of the interface.
    By default, only right side (z>0) is plotted, since gluing is most relevant
    for cross-interface transport.


Common usage
------------
List available options:
    python -m scripts.plot_interfacial_greens -h

Run with a preset:
    python -m scripts.plot_interfacial_greens --preset toy_gluing_standard
    python -m scripts.plot_interfacial_greens --preset orbitronic_gluing_standard

Override evaluation point and channels:
    python -m scripts.plot_interfacial_greens \\
      --preset toy_gluing_standard --omega 0.0 --kx 0.2 --ky 0.0 --eta 0.05 \\
      --ij1 0 0 --ij2 1 1

Fix z' on the left and limit right-side range:
    python -m scripts.plot_interfacial_greens \\
      --preset toy_gluing_standard --zp -10.0 --z-max 60.0 --downsample 2

Fix z' on the left and plot all z:
    python -m scripts.plot_interfacial_greens \\
        --preset orbitronic_gluing_standard --zp -50.0 --all-z


Notes
-----
- This script computes a *retarded* glued GF. If you later want an advanced GF
  for comparison, the typical dev approach is to flip the sign of eta and reuse
  the same retarded construction, then compare via conjugation symmetry.
- For strict reproducibility of interface physics, gluing presets should explicitly
  specify bulk_left_params, bulk_right_params, and interface_params if the registry
  is configured with no-defaults behavior.
'''

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np

from kubo.presets import GLUING_PRESETS
from kubo.models.registry import build_gluing_components
from kubo.greens import (
    realspace_kernel_retarded_with_meta,
    RealSpaceKernel,
)
from kubo.gluing import (
    precompute_gluing_from_bulk_kernels,
    glued_retarded_greens_batched,
)
from kubo.plotting import (
    plot_profile,
    plot_complex_components,
    show,
)

# region Argument parsing
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--preset",
        type=str,
        default="toy_gluing_standard",
        choices=sorted(GLUING_PRESETS.keys()),
        help="Name of a DevGluingPreset from kubo/presets.py",
    )

    # Optional overrides (deviate from preset)
    p.add_argument("--omega", type=float, default=None)
    p.add_argument("--kx", type=float, default=None)
    p.add_argument("--ky", type=float, default=None)
    p.add_argument("--eta", type=float, default=None)

    # Optional overrides for channels
    p.add_argument("--ij1", nargs=2, type=int, default=None, metavar=("I", "J"))
    p.add_argument("--ij2", nargs=2, type=int, default=None, metavar=("I", "J"))

    # Fixed z' on the LEFT side (absolute coordinate)
    p.add_argument(
        "--zp",
        type=float,
        default=None,
        help="Fix z' (absolute) on the left side (z'<0). "
             "If set, snaps to the nearest available grid point with z'<0. "
             "If omitted, uses the grid point closest to 0 from the left.",
    )

    # Choose to include all z on both sides
    p.add_argument(
        "--all-z", 
        action="store_true", # default False
        help="If set, plot for all z on both sides."
             "By default, only right side (z>0) is plotted, "
             "since gluing is most relevant for cross-interface transport."
    )

    # Limit z-range (absolute coordinate, not Δz)
    p.add_argument(
        "--z-max",
        type=float,
        default=None,
        help="If set, only plot for z with |z|<=z_max.",
    )
    p.add_argument(
        "--z-min",
        type=float,
        default=None,
        help="If set, only plot z with z>=z_min.",
    )


    # Visual clarity options
    p.add_argument("--downsample", type=int, default=1, help="Downsample factor for line plots (>=1).")

    return p.parse_args()
# endregion

# region Utility functions
def _choose_left_zp_index(zp_grid: np.ndarray, zp_target: float | None) -> tuple[int, float]:
    """
    Pick an index j into zp_grid (absolute z' grid) such that zp_grid[j] < 0.

    - If zp_target is None: pick the left grid point closest to 0 (largest negative).
    - If zp_target is provided: snap to nearest left grid point.

    Returns (j, zp_value).
    """
    left_idx = np.where(zp_grid < 0.0)[0]
    if left_idx.size == 0:
        raise ValueError("No negative z' points available on this grid (need some z'<0).")

    if zp_target is None:
        j = int(left_idx[-1])  # closest to 0 from the left
        return j, float(zp_grid[j])

    zp_target = float(zp_target)
    if zp_target >= 0.0:
        raise ValueError(f"--zp must be < 0 (left side). Got zp={zp_target}.")

    cand = zp_grid[left_idx]
    j_local = int(np.argmin(np.abs(cand - zp_target)))
    j = int(left_idx[j_local])
    return j, float(zp_grid[j])


def _frobenius_norm_matrix_field(G: np.ndarray) -> np.ndarray:
    """
    G: (..., n, n) complex
    Return: (...) real, Frobenius norm over the last two axes.
    """
    return np.sqrt(np.sum(np.abs(G) ** 2, axis=(-2, -1)))


def _frobenius_norm_selected_channels(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """
    Treat selected channels as a 1x1 or 1x2 complex "matrix" per z and compute its Frobenius norm.
    g1, g2: (Nz,) complex
    """
    if g2 is None:
        return np.abs(g1)
    return np.sqrt(np.abs(g1) ** 2 + np.abs(g2) ** 2)
# endregion

# region Main script
def main() -> None:
    # region Settings
    args = parse_args()

    preset = GLUING_PRESETS[args.preset]
    grid = preset.grid
    physics = preset.physics
    model = preset.model

    omega = preset.omega if args.omega is None else float(args.omega)
    kx = preset.kx if args.kx is None else float(args.kx)
    ky = preset.ky if args.ky is None else float(args.ky)

    (i1, j1) = preset.plot_channels[0] if args.ij1 is None else (int(args.ij1[0]), int(args.ij1[1]))
    (i2, j2) = preset.plot_channels[1] if args.ij2 is None else (int(args.ij2[0]), int(args.ij2[1]))

    if args.eta is not None:
        physics = type(physics)(**{**asdict(physics), "eta": float(args.eta)})

    # Build model-specific components needed for gluing
    H_L, H_R, m_L, m_R, V_int = build_gluing_components(model, no_defaults=preset.no_defaults)
    H_int = V_int(kx, ky, 0.0)  # (n,n) matrix; kz ignored by construction

    print(f"[run] preset={preset.name}")
    print(f"[run] model={model.name}")
    print("[run] grid   =", grid)
    print("[run] physics=", physics)
    print(f"[run] omega={omega}, kx={kx}, ky={ky}")
    print(f"[run] plot channels: ({i1}, {j1}), ({i2}, {j2})")
    print(f"[run] effective masses: m_L={m_L}, m_R={m_R}")
    if getattr(preset, "note", ""):
        print(f"[run] note: {preset.note}")
    # endregion

    # region Grids & kernels

    # -----------------------------------------------------------
    # Bulk computation
    # -----------------------------------------------------------
    def _build_bulk_kernels(
        omega_: float, kx_: float, ky_: float, cfg, physics_
    ) -> tuple[RealSpaceKernel, RealSpaceKernel]:
        gL = realspace_kernel_retarded_with_meta(
            omega=omega_,
            kx=kx_,
            ky=ky_,
            H=H_L,
            physics=physics_,
            grid=cfg,
            carry_k_info=False,
            edge_action="none",
        )
        gR = realspace_kernel_retarded_with_meta(
            omega=omega_,
            kx=kx_,
            ky=ky_,
            H=H_R,
            physics=physics_,
            grid=cfg,
            carry_k_info=False,
            edge_action="none",
        )
        return gL, gR

    gL, gR = _build_bulk_kernels(omega, kx, ky, grid, physics)

    # -----------------------------------------------------------
    # Grids
    # -----------------------------------------------------------
    pre = precompute_gluing_from_bulk_kernels(
        gL=gL,
        gR=gR,
        H_int=H_int,
        m_L=m_L,
        m_R=m_R,
    )

    # z-grid (absolute coordinates, i.e. centered FFT box not Δz)
    z_abs = pre.z_abs # (Nz,) array

    # Build the z-list 
    ## Only right side (z_max > z > 0) by default, unless --all-z or z_min<0 is set
    if args.z_min is not None:
        z_eval = z_abs[z_abs >= float(args.z_min)]
    elif args.all_z: 
        z_eval = z_abs
    else: 
        z_right = z_abs[z_abs > 0.0]
        z_eval = z_right
    if args.z_max is not None:
        z_eval = z_eval[z_eval <= float(args.z_max)]
    if z_eval.size == 0:
        raise ValueError("No z points selected after applying --z-min/--z-max and --all-z/right-side filtering.")


    # -----------------------------------------------------------
    # Gluing
    # -----------------------------------------------------------
    # Evaluate glued GF only for right-side z, but for all z' on the absolute grid
    z_out, zp_out, G_glued = glued_retarded_greens_batched(pre=pre, z=z_eval)
    # Shapes:
    #   z_out: (Nz_right,)
    #   zp_out: (Nz_full,)
    #   G_glued: (Nz_right, Nz_full, n, n)
    if not np.array_equal(z_out, z_eval):
        raise RuntimeError("Internal error: z_out does not match input z_eval.")

    # -----------------------------------------------------------
    # Gather selected channels and prepare for plotting
    # -----------------------------------------------------------
    # Choose fixed z' on the left side and extract the corresponding column
    j_zp, zp_fixed = _choose_left_zp_index(zp_out, args.zp)
    print(f"[run] fixed z' (absolute, left side): z'={zp_fixed} (index {j_zp})")

    # Extract line: G_line(z) = G_glued(z, z'=zp_fixed)
    G_line = G_glued[:, j_zp, :, :]  # (Nz_right, n, n)

    # Basic sanity on channel indices
    n = G_line.shape[-1]
    if not (0 <= i1 < n and 0 <= j1 < n and 0 <= i2 < n and 0 <= j2 < n):
        raise ValueError(f"Channel indices out of range for n={n}: "
                         f"(i1,j1)=({i1},{j1}), (i2,j2)=({i2},{j2})")

    g1 = G_line[:, i1, j1]  # (Nz_right,) complex
    g2 = G_line[:, i2, j2]  # (Nz_right,) complex

    # Δz is NOT the plotted axis (we plot absolute z), but we compute it for clarity/diagnostics.
    delta_z = z_out - zp_fixed
    print(f"[run] z range (right, absolute): [{z_out.min()}, {z_out.max()}], Nz={z_out.size}")
    print(f"[run] Δz range (z - z'): [{delta_z.min()}, {delta_z.max()}]")

    # Norms:
    # 1) Full-matrix Frobenius norm ||G(z,z')||_F over internal (n,n)
    norm_full = _frobenius_norm_matrix_field(G_line)

    # 2) Frobenius norm of the selected 1x2 "channel vector"
    # (for one channel, this reduces to |g1| if the channels match)
    if (i1, j1) == (i2, j2):
        norm_sel = np.abs(g1)
    else:
        norm_sel = _frobenius_norm_selected_channels(g1, g2)

    # Downsample for plotting
    ds = max(int(args.downsample), 1)
    z_plot = z_out[::ds]
    delta_z_plot = delta_z[::ds]
    g1_plot = g1[::ds]
    g2_plot = g2[::ds]
    norm_full_plot = norm_full[::ds]
    norm_sel_plot = norm_sel[::ds]

    # endregion

    # region Plotting
        # --- Channel plots: Re/Im vs absolute z (right side) ---
    plot_complex_components(
        z_plot,
        g1_plot,
        title=f"G^R_glued(z, z'={zp_fixed:.6g}) channel ({i1},{j1})  [x-axis: absolute z on right]",
        xlabel="z (absolute, right side)",
        ylabel="G",
        additional_markers=True,
    )

    if (i1, j1) != (i2, j2):
        plot_complex_components(
            z_plot,
            g2_plot,
            title=f"G^R_glued(z, z'={zp_fixed:.6g}) channel ({i2},{j2})  [x-axis: absolute z on right]",
            xlabel="z (absolute, right side)",
            ylabel="G",
        )

    # --- Norm plots ---
    plot_profile(
        z_plot,
        norm_full_plot,
        logy=True,
        title=f"||G^R_glued(z, z'={zp_fixed:.6g})||_F over internal matrix  [x-axis: absolute z on right]",
        xlabel="z (absolute, right side)",
        ylabel="Frobenius norm",
    )

    plot_profile(
        z_plot,
        norm_sel_plot,
        logy=True,
        title=f"Selected-channel Frobenius norm at z'={zp_fixed:.6g}: "
              f"sqrt(|G[{i1},{j1}]|^2 + |G[{i2},{j2}]|^2)  [x-axis: absolute z on right]",
        xlabel="z (absolute, right side)",
        ylabel="selected-channel norm",
    )

    show()

    # endregion

# endregion

if __name__ == "__main__":
    main()
