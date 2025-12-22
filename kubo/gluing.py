import numpy as np
import warnings
from numpy.typing import NDArray
from dataclasses import dataclass
from .greens import RealSpaceKernel


ArrayC = NDArray[np.complex128]

@dataclass
class GluingPrecompute:
    z_abs: NDArray[np.float64]      # (N,)
    delta_z: NDArray[np.float64]    # (N,)
    dz_step: float
    gL_dz: ArrayC                   # (N,n,n)
    gR_dz: ArrayC                   # (N,n,n)
    inv_GL00: ArrayC                # (n,n)
    inv_GR00: ArrayC                # (n,n)
    F_L: ArrayC                     # (N,n,n)
    Fbar_L: ArrayC                  # (N,n,n)
    F_R: ArrayC                     # (N,n,n)
    Fbar_R: ArrayC                  # (N,n,n)
    G00: ArrayC                     # (n,n) full interface matrix
    leak: float


def precompute_gluing_from_bulk_kernels(
    gL: RealSpaceKernel,          # has .delta_z, .G_dz, .diag.edge_leak_ratio
    gR: RealSpaceKernel,   
    H_int: ArrayC,                # (n,n)
    m_L: float,
    m_R: float,
) -> GluingPrecompute:
    
    delta_z = gL.delta_z  # (N,); relative z grid, delta_z = z - z'
    N = delta_z.size
    mid = N // 2
    dz_step = float(delta_z[1] - delta_z[0])
    # basic checks
    if not np.allclose(gL.delta_z, gR.delta_z):
        raise ValueError("Left/right delta_z grids differ.")
    if not np.isclose(delta_z[mid], 0.0):
        raise ValueError("delta_z grid must be centered so that delta_z[mid] == 0.")
    if not np.allclose(np.diff(delta_z), dz_step):
        raise ValueError("delta_z must be uniformly spaced.")
    if not np.allclose(delta_z, -delta_z[::-1], atol=1e-12, rtol=0):
        raise ValueError("delta_z must be symmetric: delta_z == -delta_z[::-1].")
    if gL.G_dz.shape != gR.G_dz.shape:
        raise ValueError(f"Left and right bulk GF must have same shape; got {gL.G_dz.shape} vs {gR.G_dz.shape}.")


    gL_dz = gL.G_dz       # (N,n,n); left-side bulk retarded Green's function G_L(z,z') = g_L(z - z')
    gR_dz = gR.G_dz
    assert gL_dz.shape == gR_dz.shape, f"Left and right bulk GF must have the same shape. Got {gL_dz.shape} and {gR_dz.shape}."
    
    # leak from grid choice
    leak_L = gL.diag.edge_leak_ratio
    leak_R = gR.diag.edge_leak_ratio
    leak = max(leak_L, leak_R)
    if gL.diag.edge_action in {"warn","error"} or gR.diag.edge_action in {"warn","error"}:
        # configurable thresholds (suggested defaults)
        warn_thr = min(gL.diag.edge_warn, gR.diag.edge_warn)
        err_thr = min(gL.diag.edge_error, gR.diag.edge_error)

        msg = (
            f"FFT kernel edge leakage is {leak:.3e} {leak_L:.3e} {leak_R:.3e}. "
            "You could be truncating non-negligible tails; "
            "increase z_max/nz or eta, or avoid zero-fill or faulty wrap-arounds."
        )
        if leak >= err_thr:
            raise ValueError(msg)
        if leak >= warn_thr:
            if gL.diag.edge_action == "error" or gR.diag.edge_action == "error":
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning)

    # Canonical absolute z grid equals centered delta_z grid
    z_abs = delta_z.copy()

    GL00 = gL_dz[mid]
    GR00 = gR_dz[mid]
    inv_GL00 = np.linalg.inv(GL00)
    inv_GR00 = np.linalg.inv(GR00)

    # F_side(z) = -G(z,0) @ inv(G(0,0)) 
    # G(0,z') = g(0-z') so we can use flipped index for barred partner
    gL_minus = gL_dz[::-1]  # g_L(-z')
    gR_minus = gR_dz[::-1]  # g_R(-z')

    # (N,n,n) F matrices on left and right sides
    F_L    = -(gL_dz      @ inv_GL00)          
    Fbar_L = -(inv_GL00   @ gL_minus)         
    F_R    = -(gR_dz      @ inv_GR00)
    Fbar_R = -(inv_GR00   @ gR_minus)

    # log-derivative matrices from one-sided differences at interface:
    # F_L'(0-) ≈ (F_L(0) - F_L(-dz))/dz, where -dz corresponds to index mid-1
    # F_R'(0+) ≈ (F_R(+dz) - F_R(0))/dz, where +dz corresponds to index mid+1
    F_L_0  = F_L[mid]
    F_L_m1 = F_L[mid - 1]
    F_R_0  = F_R[mid]
    F_R_p1 = F_R[mid + 1]

    # log-derivative matrices without spin-orbit coupling:
    # if spin-orbit coupling is present, additional terms would appear here
    # TODO: add spin-orbit coupling effects if needed
    L_L = ((F_L_0 - F_L_m1) / dz_step) / (2.0 * m_L) 
    L_R = ((F_R_p1 - F_R_0) / dz_step) / (2.0 * m_R) 

    G00 = np.linalg.inv(-H_int + L_R - L_L)

    return GluingPrecompute(
        z_abs=z_abs,
        delta_z=delta_z,
        dz_step=dz_step,
        gL_dz=gL_dz,
        gR_dz=gR_dz,
        inv_GL00=inv_GL00,
        inv_GR00=inv_GR00,
        F_L=F_L,
        Fbar_L=Fbar_L,
        F_R=F_R,
        Fbar_R=Fbar_R,
        G00=G00,
        leak=leak
    )

def _grid_alignment_check(dz_mat, dz_step):
    offset = dz_mat / dz_step
    idx0 = np.rint(offset).astype(int)
    if np.max(np.abs(offset - idx0)) > 1e-6:
        raise ValueError("Δz values are not aligned with dz_step.")

def _build_dz_index(
    delta_z: np.ndarray, # (N,)
    z: np.ndarray,       # (Nz,)
    zp: np.ndarray,      # (Nzp,)
    *,
    tol: float = 1e-6,   # rounding tolerance
):
    """
    Build (i,j) index matrix of (z_i - zp_j)_ij
    """
    N = delta_z.size
    mid = N // 2
    dz_step = float(delta_z[1] - delta_z[0])

    # (Nz, Nzp)
    offset = (z[:, None] - zp[None, :]) / dz_step
    idx0 = np.rint(offset).astype(int)

    if np.max(np.abs(offset - idx0)) > tol:
        raise ValueError("Δz values are not aligned with dz_step.")

    idx = idx0 + mid
    ok = (idx >= 0) & (idx < N)
    return idx, ok

def gather_kernel(
    g_dz: ArrayC,                   # (N,n,n)
    idx: np.ndarray,    # (Nz,Nzp)
    ok: np.ndarray,     # (Nz,Nzp)
    *,
    out_of_range: str = "zero",     # "zero" or "error"
) -> ArrayC:
    """
    Return g(z - z') for all pairs (z_i, zp_j).
    Output shape: (Nz, Nzp, n, n).
    """
    if out_of_range == "error":
        if not np.all(ok):
            raise ValueError("Δz contains values outside the kernel grid range.")
        return g_dz[idx]

    if out_of_range != "zero":
        raise ValueError("out_of_range must be 'zero' or 'error'.")

    # zero-fill out-of-range
    out_g_dzzp = np.zeros(idx.shape + g_dz.shape[1:], dtype=g_dz.dtype)
    out_g_dzzp[ok] = g_dz[idx[ok]]
    # zero everywhere where z-z' is not represented on the FFT grid, i.e. out of resolution bounds
    # the validity of this is heavily informed by the grid design.
    return out_g_dzzp  # (Nz, Nzp, n, n)


def glued_retarded_greens_batched(
    pre: GluingPrecompute,
    z: NDArray[np.float64] | None = None,         # (Nz,)
    *,
    out_of_range: str = "zero",
    core_only: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64], ArrayC]:
    """
    Batched glued retarded Green's function evaluated on (z, zp), where z can be custom choice.
    zp is created along FFT grid shape, z would be too ideally.
    If custom z is required, ensure it is aligned with the delta_z grid from the real space GF.

    Returns (z, zp, G) with G shape (Nz, Nzp, n, n).
    """
    delta_z = pre.delta_z
    gL_dz = pre.gL_dz
    gR_dz = pre.gR_dz
    mid = delta_z.size // 2
    dz_step = float(delta_z[1] - delta_z[0])


    if z is not None:
        warnings.warn("Custom z-grid may not align with zp and delta_z. Be careful to ensure alignment.", UserWarning)
        z = np.asarray(z, dtype=float) # (Nz,)
        zp = np.asarray(pre.z_abs, dtype=float) # (Nzp,)
        dz_mat = z[:, None] - zp[None, :] # matrix of dz_ij = z_i - zp_j, shape (Nz,Nzp)
        _grid_alignment_check(dz_mat, dz_step) # raises, if z grids are not aligned
    else:
        z = np.asarray(pre.z_abs, dtype=float)
        zp = np.asarray(pre.z_abs, dtype=float)
        dz_mat = z[:, None] - zp[None, :] # matrix of dz_ij = z_i - zp_j
    
    idx_zzp, ok_zzp = _build_dz_index(delta_z, z, zp)
    n = pre.G00.shape[0]

    # side masks
    left_z   = z < 0
    right_z  = z > 0
    left_zp  = zp < 0
    right_zp = zp > 0

    # Build F(z) and Fbar(z')
    F = np.empty((z.size, n, n), dtype=np.complex128)
    Fbar = np.empty((zp.size, n, n), dtype=np.complex128)

    # NOTE: pre.F_L etc are tabulated on the SAME delta_z grid, so index by matching z -> delta_z.
    # We assume z and zp lie on that same grid.
    # For now, simplest: map z to indices and gather from pre.F_* arrays.
    iz  = np.rint(z  / dz_step).astype(int) + mid
    izp = np.rint(zp / dz_step).astype(int) + mid
    
    # Fill defaults at z=0 and z'=0
    F[:] = -np.eye(n, dtype=np.complex128)
    Fbar[:] = -np.eye(n, dtype=np.complex128)

    # Valid index masks (in-range only)
    ok_z  = (iz  >= 0) & (iz  < delta_z.size)
    ok_zp = (izp >= 0) & (izp < delta_z.size)
    # Enforce contract for z and z' being representable on the kernel grid
    if np.any(~ok_z):
        if out_of_range == "error":
            raise ValueError("z contains values outside the kernel grid range.")
        elif out_of_range == "zero":
            # Important: prevent nonzero core from default -I
            F[~ok_z] = 0.0
    if np.any(~ok_zp):
        if out_of_range == "error":
            raise ValueError("z' contains values outside the kernel grid range.")
        elif out_of_range == "zero":
            Fbar[~ok_zp] = 0.0

    # Side-dependent selection
    # z<0 -> use left precomputed F_L; z>0 -> use right precomputed F_R
    sel = ok_z & left_z
    F[sel] = pre.F_L[iz[sel]]

    sel = ok_z & right_z
    F[sel] = pre.F_R[iz[sel]]

    sel = ok_zp & left_zp
    Fbar[sel] = pre.Fbar_L[izp[sel]]

    sel = ok_zp & right_zp
    Fbar[sel] = pre.Fbar_R[izp[sel]]

    # Core term: F(z) G00 Fbar(z')
    core = F[:, None, :, :] @ pre.G00[None, None, :, :] @ Fbar[None, :, :, :]  # (Nz, Nzp, n, n)
    
    # Early exit, if half-space contributions are not needed.
    if core_only:
        return z, zp, core
    
    # Same-side barred half-space contributions:
    # GL_bar = gL(z-z') - gL(z-0) inv_GL00 gL(0-z')
    # GR_bar = gR(z-z') - gR(z-0) inv_GR00 gR(0-z')

    GL = gather_kernel(gL_dz, idx_zzp, ok_zzp, out_of_range=out_of_range)
    GR = gather_kernel(gR_dz, idx_zzp, ok_zzp, out_of_range=out_of_range)

    zp0 = np.array([0.0], dtype=float)
    idx_z0, ok_z0 = _build_dz_index(delta_z, z, zp0)
    gL_z0 = gather_kernel(gL_dz, idx_z0, ok_z0, out_of_range=out_of_range)  # (Nz,1,n,n)
    gR_z0 = gather_kernel(gR_dz, idx_z0, ok_z0, out_of_range=out_of_range)
    z0 = np.array([0.0], dtype=float)
    idx_0zp, ok_0zp = _build_dz_index(delta_z, z0, zp)
    gL_0zp = gather_kernel(gL_dz, idx_0zp, ok_0zp, out_of_range=out_of_range)  # (1,Nzp,n,n)
    gR_0zp = gather_kernel(gR_dz, idx_0zp, ok_0zp, out_of_range=out_of_range)

    GL_bar = GL - (gL_z0 @ pre.inv_GL00[None, None, :, :] @ gL_0zp)
    GR_bar = GR - (gR_z0 @ pre.inv_GR00[None, None, :, :] @ gR_0zp)

    # Apply masks: add GL_bar only if z and z' both left; GR_bar only if both right
    same_left  = left_z[:, None]  & left_zp[None, :]
    same_right = right_z[:, None] & right_zp[None, :]

    out = core.copy()
    out[same_left]  += GL_bar[same_left]
    out[same_right] += GR_bar[same_right]

    return z, zp, out
