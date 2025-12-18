import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

ArrayC = NDArray[np.complex128]

@dataclass
class GluingPrecompute:
    z_abs: NDArray[np.float64]      # (Nz,)
    dz_step: float
    inv_GL00: ArrayC                # (n,n)
    inv_GR00: ArrayC                # (n,n)
    F_L: ArrayC                     # (Nz,n,n)
    Fbar_L: ArrayC                  # (Nz,n,n)
    F_R: ArrayC                     # (Nz,n,n)
    Fbar_R: ArrayC                  # (Nz,n,n)
    G00: ArrayC                     # (n,n) full interface matrix

def precompute_gluing_from_bulk_kernels(
    delta_z: NDArray[np.float64], # (N,); relative z grid, delta_z = z - z'
    gL_dz: ArrayC,                 # (N,n,n); left-side bulk retarded Green's function G_L(z,z') = g_L(z - z')
    gR_dz: ArrayC,                 # (N,n,n); right-side bulk retarded Green's function G_R(z,z') = g_R(z - z')
    H_int: ArrayC,                # (n,n)
    m_L: float,
    m_R: float,
) -> GluingPrecompute:
    
    N = delta_z.size
    mid = N // 2 # zero point
    dz_step = float(delta_z[1] - delta_z[0])
    z_abs = [] # source for z and z' values
    for i in range(mid):
        z_abs.append(delta_z[(mid // 2) + i])
    z_abs = np.ndarray(z_abs, dtype=np.complex128)
    # basic checks
    assert z_abs[mid // 2] == 0.0
    assert z_abs.size == mid
    assert np.allclose(dz_step, float(z_abs[-1] - z_abs[-2])) # uniform spacing

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
        dz_step=dz_step,
        inv_GL00=inv_GL00,
        inv_GR00=inv_GR00,
        F_L=F_L,
        Fbar_L=Fbar_L,
        F_R=F_R,
        Fbar_R=Fbar_R,
        G00=G00,
    )

def glued_retarded_greens_on_grid(
    pre: GluingPrecompute,
    gL_dz: ArrayC,                 # (N,n,n); left-side bulk retarded Green's function G_L(z,z') = g_L(z - z')
    gR_dz: ArrayC,                 # (N,n,n); right-side bulk retarded Green's function G_R(z,z') = g_R(z - z')
) -> ArrayC:
    """
    This is the implementation of the glued retarded Green's function, i.e. a real space retarded Green's
    function G^R(z,z') constructed from two bulk half-space Green's functions G_L and G_R
    and the interface Hamiltonian H_int. That is why it is also called the full interfacial Green's function 
    for a bilayer system with a plane interface.

    Parameters
    ----------
    pre : GluingPrecompute
        Precomputed quantities for gluing.
    gL_dz : ArrayC
        Left-side bulk retarded Green's function on delta_z grid, shape (N,n,n).
    gR_dz : ArrayC
        Right-side bulk retarded Green's function on delta_z grid, shape (N,n,n).
    
    Returns
    -------
    out : ArrayC
        Glued retarded Green's function G^R(z,z') on the (z,z') grid, shape (N,N,n,n).
    """
    z = pre.z_abs
    Nz = z.size
    midz = Nz // 2
    n = pre.G00.shape[0]

    # choose side-specific F arrays per z and z'
    # boolean masks (N,) for z<0 and z>0
    left_mask  = z < 0 
    right_mask = z > 0

    F = np.empty((Nz, n, n), dtype=np.complex128)
    Fbar = np.empty((Nz, n, n), dtype=np.complex128)

    # effective F and Fbar matrix arrays on full z grid using theta function like masks
    F[left_mask]  = pre.F_L[left_mask]
    F[right_mask] = pre.F_R[right_mask]
    F[~(left_mask | right_mask)] = -np.eye(n, dtype=np.complex128)  # z==0
    assert np.allclose(F[midz], -np.eye(n, dtype=np.complex128))

    Fbar[left_mask]  = pre.Fbar_L[left_mask]
    Fbar[right_mask] = pre.Fbar_R[right_mask]
    Fbar[~(left_mask | right_mask)] = -np.eye(n, dtype=np.complex128)  # z'==0
    assert np.allclose(Fbar[midz], -np.eye(n, dtype=np.complex128))

    # core glued term: F(z) G00 Fbar(z')
    # broadcast with new dummy axes + batched matmul:
    # core is (Nz,Nz,n,n) and the first two axes correspond to (z,z')
    core = (F[:, None, :, :] @ pre.G00[None, None, :, :] @ Fbar[None, :, :, :])  # (Nz,Nz,n,n)

    # add half-space barred contributions on same side
    # G_bar = G(z,z') - G(z,0) G(0,0)^{-1} G(0,z')
    # but in bulk: G(z,z') = g(z-z') => use g_dz with index differences i-j

    # Build index difference table once:
    idx = np.arange(Nz) # [0..Nz-1], shape (Nz,)
    # Subtracting broadcasts to shape (Nz,Nz):
    diff_idx = (idx[:, None] - idx[None, :]) + Nz  # values in [-(Nz-1)..+(Nz-1)] + (Nz-1) -> [0..2Nz-2]

    # bulk same-side terms:
    GL = gL_dz[diff_idx]  # (Nz,Nz,n,n)
    GR = gR_dz[diff_idx]  # (Nz,Nz,n,n)

    # construct barred:
    # G(z,0)=g(z), G(0,z')=g(-z')
    gL_z0   = gL_dz[:, None, :, :]        # (N,1,n,n)
    gL_0zp  = gL_dz[::-1][None, :, :, :]  # (1,N,n,n)
    GL_bar  = GL - (gL_z0 @ pre.inv_GL00 @ gL_0zp)

    gR_z0   = gR_dz[:, None, :, :]
    gR_0zp  = gR_dz[::-1][None, :, :, :]
    GR_bar  = GR - (gR_z0 @ pre.inv_GR00 @ gR_0zp)

    # apply domain masking: only add GL_bar when both indices are left; GR_bar when both are right
    same_left  = left_mask[:, None] & left_mask[None, :]
    same_right = right_mask[:, None] & right_mask[None, :]

    out = core.copy()
    out[same_left]  += GL_bar[same_left]
    out[same_right] += GR_bar[same_right]

    # ensure (0,0) equals interface G00 if you want that exact identity
    out[midz, midz] = pre.G00
    return out

