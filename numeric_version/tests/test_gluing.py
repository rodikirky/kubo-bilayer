from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pytest

from kubo.gluing import (  # type: ignore
    precompute_gluing_from_bulk_kernels,
    glued_retarded_greens_batched,
    _build_dz_index,
    gather_kernel,
)

from kubo.greens import RealSpaceKernel, KernelDiagnostics  # type: ignore

# -------------------------------
# Helpers / fixtures
# -------------------------------
def make_delta_z(N: int = 33, dz: float = 0.1) -> np.ndarray:
    """Odd-length symmetric grid with delta_z[mid] == 0."""
    assert N % 2 == 1
    mid = N // 2
    delta_z = (np.arange(N) - mid) * dz
    # sanity: symmetry + centered
    assert np.isclose(delta_z[mid], 0.0)
    assert np.allclose(delta_z, -delta_z[::-1], atol=0, rtol=0)
    return delta_z.astype(float)


def make_kernel(
    delta_z: np.ndarray,
    mat_func,
    *,
    n: int = 2,
    leak: float = 0.0,
    edge_action: str = "none",
    edge_warn: float = 1e-6,
    edge_error: float = 1e-3,
) -> RealSpaceKernel:
    """Create a RealSpaceKernel with G_dz[k] = mat_func(delta_z[k])."""
    N = delta_z.size
    mid = N // 2

    G_dz = np.empty((N, n, n), dtype=np.complex128)
    for i, dz in enumerate(delta_z):
        G_dz[i] = np.asarray(mat_func(float(dz)), dtype=np.complex128)

    diag = KernelDiagnostics(
        edge_leak_ratio=float(leak),
        center_index=int(mid),
        edge_warn=float(edge_warn),
        edge_error=float(edge_error),
        edge_action=str(edge_action),
        edge_m=10,
    )

    # kz / G_kz are optional “nice-to-have” in your design; here we fill placeholders.
    kz = np.zeros_like(delta_z, dtype=float)
    G_kz = G_dz.copy()

    return RealSpaceKernel(delta_z=delta_z, kz=kz, G_dz=G_dz, G_kz=G_kz, diag=diag)


# -------------------------------
# region _build_dz_index
# -------------------------------
def test_build_dz_index_exact_mapping_ok_all_true():
    delta_z = make_delta_z(N=33, dz=0.2)
    mid = delta_z.size // 2

    # Pick values near 0 so that all pairwise dz remain within [-zmax, zmax]
    z  = delta_z[mid-4:mid+5]   # 9 points: [-0.8, ..., +0.8]
    zp = delta_z[mid-4:mid+5]   # same range

    idx, ok = _build_dz_index(delta_z, z, zp)

    assert idx.shape == (z.size, zp.size)
    assert ok.shape == (z.size, zp.size)
    assert np.all(ok)

    # verify mapping reproduces dz (for ok entries)
    dz_step = float(delta_z[1] - delta_z[0])
    mid = delta_z.size // 2
    dz_reconstructed = (idx - mid) * dz_step

    assert np.allclose(dz_reconstructed, z[:, None] - zp[None, :], atol=1e-12, rtol=0)

def test_build_dz_index_out_of_range_pairs_exist_for_wide_spread_subsets():
    delta_z = make_delta_z(N=33, dz=0.2)
    z = delta_z[::3]
    zp = delta_z[1::4]

    _, ok = _build_dz_index(delta_z, z, zp)

    assert np.any(~ok)  # must have some out-of-range pairs
    assert np.any(ok)   # but also some in-range pairs

def test_build_dz_index_out_of_range_sets_ok_false():
    delta_z = make_delta_z(N=21, dz=0.1)
    z = np.array([-100.0, 0.0, 100.0], dtype=float)  # definitely off-grid range
    zp = np.array([0.0], dtype=float)

    idx, ok = _build_dz_index(delta_z, z, zp)
    assert ok.shape == (3, 1)
    assert ok[1, 0]  # z=0 should be in-range
    assert not ok[0, 0]
    assert not ok[2, 0]

def test_build_dz_index_raises_if_misaligned_with_step():
    delta_z = make_delta_z(N=21, dz=0.1)
    z = np.array([0.05], dtype=float)  # half-step misalignment
    zp = np.array([0.0], dtype=float)

    with pytest.raises(ValueError, match="aligned"):
        _build_dz_index(delta_z, z, zp, tol=1e-6)

# endregion
# -------------------------------

# -------------------------------
# region gather_kernel
# -------------------------------
def test_gather_kernel_zero_fill_out_of_range():
    delta_z = make_delta_z(N=21, dz=0.1)
    N = delta_z.size
    n = 1

    # simple scalar kernel: g(dz)=1+0j
    g_dz = np.ones((N, n, n), dtype=np.complex128)

    z = np.array([-100.0, 0.0, 100.0], dtype=float)
    zp = np.array([0.0], dtype=float)
    idx, ok = _build_dz_index(delta_z, z, zp)

    out = gather_kernel(g_dz, idx, ok, out_of_range="zero")
    assert out.shape == (3, 1, 1, 1)

    assert out[1, 0, 0, 0] == pytest.approx(1.0 + 0.0j)
    assert out[0, 0, 0, 0] == pytest.approx(0.0 + 0.0j)
    assert out[2, 0, 0, 0] == pytest.approx(0.0 + 0.0j)

def test_gather_kernel_error_out_of_range():
    delta_z = make_delta_z(N=21, dz=0.1)
    N = delta_z.size
    n = 1
    g_dz = np.ones((N, n, n), dtype=np.complex128)

    z = np.array([-100.0, 0.0, 100.0], dtype=float)
    zp = np.array([0.0], dtype=float)
    idx, ok = _build_dz_index(delta_z, z, zp)

    with pytest.raises(ValueError, match="outside the kernel grid range"):
        gather_kernel(g_dz, idx, ok, out_of_range="error")

# endregion
# -------------------------------

# -------------------------------
# region precompute
# -------------------------------
def test_precompute_constant_kernel_gives_expected_G00_and_F_matrices():
    delta_z = make_delta_z(N=33, dz=0.1)
    n = 2

    # constant invertible bulk kernel
    A = np.array([[2.0 + 0.2j, 0.1j], [-0.2j, 1.5 + 0.1j]], dtype=np.complex128)
    assert np.linalg.det(A) != 0

    gL = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")

    H_int = np.array([[0.3, 0.05], [0.05, -0.1]], dtype=np.complex128)
    pre = precompute_gluing_from_bulk_kernels(gL, gR, H_int, m_L=1.0, m_R=2.0)

    mid = delta_z.size // 2
    Ainv = np.linalg.inv(A)

    # inv_GL00 / inv_GR00 should match inverse of g(0)
    assert np.allclose(pre.inv_GL00, Ainv)
    assert np.allclose(pre.inv_GR00, Ainv)

    # For constant kernel: F(z) = -(g(z) @ inv(g(0))) = -I for all z
    I = np.eye(n, dtype=np.complex128)
    assert np.allclose(pre.F_L, -I[None, :, :])
    assert np.allclose(pre.F_R, -I[None, :, :])
    assert np.allclose(pre.Fbar_L, -I[None, :, :])
    assert np.allclose(pre.Fbar_R, -I[None, :, :])

    # For constant kernel: L_L=L_R=0 => G00 = inv(-H_int)
    assert np.allclose(pre.G00, np.linalg.inv(-H_int))
    assert pre.leak == pytest.approx(0.0)
    assert np.allclose(pre.delta_z[mid], 0.0)


def test_precompute_warns_on_leak_when_left_requests_warn():
    delta_z = make_delta_z(N=21, dz=0.1)
    n = 1
    A = np.array([[1.0 + 0.0j]], dtype=np.complex128)

    # leak above warn_thr but below err_thr
    gL = make_kernel(delta_z, lambda dz: A, n=n, leak=1e-5, edge_action="warn", edge_warn=1e-6, edge_error=1e-3)
    gR = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")

    with pytest.warns(RuntimeWarning, match="edge leakage"):
        precompute_gluing_from_bulk_kernels(gL, gR, H_int=np.array([[0.1]], dtype=np.complex128), m_L=1.0, m_R=1.0)


def test_precompute_warns_on_leak_when_right_requests_warn_only():
    delta_z = make_delta_z(N=21, dz=0.1)
    n = 1
    A = np.array([[1.0 + 0.0j]], dtype=np.complex128)

    gL = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(delta_z, lambda dz: A, n=n, leak=1e-5, edge_action="warn", edge_warn=1e-6, edge_error=1e-3)

    with pytest.warns(RuntimeWarning, match="edge leakage"):
        precompute_gluing_from_bulk_kernels(gL, gR, H_int=np.array([[0.1]], dtype=np.complex128), m_L=1.0, m_R=1.0)


def test_precompute_raises_on_leak_error_threshold():
    delta_z = make_delta_z(N=21, dz=0.1)
    n = 1
    A = np.array([[1.0 + 0.0j]], dtype=np.complex128)

    gL = make_kernel(delta_z, lambda dz: A, n=n, leak=1e-2, edge_action="warn", edge_warn=1e-6, edge_error=1e-3)
    gR = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")

    with pytest.raises(ValueError, match="edge leakage"):
        precompute_gluing_from_bulk_kernels(gL, gR, H_int=np.array([[0.1]], dtype=np.complex128), m_L=1.0, m_R=1.0)

def test_precompute_raises_if_GL00_singular():
    delta_z = make_delta_z(N=33, dz=0.1)
    n = 2
    # singular matrix
    A = np.array([[1.0 + 0j, 2.0 + 0j],
                  [2.0 + 0j, 4.0 + 0j]], dtype=np.complex128)

    gL = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(delta_z, lambda dz: np.eye(n, dtype=np.complex128), n=n, leak=0.0, edge_action="none")

    with pytest.raises(np.linalg.LinAlgError):
        precompute_gluing_from_bulk_kernels(gL, gR, H_int=np.eye(n, dtype=np.complex128), m_L=1.0, m_R=1.0)


def test_precompute_raises_if_GR00_singular():
    delta_z = make_delta_z(N=33, dz=0.1)
    n = 2
    A = np.array([[1.0 + 0j, 2.0 + 0j],
                  [2.0 + 0j, 4.0 + 0j]], dtype=np.complex128)

    gL = make_kernel(delta_z, lambda dz: np.eye(n, dtype=np.complex128), n=n, leak=0.0, edge_action="none")
    gR = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")

    with pytest.raises(np.linalg.LinAlgError):
        precompute_gluing_from_bulk_kernels(gL, gR, H_int=np.eye(n, dtype=np.complex128), m_L=1.0, m_R=1.0)


def test_precompute_raises_if_interface_matrix_singular_constant_kernel():
    """
    With constant bulk kernels: L_R - L_L == 0 and G00 = inv(-H_int).
    So choosing singular H_int should raise.
    """
    delta_z = make_delta_z(N=33, dz=0.1)
    n = 2

    A = np.array([[2.0 + 0.2j, 0.1j],
                  [-0.2j, 1.5 + 0.1j]], dtype=np.complex128)  # invertible
    H_singular = np.array([[1.0, 2.0],
                           [2.0, 4.0]], dtype=np.complex128)  # singular

    gL = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")

    with pytest.raises(np.linalg.LinAlgError):
        precompute_gluing_from_bulk_kernels(gL, gR, H_int=H_singular, m_L=1.0, m_R=1.0)

def test_precompute_raises_on_mismatched_delta_z_grids():
    delta_z = make_delta_z(N=33, dz=0.1)
    delta_z_bad = delta_z.copy()
    delta_z_bad[0] += 0.01  # mismatch (and breaks symmetry)

    n = 1
    A = np.array([[1.0 + 0.1j]], dtype=np.complex128)

    gL = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(delta_z_bad, lambda dz: A, n=n, leak=0.0, edge_action="none")

    with pytest.raises(ValueError, match="delta_z.*differ"):
        precompute_gluing_from_bulk_kernels(gL, gR, H_int=np.array([[0.2]], dtype=np.complex128), m_L=1.0, m_R=1.0)


def test_precompute_raises_on_nonuniform_spacing():
    delta_z = make_delta_z(N=33, dz=0.1)
    mid = delta_z.size // 2
    dz_bad = delta_z.copy()
    k = 3
    # keep antisymmetry but break uniform spacing
    dz_bad[mid + k] += 0.01
    dz_bad[mid - k] -= 0.01

    n = 1
    A = np.array([[1.0 + 0.1j]], dtype=np.complex128)
    gL = make_kernel(dz_bad, lambda dz: A, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(dz_bad, lambda dz: A, n=n, leak=0.0, edge_action="none")

    with pytest.raises(ValueError, match="uniform"):
        precompute_gluing_from_bulk_kernels(gL, gR, H_int=np.array([[0.2]], dtype=np.complex128), m_L=1.0, m_R=1.0)


def test_precompute_raises_on_nonsymmetric_delta_z():
    delta_z = make_delta_z(N=33, dz=0.1)
    dz_bad = delta_z + 0.05  # shift by half a step: still uniform, but center != 0

    # sanity: uniform spacing preserved
    assert np.allclose(np.diff(dz_bad), np.diff(delta_z)[0])

    n = 1
    A = np.array([[1.0 + 0.1j]], dtype=np.complex128)
    gL = make_kernel(dz_bad, lambda dz: A, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(dz_bad, lambda dz: A, n=n, leak=0.0, edge_action="none")

    with pytest.raises(ValueError, match="centered|delta_z\\[mid\\]"):
        precompute_gluing_from_bulk_kernels(
            gL, gR,
            H_int=np.array([[0.2]], dtype=np.complex128),
            m_L=1.0, m_R=1.0
        )


def test_precompute_raises_if_center_is_not_zero():
    delta_z = make_delta_z(N=33, dz=0.1)
    mid = delta_z.size // 2
    dz_bad = delta_z.copy()
    dz_bad[mid] = 0.05  # violates centering

    n = 1
    A = np.array([[1.0 + 0.1j]], dtype=np.complex128)
    gL = make_kernel(dz_bad, lambda dz: A, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(dz_bad, lambda dz: A, n=n, leak=0.0, edge_action="none")

    with pytest.raises(ValueError, match="centered|delta_z\\[mid\\]"):
        precompute_gluing_from_bulk_kernels(gL, gR, H_int=np.array([[0.2]], dtype=np.complex128), m_L=1.0, m_R=1.0)

# endregion
# -------------------------------

# -------------------------------
# region batched gluing
# -------------------------------
def test_glued_constant_kernel_returns_G00_everywhere_default_grid():
    delta_z = make_delta_z(N=33, dz=0.1)
    n = 2
    A = np.array([[2.0 + 0.2j, 0.1j], [-0.2j, 1.5 + 0.1j]], dtype=np.complex128)
    gL = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")
    H_int = np.array([[0.3, 0.05], [0.05, -0.1]], dtype=np.complex128)

    pre = precompute_gluing_from_bulk_kernels(gL, gR, H_int, m_L=1.0, m_R=1.0)

    z, zp, G = glued_retarded_greens_batched(pre, z=None, out_of_range="zero")
    assert z.shape == (delta_z.size,)
    assert zp.shape == (delta_z.size,)
    assert G.shape == (delta_z.size, delta_z.size, n, n)

    # For constant kernel, output should be identically G00 everywhere
    target = pre.G00
    assert np.allclose(G, target[None, None, :, :])


def test_glued_custom_z_misaligned_raises():
    delta_z = make_delta_z(N=33, dz=0.1)
    n = 1
    A = np.array([[1.0 + 0.1j]], dtype=np.complex128)
    gL = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")
    pre = precompute_gluing_from_bulk_kernels(gL, gR, H_int=np.array([[0.2]], dtype=np.complex128), m_L=1.0, m_R=1.0)

    z_bad = pre.z_abs + 0.5 * pre.dz_step  # half-step -> misaligned
    with pytest.warns(UserWarning, match="Custom"):
        with pytest.raises(ValueError, match="aligned"):
            glued_retarded_greens_batched(pre, z=z_bad, out_of_range="zero")


def test_glued_cross_side_equals_core_only_for_nonconstant_kernel():
    """
    For a non-constant kernel, GL_bar / GR_bar should only contribute on same-side blocks.
    We test: cross-side blocks match 'core' (up to tolerance), while at least one same-side
    entry differs from core.
    """
    delta_z = make_delta_z(N=41, dz=0.1)
    n = 1

    def g_of_dz(dz: float):
        # smooth, invertible at dz=0; varying with dz so barred term can be nonzero
        val = 1.0 / (1.0 + dz * dz) + 0.05j
        return np.array([[val]], dtype=np.complex128)

    gL = make_kernel(delta_z, g_of_dz, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(delta_z, g_of_dz, n=n, leak=0.0, edge_action="none")
    pre = precompute_gluing_from_bulk_kernels(gL, gR, H_int=np.array([[0.3]], dtype=np.complex128), m_L=1.0, m_R=1.0)

    z, zp, G = glued_retarded_greens_batched(pre, z=None, out_of_range="zero")

    # reconstruct core exactly as implementation does (for default grid)
    mid = delta_z.size // 2
    dz_step = pre.dz_step

    iz = np.rint(z / dz_step).astype(int) + mid
    izp = np.rint(zp / dz_step).astype(int) + mid

    F = np.full((z.size, 1, 1), -1.0 + 0.0j, dtype=np.complex128)
    Fbar = np.full((zp.size, 1, 1), -1.0 + 0.0j, dtype=np.complex128)

    left_z = z < 0
    right_z = z > 0
    left_zp = zp < 0
    right_zp = zp > 0

    ok_z = (iz >= 0) & (iz < delta_z.size)
    ok_zp = (izp >= 0) & (izp < delta_z.size)

    sel = ok_z & left_z
    F[sel] = pre.F_L[iz[sel]]
    sel = ok_z & right_z
    F[sel] = pre.F_R[iz[sel]]

    sel = ok_zp & left_zp
    Fbar[sel] = pre.Fbar_L[izp[sel]]
    sel = ok_zp & right_zp
    Fbar[sel] = pre.Fbar_R[izp[sel]]

    core = F[:, None, :, :] @ pre.G00[None, None, :, :] @ Fbar[None, :, :, :]

    # cross-side mask: z<0, zp>0 OR z>0, zp<0
    cross = (left_z[:, None] & right_zp[None, :]) | (right_z[:, None] & left_zp[None, :])
    same_left = left_z[:, None] & left_zp[None, :]
    same_right = right_z[:, None] & right_zp[None, :]

    assert np.allclose(G[cross], core[cross], atol=1e-12, rtol=0)

    # At least one same-side element should differ from core when kernel is non-constant
    diff_left = np.max(np.abs(G[same_left] - core[same_left])) if np.any(same_left) else 0.0
    diff_right = np.max(np.abs(G[same_right] - core[same_right])) if np.any(same_right) else 0.0
    assert (diff_left > 1e-10) or (diff_right > 1e-10)

def test_glued_custom_z_out_of_range_raises_valueerror():
    delta_z = make_delta_z(N=33, dz=0.1)
    n = 1
    A = np.array([[1.0 + 0.1j]], dtype=np.complex128)

    gL = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")
    pre = precompute_gluing_from_bulk_kernels(
        gL, gR, H_int=np.array([[0.2]], dtype=np.complex128), m_L=1.0, m_R=1.0
    )

    zmax = float(np.max(np.abs(pre.z_abs)))
    z_bad = np.array([0.0, zmax + pre.dz_step], dtype=float)  # aligned but out-of-range

    with pytest.warns(UserWarning, match="Custom z-grid"):
        with pytest.raises(ValueError, match="outside the kernel grid range"):
            glued_retarded_greens_batched(pre, z=z_bad, out_of_range="error")

def test_glued_out_of_range_error_raises_when_any_dz_pair_outside_grid():
    delta_z = make_delta_z(N=33, dz=0.1)
    n = 1
    A = np.array([[1.0 + 0.1j]], dtype=np.complex128)

    gL = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(delta_z, lambda dz: A, n=n, leak=0.0, edge_action="none")
    pre = precompute_gluing_from_bulk_kernels(
        gL, gR, H_int=np.array([[0.2]], dtype=np.complex128), m_L=1.0, m_R=1.0
    )

    # default z=z' uses full span, so some pairs have dz outside [-zmax,+zmax]
    with pytest.raises(ValueError, match="outside the kernel grid range"):
        glued_retarded_greens_batched(pre, z=None, out_of_range="error")

def test_boundary_z0_and_zp0_are_core_only_for_nonconstant_kernel():
    delta_z = make_delta_z(N=41, dz=0.1)
    n = 1

    def g_of_dz(dz: float):
        return np.array([[1.0 / (1.0 + dz * dz) + 0.05j]], dtype=np.complex128)

    gL = make_kernel(delta_z, g_of_dz, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(delta_z, g_of_dz, n=n, leak=0.0, edge_action="none")
    pre = precompute_gluing_from_bulk_kernels(
        gL, gR, H_int=np.array([[0.3]], dtype=np.complex128), m_L=1.0, m_R=1.0
    )

    dz = pre.dz_step
    z_custom = np.array([-dz, 0.0, dz], dtype=float)

    with pytest.warns(UserWarning, match="Custom z-grid"):
        z, zp, G = glued_retarded_greens_batched(pre, z=z_custom, out_of_range="zero")

    # Rebuild core exactly the same way as glued_retarded_greens_batched does
    mid = pre.delta_z.size // 2
    iz = np.rint(z / dz).astype(int) + mid
    izp = np.rint(zp / dz).astype(int) + mid

    F = np.empty((z.size, n, n), dtype=np.complex128)
    Fbar = np.empty((zp.size, n, n), dtype=np.complex128)

    F[:] = -np.eye(n, dtype=np.complex128)
    Fbar[:] = -np.eye(n, dtype=np.complex128)

    ok_z = (iz >= 0) & (iz < pre.delta_z.size)
    ok_zp = (izp >= 0) & (izp < pre.delta_z.size)

    left_z = z < 0
    right_z = z > 0
    left_zp = zp < 0
    right_zp = zp > 0

    sel = ok_z & left_z
    F[sel] = pre.F_L[iz[sel]]
    sel = ok_z & right_z
    F[sel] = pre.F_R[iz[sel]]

    sel = ok_zp & left_zp
    Fbar[sel] = pre.Fbar_L[izp[sel]]
    sel = ok_zp & right_zp
    Fbar[sel] = pre.Fbar_R[izp[sel]]

    core = F[:, None, :, :] @ pre.G00[None, None, :, :] @ Fbar[None, :, :, :]

    # z==0 row: must be core only
    row0 = np.where(np.isclose(z, 0.0))[0][0]
    assert np.allclose(G[row0, :, :, :], core[row0, :, :, :], atol=1e-12, rtol=0)

    # z'==0 column: must be core only
    col0 = np.where(np.isclose(zp, 0.0))[0][0]
    assert np.allclose(G[:, col0, :, :], core[:, col0, :, :], atol=1e-12, rtol=0)

def test_glued_out_of_range_zero_core_vanishes_but_barred_survives_when_dz_in_range():
    delta_z = make_delta_z(N=33, dz=0.1)
    n = 1

    # Non-constant kernel so barred term is not identically zero
    def g_of_dz(dz: float):
        return np.array([[1.0 / (1.0 + dz * dz) + 0.05j]], dtype=np.complex128)

    gL = make_kernel(delta_z, g_of_dz, n=n, leak=0.0, edge_action="none")
    gR = make_kernel(delta_z, g_of_dz, n=n, leak=0.0, edge_action="none")
    pre = precompute_gluing_from_bulk_kernels(
        gL, gR, H_int=np.array([[0.2]], dtype=np.complex128), m_L=1.0, m_R=1.0
    )

    dz_step = float(pre.dz_step)
    dz_max = float(np.max(np.abs(pre.delta_z)))  # support of kernel in Δz

    # Make z_out aligned, absolute out-of-range, and large enough that some (z_out - zp) exceed dz_max
    z_out = dz_max + 2.0 * dz_step
    z_custom = np.array([0.0, z_out], dtype=float)

    with pytest.warns(UserWarning, match="Custom z-grid"):
        z, zp, G = glued_retarded_greens_batched(pre, z=z_custom, out_of_range="zero")

    assert G.shape == (z_custom.size, pre.delta_z.size, n, n)

    # Identify out-of-range row using same index logic
    mid = pre.delta_z.size // 2
    iz = np.rint(z / dz_step).astype(int) + mid
    ok_z = (iz >= 0) & (iz < pre.delta_z.size)

    out_rows = np.where(~ok_z)[0]
    assert out_rows.size == 1
    r = int(out_rows[0])

    left_cols = zp < 0.0
    right_cols = zp > 0.0

    # 1) Cross-side block: barred is not added; Δz are huge so gathered kernels are zero; core suppressed -> zero
    assert np.allclose(G[r, left_cols, :, :], 0.0, atol=0.0, rtol=0.0)

    # 2) Same-side right block: barred survives where Δz in-range; must be zero where Δz out-of-range
    dz_vec = z[r] - zp  # (N,)

    in_range = right_cols & (np.abs(dz_vec) <= dz_max + 1e-12)
    out_range = right_cols & (np.abs(dz_vec) > dz_max + 1e-12)

    assert np.any(in_range)
    assert np.any(out_range)

    # out-of-range Δz must be zero-filled
    assert np.allclose(G[r, out_range, :, :], 0.0, atol=0.0, rtol=0.0)

    # in-range Δz must contain at least one nonzero entry (barred contribution survives)
    assert np.max(np.abs(G[r, in_range, 0, 0])) > 1e-10
# endregion
# -------------------------------