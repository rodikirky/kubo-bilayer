from __future__ import annotations

import numpy as np
import pytest

from kubo.config import GridConfig, PhysicsConfig
from kubo.greens import realspace_kernel_retarded_with_meta
from kubo.gluing import precompute_gluing_from_bulk_kernels, glued_retarded_greens_batched

# Be resilient to whether toy.py lives in kubo.toy or kubo.models.toy
try:
    from kubo.models.toy import ToyBulk, ToyInterface
except ImportError:  # pragma: no cover
    from kubo.models.toy import ToyBulk, ToyInterface


def _grid_small_gluing() -> GridConfig:
    # Keep it small so tests run fast; must be odd so z=0 is on-grid.
    # nz must be >= 3 because gluing uses mid±1 for one-sided derivatives.
    return GridConfig(nz=101, z_max=20.0, nk_parallel=1, nomega=1)


def _toy_bulk_callable(bulk: ToyBulk):
    # kubo.greens expects H(kx,ky,kz)->matrix, while ToyBulk.hamiltonian expects (kx,ky,kz) as sequence
    return lambda kx, ky, kz: bulk.hamiltonian((kx, ky, kz))


def _build_bulk_kernel(bulk: ToyBulk, omega: float, kx: float, ky: float, cfg: GridConfig, physics: PhysicsConfig):
    # Use edge_action="none" so the diagnostic doesn't warn/fail the test suite.
    return realspace_kernel_retarded_with_meta(
        omega=omega,
        kx=kx,
        ky=ky,
        H=_toy_bulk_callable(bulk),
        physics=physics,
        grid=cfg,
        carry_k_info=True,
        edge_action="none",
    )


def _reconstruct_core(pre, z: np.ndarray, zp: np.ndarray) -> np.ndarray:
    """
    Rebuild core term exactly like glued_retarded_greens_batched does:
        core(z,z') = F(z) @ G00 @ Fbar(z')
    This lets us test the assembly logic on cross-side blocks. :contentReference[oaicite:4]{index=4}
    """
    delta_z = pre.delta_z
    mid = delta_z.size // 2
    dz_step = float(pre.dz_step)
    n = pre.G00.shape[0]

    # map z and z' onto delta_z indices
    iz = np.rint(z / dz_step).astype(int) + mid
    izp = np.rint(zp / dz_step).astype(int) + mid

    ok_z = (iz >= 0) & (iz < delta_z.size)
    ok_zp = (izp >= 0) & (izp < delta_z.size)

    left_z = z < 0
    right_z = z > 0
    left_zp = zp < 0
    right_zp = zp > 0

    F = np.empty((z.size, n, n), dtype=np.complex128)
    Fbar = np.empty((zp.size, n, n), dtype=np.complex128)

    # defaults at z=0 and z'=0
    F[:] = -np.eye(n, dtype=np.complex128)
    Fbar[:] = -np.eye(n, dtype=np.complex128)

    sel = ok_z & left_z
    F[sel] = pre.F_L[iz[sel]]
    sel = ok_z & right_z
    F[sel] = pre.F_R[iz[sel]]

    sel = ok_zp & left_zp
    Fbar[sel] = pre.Fbar_L[izp[sel]]
    sel = ok_zp & right_zp
    Fbar[sel] = pre.Fbar_R[izp[sel]]

    return F[:, None, :, :] @ pre.G00[None, None, :, :] @ Fbar[None, :, :, :]


def test_toy_gluing_pipeline_shapes_and_finiteness():
    cfg = _grid_small_gluing()
    physics = PhysicsConfig(eta=0.05)  # stable “evanescent-ish” broadening regime :contentReference[oaicite:5]{index=5}

    omega, kx, ky = -2.0, 0.0, 0.0

    bulkL = ToyBulk(mass=1.0, gap=1.0)
    bulkR = ToyBulk(mass=1.0, gap=1.0)
    iface = ToyInterface(strength=0.2)

    gL = _build_bulk_kernel(bulkL, omega, kx, ky, cfg, physics)
    gR = _build_bulk_kernel(bulkR, omega, kx, ky, cfg, physics)

    pre = precompute_gluing_from_bulk_kernels(
        gL, gR,
        H_int=iface.hamiltonian(kx, ky),
        m_L=bulkL.mass,
        m_R=bulkR.mass,
    )

    # z=None uses z_abs internally; out_of_range must be "zero" because some dz pairs exceed the FFT window. :contentReference[oaicite:6]{index=6}
    z, zp, G = glued_retarded_greens_batched(pre, z=None, out_of_range="zero")

    assert z.shape == (cfg.nz,)
    assert zp.shape == (cfg.nz,)
    assert G.shape == (cfg.nz, cfg.nz, 2, 2)
    assert np.isfinite(G.real).all()
    assert np.isfinite(G.imag).all()


def test_toy_gluing_cross_side_block_equals_core_only():
    cfg = _grid_small_gluing()
    physics = PhysicsConfig(eta=0.05)
    omega, kx, ky = -2.0, 0.0, 0.0

    bulkL = ToyBulk(mass=1.0, gap=1.0)
    bulkR = ToyBulk(mass=1.0, gap=1.0)
    iface = ToyInterface(strength=0.15)

    gL = _build_bulk_kernel(bulkL, omega, kx, ky, cfg, physics)
    gR = _build_bulk_kernel(bulkR, omega, kx, ky, cfg, physics)

    pre = precompute_gluing_from_bulk_kernels(
        gL, gR,
        H_int=iface.hamiltonian(kx, ky),
        m_L=bulkL.mass,
        m_R=bulkR.mass,
    )

    z, zp, G = glued_retarded_greens_batched(pre, z=None, out_of_range="zero")

    core = _reconstruct_core(pre, z, zp)

    left_z = z < 0
    right_z = z > 0
    left_zp = zp < 0
    right_zp = zp > 0

    cross = (left_z[:, None] & right_zp[None, :]) | (right_z[:, None] & left_zp[None, :])

    # By construction, barred terms are only added on same-side blocks, so cross-side == core. :contentReference[oaicite:7]{index=7}
    assert np.allclose(G[cross], core[cross], atol=1e-10, rtol=0)


def test_toy_gluing_same_side_differs_from_core_somewhere():
    cfg = _grid_small_gluing()
    physics = PhysicsConfig(eta=0.05)
    omega, kx, ky = -2.0, 0.0, 0.0

    bulkL = ToyBulk(mass=1.0, gap=1.0)
    bulkR = ToyBulk(mass=1.0, gap=1.0)
    iface = ToyInterface(strength=0.15)

    gL = _build_bulk_kernel(bulkL, omega, kx, ky, cfg, physics)
    gR = _build_bulk_kernel(bulkR, omega, kx, ky, cfg, physics)

    pre = precompute_gluing_from_bulk_kernels(
        gL, gR,
        H_int=iface.hamiltonian(kx, ky),
        m_L=bulkL.mass,
        m_R=bulkR.mass,
    )

    z, zp, G = glued_retarded_greens_batched(pre, z=None, out_of_range="zero")
    core = _reconstruct_core(pre, z, zp)

    left_z = z < 0
    right_z = z > 0
    left_zp = zp < 0
    right_zp = zp > 0

    same_left = left_z[:, None] & left_zp[None, :]
    same_right = right_z[:, None] & right_zp[None, :]

    # We don’t require it everywhere (tails may be zero-filled), but it must differ somewhere.
    dL = np.max(np.abs(G[same_left] - core[same_left])) if np.any(same_left) else 0.0
    dR = np.max(np.abs(G[same_right] - core[same_right])) if np.any(same_right) else 0.0

    assert (dL > 1e-12) or (dR > 1e-12)


def test_toy_gluing_custom_z_emits_warning_and_keeps_shape():
    cfg = _grid_small_gluing()
    physics = PhysicsConfig(eta=0.05)
    omega, kx, ky = -2.0, 0.0, 0.0

    bulkL = ToyBulk(mass=1.0, gap=1.0)
    bulkR = ToyBulk(mass=1.0, gap=1.0)
    iface = ToyInterface(strength=0.15)

    gL = _build_bulk_kernel(bulkL, omega, kx, ky, cfg, physics)
    gR = _build_bulk_kernel(bulkR, omega, kx, ky, cfg, physics)

    pre = precompute_gluing_from_bulk_kernels(
        gL, gR,
        H_int=iface.hamiltonian(kx, ky),
        m_L=bulkL.mass,
        m_R=bulkR.mass,
    )

    # aligned subset of the canonical grid
    z_custom = pre.z_abs[::5].copy()

    with pytest.warns(UserWarning, match="Custom z-grid may not align"):
        z, zp, G = glued_retarded_greens_batched(pre, z=z_custom, out_of_range="zero")  # :contentReference[oaicite:8]{index=8}

    assert z.shape == z_custom.shape
    assert zp.shape == pre.z_abs.shape
    assert G.shape == (z_custom.size, pre.z_abs.size, 2, 2)

def test_toy_gluing_asymmetric_left_right_blocks_are_distinct_and_finite():
    cfg = _grid_small_gluing()
    physics = PhysicsConfig(eta=0.05)
    omega, kx, ky = -2.0, 0.0, 0.0

    # Asymmetric bulks
    bulkL = ToyBulk(mass=0.8, gap=0.7)
    bulkR = ToyBulk(mass=1.6, gap=1.3)
    iface = ToyInterface(strength=0.12)

    gL = _build_bulk_kernel(bulkL, omega, kx, ky, cfg, physics)
    gR = _build_bulk_kernel(bulkR, omega, kx, ky, cfg, physics)

    pre = precompute_gluing_from_bulk_kernels(
        gL, gR,
        H_int=iface.hamiltonian(kx, ky),
        m_L=bulkL.mass,
        m_R=bulkR.mass,
    )

    z, zp, G = glued_retarded_greens_batched(pre, z=None, out_of_range="zero")

    assert np.isfinite(G.real).all()
    assert np.isfinite(G.imag).all()

    # Pick interior same-side points away from z=0 and away from edges
    mid = z.size // 2
    iL = mid - 3  # z < 0
    iR = mid + 3  # z > 0
    jL = mid - 2  # z' < 0
    jR = mid + 2  # z' > 0

    assert z[iL] < 0 and zp[jL] < 0
    assert z[iR] > 0 and zp[jR] > 0

    G_LL = G[iL, jL]  # (2,2)
    G_RR = G[iR, jR]  # (2,2)

    # With asymmetric bulks, these same-side blocks should differ (not a symmetry case)
    assert np.linalg.norm(G_LL - G_RR) > 1e-10
