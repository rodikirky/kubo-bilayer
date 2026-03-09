# TODO: Altered the tests to reflect the recent changes in gluing.py and use the registry.
# TODO: Add docstrings to the new classes and methods where needed and the whole file.

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from kubo.greens import realspace_kernel_retarded_with_meta
from kubo.gluing import precompute_gluing_from_bulk_kernels, glued_retarded_greens_batched
from kubo.models.orbitronic import OrbitronicBulk, OrbitronicBulkParams


def _orbitronic_callable(bulk: OrbitronicBulk):
    return lambda kx, ky, kz: bulk.hamiltonian((kx, ky, kz))


def _build_bulk_kernel(bulk: OrbitronicBulk, omega: float, kx: float, ky: float, cfg, physics):
    return realspace_kernel_retarded_with_meta(
        omega=omega,
        kx=kx,
        ky=ky,
        H=_orbitronic_callable(bulk),
        physics=physics,
        grid=cfg,
        carry_k_info=False,
        edge_action="none",
    )

def _glued_and_core(pre, *, z=None, out_of_range="zero"):
    z1, zp1, G = glued_retarded_greens_batched(pre, z=z, out_of_range=out_of_range)
    z2, zp2, core = glued_retarded_greens_batched(pre, z=z, out_of_range=out_of_range, core_only=True)
    assert np.allclose(z1, z2)
    assert np.allclose(zp1, zp2)
    return z1, zp1, G, core

def _default_hint() -> np.ndarray:
    # Simple diagonal interface potential (Hermitian, basis-covariant)
    return np.float64(0.2) * np.eye(3, dtype=np.complex128)


def test_orbitronic_gluing_shapes_and_finiteness(
    orbitronic_params_default, physics_default, grid_small_fft, kpar_simple
):
    # Use a slightly larger eta for gluing stability on small grids
    physics = replace(physics_default, eta=0.05) 

    omega = -1.0
    kx, ky = kpar_simple

    bulkL = OrbitronicBulk.from_params(orbitronic_params_default, basis=None)
    bulkR = OrbitronicBulk.from_params(orbitronic_params_default, basis=None)
    H_int = _default_hint()

    gL = _build_bulk_kernel(bulkL, omega, kx, ky, grid_small_fft, physics)
    gR = _build_bulk_kernel(bulkR, omega, kx, ky, grid_small_fft, physics)

    pre = precompute_gluing_from_bulk_kernels(
        gL, gR,
        H_int=H_int,
        m_L=bulkL.mass,
        m_R=bulkR.mass,
    )

    z, zp, G = glued_retarded_greens_batched(pre, z=None, out_of_range="zero")

    assert z.shape == (grid_small_fft.nz,)
    assert zp.shape == (grid_small_fft.nz,)
    assert G.shape == (grid_small_fft.nz, grid_small_fft.nz, 3, 3)
    assert np.isfinite(G.real).all()
    assert np.isfinite(G.imag).all()


def test_orbitronic_gluing_cross_side_block_equals_core_only(
    orbitronic_params_default, physics_default, grid_small_fft, kpar_simple
):
    physics = replace(physics_default, eta=0.05)
    omega = -1.0
    kx, ky = kpar_simple

    bulkL = OrbitronicBulk.from_params(orbitronic_params_default, basis=None)
    bulkR = OrbitronicBulk.from_params(orbitronic_params_default, basis=None)
    H_int = _default_hint()

    gL = _build_bulk_kernel(bulkL, omega, kx, ky, grid_small_fft, physics)
    gR = _build_bulk_kernel(bulkR, omega, kx, ky, grid_small_fft, physics)
    pre = precompute_gluing_from_bulk_kernels(gL, gR, H_int=H_int, m_L=bulkL.mass, m_R=bulkR.mass)

    z, zp, G, core = _glued_and_core(pre, z=None, out_of_range="zero")

    left_z = z < 0
    right_z = z > 0
    left_zp = zp < 0
    right_zp = zp > 0
    cross = (left_z[:, None] & right_zp[None, :]) | (right_z[:, None] & left_zp[None, :])

    assert np.allclose(G[cross], core[cross], atol=1e-10, rtol=0.0)



def test_orbitronic_gluing_same_side_differs_from_core_somewhere(
    orbitronic_params_default, physics_default, grid_small_fft, kpar_simple
):
    physics = replace(physics_default, eta=0.05)
    omega = -1.0
    kx, ky = kpar_simple

    # Slightly perturb right bulk to avoid accidental cancellations
    paramsR = OrbitronicBulkParams(
        mass=orbitronic_params_default.mass * 1.2,
        gamma=orbitronic_params_default.gamma * 0.9,
        J=orbitronic_params_default.J,
        magnetisation=orbitronic_params_default.magnetisation,
    )

    bulkL = OrbitronicBulk.from_params(orbitronic_params_default, basis=None)
    bulkR = OrbitronicBulk.from_params(paramsR, basis=None)
    H_int = _default_hint()

    gL = _build_bulk_kernel(bulkL, omega, kx, ky, grid_small_fft, physics)
    gR = _build_bulk_kernel(bulkR, omega, kx, ky, grid_small_fft, physics)
    pre = precompute_gluing_from_bulk_kernels(gL, gR, H_int=H_int, m_L=bulkL.mass, m_R=bulkR.mass)

    z, zp, G, core = _glued_and_core(pre, z=None, out_of_range="zero")

    left_z = z < 0
    right_z = z > 0
    left_zp = zp < 0
    right_zp = zp > 0

    same_left = left_z[:, None] & left_zp[None, :]
    same_right = right_z[:, None] & right_zp[None, :]

    dL = np.max(np.abs(G[same_left] - core[same_left])) if np.any(same_left) else 0.0
    dR = np.max(np.abs(G[same_right] - core[same_right])) if np.any(same_right) else 0.0

    assert (dL > 1e-12) or (dR > 1e-12)


def test_orbitronic_gluing_asymmetric_left_right_blocks_are_distinct(
    physics_default, grid_small_fft, kpar_simple
):
    physics = replace(physics_default, eta=0.05)
    omega = -1.0
    kx, ky = kpar_simple

    bulkL = OrbitronicBulk.from_params(
        OrbitronicBulkParams(mass=1.0, gamma=0.7, J=0.3, magnetisation=[0.2, -0.1, 0.97]),
        basis=None,
    )
    bulkR = OrbitronicBulk.from_params(
        OrbitronicBulkParams(mass=2.0, gamma=0.5, J=0.4, magnetisation=[0.1, 0.05, 0.99]),
        basis=None,
    )
    H_int = _default_hint()

    gL = _build_bulk_kernel(bulkL, omega, kx, ky, grid_small_fft, physics)
    gR = _build_bulk_kernel(bulkR, omega, kx, ky, grid_small_fft, physics)
    pre = precompute_gluing_from_bulk_kernels(gL, gR, H_int=H_int, m_L=bulkL.mass, m_R=bulkR.mass)

    z, zp, G = glued_retarded_greens_batched(pre, z=None, out_of_range="zero")

    mid = z.size // 2
    iL, jL = mid - 2, mid - 3
    iR, jR = mid + 2, mid + 3

    assert z[iL] < 0 and zp[jL] < 0
    assert z[iR] > 0 and zp[jR] > 0

    G_LL = G[iL, jL]
    G_RR = G[iR, jR]

    assert np.linalg.norm(G_LL - G_RR) > 1e-10

@pytest.mark.parametrize("eta", [0.02, 0.05])
def test_orbitronic_gluing_retarded_advanced_two_point_relation(
    orbitronic_params_default, physics_default, grid_small_fft, kpar_simple, eta
):
    """
    For the full two-point object:
        G^A(z,z') ≈ [G^R(z',z)]†
    We generate G^A by flipping eta -> -eta through the whole pipeline.
    """
    physics_R = replace(physics_default, eta=eta)
    physics_A = replace(physics_default, eta=-eta)
    omega = -1.0
    kx, ky = kpar_simple

    bulkL_R = OrbitronicBulk.from_params(orbitronic_params_default, basis=None)
    bulkR_R = OrbitronicBulk.from_params(orbitronic_params_default, basis=None)

    H_int = _default_hint()

    gL_R = _build_bulk_kernel(bulkL_R, omega, kx, ky, grid_small_fft, physics_R)
    gR_R = _build_bulk_kernel(bulkR_R, omega, kx, ky, grid_small_fft, physics_R)
    pre_R = precompute_gluing_from_bulk_kernels(gL_R, gR_R, H_int=H_int, m_L=bulkL_R.mass, m_R=bulkR_R.mass)
    z, zp, GR = glued_retarded_greens_batched(pre_R, z=None, out_of_range="zero")

    gL_A = _build_bulk_kernel(bulkL_R, omega, kx, ky, grid_small_fft, physics_A)
    gR_A = _build_bulk_kernel(bulkR_R, omega, kx, ky, grid_small_fft, physics_A)
    pre_A = precompute_gluing_from_bulk_kernels(gL_A, gR_A, H_int=H_int, m_L=bulkL_R.mass, m_R=bulkR_R.mass)
    z2, zp2, GA = glued_retarded_greens_batched(pre_A, z=None, out_of_range="zero")

    assert np.allclose(z, z2) and np.allclose(zp, zp2)

    GR_swap_dag = GR.swapaxes(0, 1).conj().transpose(0, 1, 3, 2)
    rel = np.linalg.norm(GA - GR_swap_dag) / np.linalg.norm(GR_swap_dag)

    # Canary threshold: catches gross breaks; tolerates finite-dz discretization effects
    assert rel < 6e-2



def test_orbitronic_gluing_basis_covariance(
    orbitronic_params_default, unitary_U3, physics_default, grid_small_fft, kpar_simple
):
    """
    If orbitronic supports basis U, and H_int transforms as U† H_int U,
    then glued G transforms covariantly: G' = U† G U.
    """
    physics = replace(physics_default, eta=0.05)
    omega = -1.0
    kx, ky = kpar_simple

    U = unitary_U3
    Udag = U.conj().T

    bulkL = OrbitronicBulk.from_params(orbitronic_params_default, basis=None)
    bulkR = OrbitronicBulk.from_params(orbitronic_params_default, basis=None)
    bulkL_rot = OrbitronicBulk.from_params(orbitronic_params_default, basis=U)
    bulkR_rot = OrbitronicBulk.from_params(orbitronic_params_default, basis=U)

    H_int = _default_hint()
    H_int_rot = Udag @ H_int @ U

    gL = _build_bulk_kernel(bulkL, omega, kx, ky, grid_small_fft, physics)
    gR = _build_bulk_kernel(bulkR, omega, kx, ky, grid_small_fft, physics)
    pre = precompute_gluing_from_bulk_kernels(gL, gR, H_int=H_int, m_L=bulkL.mass, m_R=bulkR.mass)
    z, zp, G = glued_retarded_greens_batched(pre, z=None, out_of_range="zero")

    gLr = _build_bulk_kernel(bulkL_rot, omega, kx, ky, grid_small_fft, physics)
    gRr = _build_bulk_kernel(bulkR_rot, omega, kx, ky, grid_small_fft, physics)
    pre_r = precompute_gluing_from_bulk_kernels(gLr, gRr, H_int=H_int_rot, m_L=bulkL_rot.mass, m_R=bulkR_rot.mass)
    z2, zp2, G_rot = glued_retarded_greens_batched(pre_r, z=None, out_of_range="zero")

    assert np.allclose(z, z2) and np.allclose(zp, zp2)

    # Apply U† G U on the last two axes
    G_cov = Udag[None, None, :, :] @ G @ U[None, None, :, :]
    assert np.allclose(G_rot, G_cov, rtol=1e-8, atol=1e-10)


def test_orbitronic_gluing_out_of_range_error_and_zero_fill(
    orbitronic_params_default, physics_default, grid_small_fft, kpar_simple
):
    physics = replace(physics_default, eta=0.05)
    omega = -1.0
    kx, ky = kpar_simple

    bulk = OrbitronicBulk.from_params(orbitronic_params_default, basis=None)
    H_int = _default_hint()

    g = _build_bulk_kernel(bulk, omega, kx, ky, grid_small_fft, physics)
    pre = precompute_gluing_from_bulk_kernels(g, g, H_int=H_int, m_L=bulk.mass, m_R=bulk.mass)

    # aligned but includes one point outside the kernel grid
    z_custom = pre.z_abs.copy()
    z_custom[0] = pre.z_abs[0] - pre.dz_step

    with pytest.warns(UserWarning, match="Custom z-grid may not align"):
        with pytest.raises(ValueError, match="outside the kernel grid range"):
            glued_retarded_greens_batched(pre, z=z_custom, out_of_range="error")


    # your code warns whenever custom z is used; capture it explicitly
    with pytest.warns(UserWarning, match="Custom z-grid may not align"):
        z, zp, G, core = _glued_and_core(pre, z=z_custom, out_of_range="zero")

    # (B) whenever dz=z-z' is out of range, output must be zero there
    N = pre.delta_z.size
    mid = N // 2
    dz_step = float(pre.dz_step)

    idx_row = np.rint((z_custom[0] - zp) / dz_step).astype(int) + mid
    ok_row = (idx_row >= 0) & (idx_row < N)
    # (B) whenever dz=z-z' is out of range, output must be zero there
    assert np.allclose(G[0, ~ok_row], 0.0, atol=1e-14, rtol=0.0)
    assert np.allclose(core[0], 0.0, atol=1e-14, rtol=0.0)


