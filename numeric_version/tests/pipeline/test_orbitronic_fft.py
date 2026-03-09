from __future__ import annotations

import numpy as np
import pytest

from dataclasses import replace
from kubo.grids import build_delta_z_kz_grids_fft
from kubo.greens import (
    realspace_greens_retarded,
    realspace_greens_retarded_with_kz,
    _fourier_kz_to_z,
    _build_fft_Gkz_input_for_fixed_omega_kpar,
)

from kubo.models.orbitronic import OrbitronicBulk, OrbitronicBulkParams


def _inverse_fourier_z_to_kz(Gz: np.ndarray, cfg, axis: int = 0) -> np.ndarray:
    """
    Exact inverse of kubo.greens._fourier_kz_to_z (current implementation):

      Gz = fftshift( ifft(Gk, axis) * (N/L), axes=axis )

    =>  Gk = fft( ifftshift(Gz, axes=axis) * (L/N), axis=axis )
    """
    N = cfg.nz
    L = 2.0 * cfg.z_max
    tmp = np.fft.ifftshift(Gz, axes=axis) * (L / N)
    return np.fft.fft(tmp, axis=axis)


# --------------------------------------------------------------------------------------
# Basic FFT/grid sanity (orbitronic)
# --------------------------------------------------------------------------------------

def test_orbitronic_build_kz_grid_fft_sanity(grid_small_fft):
    cfg = grid_small_fft
    z, kz = build_delta_z_kz_grids_fft(cfg)

    assert z.shape == (cfg.nz,)
    assert kz.shape == (cfg.nz,)

    dz = z[1] - z[0]
    assert np.allclose(np.diff(z), dz)

    # convention: z=0 at midpoint
    assert z[cfg.nz // 2] == pytest.approx(0.0)

    # kz should match fftfreq convention
    L = 2.0 * cfg.z_max
    expected_kz = 2.0 * np.pi * np.fft.fftfreq(cfg.nz, d=L / cfg.nz)
    assert np.allclose(kz, expected_kz)


def test_orbitronic_realspace_greens_retarded_shape_and_dtype(orbitronic_H_default, physics_default, grid_small_fft, kpar_simple):
    omega = 0.5
    kx, ky = kpar_simple 

    z, Gz = realspace_greens_retarded(
        omega=omega,
        kx=kx,
        ky=ky,
        H=orbitronic_H_default,
        physics=physics_default,
        grid=grid_small_fft,
    )

    assert z.shape == (grid_small_fft.nz,)
    assert Gz.shape == (grid_small_fft.nz, 3, 3)
    assert np.iscomplexobj(Gz)
    assert np.isfinite(Gz.real).all()
    assert np.isfinite(Gz.imag).all()


def test_orbitronic_realspace_im_trace_nonpositive_at_z0(orbitronic_H_default, physics_default, grid_small_fft, kpar_simple):
    omega = 0.1
    kx, ky = kpar_simple

    z, Gz = realspace_greens_retarded(
        omega=omega,
        kx=kx,
        ky=ky,
        H=orbitronic_H_default,
        physics=physics_default,
        grid=grid_small_fft,
    )

    mid = grid_small_fft.nz // 2
    im_tr = np.imag(np.trace(Gz[mid]))
    assert im_tr <= 1e-10


# --------------------------------------------------------------------------------------
# FFT convention check: roundtrip kz -> z -> kz should be identity
# --------------------------------------------------------------------------------------

def test_orbitronic_fft_roundtrip_kz_to_z_and_back(orbitronic_H_default, physics_default, grid_small_fft, kpar_simple):
    cfg = grid_small_fft
    _, kz = build_delta_z_kz_grids_fft(cfg)

    omega = 0.6
    kx, ky = kpar_simple

    Gk = _build_fft_Gkz_input_for_fixed_omega_kpar(omega, kx, ky, kz, orbitronic_H_default, physics_default)
    Gz = _fourier_kz_to_z(Gk, cfg)
    Gk_rec = _inverse_fourier_z_to_kz(Gz, cfg)

    assert np.allclose(Gk_rec, Gk, rtol=1e-10, atol=1e-10)

@pytest.mark.parametrize("omega", [0.1, 0.6])
def test_orbitronic_fft_roundtrip_is_identity_tiny_grid(orbitronic_H_default, physics_default, grid_tiny_fft, kpar_simple, omega):
    kx, ky = kpar_simple
    _, kz = build_delta_z_kz_grids_fft(grid_tiny_fft)
    Gk = _build_fft_Gkz_input_for_fixed_omega_kpar(omega, kx, ky, kz, orbitronic_H_default, physics_default)
    Gz = _fourier_kz_to_z(Gk, grid_tiny_fft)
    Gk_rec = _inverse_fourier_z_to_kz(Gz, grid_tiny_fft)
    assert np.allclose(Gk_rec, Gk, rtol=1e-10, atol=1e-10)


def test_orbitronic_realspace_with_kz_matches_plain_realspace(orbitronic_H_default, physics_default, grid_small_fft, kpar_simple):
    omega = 0.55
    kx, ky = kpar_simple

    z1, G1 = realspace_greens_retarded(omega, kx, ky, orbitronic_H_default, physics_default, grid_small_fft)
    z2, kz2, G2, Gk2 = realspace_greens_retarded_with_kz(omega, kx, ky, orbitronic_H_default, physics_default, grid_small_fft)

    assert np.allclose(z1, z2)
    assert kz2.shape == (grid_small_fft.nz,)
    assert Gk2.shape == (grid_small_fft.nz, 3, 3)
    assert np.allclose(G1, G2, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------------------
# Orbitronic-specific canaries
# --------------------------------------------------------------------------------------

def test_orbitronic_realspace_retarded_advanced_relation_when_even_in_kz(
    physics_default, grid_small_fft, kpar_simple
):
    """
    For Hermitian H and if H(kz)=H(-kz) (true here for J=0, M=0),
    the real-space kernels satisfy:
        G^A(z) = [G^R(-z)]†
    """
    params = OrbitronicBulkParams(
        mass=1.2,
        gamma=0.9,
        J=0.0,
        magnetisation=[0.0, 0.0, 0.0],
    )
    bulk = OrbitronicBulk.from_params(params, basis=None)
    H_even = lambda kx, ky, kz: bulk.hamiltonian((kx, ky, kz))

    omega = 0.7
    kx, ky = kpar_simple

    # Retarded
    z, GRz = realspace_greens_retarded(
        omega, kx, ky, H_even, physics_default, grid_small_fft
    )

    # Advanced is retarded with eta -> -eta
    physics_adv = replace(physics_default, eta=-physics_default.eta)
    assert physics_adv.eta < 0
    z2, GAz = realspace_greens_retarded(
        omega, kx, ky, H_even, physics_adv, grid_small_fft
    )

    assert np.allclose(z, z2)

    GR_flip_dag = GRz[::-1].conj().transpose(0, 2, 1)  # [G^R(-z)]†
    assert np.allclose(GAz, GR_flip_dag, rtol=1e-9, atol=1e-11)

def test_orbitronic_realspace_fro_norm_is_even_when_even_in_kz(physics_default, grid_small_fft, kpar_simple):
    params = OrbitronicBulkParams(mass=1.2, gamma=0.9, J=0.0, magnetisation=[0.0, 0.0, 0.0])
    bulk = OrbitronicBulk.from_params(params, basis=None)
    H_even = lambda kx, ky, kz: bulk.hamiltonian((kx, ky, kz))

    omega = 0.7
    kx, ky = kpar_simple
    z, GRz = realspace_greens_retarded(omega, kx, ky, H_even, physics_default, grid_small_fft)

    amp = np.linalg.norm(GRz, axis=(1, 2))  # Frobenius norm over matrix indices
    assert np.allclose(amp, amp[::-1], rtol=1e-10, atol=1e-12)


def test_orbitronic_realspace_basis_covariance_through_fft(
    orbitronic_params_default,
    unitary_U3,
    physics_default,
    grid_small_fft,
    kpar_simple,
):
    """
    For unitary basis transform U:
      H' = U† H U  =>  G'(z) = U† G(z) U
    """
    U = unitary_U3
    Udag = U.conj().T

    bulk = OrbitronicBulk.from_params(orbitronic_params_default, basis=None)
    bulk_rot = OrbitronicBulk.from_params(orbitronic_params_default, basis=U)

    H = lambda kx, ky, kz: bulk.hamiltonian((kx, ky, kz))
    Hrot = lambda kx, ky, kz: bulk_rot.hamiltonian((kx, ky, kz))

    omega = 0.45
    kx, ky = kpar_simple

    z, Gz = realspace_greens_retarded(omega, kx, ky, H, physics_default, grid_small_fft)
    z2, Gz_rot = realspace_greens_retarded(omega, kx, ky, Hrot, physics_default, grid_small_fft)

    assert np.allclose(z, z2)

    Gz_cov = Udag[None, :, :] @ Gz @ U[None, :, :]
    assert np.allclose(Gz_rot, Gz_cov, rtol=1e-9, atol=1e-11)
