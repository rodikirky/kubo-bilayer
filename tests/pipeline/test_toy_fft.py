import numpy as np
import pytest

from kubo.config import GridConfig, PhysicsConfig
from kubo.grids import build_kz_grid_fft
from kubo.greens import realspace_greens_retarded


def _grid_small_fft() -> GridConfig:
    # keep it small so test runs fast; odd nz required by your grid builder
    return GridConfig(nz=31, z_max=10.0, nk_parallel=7, nomega=7)


def test_build_kz_grid_fft_sanity():
    cfg = _grid_small_fft()
    z, kz = build_kz_grid_fft(cfg)

    assert z.shape == (cfg.nz,)
    assert kz.shape == (cfg.nz,)

    # z is evenly spaced
    dz = z[1] - z[0]
    assert np.allclose(np.diff(z), dz)

    # your current implementation places z=0 exactly at the midpoint index
    assert z[cfg.nz // 2] == pytest.approx(0.0)

    # kz should be angular frequencies from fftfreq
    L = 2.0 * cfg.z_max
    expected_kz = 2.0 * np.pi * np.fft.fftfreq(cfg.nz, d=L / cfg.nz)
    assert np.allclose(kz, expected_kz)


def test_realspace_greens_retarded_shape_and_dtype(toy_H_spec):
    cfg = _grid_small_fft()
    physics = PhysicsConfig(eta=1e-3)

    omega = 0.5
    kx, ky = 0.2, -0.1

    z, G_z = realspace_greens_retarded(
        omega=omega, kx=kx, ky=ky, H=toy_H_spec, physics=physics, grid=cfg
    )

    assert z.shape == (cfg.nz,)
    assert G_z.ndim == 3
    assert G_z.shape[0] == cfg.nz
    assert G_z.shape[1] == G_z.shape[2]  # square matrices
    assert np.iscomplexobj(G_z)


def test_realspace_greens_retarded_im_trace_nonpositive_at_z0(toy_H_spec):
    cfg = _grid_small_fft()
    physics = PhysicsConfig(eta=1e-3)

    omega = 0.1
    kx, ky = 0.2, 0.1

    z, G_z = realspace_greens_retarded(
        omega=omega, kx=kx, ky=ky, H=toy_H_spec, physics=physics, grid=cfg
    )

    mid = cfg.nz // 2  # where z == 0 in your current convention
    im_tr = np.imag(np.trace(G_z[mid]))

    # allow tiny positive numerical noise
    assert im_tr <= 1e-10


def test_realspace_greens_retarded_is_even_in_z_for_toy(toy_H_spec):
    """
    Toy model depends on kz^2 => G(kz) is even in kz => G(z) should be even in z.
    This is a strong FFT convention check (shift + normalization).
    """
    cfg = _grid_small_fft()
    physics = PhysicsConfig(eta=1e-3)

    omega = 0.7
    kx, ky = 0.25, -0.15

    z, G_z = realspace_greens_retarded(
        omega=omega, kx=kx, ky=ky, H=toy_H_spec, physics=physics, grid=cfg
    )

    # compare G(z) and G(-z)
    G_flip = G_z[::-1, :, :]
    assert np.allclose(G_z, G_flip, rtol=1e-9, atol=1e-9)
