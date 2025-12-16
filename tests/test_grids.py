# ---------------------------------------------
# Imports 
# ---------------------------------------------
import pytest
import numpy as np
from kubo.config import GridConfig
from kubo.grids import (
    _require_odd,
    build_zp_grid,
    build_k_parallel_grid_polar,
    build_k_parallel_grid_cartesian,
    build_omega_grid,
    build_delta_z_kz_grids_fft
)
# ---------------------------------------------
# Fixtures
# ---------------------------------------------
@pytest.fixture
def cfg() -> GridConfig:
    return GridConfig()  # use defaults

# ---------------------------------------------
# region basics & helpers
# ---------------------------------------------
def test_grid_config_defaults_are_odd(cfg: GridConfig):
    """Defaults should satisfy the '0 lies on the grid' convention."""
    assert cfg.nomega % 2 == 1
    assert cfg.nz % 2 == 1
    assert cfg.nk_parallel % 2 == 1

def test_require_odd_accepts_odd():
    # Should not raise
    _require_odd("test_param", 3)

def test_require_odd_rejects_even():
    with pytest.raises(ValueError):
        _require_odd("test_param", 4)

# endregion
# ---------------------------------------------

# ---------------------------------------------
# region omega grid
# ---------------------------------------------
def test_build_omega_grid_basic_properties(cfg: GridConfig):
    omega = build_omega_grid(cfg)

    # Shape and dtype
    assert omega.shape == (cfg.nomega,)
    assert omega.dtype == np.float64

    # Endpoints and monotonicity
    assert omega[0] == pytest.approx(-cfg.omega_max)
    assert omega[-1] == pytest.approx(cfg.omega_max)
    diffs = np.diff(omega)
    assert np.all(diffs > 0)
    assert np.allclose(diffs, diffs[0])

    # Zero should be at the center index for odd nomega
    mid = cfg.nomega // 2
    assert omega[mid] == pytest.approx(0.0)

def test_build_omega_grid_requires_odd_nomega(cfg: GridConfig):
    cfg.nomega = 10  # even
    with pytest.raises(ValueError):
        build_omega_grid(cfg)

# endregion
# ---------------------------------------------

# ---------------------------------------------
# region zp grid
# ---------------------------------------------
def test_build_zp_grid_basic_properties(cfg: GridConfig):
    zp = build_zp_grid(cfg)

    # Shape and dtype
    assert zp.shape == (cfg.nz,)
    assert zp.dtype == np.float64

    # z should be monotonic increasing
    dz = np.diff(zp)
    assert np.all(dz > 0)
    # uniform spacing
    assert np.allclose(dz, dz[0])

    # Total box length from grid spacing should be 2 * z_max
    L_from_grid = dz[0] * cfg.nz
    assert L_from_grid == pytest.approx(2.0 * cfg.z_max)

    # z should span [-z_max, z_max) with center at 0
    assert zp[0] == pytest.approx(-cfg.z_max+(dz[0]/2))
    # half-open: max < z_max
    assert zp[-1] < cfg.z_max
    assert zp[-1] == pytest.approx(cfg.z_max-(dz[0]/2))
    mid = cfg.nz // 2
    assert zp[mid] == pytest.approx(0.0)

    

def test_build_zp_grid_requires_odd_nz(cfg: GridConfig):
    cfg.nz = 10  # even
    with pytest.raises(ValueError):
        build_zp_grid(cfg)

# endregion
# ---------------------------------------------

# ---------------------------------------------
# region in-plane grids
# ---------------------------------------------
def test_build_k_parallel_polar_grid_basic_properties(cfg: GridConfig):
    k_par, phi = build_k_parallel_grid_polar(cfg)

    # Shapes and dtypes
    assert k_par.shape == (cfg.nk_parallel,)
    assert phi.shape == (cfg.nphi,)
    assert k_par.dtype == np.float64
    assert phi.dtype == np.float64

    # k_parallel: [0, k_max], inclusive, monotonic, uniform
    assert k_par[0] == pytest.approx(0.0)
    assert k_par[-1] == pytest.approx(cfg.k_max)
    dk = np.diff(k_par)
    assert np.all(dk > 0)
    assert np.allclose(dk, dk[0])

    # phi: [0, 2π) with endpoint excluded, uniform spacing
    assert phi[0] == pytest.approx(0.0)
    assert np.all(phi >= 0.0)
    assert np.all(phi < 2.0 * np.pi)

    dphi = np.diff(phi)
    assert np.allclose(dphi, dphi[0])

    # Last phi should be 2π - Δφ
    expected_last = 2.0 * np.pi - dphi[0]
    assert phi[-1] == pytest.approx(expected_last)

def test_build_k_parallel_grid_cartesian_basic_properties(cfg: GridConfig):
    k_x, k_y = build_k_parallel_grid_cartesian(cfg)

    # Shapes and dtypes
    assert k_x.shape == (cfg.nk_parallel,)
    assert k_y.shape == (cfg.nk_parallel,)
    assert k_x.dtype == np.float64
    assert k_y.dtype == np.float64

    # Both are symmetric linspaces in [-k_max, k_max], inclusive
    for k in (k_x, k_y):
        assert k[0] == pytest.approx(-cfg.k_max)
        assert k[-1] == pytest.approx(cfg.k_max)
        dk = np.diff(k)
        assert np.all(dk > 0)
        assert np.allclose(dk, dk[0])

        # With odd nk_parallel, zero must be at the center index
        mid = cfg.nk_parallel // 2
        assert k[mid] == pytest.approx(0.0)

def test_build_k_parallel_grid_requires_odd_nk_parallel(cfg: GridConfig):
    cfg.nk_parallel = 10  # even
    with pytest.raises(ValueError):
        build_k_parallel_grid_cartesian(cfg)

# endregion
# ---------------------------------------------

# ---------------------------------------------
# region FFT grid pair
# ---------------------------------------------
def test_build_kz_grid_fft_basic_properties(cfg: GridConfig):
    z, kz = build_delta_z_kz_grids_fft(cfg)

    # Shapes and dtypes
    assert z.shape == (cfg.nz,)
    assert kz.shape == (cfg.nz,)
    assert z.dtype == np.float64
    assert kz.dtype == np.float64  # fftfreq -> float64

    # z should be monotonic increasing
    dz = np.diff(z)
    assert np.all(dz > 0)
    # uniform spacing
    assert np.allclose(dz, dz[0])

    # Total box length from grid spacing should be 2 * z_max
    L_from_grid = dz[0] * cfg.nz
    assert L_from_grid == pytest.approx(2.0 * cfg.z_max)

    # z should span [-z_max, z_max) with center at 0
    assert z[0] == pytest.approx(-cfg.z_max+(dz[0]/2)) # cell-centered
    # half-open: max < z_max
    assert z[-1] < cfg.z_max
    assert z[-1] == pytest.approx(cfg.z_max-(dz[0]/2)) # cell-centered
    mid = cfg.nz // 2
    assert z[mid] == pytest.approx(0.0)

    # kz[0] should be 0 (DC mode)
    assert kz[0] == pytest.approx(0.0)

    # Magnitude of frequency spacing should be consistent with 2π / L
    # (kz is 2π * fftfreq(N, d=dz))
    # Ignore the sign jump and just check spacing magnitude sorted.
    kz_sorted = np.sort(kz)
    dkz = np.diff(kz_sorted)
    # Not all dkz will be equal around the wrap, but magnitudes should be ~constant
    dkz_abs = np.abs(dkz)
    assert np.allclose(dkz_abs, dkz_abs[0])


def test_build_kz_grid_fft_requires_odd_nz(cfg: GridConfig):
    cfg.nz = 10  # even
    with pytest.raises(ValueError):
        build_delta_z_kz_grids_fft(cfg)

def _grid_small_fft() -> GridConfig:
    # mirror what you used in FFT tests
    return GridConfig(nz=31, z_max=10.0, nk_parallel=7, nomega=7)

def test_zp_grid_aligns_with_fft_delta_z_grid():
    """
    Guardrail: the z'-integration grid (zp) should align with the FFT real-space grid
    used for delta_z = z - z'. This avoids mixing two different discretizations of the
    same box later in gluing / Kubo traces.
    """
    cfg = _grid_small_fft()

    zp = build_zp_grid(cfg)
    delta_z, _kz = build_delta_z_kz_grids_fft(cfg)

    # same length
    assert zp.shape == delta_z.shape == (cfg.nz,)

    # same spacing
    dz_zp = zp[1] - zp[0]
    dz_dz = delta_z[1] - delta_z[0]
    assert dz_zp == pytest.approx(dz_dz)

    # pointwise alignment (this is the big one)
    assert np.allclose(zp, delta_z, rtol=0.0, atol=1e-12)