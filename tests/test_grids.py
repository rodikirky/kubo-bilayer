from kubo.config import GridConfig
from kubo.grids import (
    build_kz_grid,
    build_k_parallel_grid,
    build_z_grid,
    build_omega_grid,
)


def test_grid_shapes():
    cfg = GridConfig(
        nk_parallel=10,
        nphi=5,
        nkz=7,
        nz=9,
        nomega=11,
        k_max=1.0,
        z_max=2.0,
        omega_max=3.0,
    )

    kz = build_kz_grid(cfg)
    k_par, phi = build_k_parallel_grid(cfg)
    z = build_z_grid(cfg)
    omega = build_omega_grid(cfg)

    assert kz.shape == (7,)
    assert k_par.shape == (10,)
    assert phi.shape == (5,)
    assert z.shape == (9,)
    assert omega.shape == (11,)
