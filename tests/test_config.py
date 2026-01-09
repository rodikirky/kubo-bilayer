# ---------------------------------------------
# Imports
# ---------------------------------------------
from dataclasses import is_dataclass

import pytest
import numpy as np

from kubo.config import (
    GridConfig,
    ModelConfig,
    PhysicsConfig,
    KuboConfig,
)

# ---------------------------------------------
# region GridConfig
# ---------------------------------------------
def test_grid_config_is_dataclass():
    assert is_dataclass(GridConfig)

def test_grid_config_defaults_are_odd_and_positive():
    cfg = GridConfig()

    # odd sizes so that 0 is on the grid
    assert cfg.nomega % 2 == 1
    assert cfg.nz % 2 == 1
    assert cfg.nk_parallel % 2 == 1

    # positive sizes
    assert cfg.nomega > 0
    assert cfg.nz > 0
    assert cfg.nk_parallel > 0
    assert cfg.nphi > 0

    # positive extents
    assert cfg.k_max > 0.0
    assert cfg.z_max > 0.0
    assert cfg.omega_max > 0.0

def test_grid_config_custom_values():
    cfg = GridConfig(
        nomega=31,
        nz=33,
        nk_parallel=35,
        nphi=12,
        k_max=3.0,
        z_max=15.0,
        omega_max=7.0,
    )

    assert cfg.nomega == 31
    assert cfg.nz == 33
    assert cfg.nk_parallel == 35
    assert cfg.nphi == 12
    assert cfg.k_max == pytest.approx(3.0)
    assert cfg.z_max == pytest.approx(15.0)
    assert cfg.omega_max == pytest.approx(7.0)

# endregion
# ---------------------------------------------

# ---------------------------------------------
# region ModelConfig
# ---------------------------------------------
def test_model_config_is_dataclass():
    assert is_dataclass(ModelConfig)

def test_model_config_defaults():
    mc = ModelConfig()

    assert mc.name == "toy"
    assert mc.bulk_left_params is None
    assert mc.bulk_right_params is None
    assert mc.interface_params is None
    assert mc.hamiltonian_function is None

def test_model_config_does_not_allow_custom_params_and_factory():
    class DummyBulkParams:
        pass

    bulk_l = DummyBulkParams()
    bulk_r = DummyBulkParams()
    interface = object()

    def dummy_function(kx,ky,kz):
        return np.array([[1]])

    with pytest.raises(TypeError, match="must be OrbitronicBulkParams or None"):
        mc = ModelConfig(
            name="orbitronic",
            bulk_left_params=bulk_l,
            bulk_right_params=bulk_r,
            interface_params=interface,
            hamiltonian_function=dummy_function,
        )

# endregion
# ---------------------------------------------

# ---------------------------------------------
# region PhysicsConfig
# ---------------------------------------------
def test_physics_config_is_dataclass():
    assert is_dataclass(PhysicsConfig)

def test_physics_config_defaults():
    pc = PhysicsConfig()

    assert pc.eta == pytest.approx(1e-6)
    assert pc.mu == pytest.approx(0.0)
    assert pc.temperature == pytest.approx(0.0)

def test_physics_config_custom_values():
    pc = PhysicsConfig(eta=0.1, mu=1.5, temperature=0.01)

    assert pc.eta == pytest.approx(0.1)
    assert pc.mu == pytest.approx(1.5)
    assert pc.temperature == pytest.approx(0.01)

# endregion
# ---------------------------------------------

# ---------------------------------------------
# region KuboConfig
# ---------------------------------------------
def test_kubo_config_is_dataclass():
    assert is_dataclass(KuboConfig)

def test_kubo_config_defaults_types():
    kc = KuboConfig()

    assert isinstance(kc.grid, GridConfig)
    assert isinstance(kc.model, ModelConfig)
    assert isinstance(kc.physics, PhysicsConfig)

def test_kubo_config_uses_default_factories_independently():
    kc1 = KuboConfig()
    kc2 = KuboConfig()

    # Different KuboConfig instances
    assert kc1 is not kc2

    # Nested configs should not be shared objects
    assert kc1.grid is not kc2.grid
    assert kc1.model is not kc2.model
    assert kc1.physics is not kc2.physics

def test_kubo_config_accepts_custom_subconfigs():
    grid = GridConfig(nomega=31)
    model = ModelConfig(name="orbitronic")
    physics = PhysicsConfig(eta=0.05)

    kc = KuboConfig(grid=grid, model=model, physics=physics)

    assert kc.grid is grid
    assert kc.model is model
    assert kc.physics is physics
    assert kc.grid.nomega == 31
    assert kc.model.name == "orbitronic"
    assert kc.physics.eta == pytest.approx(0.05)

# endregion
# ---------------------------------------------
