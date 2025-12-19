from __future__ import annotations

import dataclasses
import pytest

from kubo.config import GridConfig, PhysicsConfig
from kubo.presets import (
    DevPreset,
    TOY_FFT_DEBUG_EVASANESCENT,
    TOY_FFT_NEAR_SHELL_WRAP_SAFE,
    TOY_FFT_NEAR_SHELL_MID,
    PRESETS,
)


# ---------------------------------------------
# region DevPreset dataclass behavior
# ---------------------------------------------
def test_devpreset_is_frozen():
    p = TOY_FFT_DEBUG_EVASANESCENT
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.omega = -123.0  # type: ignore[misc]


def test_devpreset_field_types():
    p = TOY_FFT_DEBUG_EVASANESCENT
    assert isinstance(p, DevPreset)
    assert isinstance(p.name, str)
    assert isinstance(p.grid, GridConfig)
    assert isinstance(p.physics, PhysicsConfig)
    assert isinstance(p.omega, float)
    assert isinstance(p.kx, float)
    assert isinstance(p.ky, float)


# endregion
# ---------------------------------------------


# ---------------------------------------------
# region Preset registry consistency
# ---------------------------------------------
@pytest.mark.parametrize(
    "preset",
    [
        TOY_FFT_DEBUG_EVASANESCENT,
        TOY_FFT_NEAR_SHELL_WRAP_SAFE,
        TOY_FFT_NEAR_SHELL_MID,
    ],
)
def test_each_preset_is_registered_by_name(preset: DevPreset):
    assert preset.name in PRESETS
    # registry should point to the same object (not a copy)
    assert PRESETS[preset.name] is preset


def test_registered_names_are_unique():
    names = [p.name for p in PRESETS.values()]
    assert len(names) == len(set(names))


def test_presets_is_dict_of_devpresets():
    assert isinstance(PRESETS, dict)
    assert all(isinstance(k, str) for k in PRESETS.keys())
    assert all(isinstance(v, DevPreset) for v in PRESETS.values())


# endregion
# ---------------------------------------------


# ---------------------------------------------
# region Toy preset invariants / sanity checks
# ---------------------------------------------
@pytest.mark.parametrize(
    "preset",
    [
        TOY_FFT_DEBUG_EVASANESCENT,
        TOY_FFT_NEAR_SHELL_WRAP_SAFE,
        TOY_FFT_NEAR_SHELL_MID,
    ],
)
def test_toy_presets_have_basic_grid_sanity(preset: DevPreset):
    g = preset.grid
    assert g.nomega == 1
    assert g.nz > 0
    assert g.nz % 2 == 1  # odd nz => well-defined center at nz//2
    assert g.k_max > 0
    assert g.z_max > 0
    assert g.omega_max > 0
    assert g.nk_parallel >= 1
    assert g.nphi >= 1


@pytest.mark.parametrize(
    "preset",
    [
        TOY_FFT_DEBUG_EVASANESCENT,
        TOY_FFT_NEAR_SHELL_WRAP_SAFE,
        TOY_FFT_NEAR_SHELL_MID,
    ],
)
def test_toy_presets_have_basic_physics_sanity(preset: DevPreset):
    ph = preset.physics
    assert ph.eta > 0
    # allow mu/temperature to be zero, but not NaN
    assert ph.mu == ph.mu
    assert ph.temperature == ph.temperature


def test_toy_evanescent_regime_is_below_band_edge_convention():
    # based on the docstring convention in presets.py: "evanescent" means omega < -1
    assert TOY_FFT_DEBUG_EVASANESCENT.omega < -1.0


def test_toy_near_shell_presets_use_omega_zero():
    assert TOY_FFT_NEAR_SHELL_WRAP_SAFE.omega == pytest.approx(0.0)
    assert TOY_FFT_NEAR_SHELL_MID.omega == pytest.approx(0.0)


# endregion
# ---------------------------------------------
