# TODO: Old tests need to be altered and new added in test_preset.py to reflect the recent changes.
# TODO: Add docstrings to the new classes and methods where needed and the whole file.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import GridConfig, PhysicsConfig, ModelConfig

# --------------------------------------------
# region Types and helper functions
# --------------------------------------------
SideChoice = Literal["left", "right"]

@dataclass(frozen=True)
class DevBulkPreset:
    """
    Fully-specified development preset for diagnostics/plotting scripts of bulk Green's functions.

    Goal: plotting + small pipeline scripts only need:
      preset.grid, preset.physics, preset.model, preset.bulk_side, preset.plot_channels, preset.omega/kx/ky
    """
    name: str
    grid: GridConfig
    physics: PhysicsConfig
    model: ModelConfig

    omega: float
    kx: float = 0.0
    ky: float = 0.0

    # Which bulk to use for bulk-only plots/FFTs (no interface yet)
    bulk_side: SideChoice = "left"

    # Default matrix entries to show in component plots and kz diagnostics
    # (choose sensible defaults per model by setting this per-preset)
    plot_channels: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (1, 1))

    # Optional small note for humans
    note: str = ""

@dataclass(frozen=True)
class DevGluingPreset:
    """
    Fully-specified development preset for diagnostics/plotting scripts of interfacial Green's functions.

    The interfacial GF is built from two bulks + an interface, so we need more info than for bulk-only presets.

    Goal: plotting + small pipeline scripts only need:
    TODO: finish docstring
    """
    name: str
    grid: GridConfig
    physics: PhysicsConfig
    model: ModelConfig

    omega: float
    # TODO: Introduce a batched version later for kx/ky integrals 
    kx: float = 0.0
    ky: float = 0.0

    # Allow defaults for bulk/interface params to be used if not explicitly set in model 
    # or force explicit setting
    # Outside of dev presets, prefer explicit setting to avoid silent errors or inconsistencies.
    # if True, using falling back on defaults is not allowed and bulk/interface params must be explicitly set in model
    # otherwise a ValueError is raised in build_gluing_components in the registry
    no_defaults: bool = False  

    # Default matrix entries to show in component plots
    # (choose sensible defaults per model by setting this per-preset)
    plot_channels: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (1, 1))

    # Optional small note for humans
    note: str = ""

def _make_presets_for_bulk(*items: DevBulkPreset) -> dict[str, DevBulkPreset]:
    # convenience + prevents typos in dict keys
    return {p.name: p for p in items}

def _make_presets_for_gluing(*items: DevGluingPreset) -> dict[str, DevGluingPreset]:
    # convenience + prevents typos in dict keys
    return {p.name: p for p in items}
# endregion
# -------------------------------------------------------------


# -------------------------------------------------------------
# region Toy presets
# -------------------------------------------------------------
# Conventions: Toy defaults are m=1, gap(Δ)=1.
# Lower band minimum at k=0 is E_- = -Δ = -1.
# "Evanescent" regime: choose omega < -1.

# evanescent: omega = -2 is below the energy minimum in the toy model, hence no poles are present
# "evanescent" = below band edge
TOY_FFT_DEBUG_EVASANESCENT = DevBulkPreset(
    name="toy_fft_debug_evanescent",
    model=ModelConfig(name="toy"),
    bulk_side="left",
    plot_channels=((0, 0), (1, 1)),
    grid=GridConfig(
        nomega=1,
        nz=513,
        nk_parallel=1,
        nphi=1,
        k_max=5.0,
        z_max=50.0,
        omega_max=1.0,
    ),
    physics=PhysicsConfig(eta=0.05, mu=0.0, temperature=0.0),
    omega=-2.0,
    kx=0.0,
    ky=0.0,
    note="Below band edge; no poles expected.",
)

# "Near-shell" regime at omega=0 for the lower band:
# E_-(k)=k^2/2 - 1 => on-shell at |k|=sqrt(2) ≈ 1.414.
# This preset is chosen to make wrap-around negligible while still seeing the on-shell structure.
TOY_FFT_NEAR_SHELL_WRAP_SAFE = DevBulkPreset(
    name="toy_fft_near_shell_wrap_safe",
    model=ModelConfig(name="toy"),
    bulk_side="left",
    plot_channels=((0, 0), (1, 1)),
    grid=GridConfig(
        nomega=1,
        nz=2051,
        nk_parallel=1,
        nphi=1,
        k_max=5.0,
        z_max=1000.0,
        omega_max=1.0,
    ),
    physics=PhysicsConfig(eta=0.02, mu=0.0, temperature=0.0),
    omega=0.0,
    kx=0.0,
    ky=0.0,
    note="Near-shell at ω=0 with a huge box to make wrap-around negligible.",
)

# Middle ground: near-shell at omega=0 but with larger eta and moderate box.
# Intended for everyday dev runs where you want "somewhat physical" behavior
# without the huge cost of the fully wrap-safe near-shell preset.
TOY_FFT_NEAR_SHELL_MID = DevBulkPreset(
    name="toy_fft_near_shell_mid",
    model=ModelConfig(name="toy"),
    bulk_side="left",
    plot_channels=((0, 0), (1, 1)),
    grid=GridConfig(
        nomega=1,
        nz=1025,
        nk_parallel=1,
        nphi=1,
        k_max=5.0,
        z_max=200.0,
        omega_max=1.0,
    ),
    physics=PhysicsConfig(eta=0.1, mu=0.0, temperature=0.0),
    omega=0.0,
    kx=0.0,
    ky=0.0,
    note="Everyday dev run: larger η + moderate box.",
)

TOY_GLUING_STANDARD = DevGluingPreset(
    name="toy_gluing_standard",
    model=ModelConfig(name="toy"),
    plot_channels=((0, 0), (1, 1)),
    grid=GridConfig(
        nomega=1,
        nz=1025,
        nk_parallel=1,
        nphi=1,
        k_max=5.0,
        z_max=200.0,
        omega_max=1.0,
    ),
    physics=PhysicsConfig(eta=0.1, mu=0.0, temperature=0.0),
    omega=0.0,
    kx=0.0,
    ky=0.0,
    no_defaults=False,
    note="Standard toy gluing preset with moderate box and defaults allowed.",
)

# Collect everything
TOY_PRESETS_bulk = _make_presets_for_bulk(
    TOY_FFT_DEBUG_EVASANESCENT,
    TOY_FFT_NEAR_SHELL_WRAP_SAFE,
    TOY_FFT_NEAR_SHELL_MID,
)
TOY_PRESETS_gluing = _make_presets_for_gluing(
    TOY_GLUING_STANDARD,)

TOY_PRESETS: dict[str, object] = {**TOY_PRESETS_bulk, **TOY_PRESETS_gluing}

# endregion
# -------------------------------------------------------------

# -------------------------------------------------------------
#region Orbitronic presets
# -------------------------------------------------------------
# Keep model params None to use model defaults in your registry.
# Later you can set bulk_left_params/bulk_right_params/interface_params in additional presets,
# e.g. model=ModelConfig(name="orbitronic", bulk_left_params=OrbitronicBulkParams(...)))

ORBITRONIC_FFT_MID = DevBulkPreset(
    name="orbitronic_fft_mid",
    model=ModelConfig(name="orbitronic"), # no params yields defaults in the registry
    bulk_side="left",
    # choose channels you care about (update once you know what’s most informative)
    plot_channels=((0, 0), (2, 2)),
    grid=GridConfig(
        nomega=1,
        nz=1025,
        nk_parallel=1,
        nphi=1,
        k_max=5.0,
        z_max=200.0,
        omega_max=1.0,
    ),
    physics=PhysicsConfig(eta=0.1, mu=0.0, temperature=0.0),
    omega=0.0,
    kx=0.0,
    ky=0.0,
    note="Orbitronic bulk FFT sanity run (mid-size box).",
)

ORBITRONIC_GLUING_STANDARD = DevGluingPreset(
    name="orbitronic_gluing_standard",
    model=ModelConfig(name="orbitronic"), # no params yields defaults in the registry
    plot_channels=((0, 0), (2, 2)),
    grid=GridConfig(
        nomega=1,
        nz=1025,
        nk_parallel=1,
        nphi=1,
        k_max=5.0,
        z_max=200.0,
        omega_max=1.0,
    ),  
    physics=PhysicsConfig(eta=0.1, mu=0.0, temperature=0.0),
    omega=0.0,  
    kx=0.0,
    ky=0.0,
    no_defaults=False,
    note="Standard orbitronic gluing preset with moderate box and defaults allowed.",
)

ORBITRONIC_GLUING_AROUND_ZERO = DevGluingPreset(
    name="orbitronic_gluing_around_zero",
    model=ModelConfig(name="orbitronic"), # no params yields defaults in the registry
    plot_channels=((0, 0), (2, 2)),
    grid=GridConfig(
        nomega=1,
        nz=1025,
        nk_parallel=1,
        nphi=1,
        k_max=5.0,
        z_max=55.0,
        omega_max=1.0,
    ),  
    physics=PhysicsConfig(eta=0.1, mu=0.0, temperature=0.0),
    omega=0.0,  
    kx=0.0,
    ky=0.0,
    no_defaults=False,
    note="Fine orbitronic gluing preset with moderate box and defaults allowed continuity observations.",
)

# Collect everything
ORBITRONIC_PRESETS_bulk = _make_presets_for_bulk(
    ORBITRONIC_FFT_MID,
)
ORBITRONIC_PRESETS_gluing = _make_presets_for_gluing(
    ORBITRONIC_GLUING_STANDARD, ORBITRONIC_GLUING_AROUND_ZERO)

ORBITRONIC_PRESETS: dict[str, object] = {**ORBITRONIC_PRESETS_bulk, **ORBITRONIC_PRESETS_gluing}

# endregion
# -------------------------------------------------------------


# region Collections of all presets
BULK_PRESETS: dict[str, DevBulkPreset] = {**TOY_PRESETS_bulk, **ORBITRONIC_PRESETS_bulk}
TOY_BULK_PRESET_NAMES = tuple(TOY_PRESETS_bulk.keys())
ORBITRONIC_BULK_PRESET_NAMES = tuple(ORBITRONIC_PRESETS_bulk.keys())

GLUING_PRESETS: dict[str, DevGluingPreset] = {**TOY_PRESETS_gluing, **ORBITRONIC_PRESETS_gluing}
TOY_GLUING_PRESET_NAMES = tuple(TOY_PRESETS_gluing.keys())
ORBITRONIC_GLUING_PRESET_NAMES = tuple(ORBITRONIC_PRESETS_gluing.keys())

# endregion
