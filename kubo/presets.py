from __future__ import annotations

from dataclasses import dataclass

from .config import GridConfig, PhysicsConfig


@dataclass(frozen=True)
class DevPreset:
    name: str
    grid: GridConfig
    physics: PhysicsConfig
    omega: float
    kx: float = 0.0
    ky: float = 0.0


# ---- Toy presets ----
# Conventions: Toy defaults are m=1, gap(Δ)=1.
# Lower band minimum at k=0 is E_- = -Δ = -1.
# "Evanescent" regime: choose omega < -1.

# evanescent: omega = -2 is below the energy minimum in the toy model, hence no poles are present
# "evanescent" = below band edge
TOY_FFT_DEBUG_EVASANESCENT = DevPreset(
    name="toy_fft_debug_evanescent",
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
)

# "Near-shell" regime at omega=0 for the lower band:
# E_-(k)=k^2/2 - 1 => on-shell at |k|=sqrt(2) ≈ 1.414.
# This preset is chosen to make wrap-around negligible while still seeing the on-shell structure.
TOY_FFT_NEAR_SHELL_WRAP_SAFE = DevPreset(
    name="toy_fft_near_shell_wrap_safe",
    grid=GridConfig(
        nomega=1,
        nz=5013,
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
)

# Middle ground: near-shell at omega=0 but with larger eta and moderate box.
# Intended for everyday dev runs where you want "somewhat physical" behavior
# without the huge cost of the fully wrap-safe near-shell preset.
TOY_FFT_NEAR_SHELL_MID = DevPreset(
    name="toy_fft_near_shell_mid",
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
)

PRESETS = {
    TOY_FFT_DEBUG_EVASANESCENT.name: TOY_FFT_DEBUG_EVASANESCENT,
    TOY_FFT_NEAR_SHELL_WRAP_SAFE.name: TOY_FFT_NEAR_SHELL_WRAP_SAFE,
    TOY_FFT_NEAR_SHELL_MID.name: TOY_FFT_NEAR_SHELL_MID,
}