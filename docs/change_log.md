# Changelog

**Scope:** This changelog tracks **breaking** and **user-facing** changes (plus a short “Internal” section for notable refactors).  
For full detail, see the git history.

## Doc
### Release workflow (optional but recommended)
```bash
git add -A
git commit -m "..."
git tag -a v0.2.0 -m "v0.2.0"
git push
git push --tags
```
### Template
Paste this in `## Unreleased` for a new day.
<!--
#### Breaking
- ...

#### User-facing
- ...

#### Internal
- ...
-->
and then, when ready for release, move it to the top of the log, f.ex. in  `### 0.2.0 (2025-12-29)`
### Naming convention
Major.MINOR.PATCH
- MAJOR (the first number): breaking changes
- MINOR (second): new features that don’t break existing usage
- PATCH (third): bug fixes / small improvements, no breaking changes, no new features required

So: 1.2.1 means: major version 1, minor version 2, patch 1.

Special rule of thumb in development:
- 0.x.y = “pre-1.0 / unstable API”
Breaking changes are expected and don’t force you to bump MAJOR (because MAJOR is 0).
Hence, “0.x” is read as “still evolving”.
- 1.0.0 and above = “stable public API promise”
From here on, breaking changes should bump MAJOR.

## Unreleased
#### Breaking
- Deleted functional_trace.py and integrated it as a helper in streda.py


#### User-facing
- Implemented a bare-bones version of streda.py with all its core structure but without the integration algorithms

#### Internal
- ...

## Logbook

### 0.1.0 (2025-12-29)

#### Breaking
- Renamed `ModelConfig.hamiltonian_factory` → `ModelConfig.hamiltonian_function`.
- Added `DevPreset.bulk_side` and `DevPreset.plot_channels`.
- Updated the presets accordingly


#### User-facing
- Added new model-agnostic plotting script `plot_bulk_greens` to replace toy-specific `plot_fft_decay_toy`
- Added \docs folder and md files there on `diagnostics`, `status` and this `changelog`
- Added kubo\diagnostics folder and `kz_coverage` module to compute diagnostic metrics
- Added `registry`in kubo\models folder
- Added `plot_kz_diagnostic_with_fft_coverage` function in kubo\plotting.py
- Updated README.md in the root folder
- Added `ORBITRONIC_FFT_MID` as the first orbitronic model preset in presets.py
- Added sub-dicts to `PRESETS` called `TOY_PRESETS` and `ORBITRONIC_PRESETS`
- Added defaults to `OrbitronicBulkParams` and `OrbitronicInterfaceParams`

#### Internal
- Refactored registry imports to relative imports.
- Updated docstrings
- Added type checking safety nets in ModelConfig in config.py
- Created VS Code style regions in toy.py and orbitronic.py
