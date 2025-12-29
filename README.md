> 🚧 **Status:** Under active development — APIs and file layout may change.

# Numeric GF Bilayer Kubo Response

Numerical computation of Kubo (DC) observables from Green’s functions for **bilayer systems** that break translation invariance in **one spatial direction** (the interface normal). The core package (`kubo/`) builds bulk Green’s functions in momentum space, transforms them to real space via FFTs, and **glues** the two half-spaces into a full interface Green’s function. The Kubo response is evaluated using the **Středa decomposition** into Fermi-surface and Fermi-sea contributions in the DC limit.

This repository currently targets an **orbitronic bilayer** model (3×3 bulk Hamiltonian) with a planar interface between two metals, but the codebase is structured so you can swap in other models that share the same “two bulks + interface” geometry.

---

## What the code does

At a high level, the pipeline is:

1. **Model definition** (`kubo/models/`)
   - Bulk Hamiltonians \(H(\mathbf{k})\) for the left/right materials
   - Interface coupling / boundary conditions
   - Observable operators (model-specific)

2. **Bulk Green’s functions** (`kubo/greens.py`)
   - Construct retarded bulk Green’s functions \(G^R(\omega,\mathbf{k})\)
   - Diagnostic kz scans on custom grids

3. **FFT to real space** (`kubo/greens.py`, `kubo/grids.py`)
   - FFT over \(k_z\) to obtain \(G^R(\omega, k_\parallel; \Delta z)\)
   - Careful grid conventions (FFT kz grid vs wide diagnostic kz grid)

4. **Gluing / interface Green’s function** (`kubo/gluing.py`)
   - Combine left/right bulk kernels and interface conditions into a full bilayer Green’s function
   - Produces the real-space Green’s function needed by later Kubo integrals

5. **Kubo / Středa evaluation** (`kubo/streda.py`, integration modules)
   - Split into Fermi-surface and Fermi-sea terms in the DC limit
   - Perform the remaining numerical sums/integrals on the chosen grids

---

## Repository structure (important entry points)

- `kubo/config.py`  
  Shared configuration dataclasses (grid, physics, model selection).

- `kubo/presets.py`  
  Reproducible **development presets** (`DevPreset`) bundling grid + physics + model + default bulk side + default plot channels.

- `kubo/models/registry.py`  
  Model dispatch layer that constructs a bulk Hamiltonian callable from a `ModelConfig` + side.

- `scripts/`  
  Small developer scripts for diagnostics and sanity checks (run as `python -m ...`).

- `examples/`
  Small runners showcasing the use of the kubo package.

- `tests/`
  Unit and pipeline tests for all modules and models.

- `docs/`  
  Project notes and diagnostic explanations.

---

## Diagnostics

FFT-based real-space kernels require *two* sanity checks:

- **kz-range coverage** (does the FFT kz window cover the relevant structure of \(|G_{ij}(k_z)|\)?)
- **real-space wrap-around** (has \(G(\Delta z)\) decayed enough at the FFT box edges?)

See `docs/diagnostics.md` for details and interpretation.

Recommended dev entry point with default values:

```bash
python -m kubo.scripts.plot_bulk_greens --preset toy_fft_near_shell_mid
python -m kubo.scripts.plot_bulk_greens --preset orbitronic_fft_mid

