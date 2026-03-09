# kubo_bilayer

A continuum linear response framework for computing cross-interface transport
in planar bilayer systems, based on interfacial Green's functions and the
Kubo-Streda formalism.

---

## What it does

<!-- Brief description of the physical problem and the key result:
     given a bilayer system defined by two bulk Hamiltonians and an interface
     Hamiltonian, kubo_bilayer computes the spatially resolved response
     σ_A(z) of an observable A to a perturbation on the other side of the
     interface. -->

---

## Computational pipeline

```
setup/          →     numerics/      →     analytics/     →     integrals/
                                                                     ↓
                                                               σ_A(z) profile
```

<!-- Expand this section with the Mermaid flowchart once the pipeline is complete. -->

---

## Package structure

```
kubo_bilayer/
├── setup/            # Hamiltonian dataclasses and coefficient containers
│                     # Operators for response and perturbation
│                     # Grids
│
├── numerics/         # Pole computation via companion linearization
│                     # and residue construction via SVD
│
├── analytics/        # Analytical expressions for the interfacial
│                     # Green's function and response terms I–Vb
│
├── integrals/        # k∥ grid, quadrature, and parallelization
│                     # returns σ_A(z)
│
├── config.py         # Loads and validates config.yaml → Config dataclass
├── presets.py        # Loads presets.yaml → BilayerSystem instances
└── README.md
```

---

## Installation

<!-- pip install instructions once packaged -->

---

## Quickstart

<!-- Minimal working example:
     - define a BilayerSystem from BulkCoeffs and InterfaceCoeffs
     - load a config
     - run the integral
     - plot σ_A(z) -->

---

## Configuration

<!-- Description of config.yaml fields: grid size, quadrature,
     parallelization backend, output directory, tolerances. -->

---

## Presets

<!-- How to define a named physical system in presets.yaml and
     load it via load_preset(). Point to showcase/ for worked examples. -->

---

## Testing

<!-- How to run the test suite. Note the role of the free-electron
     benchmark as the primary integration test. -->

---

## References

<!-- Kubo formula, Streda decomposition, companion linearization,
     and any papers the implementation is based on. -->