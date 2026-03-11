# kubo_bilayer

A Python package for computing spatially resolved linear response coefficients
across a planar interface in bilayer systems, using interfacial Green's functions
and the Kubo–Streda formalism.

Based on the master's thesis of Leona C. Rodenkirchen (RWTH Aachen, March 2026),
supervised by Prof. Dr. Dante Kennes and Prof. Dr. Yuriy Mokrousov.

---

## Physical Setting

A left bulk half-space (z < 0) and a right bulk half-space (z > 0) are joined at
z = 0 by a 2D interface Hamiltonian. On each side, translation invariance is
maintained in the (x, y) plane and the bulk Hamiltonian is a second-order
polynomial in k:

```
H(kx, ky, kz) = ½ Ax kx² + ½ Ay ky² + ½ Az kz²
              + Bx kx + By ky + Bz kz
              + Cxy kx ky + Cyz ky kz + Czx kz kx
              + D
```

where each coefficient is a hermitian matrix of the orbital/spin dimension.

The system is perturbed by an electric field on the left (z < 0), and the
spatially resolved response of an observable on the right (z > 0) is computed
using the Streda decomposition of the Kubo formula.

---

## Package Structure

```
kubo_bilayer/
├── setup/
│   ├── hamiltonians.py      BulkHamiltonian, InterfaceHamiltonian dataclasses
│   └── operators.py         ObservableCoeffs, ObservableOperator
├── numerics/
│   ├── poles.py             Companion-matrix pole finder (upper/lower half-plane)
│   └── residues.py          Residue matrices via SVD null vectors
└── greens/
    ├── bulk.py              evaluate(), coincidence_value(), coincidence_derivative()
    ├── interface.py         boundary_derivative(), assemble_G00()
    ├── halfspace.py         compute_G_bar_R(), compute_G_bar_L()
    └── bilayer.py           compute_G_bilayer() — full G(z, z') for all sign cases

showcases/
├── single_channel.py        Scalar benchmark system with analytic solution
├── orbitronics.py           p-orbital bilayer system (OAM transport showcase)
├── toy_trivial.py           Scalar toy model for unit tests
└── toy_degenerate.py        2×2 degenerate-pole toy model

scripts/
├── benchmark.py             Numeric vs. analytic comparison (single-channel)
├── plot_benchmark.py        Visual benchmark: G(z, z') vs. z for fixed z'
└── plot_orbitronics.py      Green's function plots for the orbitronics model

tests/
├── conftest.py              Shared fixtures and tolerances
├── test_poles.py
├── test_residues.py
├── test_bulk.py
├── test_interface.py
├── test_halfspace.py
└── test_bilayer.py

zp_weights.py                Scalar z'-integral weight factors (I–Vb)
zp_chains.py                 Matrix chain assembly for the response formula
zp_assembly.py               Full σ^ra_I integrand assembler (quadruple pole sum)
```

---

## Installation

The core package requires Python ≥ 3.10 with NumPy, SciPy, and Numba.

```bash
# Core + tests + linting
pip install -e ".[dev]"

# Core + tests + plotting
pip install -e ".[dev,plot]"
```

The `showcases/` and `scripts/` directories are not installed as packages.
Scripts add the project root to `sys.path` automatically, so they can be run
directly from anywhere:

```bash
python scripts/benchmark.py
python scripts/plot_benchmark.py -1 1.0
python scripts/plot_orbitronics.py --omega 0.0 --kx 0.2
```

---

## Running the Tests

```bash
pytest
```

All 83 tests should pass. Tolerances are defined in `tests/conftest.py`:

| Constant     | Value  | Used for                          |
|--------------|--------|-----------------------------------|
| `ATOL_STRICT`| 1e-10  | Exact known-answer tests          |
| `ATOL_APPROX`| 1e-5   | Approximate tests (η → 0 limit)   |
| `ETA`        | 1e-6   | Default broadening in tests       |

---

## Computational Pipeline

For a given `(kx, ky, omega, eta)` grid point, the pipeline runs as follows:

```python
from kubo_bilayer.numerics.poles    import compute_poles
from kubo_bilayer.numerics.residues import compute_residues
from kubo_bilayer.greens.bulk       import coincidence_value, coincidence_derivative
from kubo_bilayer.greens.interface  import boundary_derivative, assemble_G00
from kubo_bilayer.greens.bilayer    import compute_G_bilayer

# 1. Poles (upper half-plane for right side, lower for left)
poles_R, orders_R = compute_poles(ham_R, kx, ky, omega, eta, tol_filter=..., tol_cluster=..., halfplane='upper')
poles_L, orders_L = compute_poles(ham_L, kx, ky, omega, eta, tol_filter=..., tol_cluster=..., halfplane='lower')

# 2. Residue matrices
residues_R = compute_residues(ham_R, poles_R, orders_R, kx, ky, omega, eta, tol=...)
residues_L = compute_residues(ham_L, poles_L, orders_L, kx, ky, omega, eta, tol=...)

# 3. Boundary derivatives L_R, L_L
_, H1_R, H2_R = ham_R.hamiltonian_kz_polynomial(kx, ky)
_, H1_L, H2_L = ham_L.hamiltonian_kz_polynomial(kx, ky)
L_R = boundary_derivative(2*H2_R, H1_R, coincidence_derivative(poles_R, residues_R, 'upper'),
                                         coincidence_value(residues_R, 'upper'))
L_L = boundary_derivative(2*H2_L, H1_L, coincidence_derivative(poles_L, residues_L, 'lower'),
                                         coincidence_value(residues_L, 'lower'))

# 4. Interface and full bilayer Green's function
G00 = assemble_G00(h_int, L_R, L_L)
G   = compute_G_bilayer(z, z_prime, poles_R, residues_R, poles_L, residues_L, G00)
```

There are no package-level default tolerances. All tolerances must be supplied
explicitly at each call site.

---

## Key Design Decisions

- **Upper half-plane poles** belong to the right bulk Green's function (z > 0);
  **lower half-plane poles** belong to the left (z < 0). The contour closes
  clockwise in the lower half-plane, introducing a sign factor of −i instead of +i.
- **Real-axis poles** raise `ValueError` immediately.
- **Genuine second-order poles** raise `NotImplementedError`.
- **No imports between** `greens/` modules — all dependencies are injected.
- `assemble_G00` returns `inv(-h_int + L_R - L_L)` with no leading minus sign.
- The dagger relation G^r(z,z')† = G^a(z',z) gives the *advanced* Green's
  function and must never be used as a shortcut for the retarded one.

---

## Showcase: Orbitronics

`showcases/orbitronics.py` defines a p-orbital (L=1, 3×3) bilayer system
modelling orbital angular momentum transport. The bulk Hamiltonian includes
kinetic, crystal-field, and exchange terms; the interface Hamiltonian adds
Rashba-like and crystal-field symmetry breaking. Factory functions:

```python
from showcases.orbitronics import (
    make_bulk_hamiltonian,
    make_interface_hamiltonian,
    make_orbital_current_observable,
)
```

Plots are generated by `scripts/plot_orbitronics.py`. See its `--help` for
all CLI options.

---

## Response Formula

The Fermi-surface contribution σ^ra_I(z) to the spatially resolved linear
response is assembled in three layers:

| Module          | Responsibility                                              |
|-----------------|-------------------------------------------------------------|
| `zp_weights.py` | Ten scalar weight factors (I–Vb) from the z'-integral       |
| `zp_chains.py`  | Matrix chain `ResαR · GR(0)⁻¹ · G(0,0) · GL(0)⁻¹ · ResαL` |
| `zp_assembly.py`| Full quadruple pole sum; returns the trace for one grid point|

The 1/(64π³) prefactor and the k∥ integral are applied by the higher-level
driver (not yet implemented).

---

## References

- L. C. Rodenkirchen, *A Continuum Linear Response Model for Transport Phenomena
  in Planar Interface Bilayer Systems*, Master's Thesis, RWTH Aachen, March 2026.