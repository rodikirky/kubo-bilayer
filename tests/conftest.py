# tests/conftest.py
import numpy as np
import pytest

from kubo.config import PhysicsConfig, GridConfig
from kubo.models.toy import ToyBulk, ToyBulkParams
from kubo.models.orbitronic import OrbitronicBulk, OrbitronicBulkParams

@pytest.fixture
def physics_default() -> PhysicsConfig:
    return PhysicsConfig(eta=1e-3)

@pytest.fixture
def physics_small_eta() -> PhysicsConfig:
    # slightly larger than 1e-12 to avoid brittle floating-point edge cases
    return PhysicsConfig(eta=1e-6)

@pytest.fixture
def toy_bulk_default() -> ToyBulk:
    # mass = gap = 1.0
    return ToyBulk.from_params(ToyBulkParams())

@pytest.fixture
def toy_H_default(toy_bulk_default: ToyBulk):
    # Adapt ToyBulk.hamiltonian(k=(kx,ky,kz)) to BulkHamiltonian(kx,ky,kz)
    return lambda kx, ky, kz: toy_bulk_default.hamiltonian((kx, ky, kz))

@pytest.fixture
def toy_bulk_spec() -> ToyBulk:
    return ToyBulk.from_params(ToyBulkParams(mass=1.5, gap=0.7))

@pytest.fixture
def toy_H_spec(toy_bulk_spec: ToyBulk):
    # Adapt ToyBulk.hamiltonian(k=(kx,ky,kz)) to BulkHamiltonian(kx,ky,kz)
    return lambda kx, ky, kz: toy_bulk_spec.hamiltonian((kx, ky, kz))


@pytest.fixture
def k_point_simple():
    # a deterministic k-point used across tests
    return (0.3, -0.2, 0.5)

@pytest.fixture
def kpar_simple():
    # for just kx, ky
    return (0.2, -0.1)

@pytest.fixture
def grid_small_fft() -> GridConfig:
    return GridConfig(nz=31, z_max=10.0, nk_parallel=5, nomega=5)

@pytest.fixture
def grid_tiny_fft() -> GridConfig:
    # ultra-fast canary grid (odd nz so z=0 at mid)
    return GridConfig(nz=17, z_max=6.0, nk_parallel=3, nomega=3)

# ADD: orbitronic defaults
@pytest.fixture
def orbitronic_params_default() -> OrbitronicBulkParams:
    return OrbitronicBulkParams(
        mass=1.3,
        gamma=0.7,
        J=0.4,
        magnetisation=[0.2, -0.1, 0.97],
    )

@pytest.fixture
def orbitronic_bulk_default(orbitronic_params_default: OrbitronicBulkParams) -> OrbitronicBulk:
    return OrbitronicBulk.from_params(orbitronic_params_default, basis=None)

@pytest.fixture
def orbitronic_H_default(orbitronic_bulk_default: OrbitronicBulk):
    return lambda kx, ky, kz: orbitronic_bulk_default.hamiltonian((kx, ky, kz))

# ADD (optional): a deterministic random unitary for basis-covariance tests
@pytest.fixture
def unitary_U3() -> np.ndarray:
    rng = np.random.default_rng(123)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    ph = np.exp(-1j * np.angle(np.diag(R)))
    return Q @ np.diag(ph)