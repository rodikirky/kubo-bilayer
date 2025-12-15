# tests/conftest.py
import numpy as np
import pytest

from kubo.config import PhysicsConfig, GridConfig
from kubo.models.toy import ToyBulk, ToyBulkParams


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
