import numpy as np

from kubo.config import PhysicsConfig
from kubo.greens import bulk_greens_retarded
from kubo.models.toy import toy_bulk_hamiltonian


def test_bulk_green_retarded_inverse_property():
    physics = PhysicsConfig(eta=1e-3)
    omega = 0.5
    kx = ky = kz = 0.1

    G = bulk_greens_retarded(omega, kx, ky, kz, toy_bulk_hamiltonian, physics)

    # Check that ((ω + iη)I - H) G ≈ I
    H = toy_bulk_hamiltonian(kx, ky, kz)
    dim = H.shape[0]
    mat = (omega + 1j * physics.eta) * np.eye(dim, dtype=complex) - H
    ident_approx = mat @ G

    assert ident_approx.shape == (dim, dim)
    assert np.allclose(ident_approx, np.eye(dim), atol=1e-8)
