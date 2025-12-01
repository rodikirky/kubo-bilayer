import numpy as np

from kubo.config import PhysicsConfig
from kubo.models.toy import toy_bulk_hamiltonian
from kubo.greens import bulk_greens_retarded


def main() -> None:
    physics = PhysicsConfig(eta=1e-3)
    omega = 0.5
    kx = ky = kz = 0.1

    G = bulk_greens_retarded(omega, kx, ky, kz, toy_bulk_hamiltonian, physics)
    print("G^R(ω,k) for toy model:")
    print(G)


if __name__ == "__main__":
    main()