from __future__ import annotations

from pathlib import Path

from kubo import KuboConfig, PhysicsConfig
from kubo.logging_utils import setup_logging
from kubo.models.toy import toy_bulk_hamiltonian
from kubo.greens import bulk_greens_retarded


def main() -> None:
    logs_dir = Path("logs")
    logger = setup_logging(level="INFO", log_file=str(logs_dir / "streda_dc_demo.log"))

    logger.info("Starting Streda-DC demo run (toy model).")

    cfg = KuboConfig(physics=PhysicsConfig(eta=1e-3))

    omega = 0.5
    kx = ky = kz = 0.1
    G = bulk_greens_retarded(omega, kx, ky, kz, toy_bulk_hamiltonian, cfg.physics)

    logger.info("Computed G^R(ω,k) for toy model with shape %s", G.shape)
    logger.info("Run finished successfully.")


if __name__ == "__main__":
    main()