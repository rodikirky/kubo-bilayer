from __future__ import annotations

import argparse
import numpy as np

from kubo.config import GridConfig, PhysicsConfig
from kubo.models.toy import ToyBulk, ToyBulkParams
from kubo.greens import realspace_greens_retarded_with_kz

from kubo.plotting import (
    profile_amplitude_over_first_axis,
    edge_leak_ratio,
    plot_profile,
    plot_complex_components,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--omega", type=float, default=0.0)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--kx", type=float, default=0.0)
    p.add_argument("--ky", type=float, default=0.0)

    p.add_argument("--nz", type=int, default=513)
    p.add_argument("--z-max", type=float, default=50.0)

    # The rest are required by your GridConfig even if unused here.
    p.add_argument("--k-max", type=float, default=5.0)
    p.add_argument("--nk-parallel", type=int, default=1)
    p.add_argument("--nphi", type=int, default=1)
    p.add_argument("--omega-max", type=float, default=1.0)
    p.add_argument("--nomega", type=int, default=1)
    return p.parse_args()
