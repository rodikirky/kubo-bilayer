from __future__ import annotations

"""
kubo: Numerical Kubo observable computation.

Main modules:
- config: configs and dataclasses
- grids: k, z, omega grids
- greens: Green's-function kernels
- gluing: interface / bilayer gluing
- functional_trace: assembling traces
- streda: σ_I^A / σ_II^A etc.
- integrate: numerical integration helpers
- io: saving / loading results
- logging_utils: logging helpers
- models: concrete Hamiltonians (toy, orbitronic, ...) and operators (velocity, angular momentum current, ...)
"""

from . import (  # noqa: F401
    config,
    grids,
    greens,
    gluing,
    functional_trace,
    streda,
    integrate,
    io,
    logging_utils,
    models,
)

from .config import GridConfig, PhysicsConfig, ModelConfig, KuboConfig  # noqa: F401

__all__ = [
    "config",
    "grids",
    "greens",
    "gluing",
    "functional_trace",
    "streda",
    "integrate",
    "io",
    "logging_utils",
    "models",
    "GridConfig",
    "PhysicsConfig",
    "ModelConfig",
    "KuboConfig",
]
