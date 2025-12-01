from __future__ import annotations

"""
Model library for kubo_numerics.

- toy: simple 2x2 bulk + interface Hamiltonians for testing
- orbitronic: 3x3 orbitronic bulk + interface Hamiltonians
"""

from . import toy, orbitronic  # noqa: F401

__all__ = ["toy", "orbitronic"]
