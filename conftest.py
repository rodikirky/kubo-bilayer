import sys
import os

import pytest
import numpy as np
from showcases.toy_trivial import make_scalar_hamiltonian
from showcases.toy_degenerate import make_degenerate_pole_hamiltonian


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture
def scalar_hamiltonian():
    return make_scalar_hamiltonian()

@pytest.fixture
def degenerate_hamiltonian():
    return make_degenerate_pole_hamiltonian()

@pytest.fixture
def expected_upper_pole():
    return -0.5 + 1j * np.sqrt(3) / 2

@pytest.fixture
def expected_lower_pole():
    return -0.5 - 1j * np.sqrt(3) / 2

@pytest.fixture
def expected_degenerate_pole():
    return 1. + 0j

#TODO: Transfer these to config.yaml once in existence:

# Numerical tolerances
ATOL_STRICT = 1e-10    # exact known-answer tests e.g. companion matrices
ATOL_APPROX = 1e-5     # approximate tests e.g. pole locations with eta→0

# Default numerical parameters for tests
ETA = 1e-6             # broadening, small enough to approximate eta→0
OMEGA = 0.             # default frequency
OMEGA_DEGENERATE = 1.  # omega for degenerate pole tests where kz=0 must be avoided