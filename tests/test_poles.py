from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from setup.hamiltonians import BulkHamiltonian
from kubo_bilayer.numerics.poles import ArrayC, build_companion_matrices

def test_companion_shape():
    # for an n×n Hamiltonian, L and M should be 2n×2n
    L, M = build_companion_matrices(hamiltonian, kx=0., ky=0., omega=0., eta=0.1)
    n = hamiltonian.matrix_dim
    assert L.shape == (2*n, 2*n)
    assert M.shape == (2*n, 2*n)