""" toy_double_pole.py
    ------------------
    A trivial, non-physical scalar Hamiltonian with a known double pole
    for unit testing the pole clustering and order detection machinery
    in kubo_bilayer.numerics.poles.

    System
    ------
    ...

    Analytic Solution
    -----------------
    ...

    Expected Clustering Behaviour
    ------------------------------
    ...

    Usage
    -----
    The factory function make_double_pole_hamiltonian() returns a
    BulkHamiltonian instance ready to be passed directly to
    build_companion_matrices() and compute_poles().

        from showcases.toy_double_pole import make_double_pole_hamiltonian
        hamiltonian = make_double_pole_hamiltonian()

    Notes
    -----
    - All in-plane momenta kx, ky should be set to 0. when using
    this Hamiltonian, as all in-plane coefficients are zero and
    nonzero values have no physical meaning here.
    - A small eta > 0 is required by build_companion_matrices().
    With eta > 0 the double pole splits into two distinct but
    close eigenvalues, which is the numerically realistic scenario
    that cluster_poles() is designed to handle.
    - The exact splitting distance depends on eta — smaller eta
    gives closer eigenvalues and tests the clustering tolerance
    more stringently.

    Dependencies
    ------------
        BulkCoeffs, BulkHamiltonian  — from kubo_bilayer.setup.hamiltonians
"""
import numpy as np
from kubo_bilayer.setup.hamiltonians import BulkCoeffs, BulkHamiltonian
# TODO: genuine second-order pole toy model

#def make_double_pole_hamiltonian() -> BulkHamiltonian:
    