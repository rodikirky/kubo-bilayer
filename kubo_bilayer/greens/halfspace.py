"""
halfspace.py
------------
Construction of the half-space Green's functions Ḡ^r_L and Ḡ^r_R,
describing propagation within a single half-space.

These contributions appear in the full interfacial Green's function:

    G^r(z,z') = Ḡ^r_L(z,z') + Ḡ^r_R(z,z') + F^r(z)·G(0,0)·F̄^r(z')

For z > 0 and z' < 0 (cross-interface configuration), both
Ḡ^r_L and Ḡ^r_R vanish identically, and only the
last term contributes. This is the configuration used in the
OAM showcase.

# TODO: Implement Ḡ^r_L and Ḡ^r_R for same-side response
# calculations. Requires:
#   - evaluate_halfspace(z, z', poles, residues, halfplane)
#   - Both upper and lower half-plane poles needed.
"""

# Placeholder — not implemented.
# See module docstring for details.