""" bilayer.py
    ----------
    Construction of the full cross-interface retarded Green's function
    for z > 0, z' < 0, by combining the propagation factors F^r_R(z)
    and F̄^r_L(z') with the coincidence value G(0,0) from interface.py.
    #TODO: All in one half-space extension to be added once halspace.py is implemented.

    Physical Background
    -------------------
    For z > 0 and z' < 0, the full interfacial retarded Green's
    function factorizes into three terms (eq. 54 in supervisor's notes):

        G^r(z,z') = F^r_R(z) · G(0,0) · F̄^r_L(z')

    where the propagation factors are defined as (eqs. 52, 53):

        F^r_R(z)  = -G^(0)r_R(z,0) · [G^(0)r_R(0,0)]^{-1}
        F̄^r_L(z') = -[G^(0)r_L(0,0)]^{-1} · G^(0)r_L(0,z')

    and G^(0)r_{R/L}(z,0) are the translational invariant bulk
    Green's functions evaluated via bulk.py.

    Propagation Factors
    -------------------
    Expanding in terms of poles and residues:

        F^r_R(z)  = -[Σ_α e^(i·kαR·z) · ResαR] · [G^r_R(0)]^{-1}
                = -(1/i) · evaluate(z, poles_R, residues_R, 'upper')
                        · [coincidence_value(residues_R, 'upper')]^{-1}

        F̄^r_L(z') = -[G^r_L(0)]^{-1} · [Σ_α e^(i·kαL·z') · ResαL]
                = -(1/i) · [coincidence_value(residues_L, 'lower')]^{-1}
                        · evaluate(z', poles_L, residues_L, 'lower') · (-1)

    Note the sign conventions:
        - F^r_R(z)  requires z  > 0, upper half-plane poles
        - F̄^r_L(z') requires z' < 0, lower half-plane poles
        - Both carry a leading minus sign from eqs. 52, 53

    Functions
    ---------
    compute_F_R(z, poles_R, residues_R)
        Computes the right propagation factor F^r_R(z) for z > 0:

            F^r_R(z) = -G^(0)r_R(z,0) · [G^(0)r_R(0,0)]^{-1}

    compute_F_bar_L(z_prime, poles_L, residues_L)
        Computes the left propagation factor F̄^r_L(z') for z' < 0:

            F̄^r_L(z') = -[G^(0)r_L(0,0)]^{-1} · G^(0)r_L(0,z')

    compute_G_bilayer(z, z_prime, poles_R, residues_R,
                    poles_L, residues_L, G00)
        Assembles the full cross-interface Green's function:

            G^r(z,z') = F^r_R(z) · G(0,0) · F̄^r_L(z')

        for z > 0, z' < 0.

    Notes
    -----
    - z must be strictly positive and z' must be strictly negative.
    These constraints are enforced by evaluate() in bulk.py.
    - The half-space contributions Ḡ^r_L and Ḡ^r_R vanish for
    z > 0, z' < 0 and are therefore not included here. See
    halfspace.py for their implementation (TODO).
    - G00 must have been computed at the same (kx, ky, omega, eta)
    as the poles and residues passed here.

    Dependencies
    ------------
        bulk.py      — provides evaluate() and coincidence_value()
        interface.py — provides assemble_G00()

    Usage
    ------
    - Plot the full spatial Green's function
    - Provide a pedagogical illustration of the matching method

# TODO: Implement compute_F_R, compute_F_bar_L, compute_G_bilayer
"""

# Placeholder — not implemented.
# See module docstring for details.