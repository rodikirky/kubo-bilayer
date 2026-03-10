"""
analytics/
----------
Analytical construction of the interfacial retarded Green's function
and the spatially resolved linear response, following the derivation
in Appendix A of the thesis.

Module Structure and Dependency Chain
--------------------------------------
The computation proceeds in four stages, each building on the previous:

    bulk.py  →  interface.py  →  bilayer.py  →  zp_integral.py

bulk.py
    Constructs the bulk retarded Green's function on each side
    of the interface from the poles and residues computed in
    kubo_bilayer.numerics:

        G^r_{L/R}(Δz) = i · Σ_α e^(i·kα·Δz) · Res_α

    and its coincidence value G^r_{L/R}(0).

interface.py
    Constructs the coincidence value G(0,0) by gluing the two
    half-space Green's functions at the interface, incorporating
    the interface Hamiltonian H_int:

        G(0,0) = -[H_int + L^r_R - L^r_L]⁻¹

    where L^r_{L/R} are the boundary derivatives of the bulk
    Green's functions.

halfspace.py
    Constructs the half-space Green's functions Ḡ^r_L and Ḡ^r_R,
    which describe propagation within a single half-space and
    contribute to the full interfacial Green's function when
    perturbation and observable are on the same side of the
    interface. Concretely:

        Ḡ^r_L(z,z') = G^r_L(z-z') · Θ(-z) · Θ(-z')
        Ḡ^r_R(z,z') = G^r_R(z-z') · Θ(z)  · Θ(z')

    These contributions vanish identically for z > 0, z' < 0,
    which is the cross-interface configuration studied in the
    OAM showcase. They are included here for completeness and
    to make the framework applicable to same-side response
    calculations in future applications.
    
bilayer.py
    Assembles the full interfacial retarded Green's function
    for z > 0 and z' < 0:

        G^r(z,z') = Σ_αR e^(i·kαR·z) ResαR · G_R(0)⁻¹
                    · G(0,0) · G_L(0)⁻¹
                    · Σ_αL e^(-i·kαL·z') ResαL

Dependencies
------------
    kubo_bilayer.numerics.poles    — provides poles {kα}
    kubo_bilayer.numerics.residues — provides residues {Res_α}
    kubo_bilayer.setup.hamiltonians — provides BulkHamiltonian,
                                      InterfaceHamiltonian
"""