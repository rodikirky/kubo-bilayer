"""
bulk.py
-------
Construction of the bulk retarded Green's function in real space
from the poles and residues computed in kubo_bilayer.numerics.

Physical Background
-------------------
In a translation invariant bulk system, the retarded Green's
function in k-space is:

    G̃^r(kz) = [(ω + i·eta)·I - H(kx, ky, kz)]⁻¹

Its real-space form is obtained via contour integration over kz,
closing in the upper half-plane for Δz > 0. By the residue
theorem, this yields a sum over the poles {kα} in the upper
half-plane:

    G^r(Δz) = i · Σ_α e^(i·kα·Δz) · Res_α

where Res_α = Res(G̃^r; kα) are the n×n residue matrices
computed in kubo_bilayer.numerics.residues.

Functions
---------
evaluate(Δz, poles, residues)
    Evaluates G^r(Δz) at a given Δz > 0 as the sum:

        G^r(Δz) = i · Σ_α e^(i·kα·Δz) · Res_α

    This is the building block for both the half-space Green's
    function in halfspace.py and the interface coincidence value
    in interface.py.

coincidence_value(poles, residues)
    Returns the coincidence value G^r(Δz=0):

        G^r(0) = i · Σ_α Res_α

    This is a special case of evaluate() at Δz=0, but is used
    frequently enough to warrant its own function. It appears
    directly in the gluing formula for G(0,0) in interface.py.

boundary_derivative(poles, residues)
    Returns the boundary derivative L^r = 1/2 · Az · ∂_Δz G^r(Δz)|_{Δz=0}:

        L^r = 1/2 · Az · i · Σ_α i·kα · Res_α
            = -1/2 · Az · Σ_α kα · Res_α

    where Az is the kz² coefficient matrix of the bulk Hamiltonian.
    This quantity appears in the gluing formula for G(0,0) in
    interface.py.

Notes
-----
- Δz must be strictly positive for the sum to converge, since
  Im(kα) > 0 for all poles in the upper half-plane ensures
  exponential decay.
- The poles and residues passed here must have been computed at
  the same (kx, ky, ω, eta) as will be used in the response
  calculation.
- This module is agnostic to whether it is computing the left
  or right bulk Green's function — the distinction is carried
  entirely by which poles and residues are passed in.

Dependencies
------------
    kubo_bilayer.numerics.poles    — provides poles {kα}
    kubo_bilayer.numerics.residues — provides residues {Res_α}
"""