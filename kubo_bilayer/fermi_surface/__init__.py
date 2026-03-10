"""
fermi_surface/
--------------
Analytical evaluation of the Fermi-surface contribution to the
Streda decomposition of the spatially resolved linear response σ_A(z).

Physical Background
-------------------
In the DC limit and at T=0, the Fermi-surface contribution to the
response of observable current density σ̂^β_A(z) to a perturbation
B̂^λ(z') on the other side of the interface is made up of two parts
σ^ra_I(z) and σ^rr_I(z), where

    σ^ra_I(z) = ∫ d²k∥ / (64π³)
                · tr[ 4{A,Wβ}·(I)
                     + 2i{A,Wβ}·(IIa - IIb)
                     + 2i{A,Vβ}·(IIIa - IIIb)
                     -   {A,Vβ}·(IVa + IVb - Va - Vb) ]
and
    σ^rr_I(z) #TODO: To be added 

(I)–(Vb) are matrix-valued quantities, each a quadruple sum
over pole indices (αR, α'R, αL, α'L):

    (I) = Σ_{αR,α'R,αL,α'L}  w_I(kαL, kα'L)
              · e^{i(kαR - k*α'R)z}
              · Chain(αR, αL) · Wλ · Chain(α'R, α'L)†

and analogously for (IIa)–(Vb) with weights from zp_weights.py
and operator matrices Wλ or Vλ from the perturbation velocity.

The key structural observation is that every term shares the same
matrix building block:

    Chain(αR, αL) = ResαR · G_R(0)⁻¹ · G(0,0) · G_L(0)⁻¹ · ResαL

which depends on the pole indices but not on which term (I)–(Vb)
is being computed. This is precomputed in zp_chains.py.
The complex conjugate is performed later in assembly.

Module Structure
----------------
The computation is split across three modules:

zp_weights.py
    Scalar weight factors for terms (I)–(Vb), arising from the
    analytical evaluation of the z'-integral:

        ∫_{-∞}^{0} dz' e^{i(k*_{α'L} - k_{αL}) z'} = i / (kαL - k*_{α'L})

    Each weight is a function of one or two pole values only — no
    matrix content. All ten weight functions share a common helper
    _zp_building_blocks() that precomputes the scalar denominators.

zp_chains.py
    Construction of the matrix chain

        Chain(αR, αL) = ResαR · G_R(0)⁻¹ · G(0,0) · G_L(0)⁻¹ · ResαL

    which is the matrix skeleton shared by all ten terms. The middle
    factor G_R(0)⁻¹ · G(0,0) · G_L(0)⁻¹ is independent of the pole
    indices and is precomputed once per (kx, ky) point. Each chain is
    then assembled by two matrix multiplications:

        Chain(αR, αL) = ResαR · middle · ResαL

    The conjugate transpose Chain(α'R, α'L)† needed for the advanced
    GF side follows immediately as Chain(α'R, α'L).conj().T.

zp_assembly.py
    The quadruple sum over (αR, α'R, αL, α'L) that assembles the
    ten terms (I)–(Vb) from the scalar weights, phase factors
    e^{i(kαR - k*α'R)z}, operator matrices, and chain products.
    Returns the trace integrand at a fixed (z, k∥) point, ready
    for the k∥ quadrature in integrals/.

Dependencies
------------
    kubo_bilayer.numerics.poles      — provides poles {kαL}, {kαR}
    kubo_bilayer.numerics.residues   — provides residues {ResαL}, {ResαR}
    kubo_bilayer.greens.bulk         — provides G_L(0), G_R(0)
    kubo_bilayer.greens.interface    — provides G(0,0)
    kubo_bilayer.setup.operators     — provides {A,Wβ}, {A,Vβ}
    kubo_bilayer.setup.hamiltonians  — provides Wλ, Vλ via
                                       velocity_kz_polynomial()
"""