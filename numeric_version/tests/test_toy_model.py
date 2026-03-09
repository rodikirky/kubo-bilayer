import numpy as np
import pytest

from kubo.models.toy import (  
    ToyBulkParams,
    ToyInterfaceParams,
    ToyBulk,
    ToyInterface,
)

# -------------------------------------------------------
# region Params 
# -------------------------------------------------------

def test_toy_bulk_params_defaults():
    params = ToyBulkParams()
    assert params.mass == 1.0
    assert params.gap == 1.0


def test_toy_interface_params_defaults():
    params = ToyInterfaceParams()
    assert params.strength == 0.0


def test_toy_bulk_from_params_copies_values():
    params = ToyBulkParams(mass=2.5, gap=0.7)
    bulk = ToyBulk.from_params(params)
    assert bulk.mass == pytest.approx(2.5)
    assert bulk.gap == pytest.approx(0.7)

def test_toy_bulk_from_params_raises_at_zero_mass():
    params = ToyBulkParams(mass=1e-13, gap=0.7)
    with pytest.raises(ValueError):
        bulk = ToyBulk.from_params(params)


def test_toy_interface_from_params_copies_values():
    params = ToyInterfaceParams(strength=3.14)
    interface = ToyInterface.from_params(params)
    assert interface.strength == pytest.approx(3.14)

# endregion


# -------------------------------------------------------
# region ToyBulk 
# -------------------------------------------------------

def test_toy_bulk_identity_and_sigma_z():
    bulk = ToyBulk(mass=1.0, gap=1.0)

    I = bulk.identity
    sz = bulk.sigma_z

    assert I.shape == (2, 2)
    assert sz.shape == (2, 2)
    assert I.dtype == np.complex128
    assert sz.dtype == np.complex128

    np.testing.assert_allclose(I, np.eye(2, dtype=np.complex128))
    np.testing.assert_allclose(sz, np.array([[1, 0], [0, -1]], dtype=np.complex128))


@pytest.mark.parametrize(
    "k, mass, gap",
    [
        ((0.0, 0.0, 0.0), 1.0, 1.7),          # k=0 pure gap case
        ((1.0, 0.0, 0.0), 1.0, 1.0),
        ((0.0, 2.0, 0.0), 2.0, 0.5),
        ((1.0, 2.0, 2.0), 0.5, 1.2),
    ],
)
def test_toy_bulk_hamiltonian_diagonal_form(k, mass, gap):
    bulk = ToyBulk(mass=mass, gap=gap)
    H = bulk.hamiltonian(k)

    kx, ky, kz = k
    k2 = kx*kx + ky*ky + kz*kz
    ek = k2 / (2.0 * mass)

    expected = np.array([[ek + gap, 0.0], [0.0, ek - gap]], dtype=np.complex128)
    np.testing.assert_allclose(H, expected)

@pytest.mark.parametrize(
    "k, mass, gap",
    [
        ((0.1, 0.2, 0.3), 1.0, 1.0),
        ((-1.0, 2.0, 0.5), -2.0, 0.3),   # negative mass allowed by code
        ((10.0, -5.0, 3.0), 0.7, -1.1),  # negative gap
    ],
)
def test_toy_bulk_hamiltonian_is_hermitian(k, mass, gap):
    bulk = ToyBulk(mass=mass, gap=gap)
    H = bulk.hamiltonian(k)
    np.testing.assert_allclose(H, H.conj().T)

def test_toy_bulk_zero_gap_gives_degenerate_bands():
    bulk = ToyBulk(mass=1.0, gap=0.0)
    k = (1.0, 2.0, 3.0)

    H = bulk.hamiltonian(k)
    eigvals = np.linalg.eigvalsh(H)

    # Both eigenvalues equal ek = k^2 / (2m)
    kx, ky, kz = k
    k2 = kx * kx + ky * ky + kz * kz
    ek = k2 / 2.0

    assert eigvals.shape == (2,)
    np.testing.assert_allclose(eigvals, [ek, ek])

@pytest.mark.parametrize("bad_k", [
    (0.0,),           # too short
    (0.0, 1.0),       # too short
    (0.0, 1.0, 2.0, 3.0),  # too long
])
def test_toy_bulk_hamiltonian_raises_on_bad_k_length(bad_k):
    bulk = ToyBulk(mass=1.0, gap=1.0)
    with pytest.raises(ValueError):
        # ValueError from tuple unpacking in hamiltonian
        bulk.hamiltonian(bad_k)

def test_toy_bulk_hamiltonian_finite_for_large_k():
    # just a sanity check to ensure no overflow occurs
    bulk = ToyBulk(mass=1.0, gap=1.0)
    k = (1e3, -2e3, 3e3)  # still safe for float64

    H = bulk.hamiltonian(k)
    assert np.isfinite(H).all()

def test_toy_bulk_rejects_zero_or_tiny_mass():
    # exact zero
    with pytest.raises(ValueError):
        ToyBulk(mass=0.0, gap=1.0)

    # below threshold
    with pytest.raises(ValueError):
        ToyBulk(mass=1e-13, gap=1.0)


def test_toy_bulk_accepts_mass_above_threshold():
    # just above the gate
    bulk = ToyBulk(mass=2e-12, gap=1.0)
    assert bulk.mass == pytest.approx(2e-12)

# endregion

# -------------------------------------------------------
# region ToyInterface  
# -------------------------------------------------------

def test_toy_interface_hamiltonian_basic():
    interface = ToyInterface(strength=2.5)
    H = interface.hamiltonian(kx=0.1, ky=-0.2)

    expected = 2.5 * np.eye(2, dtype=np.complex128)
    np.testing.assert_allclose(H, expected)
    assert H.shape == (2, 2)
    assert H.dtype == np.complex128

@pytest.mark.parametrize(
    "strength",
    [0.0, 1.0, -3.5, 10.0],
)
def test_toy_interface_proportional_to_identity_for_any_strength(strength):
    interface = ToyInterface(strength=strength)

    for kx, ky in [(0.0, 0.0), (1.0, -1.0), (10.0, 5.0)]:
        H = interface.hamiltonian(kx, ky)
        expected = strength * np.eye(2, dtype=np.complex128)
        np.testing.assert_allclose(H, expected)
        np.testing.assert_allclose(H, H.conj().T)  # Hermitian

# endregion