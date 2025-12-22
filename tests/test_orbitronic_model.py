# ---------------------------------------------
# Imports
# ---------------------------------------------
import numpy as np
import pytest

from kubo.models.orbitronic import (
    _as_real3, 
    _check_unitary, 
    canonical_L_matrices, 
    OrbitronicBulk, 
    OrbitronicBulkParams, 
    OrbitronicInterface, 
    OrbitronicInterfaceParams, 
)

# ---------------------------------------------------------------------
# region test helpers
# ---------------------------------------------------------------------
def test_as_real3_basic():
    v_list = [1.0, 2.0, 3.0]
    arr = _as_real3(v_list)
    assert arr.shape == (3,)
    assert arr.dtype == np.float64
    assert np.allclose(arr, np.array(v_list, dtype=float))

def test_as_real3_raises_on_wrong_length():
    with pytest.raises(ValueError):
        _ = _as_real3([1.0, 2.0, 3.0, 4.0])

def test_check_unitary_accepts_identity():
    U = np.eye(3, dtype=np.complex128)
    # Should not raise
    _check_unitary(U)


def test_check_unitary_accepts_diagonal_phases():
    phases = np.exp(1j * np.array([0.0, 0.3, 0.7]))
    U = np.diag(phases).astype(np.complex128)
    _check_unitary(U)  # no error


def test_check_unitary_rejects_non_unitary():
    U = np.ones((3, 3), dtype=np.complex128)  # clearly not unitary
    with pytest.raises(ValueError):
        _check_unitary(U)

def test_check_unitary_rejects_wrong_shape():
    U = np.eye(2, dtype=np.complex128)
    with pytest.raises(ValueError):
        _check_unitary(U)

def test_canonical_L_matrices_rotated_basis_preserves_commutation():
    # Simple unitary: diagonal phases
    phases = np.exp(1j * np.array([0.1, -0.2, 0.4]))
    U = np.diag(phases).astype(np.complex128)

    Lx_p, Ly_p, Lz_p = canonical_L_matrices(U)

    # Still Hermitian
    for L in (Lx_p, Ly_p, Lz_p):
        assert np.allclose(L.conj().T, L)

    # Still satisfy su(2) commutation (angular momentum algebra)
    comm_xy = Lx_p @ Ly_p - Ly_p @ Lx_p
    comm_yz = Ly_p @ Lz_p - Lz_p @ Ly_p
    comm_zx = Lz_p @ Lx_p - Lx_p @ Lz_p

    assert np.allclose(comm_xy, 1j * Lz_p, atol=1e-12)
    assert np.allclose(comm_yz, 1j * Lx_p, atol=1e-12)
    assert np.allclose(comm_zx, 1j * Ly_p, atol=1e-12)
# endregion
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# region test OrbitronicBulk
# ---------------------------------------------------------------------
def test_orbitronic_bulk_from_params_positive_mass_and_shape():
    params = OrbitronicBulkParams(
        mass=2.0,
        gamma=0.5,
        J=0.3,
        magnetisation=[0.1, 0.2, 0.3],
    )
    bulk = OrbitronicBulk.from_params(params=params)
    assert isinstance(bulk.mass, float)
    assert bulk.mass == pytest.approx(2.0)
    assert bulk.gamma == pytest.approx(0.5)
    assert bulk.J == pytest.approx(0.3)

    # magnetisation is a real 3-vector
    assert bulk.magnetisation.shape == (3,)
    assert bulk.magnetisation.dtype == np.float64

    # L is a triple of 3x3 matrices
    assert len(bulk.L) == 3
    for L in bulk.L:
        assert L.shape == (3, 3)

def test_orbitronic_bulk_from_params_rejects_non_positive_mass():
    with pytest.raises(ValueError):
        params= OrbitronicBulkParams(mass=-1.0,
            gamma=0.5,
            J=0.3,
            magnetisation=_as_real3([0.0, 0.0, 1.0]))
        bulk = OrbitronicBulk.from_params(params)

def test_orbitronic_bulk_potential_k_zero_reduces_to_J_M_dot_L():
    gamma = 0.7
    J = 0.4
    M = np.array([0.1, -0.2, 0.3])
    bulk = OrbitronicBulk(
        mass=1.0,
        gamma=gamma,
        J=J,
        magnetisation=M,
    )
    Lx, Ly, Lz = bulk.L
    ML = M[0] * Lx + M[1] * Ly + M[2] * Lz

    V0 = bulk.potential([0.0, 0.0, 0.0])

    assert np.allclose(V0, J * ML)


def test_orbitronic_bulk_potential_zero_when_gamma_and_J_zero():
    bulk = OrbitronicBulk(
        mass=1.0,
        gamma=0.0,
        J=0.0,
        magnetisation=_as_real3([1.0, 0.0, 0.0]),
    )
    k = np.array([0.1, 0.2, 0.3])
    V = bulk.potential(k)
    H = bulk.hamiltonian(k) # pure kinetic
    k2 = float(np.dot(k, k))
    expected = (k2 / (2.0 * bulk.mass)) * np.eye(3, dtype=np.complex128)

    assert np.allclose(H, expected)
    assert np.allclose(V, np.zeros((3, 3), dtype=np.complex128))


def test_orbitronic_bulk_hamiltonian_allows_neg_gamma_and_J():
    bulk = OrbitronicBulk(
        mass=1.5,
        gamma=-0.8,
        J=-0.2,
        magnetisation=_as_real3([0.1, 0.2, 0.3]),
    )
    k = [0.4, -0.1, 0.2]
    H = bulk.hamiltonian(k) 
    

def test_orbitronic_bulk_hamiltonian_is_hermitian():
    bulk = OrbitronicBulk(
        mass=1.5,
        gamma=0.8,
        J=0.2,
        magnetisation=_as_real3([0.1, 0.2, 0.3]),
    )
    k = [0.4, -0.1, 0.2]
    H = bulk.hamiltonian(k)

    assert H.shape == (3, 3)
    assert np.allclose(H.conj().T, H)

def test_velocity_components_shape_and_hermiticity():
    bulk = OrbitronicBulk(
        mass=1.3,
        gamma=0.0,
        J=0.0,
        magnetisation=_as_real3([0.0, 0.0, 1.0]),
    )
    k = _as_real3([0.2, 0.3, -0.1])
    v_x, v_y, v_z = bulk.velocity_components(k)
    # expectation: v_i(k) = (1/ħ) ∂H/∂k_i.
    hamilonian = bulk.hamiltonian(k)
    for i,v in enumerate([v_x, v_y, v_z]):
        assert v.shape == (3, 3)
        assert np.allclose(v.conj().T, v) # Hermitian
        assert np.allclose(v[0,0], k[i]/bulk.mass) # expected value for free particle

@pytest.mark.parametrize("direction", ["x", "y", "z"])
def test_velocity_components_match_finite_difference(direction):
    bulk = OrbitronicBulk(
        mass=1.0,
        gamma=0.5,
        J=0.2,
        magnetisation=_as_real3([0.1, 0.0, 0.3]),
    )
    hbar = 1.0
    k = _as_real3([0.4, -0.2, 0.1])
    eps = 1e-5

    vx, vy, vz = bulk.velocity_components(k, hbar=hbar)
    v_map = {"x": vx, "y": vy, "z": vz}
    v_analytic = v_map[direction]

    # direction unit vector
    ei = {"x": np.array([1.0, 0.0, 0.0]),
          "y": np.array([0.0, 1.0, 0.0]),
          "z": np.array([0.0, 0.0, 1.0])}[direction]

    # finite difference
    H_plus = bulk.hamiltonian(k + eps * ei)
    H_minus = bulk.hamiltonian(k - eps * ei)
    v_fd = (H_plus - H_minus) / (2 * eps * hbar)

    assert np.allclose(v_analytic, v_fd, atol=1e-6)

@pytest.mark.parametrize("flow_dir", ["x", "y", "z"])
@pytest.mark.parametrize("L_comp", ["x", "y", "z"])
def test_orbital_current_operator_basic(flow_dir, L_comp):
    bulk = OrbitronicBulk(
        mass=1.0,
        gamma=0.4,
        J=0.3,
        magnetisation=_as_real3([0.1, 0.2, 0.0]),
    )
    k = [0.3, -0.2, 0.5]

    J_op = bulk.orbital_current_operator(k, flow_dir=flow_dir, L_comp=L_comp)

    assert J_op.shape == (3, 3)
    # should be Hermitian
    assert np.allclose(J_op.conj().T, J_op)


def test_orbital_current_operator_zero_at_k_zero():
    bulk = OrbitronicBulk(
        mass=1.0,
        gamma=0.4,
        J=0.3,
        magnetisation=_as_real3([0.1, 0.2, 0.0]),
    )
    k_zero = [0.0, 1.0, 0.0] # non-flow direction and non-L direction can be non-zero

    J_op = bulk.orbital_current_operator(k_zero, flow_dir="x", L_comp="z")
    assert np.allclose(J_op, np.zeros((3, 3), dtype=np.complex128), atol=1e-12)

# endregion
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# region test OrbitronicInterface
# ---------------------------------------------------------------------
def test_orbitronic_interface_from_params_shapes():
    interface = OrbitronicInterface.from_params(
        m_int=1.2,
        gamma_int=0.5,
        alpha=0.3,
        beta=0.1,
        delta_CF=0.8,
        L0=2.0,
    )

    assert interface.m_int == pytest.approx(1.2)
    assert len(interface.L) == 3
    for L in interface.L:
        assert L.shape == (3, 3)

def test_orbitronic_interface_hamiltonian_is_hermitian():
    interface = OrbitronicInterface.from_params(
        m_int=1.0,
        gamma_int=0.7,
        alpha=0.2,
        beta=0.1,
        delta_CF=0.6,
        L0=1.5,
    )
    H_int = interface.hamiltonian(0.2, -0.3)

    assert H_int.shape == (3, 3)
    assert np.allclose(H_int.conj().T, H_int)


def test_orbitronic_interface_hamiltonian_special_case_gamma_alpha_beta_zero():
    m_int = 1.4
    delta_CF = 0.9
    L0 = 1.7

    interface = OrbitronicInterface.from_params(
        m_int=m_int,
        gamma_int=0.0,
        alpha=0.0,
        beta=0.0,
        delta_CF=delta_CF,
        L0=L0,
    )

    kx, ky = 0.3, -0.4
    k_par_sq = kx * kx + ky * ky

    _, _, Lz = interface.L
    I = np.eye(3, dtype=np.complex128)
    Lz2 = Lz @ Lz

    expected = L0 * (
        (k_par_sq / (2.0 * m_int)) * I
        + delta_CF * (I - Lz2)
    )

    H_int = interface.hamiltonian(kx, ky)

    assert np.allclose(H_int, expected)


def test_orbitronic_interface_hamiltonian_at_k_zero():
    m_int = 1.1
    gamma_int = 0.2
    alpha = 0.3
    beta = 0.4
    delta_CF = 0.5
    L0 = 1.0

    interface = OrbitronicInterface.from_params(
        m_int=m_int,
        gamma_int=gamma_int,
        alpha=alpha,
        beta=beta,
        delta_CF=delta_CF,
        L0=L0,
    )

    H0 = interface.hamiltonian(0.0, 0.0)

    _, _, Lz = interface.L
    I = np.eye(3, dtype=np.complex128)
    Lz2 = Lz @ Lz
    expected = L0 * delta_CF * (I - Lz2)

    assert np.allclose(H0, expected)

# endregion
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# region test Params
# ---------------------------------------------------------------------
def test_orbitronic_bulk_params_stores_values():
    M = np.array([0.1, 0.2, 0.3])
    params = OrbitronicBulkParams(
        mass=1.2,
        gamma=0.4,
        J=0.5,
        magnetisation=M,
    )
    assert params.mass == pytest.approx(1.2)
    assert params.gamma == pytest.approx(0.4)
    assert params.J == pytest.approx(0.5)
    assert np.allclose(params.magnetisation, M)

def test_orbitronic_bulk_from_params_roundtrip():
    M = np.array([0.1, 0.2, 0.3], dtype=float)
    params = OrbitronicBulkParams(
        mass=1.2,
        gamma=0.4,
        J=0.5,
        magnetisation=M,
    )

    # params -> model
    bulk = OrbitronicBulk.from_params(params)

    # model -> params again
    params_rt = OrbitronicBulkParams(
        mass=bulk.mass,
        gamma=bulk.gamma,
        J=bulk.J,
        magnetisation=bulk.magnetisation,
    )

    # compare
    assert params_rt.mass == params.mass
    assert params_rt.gamma == params.gamma
    assert params_rt.J == params.J
    np.testing.assert_allclose(params_rt.magnetisation, params.magnetisation)

def test_orbitronic_interface_params_stores_values():
    params = OrbitronicInterfaceParams(
        m_int=1.0,
        gamma_int=0.2,
        alpha=0.3,
        beta=0.4,
        delta_CF=0.5,
        L0=1.6,
    )
    assert params.m_int == pytest.approx(1.0)
    assert params.gamma_int == pytest.approx(0.2)
    assert params.alpha == pytest.approx(0.3)
    assert params.beta == pytest.approx(0.4)
    assert params.delta_CF == pytest.approx(0.5)
    assert params.L0 == pytest.approx(1.6)

def test_orbitronic_interface_from_params_roundtrip():
    params = OrbitronicInterfaceParams(
        m_int=1.0,
        gamma_int=0.2,
        alpha=0.3,
        beta=0.4,
        delta_CF=0.5,
        L0=1.6,
    )

    # params -> model
    interface = OrbitronicInterface.from_params(**vars(params))

    # model -> params again
    params_rt = OrbitronicInterfaceParams(
        m_int=interface.m_int,
        gamma_int=interface.gamma_int,
        alpha=interface.alpha,
        beta=interface.beta,
        delta_CF=interface.delta_CF,
        L0=interface.L0,
    )

    # compare (these are just copied scalars, so == is fine)
    assert params_rt.m_int == params.m_int
    assert params_rt.gamma_int == params.gamma_int
    assert params_rt.alpha == params.alpha
    assert params_rt.beta == params.beta
    assert params_rt.delta_CF == params.delta_CF
    assert params_rt.L0 == params.L0

# endregion
# ---------------------------------------------------------------------