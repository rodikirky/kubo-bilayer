import numpy as np
from numpy.typing import NDArray

# ------------------------------------------
# region Integration helpers & fixtures
# ------------------------------------------
def fermi_dirac_distribution(
    energy: float,
    chemical_potential: float,
    temperature: float,
) -> float:
    """Compute the Fermi-Dirac distribution."""
    k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
    exponent = (energy - chemical_potential) / (k_B * temperature)
    return 1 / (1 + np.exp(exponent))

def derivative_fermi_dirac(
    energy: float,
    chemical_potential: float,
    temperature: float,
) -> float:
    """Compute the derivative of the Fermi-Dirac distribution by energy."""
    if temperature == 0:
        raise ValueError("Temperature must be greater than zero for fermi-dirac derivative. At zero it becomes a delta function.")
    k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
    exponent = (energy - chemical_potential) / (k_B * temperature)
    f = 1 / (1 + np.exp(exponent))
    return f * (1 - f) / (k_B * temperature)

def _functional_trace(matrix: NDArray) -> float:
    """Compute the functional trace of a matrix."""
    # TODO: Extend with functional trace integrals
    return np.trace(matrix)

def _derivative_wrt_energy(matrix: NDArray, energy: float) -> NDArray:
    """Compute the derivative of a matrix with respect to energy."""
    return matrix # Placeholder for actual derivative implementation

def _omega_integration(integrand: float, energy: float, omega_grid: NDArray) -> float:
    """Perform numerical omega integration over the given integrand on the given grid."""
    return integrand # Placeholder for actual integration implementation

# endregion
# ------------------------------------------

# ------------------------------------------
# region Fermi Surface
# ------------------------------------------
def _build_fermi_surface_integrand(
    energy: float,
    chemical_potential: float,
    temperature: float,
    observable: float, # for example one component of the angular momentum operator
    perturbation: NDArray, # matrix representing the perturbation, e.g. velocity operator
    greens_retarded: NDArray, # retarded Green's function matrix of the full system
)-> float:
    """Build the integrand for the Fermi surface contribution."""
    A = observable
    B = perturbation
    G_r = greens_retarded
    G_a = np.conj(G_r)
    integrand_rr = A * G_r @ B @ G_r
    integrand_ra = A * G_r @ B @ G_a

    fermi_surface_integrand = _functional_trace(integrand_ra) - np.real(_functional_trace(integrand_rr))
    if temperature == 0:
        return fermi_surface_integrand
    dfdE = derivative_fermi_dirac(energy, chemical_potential, temperature)
    return dfdE * fermi_surface_integrand 

def kubo_streda_fermi_surface(
    energy: float, # external energy omega at which the observable is computed
    chemical_potential: float,
    temperature: float,
    observable: float, # for example one component of the angular momentum operator
    perturbation: NDArray, # matrix representing the perturbation, e.g. velocity operator
    greens_retarded: NDArray, # retarded Green's function matrix of the full system
    omega_grid: NDArray, # omega' grid for integration
) -> float:
    """Compute the Kubo-Streda Fermi surface contribution by integrating over omega'."""
    fermi_surface_integrand = _build_fermi_surface_integrand(
        energy,
        chemical_potential,
        temperature,
        observable,
        perturbation,
        greens_retarded,
    )
    if temperature == 0:
        return fermi_surface_integrand
    return _omega_integration(fermi_surface_integrand, energy, omega_grid)

# endregion
# ------------------------------------------

# ------------------------------------------
# region Fermi sea
# ------------------------------------------
def _build_fermi_sea_integrand(
    energy: float,
    chemical_potential: float,
    temperature: float,
    observable: float, # for example one component of the angular momentum operator
    perturbation: NDArray, # matrix representing the perturbation, e.g. velocity operator
    greens_retarded: NDArray, # retarded Green's function matrix of the full system
)-> float:
    """Build the integrand for Fermi sea contributions."""
    omega = energy
    A = observable
    B = perturbation
    G_r = greens_retarded
    dGr_domega = _derivative_wrt_energy(G_r, energy=omega) 

    fermi_sea_term = np.real(_functional_trace(A*dGr_domega @ B @ G_r - A @ G_r @ B @ dGr_domega))
    fermi_dirac = fermi_dirac_distribution(energy, chemical_potential, temperature)
    return fermi_dirac * fermi_sea_term

def kubo_streda_fermi_sea(
    energy: float, # external energy omega at which the observable is computed
    chemical_potential: float,
    temperature: float,
    observable: float, # for example one component of the angular momentum operator
    perturbation: NDArray, # matrix representing the perturbation, e.g. velocity operator
    greens_retarded: NDArray, # retarded Green's function matrix of the full system
    omega_grid: NDArray, # omega' grid for integration
) -> float:
    """Compute the Kubo-Streda Fermi sea contribution by integrating over omega'."""
    fermi_sea_integrand = _build_fermi_sea_integrand(
        energy,
        chemical_potential,
        temperature,
        observable,
        perturbation,
        greens_retarded,
    )
    return _omega_integration(fermi_sea_integrand, energy, omega_grid)
# endregion
# ------------------------------------------

# ------------------------------------------
# region Kubo-Streda full
# ------------------------------------------
def kubo_streda_total(
    energy: float, # external energy omega at which the observable is computed
    chemical_potential: float,
    temperature: float,
    observable: float, # for example one component of the angular momentum operator
    perturbation: NDArray, # matrix representing the perturbation, e.g. velocity operator
    greens_retarded: NDArray, # retarded Green's function matrix of the full system
    omega_grid: NDArray, # omega' grid for integration
) -> float:
    """Compute the total Kubo-Streda contribution by summing Fermi surface and Fermi sea parts."""
    if temperature == 0:
        fermi_surface_contribution = kubo_streda_fermi_surface(
            energy,
            chemical_potential,
            temperature,
            observable,
            perturbation,
            greens_retarded,
            omega_grid,
        )
        fermi_sea_contribution = kubo_streda_fermi_sea(
            energy,
            chemical_potential,
            temperature,
            observable,
            perturbation,
            greens_retarded,
            omega_grid,
        )
        return fermi_surface_contribution + fermi_sea_contribution
    
    # For finite temperature, we can directly sum the contributions as integrands and perform one integration
    fermi_surface_integrand = _build_fermi_surface_integrand(   
        energy,
        chemical_potential,
        temperature,
        observable,
        perturbation,
        greens_retarded,
    )
    fermi_sea_integrand = _build_fermi_sea_integrand(
        energy,
        chemical_potential,
        temperature,
        observable,
        perturbation,
        greens_retarded,
    )      
    total_integrand = fermi_surface_integrand + fermi_sea_integrand
    return _omega_integration(total_integrand, energy, omega_grid)
    
# endregion
# ------------------------------------------
