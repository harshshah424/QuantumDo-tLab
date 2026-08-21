"""Tests for numerical finite-difference radial Schrödinger solver with BenDaniel-Duke boundary conditions.
"""

import pytest
import numpy as np
from physics.core_shell import CoreShellStructure
from physics.schrodinger_solver import (
    RadialSchrodingerSolver,
    solve_core_shell_system
)


@pytest.fixture
def cdse_zns_system():
    core_p = {
        "bulk_bandgap_eV": 1.74,
        "electron_effective_mass": 0.13,
        "hole_effective_mass": 0.45,
        "relative_dielectric_constant": 9.4,
        "conduction_band_edge_eV": -4.30,
        "valence_band_edge_eV": -6.04
    }
    shell_p = {
        "bulk_bandgap_eV": 3.65,
        "electron_effective_mass": 0.25,
        "hole_effective_mass": 0.60,
        "relative_dielectric_constant": 8.3,
        "conduction_band_edge_eV": -3.40,
        "valence_band_edge_eV": -7.05
    }
    return CoreShellStructure(
        core_material="CdSe",
        shell_material="ZnS",
        core_radius_nm=2.0,
        shell_thickness_nm=1.0,
        core_params=core_p,
        shell_params=shell_p,
        delta_ec_eV=0.90,
        delta_ev_eV=0.70
    )


def test_radial_solver_normalization_and_eigenvalues(cdse_zns_system):
    """Verify that numerical solver returns physically sensible eigenvalues and normalized eigenvectors."""
    res = solve_core_shell_system(cdse_zns_system, num_grid_points=300)

    # Confinement energies must be positive
    assert res["electron_ground_energy_eV"] > 0
    assert res["hole_ground_energy_eV"] > 0

    # Normalization check: sum(|u_i|^2 * dr) ≈ 1.0
    dr = res["dr_nm"]
    norm_e = np.sum(res["u_e"] ** 2) * dr
    norm_h = np.sum(res["u_h"] ** 2) * dr
    assert pytest.approx(norm_e, abs=1e-3) == 1.0
    assert pytest.approx(norm_h, abs=1e-3) == 1.0

    # Total QD gap should exceed core bulk bandgap
    assert res["qd_bandgap_eV"] > res["bulk_core_bandgap_eV"]


def test_type1_carrier_localization(cdse_zns_system):
    """In CdSe/ZnS Type I system, both electron and hole should be predominantly confined to the core."""
    res = solve_core_shell_system(cdse_zns_system, num_grid_points=350)
    e_loc = res["electron_localization"]
    h_loc = res["hole_localization"]

    assert e_loc["core_percent"] > 60.0
    assert h_loc["core_percent"] > 70.0
    assert e_loc["outer_percent"] < 5.0
    assert h_loc["outer_percent"] < 5.0


def test_grid_convergence(cdse_zns_system):
    """Verify grid convergence: doubling grid resolution changes eigenvalue by < 3 meV."""
    res_coarse = solve_core_shell_system(cdse_zns_system, num_grid_points=200)
    res_fine = solve_core_shell_system(cdse_zns_system, num_grid_points=400)

    delta_ee = abs(res_coarse["electron_ground_energy_eV"] - res_fine["electron_ground_energy_eV"])
    delta_eh = abs(res_coarse["hole_ground_energy_eV"] - res_fine["hole_ground_energy_eV"])

    # Convergence check: energy difference should be within 3 meV (0.003 eV)
    assert delta_ee < 0.003
    assert delta_eh < 0.003


def test_infinite_well_analytical_comparison():
    """Verify solver against analytical infinite spherical well particle-in-a-box:
    E_1 = (hbar^2 * pi^2) / (2 * m0 * m* * R^2)
    """
    r_well = 3.0  # nm
    m_star = 0.20  # m0
    # Analytical: (0.03809982 * pi^2) / (0.20 * 9.0) = 0.3760237 / 1.8 = 0.20890 eV
    expected_e = (0.03809982 * (np.pi ** 2)) / (m_star * (r_well ** 2))

    solver = RadialSchrodingerSolver(r_max_nm=r_well, num_grid_points=500)
    v_zero = np.zeros(solver.n)
    m_const = np.full(solver.n, m_star)

    evals, evecs = solver.solve_single_particle(v_zero, m_const, num_states=1)
    computed_e = evals[0]

    assert pytest.approx(computed_e, rel=1e-3) == expected_e
