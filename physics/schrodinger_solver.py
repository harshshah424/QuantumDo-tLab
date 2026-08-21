"""1D Radial Schrödinger solver for spherical Quantum Dot heterostructures.

Uses finite-difference method with symmetrized BenDaniel-Duke boundary conditions
for position-dependent effective mass across material interfaces.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.linalg import eigh_tridiagonal
from physics.constants import (
    HBAR2_OVER_2M0_EV_NM2,
    COULOMB_PREFACTOR_EV_NM
)
from physics.core_shell import CoreShellStructure


class RadialSchrodingerSolver:
    """Solves the single-particle radial Schrödinger equation in spherical coordinates."""

    def __init__(
        self,
        r_max_nm: float,
        num_grid_points: int = 400
    ):
        """Initialize solver grid.

        Args:
            r_max_nm: Maximum radial extent in nm.
            num_grid_points: Number of spatial discretization points N.
        """
        if r_max_nm <= 0:
            raise ValueError(f"r_max_nm must be positive, got {r_max_nm}")
        if num_grid_points < 50:
            raise ValueError(f"num_grid_points must be at least 50, got {num_grid_points}")

        self.r_max = float(r_max_nm)
        self.n = int(num_grid_points)
        self.dr = self.r_max / (self.n + 1)
        # Radial grid ri = i * dr, i = 1...N (excluding r=0 and r=Rmax where u=0)
        self.r_grid = np.linspace(self.dr, self.r_max - self.dr, self.n)

    def solve_single_particle(
        self,
        potential_profile_eV: np.ndarray,
        effective_mass_profile: np.ndarray,
        num_states: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve for ground (and low-lying) state eigenvalues and eigenvectors.

        Discretizes:
            [-hbar^2 / (2 * m0) * d/dr (1/m*(r) * d/dr) + V(r)] u(r) = E u(r)
        with BenDaniel-Duke interface matching:
            H_{i,i} = (hbar^2 / (2 * m0 * dr^2)) * (1/m_{i+1/2} + 1/m_{i-1/2}) + V_i
            H_{i,i+1} = - (hbar^2 / (2 * m0 * dr^2)) * (1/m_{i+1/2})

        Args:
            potential_profile_eV: Potential V(r) array of length N in eV.
            effective_mass_profile: Relative mass m*(r)/m0 array of length N.
            num_states: Number of lowest eigenstates to return.

        Returns:
            Tuple of:
                eigenvalues: 1D array of energies in eV
                eigenvectors: 2D array of shape (N, num_states) with normalized u(r)
        """
        v = np.asarray(potential_profile_eV, dtype=float)
        m = np.asarray(effective_mass_profile, dtype=float)
        n = self.n
        dr = self.dr

        if len(v) != n or len(m) != n:
            raise ValueError(f"Input profiles must have length {n}, got V:{len(v)}, m:{len(m)}")

        # Midpoint effective masses m_{i+1/2} using harmonic mean for smoothness
        # m_{i+1/2} = 2 * m_i * m_{i+1} / (m_i + m_{i+1})
        m_half = 2.0 * m[:-1] * m[1:] / (m[:-1] + m[1:])  # length N-1

        # Kinetic coefficient
        c_kin = HBAR2_OVER_2M0_EV_NM2 / (dr ** 2)

        # Off-diagonal elements (length N-1)
        subdiag = -c_kin / m_half

        # Main diagonal elements (length N)
        diag = np.zeros(n, dtype=float)

        # i = 0 (first interior grid point r_1)
        diag[0] = c_kin * (1.0 / m_half[0] + 1.0 / m[0]) + v[0]

        # i = 1 to N-2
        diag[1:-1] = c_kin * (1.0 / m_half[1:] + 1.0 / m_half[:-1]) + v[1:-1]

        # i = N-1 (last interior grid point r_N)
        diag[-1] = c_kin * (1.0 / m[n-1] + 1.0 / m_half[-1]) + v[-1]

        # Solve symmetric tridiagonal eigensystem
        evals, evecs = eigh_tridiagonal(
            diag,
            subdiag,
            select='i',
            select_range=(0, num_states - 1)
        )

        # Normalize wavefunctions: integral |u(r)|^2 dr = sum(|u_i|^2 * dr) = 1
        # Also ensure positive convention for ground state
        for k in range(evecs.shape[1]):
            norm = np.sqrt(np.sum(evecs[:, k] ** 2) * dr)
            if norm > 0:
                evecs[:, k] /= norm
            if np.sum(evecs[:, k]) < 0:
                evecs[:, k] *= -1.0

        return evals, evecs

    def compute_coulomb_binding_energy(
        self,
        u_e: np.ndarray,
        u_h: np.ndarray,
        dielectric_profile: np.ndarray
    ) -> float:
        """Compute the electron-hole Coulomb binding energy using the two-particle overlap integral.

        Formula:
            E_b = (e^2 / (4 * pi * eps_0)) * integral integral [ |u_e(r_e)|^2 * |u_h(r_h)|^2 / (eps_r * max(r_e, r_h)) ] dr_e dr_h

        Args:
            u_e: Electron ground state reduced wavefunction u(r).
            u_h: Hole ground state reduced wavefunction u(r).
            dielectric_profile: Position-dependent relative permittivity eps_r(r).

        Returns:
            Coulomb binding energy in eV.
        """
        dr = self.dr
        r = self.r_grid
        prob_e = (u_e ** 2) * dr
        prob_h = (u_h ** 2) * dr

        # Outer product of probabilities: P_e(i) * P_h(j)
        prob_matrix = np.outer(prob_e, prob_h)

        # 2D max(r_e, r_h) matrix
        r_e_mat, r_h_mat = np.meshgrid(r, r, indexing='ij')
        r_max_mat = np.maximum(r_e_mat, r_h_mat)

        # Effective dielectric matrix (geometric average between points)
        eps_e_mat, eps_h_mat = np.meshgrid(dielectric_profile, dielectric_profile, indexing='ij')
        eps_eff_mat = np.sqrt(eps_e_mat * eps_h_mat)

        # Coulomb energy sum
        integrand = prob_matrix / (eps_eff_mat * r_max_mat)
        e_coulomb = COULOMB_PREFACTOR_EV_NM * np.sum(integrand)
        return float(e_coulomb)


def solve_core_shell_system(
    core_shell_struct: CoreShellStructure,
    num_grid_points: int = 400,
    r_max_factor: float = 2.5
) -> Dict[str, Any]:
    """Execute complete numerical simulation for a Core/Shell quantum dot.

    Solves electron and hole radial equations independently, computes wavefunctions,
    carrier localization, Coulomb binding correction, and overall optical bandgap.

    Args:
        core_shell_struct: CoreShellStructure instance.
        num_grid_points: Number of spatial radial grid points.
        r_max_factor: Multiplier for total QD radius to place outer boundary (default 2.5x).

    Returns:
        Dictionary with eigenvalues, wavefunctions, optical gap, and localization probabilities.
    """
    r_total = core_shell_struct.r_total
    r_max = max(r_total * r_max_factor, r_total + 2.0)

    solver = RadialSchrodingerSolver(r_max_nm=r_max, num_grid_points=num_grid_points)
    r_grid = solver.r_grid

    ve, vh, me, mh, eps = core_shell_struct.generate_radial_profiles(r_grid)

    # Solve electron ground state
    ee_vals, ee_vecs = solver.solve_single_particle(ve, me, num_states=1)
    e_e = float(ee_vals[0])
    u_e = ee_vecs[:, 0]

    # Solve hole ground state
    eh_vals, eh_vecs = solver.solve_single_particle(vh, mh, num_states=1)
    e_h = float(eh_vals[0])
    u_h = eh_vecs[:, 0]

    # Coulomb binding energy
    e_coulomb = solver.compute_coulomb_binding_energy(u_e, u_h, eps)

    # Carrier localization percentages
    dr = solver.dr
    core_mask = r_grid <= core_shell_struct.r_core
    shell_mask = (r_grid > core_shell_struct.r_core) & (r_grid <= core_shell_struct.r_total)
    outer_mask = r_grid > core_shell_struct.r_total

    p_e_core = float(np.sum(u_e[core_mask] ** 2) * dr * 100.0)
    p_e_shell = float(np.sum(u_e[shell_mask] ** 2) * dr * 100.0)
    p_e_outer = float(np.sum(u_e[outer_mask] ** 2) * dr * 100.0)

    p_h_core = float(np.sum(u_h[core_mask] ** 2) * dr * 100.0)
    p_h_shell = float(np.sum(u_h[shell_mask] ** 2) * dr * 100.0)
    p_h_outer = float(np.sum(u_h[outer_mask] ** 2) * dr * 100.0)

    # Effective optical bandgap
    eg_core_bulk = float(core_shell_struct.core_params["bulk_bandgap_eV"])
    eg_qd = eg_core_bulk + e_e + e_h - e_coulomb

    return {
        "r_grid_nm": r_grid,
        "dr_nm": dr,
        "r_core_nm": core_shell_struct.r_core,
        "r_total_nm": core_shell_struct.r_total,
        "potential_e_eV": ve,
        "potential_h_eV": vh,
        "mass_e_profile": me,
        "mass_h_profile": mh,
        "dielectric_profile": eps,
        "electron_ground_energy_eV": e_e,
        "hole_ground_energy_eV": e_h,
        "u_e": u_e,
        "u_h": u_h,
        "prob_density_e": u_e ** 2,
        "prob_density_h": u_h ** 2,
        "coulomb_binding_eV": e_coulomb,
        "bulk_core_bandgap_eV": eg_core_bulk,
        "qd_bandgap_eV": eg_qd,
        "electron_localization": {
            "core_percent": p_e_core,
            "shell_percent": p_e_shell,
            "outer_percent": p_e_outer
        },
        "hole_localization": {
            "core_percent": p_h_core,
            "shell_percent": p_h_shell,
            "outer_percent": p_h_outer
        }
    }
