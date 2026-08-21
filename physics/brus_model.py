"""Analytical Brus model for quantum dot bandgap calculation.

Implements the Effective Mass Approximation (EMA) for spherical quantum dots:
E_g(R) = E_{g,bulk} + (hbar^2 * pi^2 / (2 * R^2)) * (1/me* + 1/mh*) - (1.8 * e^2 / (4 * pi * eps_0 * eps_r * R))
"""

from typing import Dict, Any, Tuple
import numpy as np
from physics.constants import (
    HBAR2_OVER_2M0_EV_NM2,
    COULOMB_PREFACTOR_EV_NM,
    BOHR_RADIUS_NM,
    EMA_MIN_RADIUS_NM
)


def calculate_exciton_bohr_radius(
    me_eff: float,
    mh_eff: float,
    epsilon_r: float
) -> float:
    """Calculate the exciton Bohr radius in nanometers.

    a_B = eps_r * (m0 / mu) * a_0
    where mu = (me_eff * mh_eff) / (me_eff + mh_eff) * m0 is the reduced exciton mass
    and a_0 is the atomic Bohr radius (0.0529 nm).

    Args:
        me_eff: Electron effective mass relative to m0.
        mh_eff: Hole effective mass relative to m0.
        epsilon_r: Relative static dielectric constant.

    Returns:
        Exciton Bohr radius in nm.
    """
    if me_eff <= 0 or mh_eff <= 0 or epsilon_r <= 0:
        raise ValueError("Effective masses and dielectric constant must be strictly positive.")
    
    inv_mu = (1.0 / me_eff) + (1.0 / mh_eff)
    a_B_nm = epsilon_r * inv_mu * BOHR_RADIUS_NM
    return float(a_B_nm)


def classify_confinement_regime(
    radius_nm: float,
    bohr_radius_nm: float
) -> Tuple[str, str]:
    """Classify the quantum confinement regime based on radius vs exciton Bohr radius.

    Args:
        radius_nm: Quantum dot radius in nm.
        bohr_radius_nm: Exciton Bohr radius in nm.

    Returns:
        Tuple of (regime_name, explanation).
    """
    ratio = radius_nm / bohr_radius_nm
    if ratio < 1.0:
        regime = "Strong Confinement"
        desc = (
            f"R ({radius_nm:.2f} nm) < a_B ({bohr_radius_nm:.2f} nm). "
            "Electrons and holes are independently confined; kinetic energy dominates Coulomb attraction. "
            "Brus model is most applicable here."
        )
    elif ratio <= 2.0:
        regime = "Intermediate Confinement"
        desc = (
            f"R ({radius_nm:.2f} nm) ≈ a_B ({bohr_radius_nm:.2f} nm). "
            "Carrier motion is correlated; single-particle EMA begins to experience perturbative Coulomb corrections."
        )
    else:
        regime = "Weak Confinement"
        desc = (
            f"R ({radius_nm:.2f} nm) > a_B ({bohr_radius_nm:.2f} nm). "
            "Exciton center-of-mass motion is quantized; bandgap approaches bulk value with minor confinement shift."
        )
    return regime, desc


def calculate_bandgap(
    radius_nm: float | np.ndarray,
    bulk_bandgap_eV: float,
    electron_effective_mass: float,
    hole_effective_mass: float,
    relative_dielectric_constant: float
) -> Dict[str, Any] | float | np.ndarray:
    """Calculate the quantum dot bandgap using the Brus equation.

    Formula:
        Eg(R) = Eg_bulk + E_conf(R) - E_coulomb(R)
        where:
            E_conf = (hbar^2 * pi^2 / (2 * m0 * R^2)) * (1/me + 1/mh) [eV]
            E_coulomb = (1.8 * e^2 / (4 * pi * eps0 * eps_r * R)) [eV]

    Args:
        radius_nm: Quantum dot radius in nm (float or numpy array).
        bulk_bandgap_eV: Bulk semiconductor bandgap in eV.
        electron_effective_mass: Electron effective mass (m_e* / m_0).
        hole_effective_mass: Hole effective mass (m_h* / m_0).
        relative_dielectric_constant: Relative dielectric constant (epsilon_r).

    Returns:
        If scalar radius: Dictionary with detailed breakdown (Eg, E_conf, E_coulomb, warnings, etc.)
        If array radius: Array of calculated bandgaps in eV.
    """
    if electron_effective_mass <= 0:
        raise ValueError(f"Electron effective mass must be positive, got {electron_effective_mass}")
    if hole_effective_mass <= 0:
        raise ValueError(f"Hole effective mass must be positive, got {hole_effective_mass}")
    if relative_dielectric_constant <= 0:
        raise ValueError(f"Relative dielectric constant must be positive, got {relative_dielectric_constant}")
    if bulk_bandgap_eV < 0:
        raise ValueError(f"Bulk bandgap cannot be negative, got {bulk_bandgap_eV}")

    is_array = isinstance(radius_nm, np.ndarray) or isinstance(radius_nm, list)
    r = np.asarray(radius_nm, dtype=float)

    if np.any(r <= 0):
        raise ValueError("Quantum dot radius must be strictly positive (> 0 nm).")

    # Kinetic confinement shift: E_conf = (pi^2 * hbar^2 / (2 * m0 * R^2)) * (1/me + 1/mh)
    inv_mass_sum = (1.0 / electron_effective_mass) + (1.0 / hole_effective_mass)
    e_conf = (np.pi ** 2 * HBAR2_OVER_2M0_EV_NM2 / (r ** 2)) * inv_mass_sum

    # Coulomb attraction correction: E_coulomb = 1.8 * (e^2 / (4 * pi * eps0 * eps_r * R))
    e_coulomb = (1.8 * COULOMB_PREFACTOR_EV_NM) / (relative_dielectric_constant * r)

    # Total quantum dot bandgap
    eg_qd = bulk_bandgap_eV + e_conf - e_coulomb

    if is_array:
        return eg_qd

    # Scalar detailed analysis
    scalar_r = float(r)
    scalar_eg = float(eg_qd)
    scalar_conf = float(e_conf)
    scalar_coulomb = float(e_coulomb)

    bohr_radius = calculate_exciton_bohr_radius(
        electron_effective_mass, hole_effective_mass, relative_dielectric_constant
    )
    regime, regime_desc = classify_confinement_regime(scalar_r, bohr_radius)

    warnings = []
    if scalar_r < EMA_MIN_RADIUS_NM:
        warnings.append(
            f"Radius R = {scalar_r:.2f} nm is below the standard EMA validity threshold ({EMA_MIN_RADIUS_NM} nm). "
            "Parabolic effective-mass approximation systematically overestimates confinement shift for sub-nanometer clusters."
        )

    return {
        "radius_nm": scalar_r,
        "bulk_bandgap_eV": bulk_bandgap_eV,
        "qd_bandgap_eV": scalar_eg,
        "confinement_energy_eV": scalar_conf,
        "coulomb_energy_eV": scalar_coulomb,
        "net_shift_eV": scalar_eg - bulk_bandgap_eV,
        "exciton_bohr_radius_nm": bohr_radius,
        "confinement_regime": regime,
        "confinement_description": regime_desc,
        "warnings": warnings,
    }
