"""Lattice mismatch strain and hydrostatic deformation potential shifts in Core/Shell quantum dots.
"""

from typing import Dict, Any, Tuple


def calculate_lattice_mismatch(
    a_core_angstrom: float,
    a_shell_angstrom: float
) -> float:
    """Calculate the fractional lattice mismatch parameter eta.

    Formula:
        eta = (a_shell - a_core) / a_core

    Args:
        a_core_angstrom: Bulk unstrained lattice constant of core in Angstroms.
        a_shell_angstrom: Bulk unstrained lattice constant of shell in Angstroms.

    Returns:
        Fractional lattice mismatch eta (e.g., +0.04 for +4%, -0.12 for -12%).
    """
    if a_core_angstrom <= 0 or a_shell_angstrom <= 0:
        raise ValueError("Lattice constants must be strictly positive.")
    return (a_shell_angstrom - a_core_angstrom) / a_core_angstrom


def calculate_hydrostatic_strain_shift(
    core_radius_nm: float,
    shell_thickness_nm: float,
    a_core_angstrom: float,
    a_shell_angstrom: float,
    deformation_potential_eV: float = -3.5,
    enabled: bool = False
) -> Dict[str, Any]:
    """Calculate the hydrostatic strain-induced bandgap shift in a spherical core/shell QD.

    Uses spherical continuum elasticity approximation:
        eta = (a_shell - a_core) / a_core
        eps_hydro ≈ eta * (1 - (R_core / R_total)^3)
        Delta_Eg_strain = a_cv * 3 * eps_hydro

    Args:
        core_radius_nm: Core radius in nm.
        shell_thickness_nm: Shell thickness in nm.
        a_core_angstrom: Core lattice constant.
        a_shell_angstrom: Shell lattice constant.
        deformation_potential_eV: Hydrostatic bandgap deformation potential a_cv (eV).
        enabled: Toggle flag whether strain correction is actively applied.

    Returns:
        Dictionary with mismatch percentage, strain tensor trace, and energy shift in eV.
    """
    if core_radius_nm <= 0:
        raise ValueError("Core radius must be positive.")
    if shell_thickness_nm < 0:
        raise ValueError("Shell thickness cannot be negative.")

    eta = calculate_lattice_mismatch(a_core_angstrom, a_shell_angstrom)
    r_total = core_radius_nm + shell_thickness_nm

    if shell_thickness_nm == 0.0 or not enabled:
        return {
            "enabled": enabled,
            "lattice_mismatch_fraction": eta,
            "lattice_mismatch_percent": eta * 100.0,
            "volumetric_strain": 0.0,
            "strain_shift_eV": 0.0,
            "notes": "Strain calculation disabled or zero shell thickness."
        }

    # Geometric volume fraction term for spherical shell clamping
    vol_ratio = (core_radius_nm / r_total) ** 3
    eps_vol = eta * (1.0 - vol_ratio)

    # Hydrostatic bandgap shift: a_cv * Tr(eps) = a_cv * 3 * eps_hydro
    delta_eg = deformation_potential_eV * 3.0 * (eps_vol / 3.0)

    strain_nature = "Compressive" if eta < 0 else "Tensile"
    shift_dir = "Blue-shift" if delta_eg > 0 else "Red-shift"

    return {
        "enabled": enabled,
        "lattice_mismatch_fraction": eta,
        "lattice_mismatch_percent": eta * 100.0,
        "volumetric_strain": eps_vol,
        "strain_shift_eV": float(delta_eg),
        "strain_nature": strain_nature,
        "shift_direction": shift_dir,
        "deformation_potential_eV": deformation_potential_eV,
        "notes": (
            f"{strain_nature} interfacial strain ({eta*100.0:+.2f}% mismatch) produces a {shift_dir} of "
            f"{abs(delta_eg):.3f} eV in the core bandgap. (Elastic continuum approximation with a_cv = {deformation_potential_eV:.2f} eV)."
        )
    }
