"""Alloy composition model for ternary semiconductor quantum dots.

Interpolates bulk bandgap with quadratic bowing:
    Eg(x) = x * Eg(A) + (1 - x) * Eg(B) - b * x * (1 - x)
and applies Vegard's law for effective masses, dielectric constant, and lattice constants.
"""

from typing import Dict, Any, Tuple
import numpy as np
from physics.brus_model import calculate_bandgap


def interpolate_alloy_properties(
    mat_a_props: Dict[str, Any],
    mat_b_props: Dict[str, Any],
    x_composition: float,
    bowing_parameter_eV: float = 0.0
) -> Dict[str, Any]:
    """Calculate effective material parameters for a ternary alloy A_x B_(1-x).

    Args:
        mat_a_props: Property dict for Material A (x = 1 limit).
        mat_b_props: Property dict for Material B (x = 0 limit).
        x_composition: Fraction x of Material A (0.0 <= x <= 1.0).
        bowing_parameter_eV: Optical bandgap bowing parameter b in eV.

    Returns:
        Dictionary of interpolated alloy properties.
    """
    if not (0.0 <= x_composition <= 1.0):
        raise ValueError(f"Alloy composition x must be in range [0.0, 1.0], got {x_composition}")

    x = float(x_composition)
    one_minus_x = 1.0 - x

    # Bandgap with optical bowing
    eg_a = float(mat_a_props["bulk_bandgap_eV"])
    eg_b = float(mat_b_props["bulk_bandgap_eV"])
    eg_alloy = x * eg_a + one_minus_x * eg_b - bowing_parameter_eV * x * one_minus_x

    # Linear Vegard's law for other electronic/structural properties
    me_a = float(mat_a_props["electron_effective_mass"])
    me_b = float(mat_b_props["electron_effective_mass"])
    me_alloy = x * me_a + one_minus_x * me_b

    mh_a = float(mat_a_props["hole_effective_mass"])
    mh_b = float(mat_b_props["hole_effective_mass"])
    mh_alloy = x * mh_a + one_minus_x * mh_b

    eps_a = float(mat_a_props["relative_dielectric_constant"])
    eps_b = float(mat_b_props["relative_dielectric_constant"])
    eps_alloy = x * eps_a + one_minus_x * eps_b

    lat_a = float(mat_a_props.get("lattice_constant_angstrom", 5.0))
    lat_b = float(mat_b_props.get("lattice_constant_angstrom", 5.0))
    lat_alloy = x * lat_a + one_minus_x * lat_b

    cbm_a = float(mat_a_props.get("conduction_band_edge_eV", -4.0))
    cbm_b = float(mat_b_props.get("conduction_band_edge_eV", -4.0))
    cbm_alloy = x * cbm_a + one_minus_x * cbm_b

    vbm_a = float(mat_a_props.get("valence_band_edge_eV", -6.0))
    vbm_b = float(mat_b_props.get("valence_band_edge_eV", -6.0))
    vbm_alloy = x * vbm_a + one_minus_x * vbm_b

    def_a = float(mat_a_props.get("hydrostatic_deformation_pot_eV", -3.0))
    def_b = float(mat_b_props.get("hydrostatic_deformation_pot_eV", -3.0))
    def_alloy = x * def_a + one_minus_x * def_b

    name_a = mat_a_props.get("formula", mat_a_props.get("material_name", "A"))
    name_b = mat_b_props.get("formula", mat_b_props.get("material_name", "B"))
    alloy_formula = f"{name_a}({x:.2f}){name_b}({1.0-x:.2f})"

    return {
        "material_name": f"Alloy {alloy_formula}",
        "formula": alloy_formula,
        "composition_x": x,
        "bowing_parameter_eV": bowing_parameter_eV,
        "bulk_bandgap_eV": eg_alloy,
        "electron_effective_mass": me_alloy,
        "hole_effective_mass": mh_alloy,
        "relative_dielectric_constant": eps_alloy,
        "lattice_constant_angstrom": lat_alloy,
        "conduction_band_edge_eV": cbm_alloy,
        "valence_band_edge_eV": vbm_alloy,
        "hydrostatic_deformation_pot_eV": def_alloy,
        "notes": f"Interpolated ternary alloy between {name_a} and {name_b} with bowing b = {bowing_parameter_eV:.3f} eV.",
        "source": "Vegard's Law + Empirical Bowing Model (Ref Doc Section 2.2)"
    }


def compute_alloy_grid_bandgap(
    mat_a_props: Dict[str, Any],
    mat_b_props: Dict[str, Any],
    radius_array_nm: np.ndarray,
    composition_array_x: np.ndarray,
    bowing_parameter_eV: float = 0.0
) -> np.ndarray:
    """Compute a 2D matrix of Quantum Dot bandgaps across a 2D (Radius x Composition) grid.

    Args:
        mat_a_props: Properties of end-member A.
        mat_b_props: Properties of end-member B.
        radius_array_nm: 1D array of QD radii in nm.
        composition_array_x: 1D array of compositions x in [0, 1].
        bowing_parameter_eV: Bowing parameter b in eV.

    Returns:
        2D numpy array of shape (len(composition_array_x), len(radius_array_nm)) with Eg(x, R) in eV.
    """
    n_comp = len(composition_array_x)
    n_rad = len(radius_array_nm)
    grid_eg = np.zeros((n_comp, n_rad), dtype=float)

    for i, x in enumerate(composition_array_x):
        alloy_p = interpolate_alloy_properties(mat_a_props, mat_b_props, float(x), bowing_parameter_eV)
        eg_row = calculate_bandgap(
            radius_nm=radius_array_nm,
            bulk_bandgap_eV=alloy_p["bulk_bandgap_eV"],
            electron_effective_mass=alloy_p["electron_effective_mass"],
            hole_effective_mass=alloy_p["hole_effective_mass"],
            relative_dielectric_constant=alloy_p["relative_dielectric_constant"]
        )
        grid_eg[i, :] = eg_row

    return grid_eg
