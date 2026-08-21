"""QuantumDotLab physics package.

Contains physical models for quantum dot bandgap engineering:
- Analytical Brus effective-mass model
- Optical emission, wavelength, and visible color mapping
- Ternary alloy composition and optical bowing
- Band alignment classification (Type I, Type II, Quasi-Type II)
- Core/Shell heterostructure radial potential landscapes
- 1D Radial Schrödinger finite-difference solver with BenDaniel-Duke conditions
- Interfacial lattice mismatch and hydrostatic strain corrections
"""

from physics.constants import (
    HBAR_SI,
    M0_SI,
    Q_E,
    EPSILON_0,
    SPEED_OF_LIGHT,
    HC_EV_NM,
    BOHR_RADIUS_NM,
    EMA_MIN_RADIUS_NM
)
from physics.brus_model import (
    calculate_bandgap,
    calculate_exciton_bohr_radius,
    classify_confinement_regime
)
from physics.emission import (
    bandgap_to_wavelength,
    wavelength_to_bandgap,
    wavelength_to_color_category,
    wavelength_to_rgb,
    gaussian_spectrum,
    get_emission_summary
)
from physics.alloy import (
    interpolate_alloy_properties,
    compute_alloy_grid_bandgap
)
from physics.band_alignment import (
    calculate_natural_band_offsets,
    classify_band_alignment
)
from physics.core_shell import CoreShellStructure
from physics.schrodinger_solver import (
    RadialSchrodingerSolver,
    solve_core_shell_system
)
from physics.strain import (
    calculate_lattice_mismatch,
    calculate_hydrostatic_strain_shift
)

__all__ = [
    "HBAR_SI",
    "M0_SI",
    "Q_E",
    "EPSILON_0",
    "SPEED_OF_LIGHT",
    "HC_EV_NM",
    "BOHR_RADIUS_NM",
    "EMA_MIN_RADIUS_NM",
    "calculate_bandgap",
    "calculate_exciton_bohr_radius",
    "classify_confinement_regime",
    "bandgap_to_wavelength",
    "wavelength_to_bandgap",
    "wavelength_to_color_category",
    "wavelength_to_rgb",
    "gaussian_spectrum",
    "get_emission_summary",
    "interpolate_alloy_properties",
    "compute_alloy_grid_bandgap",
    "calculate_natural_band_offsets",
    "classify_band_alignment",
    "CoreShellStructure",
    "RadialSchrodingerSolver",
    "solve_core_shell_system",
    "calculate_lattice_mismatch",
    "calculate_hydrostatic_strain_shift",
]
