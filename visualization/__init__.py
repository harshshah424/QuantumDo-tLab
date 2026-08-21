"""Visualization package for QuantumDotLab.
"""

from visualization.plots import (
    plot_bandgap_vs_radius,
    plot_wavelength_vs_radius,
    plot_material_comparison,
    plot_alloy_composition_curves,
    plot_alloy_heatmap
)
from visualization.spectrum import plot_emission_spectrum
from visualization.band_diagram import plot_core_shell_band_diagram
from visualization.wavefunctions import plot_probability_densities

__all__ = [
    "plot_bandgap_vs_radius",
    "plot_wavelength_vs_radius",
    "plot_material_comparison",
    "plot_alloy_composition_curves",
    "plot_alloy_heatmap",
    "plot_emission_spectrum",
    "plot_core_shell_band_diagram",
    "plot_probability_densities",
]
