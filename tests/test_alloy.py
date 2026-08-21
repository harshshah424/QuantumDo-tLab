"""Tests for alloy composition interpolation and optical bowing model.
"""

import pytest
import numpy as np
from physics.alloy import (
    interpolate_alloy_properties,
    compute_alloy_grid_bandgap
)


@pytest.fixture
def cdse_props():
    return {
        "formula": "CdSe",
        "bulk_bandgap_eV": 1.74,
        "electron_effective_mass": 0.13,
        "hole_effective_mass": 0.45,
        "relative_dielectric_constant": 9.4,
        "lattice_constant_angstrom": 4.30,
        "conduction_band_edge_eV": -4.30,
        "valence_band_edge_eV": -6.04
    }


@pytest.fixture
def cds_props():
    return {
        "formula": "CdS",
        "bulk_bandgap_eV": 2.45,
        "electron_effective_mass": 0.20,
        "hole_effective_mass": 0.70,
        "relative_dielectric_constant": 9.0,
        "lattice_constant_angstrom": 4.14,
        "conduction_band_edge_eV": -4.15,
        "valence_band_edge_eV": -6.60
    }


def test_alloy_composition_endpoints(cdse_props, cds_props):
    """Verify that x = 1 yields Material A and x = 0 yields Material B."""
    alloy_a = interpolate_alloy_properties(cdse_props, cds_props, x_composition=1.0, bowing_parameter_eV=0.3)
    assert pytest.approx(alloy_a["bulk_bandgap_eV"], abs=1e-4) == cdse_props["bulk_bandgap_eV"]
    assert pytest.approx(alloy_a["electron_effective_mass"], abs=1e-4) == cdse_props["electron_effective_mass"]

    alloy_b = interpolate_alloy_properties(cdse_props, cds_props, x_composition=0.0, bowing_parameter_eV=0.3)
    assert pytest.approx(alloy_b["bulk_bandgap_eV"], abs=1e-4) == cds_props["bulk_bandgap_eV"]
    assert pytest.approx(alloy_b["electron_effective_mass"], abs=1e-4) == cds_props["electron_effective_mass"]


def test_bowing_effect(cdse_props, cds_props):
    """Verify that optical bowing reduces the intermediate bandgap below linear interpolation."""
    b = 0.30  # eV
    alloy_bowed = interpolate_alloy_properties(cdse_props, cds_props, x_composition=0.5, bowing_parameter_eV=b)
    alloy_linear = interpolate_alloy_properties(cdse_props, cds_props, x_composition=0.5, bowing_parameter_eV=0.0)

    # For x = 0.5: delta = b * 0.5 * 0.5 = 0.25 * b = 0.075 eV
    expected_diff = b * 0.25
    assert pytest.approx(alloy_linear["bulk_bandgap_eV"] - alloy_bowed["bulk_bandgap_eV"], abs=1e-4) == expected_diff


def test_invalid_composition_range(cdse_props, cds_props):
    """Verify that x < 0 or x > 1 raises ValueError."""
    with pytest.raises(ValueError, match="range"):
        interpolate_alloy_properties(cdse_props, cds_props, x_composition=1.2)

    with pytest.raises(ValueError, match="range"):
        interpolate_alloy_properties(cdse_props, cds_props, x_composition=-0.1)


def test_alloy_2d_grid(cdse_props, cds_props):
    """Verify 2D grid bandgap generation."""
    r_arr = np.array([2.0, 3.0, 4.0])
    x_arr = np.array([0.0, 0.5, 1.0])
    grid = compute_alloy_grid_bandgap(cdse_props, cds_props, r_arr, x_arr, bowing_parameter_eV=0.3)

    assert grid.shape == (3, 3)
    # Bandgap should decrease with radius across all rows
    for row in grid:
        assert np.all(np.diff(row) < 0)
