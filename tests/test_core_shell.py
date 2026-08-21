"""Tests for Core/Shell band alignment, offsets, and strain calculations.
"""

import pytest
from physics.band_alignment import (
    calculate_natural_band_offsets,
    classify_band_alignment
)
from physics.core_shell import CoreShellStructure
from physics.strain import (
    calculate_lattice_mismatch,
    calculate_hydrostatic_strain_shift
)


def test_band_alignment_classification():
    """Verify Type I, Quasi-Type II, and Type II classification logic."""
    # Type I (both positive and large)
    t1 = classify_band_alignment(delta_ec_eV=0.90, delta_ev_eV=0.70)
    assert "Type I" in t1["alignment_type"]
    assert t1["electron_localization"] == "Confined to Core"
    assert t1["hole_localization"] == "Confined to Core"

    # Quasi-Type II (small CB offset, large VB offset)
    qt2 = classify_band_alignment(delta_ec_eV=0.10, delta_ev_eV=0.60)
    assert "Quasi-Type II" in qt2["alignment_type"]
    assert "Delocalized" in qt2["electron_localization"]
    assert qt2["hole_localization"] == "Confined to Core"

    # Type II (staggered)
    t2 = classify_band_alignment(delta_ec_eV=-0.40, delta_ev_eV=0.60)
    assert "Type II" in t2["alignment_type"]
    assert t2["electron_localization"] == "Confined to Shell"
    assert t2["hole_localization"] == "Confined to Core"


def test_lattice_mismatch_and_strain():
    """Verify lattice mismatch and strain calculations."""
    # CdSe (4.30 A) / ZnS (5.41 A in ZB or 3.82 A in Wurtzite)
    eta = calculate_lattice_mismatch(a_core_angstrom=4.30, a_shell_angstrom=4.14)
    assert eta < 0  # Compressive mismatch

    # Strain shift with non-zero thickness
    strain_res = calculate_hydrostatic_strain_shift(
        core_radius_nm=2.0,
        shell_thickness_nm=1.0,
        a_core_angstrom=4.30,
        a_shell_angstrom=4.14,
        deformation_potential_eV=-3.0,
        enabled=True
    )
    assert strain_res["enabled"] is True
    assert strain_res["strain_nature"] == "Compressive"
    assert strain_res["strain_shift_eV"] > 0  # Compressive strain blue-shifts the core gap


def test_core_shell_structure_profiles():
    """Verify spatial profile generation for CoreShellStructure."""
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

    cs = CoreShellStructure(
        core_material="CdSe",
        shell_material="ZnS",
        core_radius_nm=2.0,
        shell_thickness_nm=1.0,
        core_params=core_p,
        shell_params=shell_p
    )

    import numpy as np
    r_grid = np.linspace(0.1, 5.0, 50)
    ve, vh, me, mh, eps = cs.generate_radial_profiles(r_grid)

    assert len(ve) == 50
    # Core point r = 1.0 nm
    assert ve[r_grid <= 2.0][0] == 0.0
    assert me[r_grid <= 2.0][0] == 0.13
    # Shell point 2.0 < r <= 3.0 nm
    shell_idx = (r_grid > 2.0) & (r_grid <= 3.0)
    assert pytest.approx(ve[shell_idx][0], abs=1e-3) == cs.delta_ec
    assert me[shell_idx][0] == 0.25
