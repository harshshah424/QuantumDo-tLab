"""Tests for the analytical Brus quantum dot bandgap model.
"""

import pytest
import numpy as np
from physics.brus_model import (
    calculate_bandgap,
    calculate_exciton_bohr_radius,
    classify_confinement_regime
)


def test_radius_must_be_positive():
    """Verify that zero or negative radii raise ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_bandgap(radius_nm=0.0, bulk_bandgap_eV=1.74, electron_effective_mass=0.13, hole_effective_mass=0.45, relative_dielectric_constant=9.4)

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_bandgap(radius_nm=-2.0, bulk_bandgap_eV=1.74, electron_effective_mass=0.13, hole_effective_mass=0.45, relative_dielectric_constant=9.4)


def test_invalid_material_parameters():
    """Verify that non-positive masses or dielectric constant raise ValueError."""
    with pytest.raises(ValueError, match="effective mass"):
        calculate_bandgap(radius_nm=2.0, bulk_bandgap_eV=1.74, electron_effective_mass=-0.1, hole_effective_mass=0.45, relative_dielectric_constant=9.4)

    with pytest.raises(ValueError, match="Relative dielectric constant"):
        calculate_bandgap(radius_nm=2.0, bulk_bandgap_eV=1.74, electron_effective_mass=0.13, hole_effective_mass=0.45, relative_dielectric_constant=-5.0)


def test_confinement_shift_increases_as_radius_decreases():
    """Verify that smaller quantum dots have higher bandgap (blue shift) in the Brus model."""
    res_large = calculate_bandgap(radius_nm=5.0, bulk_bandgap_eV=1.74, electron_effective_mass=0.13, hole_effective_mass=0.45, relative_dielectric_constant=9.4)
    res_medium = calculate_bandgap(radius_nm=3.0, bulk_bandgap_eV=1.74, electron_effective_mass=0.13, hole_effective_mass=0.45, relative_dielectric_constant=9.4)
    res_small = calculate_bandgap(radius_nm=1.5, bulk_bandgap_eV=1.74, electron_effective_mass=0.13, hole_effective_mass=0.45, relative_dielectric_constant=9.4)

    assert res_small["qd_bandgap_eV"] > res_medium["qd_bandgap_eV"] > res_large["qd_bandgap_eV"]
    assert res_large["qd_bandgap_eV"] > 1.74  # Must exceed bulk gap


def test_asymptotic_bulk_limit():
    """Verify that as radius becomes very large, QD bandgap approaches bulk bandgap."""
    res_giant = calculate_bandgap(radius_nm=100.0, bulk_bandgap_eV=1.74, electron_effective_mass=0.13, hole_effective_mass=0.45, relative_dielectric_constant=9.4)
    assert pytest.approx(res_giant["qd_bandgap_eV"], abs=0.01) == 1.74


def test_array_input_support():
    """Verify that calculate_bandgap supports numpy arrays for plotting."""
    r_arr = np.array([2.0, 3.0, 4.0, 5.0])
    eg_arr = calculate_bandgap(radius_nm=r_arr, bulk_bandgap_eV=1.74, electron_effective_mass=0.13, hole_effective_mass=0.45, relative_dielectric_constant=9.4)
    assert isinstance(eg_arr, np.ndarray)
    assert len(eg_arr) == 4
    assert np.all(np.diff(eg_arr) < 0)  # Monotonically decreasing with R


def test_exciton_bohr_radius():
    """Verify exciton Bohr radius calculation against analytical formula."""
    # CdSe: me=0.13, mh=0.45, eps=9.4
    # 1/mu = 1/0.13 + 1/0.45 = 7.6923 + 2.2222 = 9.9145
    # a_B = 9.4 * 9.9145 * 0.0529177 = 4.93 nm
    a_b = calculate_exciton_bohr_radius(0.13, 0.45, 9.4)
    assert 4.5 < a_b < 5.5

    # Check confinement classification
    regime_strong, _ = classify_confinement_regime(2.0, a_b)
    assert "Strong" in regime_strong

    regime_weak, _ = classify_confinement_regime(15.0, a_b)
    assert "Weak" in regime_weak
