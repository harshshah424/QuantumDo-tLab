"""Tests for emission wavelength, color mapping, and Gaussian photoluminescence spectrum.
"""

import pytest
import numpy as np
from physics.emission import (
    bandgap_to_wavelength,
    wavelength_to_bandgap,
    wavelength_to_color_category,
    wavelength_to_rgb,
    gaussian_spectrum,
    get_emission_summary
)


def test_bandgap_to_wavelength_conversion():
    """Verify lambda = 1240 / Eg conversion."""
    # Eg = 2.0 eV -> lambda ≈ 620 nm
    wl = bandgap_to_wavelength(2.0)
    assert pytest.approx(wl, abs=1.0) == 620.0

    # Eg = 1.0 eV -> lambda ≈ 1240 nm
    wl2 = bandgap_to_wavelength(1.0)
    assert pytest.approx(wl2, abs=1.0) == 1240.0

    # Test inverse conversion
    eg_recovered = wavelength_to_bandgap(wl)
    assert pytest.approx(eg_recovered, rel=1e-3) == 2.0


def test_invalid_bandgap_raises_error():
    """Verify non-positive bandgap or wavelength raises ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        bandgap_to_wavelength(0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        bandgap_to_wavelength(-1.5)

    with pytest.raises(ValueError, match="strictly positive"):
        wavelength_to_bandgap(-500.0)


def test_wavelength_color_categories():
    """Verify visible and non-visible spectral region classifications."""
    assert wavelength_to_color_category(300.0) == "Ultraviolet (UV)"
    assert wavelength_to_color_category(420.0) == "Violet"
    assert wavelength_to_color_category(470.0) == "Blue"
    assert wavelength_to_color_category(530.0) == "Green"
    assert wavelength_to_color_category(580.0) == "Yellow"
    assert wavelength_to_color_category(605.0) == "Orange"
    assert wavelength_to_color_category(660.0) == "Red"
    assert wavelength_to_color_category(850.0) == "Infrared (IR)"


def test_wavelength_to_rgb_and_hex():
    """Verify RGB and Hex formatting."""
    r, g, b, hex_c = wavelength_to_rgb(530.0)
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255
    assert hex_c.startswith("#")
    assert len(hex_c) == 7
    # For 530 nm (Green), green component should dominate
    assert g > r and g > b


def test_gaussian_spectrum():
    """Verify Gaussian PL spectrum characteristics."""
    w_grid = np.linspace(400, 700, 301)
    peak = 550.0
    fwhm = 30.0
    intensity = gaussian_spectrum(w_grid, peak_wavelength_nm=peak, fwhm_nm=fwhm, peak_intensity=1.0)

    assert len(intensity) == 301
    assert pytest.approx(np.max(intensity), rel=1e-3) == 1.0
    peak_idx = np.argmax(intensity)
    assert pytest.approx(w_grid[peak_idx], abs=1.0) == peak

    # Half maximum check
    half_max_points = w_grid[intensity >= 0.5]
    computed_fwhm = half_max_points[-1] - half_max_points[0]
    assert pytest.approx(computed_fwhm, abs=2.0) == fwhm


def test_emission_summary():
    """Verify emission summary dict structure."""
    summary = get_emission_summary(2.35)
    assert "wavelength_nm" in summary
    assert "color_category" in summary
    assert "hex_color" in summary
    assert summary["is_visible"] is True
