"""Optical emission properties, wavelength conversion, color classification, and Gaussian PL spectrum.
"""

from typing import Dict, Any, Tuple
import numpy as np
from physics.constants import HC_EV_NM


def bandgap_to_wavelength(eg_eV: float | np.ndarray) -> float | np.ndarray:
    """Convert electronic bandgap in eV to optical emission wavelength in nm.

    Formula:
        lambda (nm) = hc / Eg (eV) ≈ 1239.84 / Eg

    Args:
        eg_eV: Bandgap in eV (scalar or numpy array).

    Returns:
        Emission wavelength in nm.
    """
    arr = np.asarray(eg_eV, dtype=float)
    if np.any(arr <= 0):
        raise ValueError("Bandgap must be strictly positive (> 0 eV) to compute emission wavelength.")
    
    wavelength = HC_EV_NM / arr
    if np.isscalar(eg_eV) or arr.ndim == 0:
        return float(wavelength)
    return wavelength


def wavelength_to_bandgap(wavelength_nm: float | np.ndarray) -> float | np.ndarray:
    """Convert optical emission wavelength in nm to photon energy / bandgap in eV.

    Formula:
        Eg (eV) = hc / lambda (nm) ≈ 1239.84 / lambda

    Args:
        wavelength_nm: Wavelength in nm (scalar or numpy array).

    Returns:
        Bandgap / photon energy in eV.
    """
    arr = np.asarray(wavelength_nm, dtype=float)
    if np.any(arr <= 0):
        raise ValueError("Wavelength must be strictly positive (> 0 nm).")
    
    eg = HC_EV_NM / arr
    if np.isscalar(wavelength_nm) or arr.ndim == 0:
        return float(eg)
    return eg


def wavelength_to_color_category(wavelength_nm: float) -> str:
    """Classify emission wavelength into approximate spectral colour bands.

    Categories:
        - Ultraviolet (UV): < 380 nm
        - Violet: 380 nm – 450 nm
        - Blue: 450 nm – 495 nm
        - Green: 495 nm – 570 nm
        - Yellow: 570 nm – 590 nm
        - Orange: 590 nm – 620 nm
        - Red: 620 nm – 750 nm
        - Infrared (IR): > 750 nm

    Args:
        wavelength_nm: Emission wavelength in nanometers.

    Returns:
        String name of the colour classification.
    """
    w = float(wavelength_nm)
    if w < 380.0:
        return "Ultraviolet (UV)"
    elif w < 450.0:
        return "Violet"
    elif w < 495.0:
        return "Blue"
    elif w < 570.0:
        return "Green"
    elif w < 590.0:
        return "Yellow"
    elif w < 620.0:
        return "Orange"
    elif w <= 750.0:
        return "Red"
    else:
        return "Infrared (IR)"


def wavelength_to_rgb(wavelength_nm: float) -> Tuple[int, int, int, str]:
    """Convert optical wavelength in nm to RGB color tuple and hex string.

    Uses standard Dan Bruton visible spectrum approximation algorithm.
    Falls back to dimmed boundary tones for UV (< 380 nm) and IR (> 750 nm).

    Args:
        wavelength_nm: Wavelength in nanometers.

    Returns:
        Tuple of (R, G, B, hex_string).
    """
    w = float(wavelength_nm)
    gamma = 0.80

    if w < 380:
        # Near UV representation (deep indigo/violet glow)
        r, g, b = 0.4, 0.0, 0.6
        intensity = max(0.2, min(1.0, 1.0 - (380 - w) / 200.0))
    elif 380 <= w < 440:
        r = -(w - 440) / (440 - 380)
        g = 0.0
        b = 1.0
        intensity = 0.3 + 0.7 * (w - 380) / (440 - 380)
    elif 440 <= w < 490:
        r = 0.0
        g = (w - 440) / (490 - 440)
        b = 1.0
        intensity = 1.0
    elif 490 <= w < 510:
        r = 0.0
        g = 1.0
        b = -(w - 510) / (510 - 490)
        intensity = 1.0
    elif 510 <= w < 580:
        r = (w - 510) / (580 - 510)
        g = 1.0
        b = 0.0
        intensity = 1.0
    elif 580 <= w < 645:
        r = 1.0
        g = -(w - 645) / (645 - 580)
        b = 0.0
        intensity = 1.0
    elif 645 <= w <= 750:
        r = 1.0
        g = 0.0
        b = 0.0
        intensity = 0.3 + 0.7 * (750 - w) / (750 - 645)
    else:
        # Near IR representation (deep dark wine red)
        r, g, b = 0.5, 0.0, 0.0
        intensity = max(0.15, min(1.0, 1.0 - (w - 750) / 400.0))

    r_final = int(round(255 * (r * intensity) ** gamma))
    g_final = int(round(255 * (g * intensity) ** gamma))
    b_final = int(round(255 * (b * intensity) ** gamma))

    # Clamp values
    r_final = max(0, min(255, r_final))
    g_final = max(0, min(255, g_final))
    b_final = max(0, min(255, b_final))

    hex_code = f"#{r_final:02x}{g_final:02x}{b_final:02x}"
    return r_final, g_final, b_final, hex_code


def gaussian_spectrum(
    wavelength_grid_nm: np.ndarray,
    peak_wavelength_nm: float,
    fwhm_nm: float = 25.0,
    peak_intensity: float = 1.0
) -> np.ndarray:
    """Generate a phenomenological Gaussian photoluminescence (PL) emission spectrum.

    I(lambda) = I_0 * exp( - (lambda - lambda_0)^2 / (2 * sigma^2) )
    where sigma = FWHM / (2 * sqrt(2 * ln(2))) ≈ FWHM / 2.35482

    Args:
        wavelength_grid_nm: 1D array of wavelength sample points in nm.
        peak_wavelength_nm: Emission peak wavelength lambda_0 in nm.
        fwhm_nm: Full Width at Half Maximum in nm (typical QD PL: 20-40 nm).
        peak_intensity: Relative peak intensity I_0 (default 1.0).

    Returns:
        1D array of spectral intensities.
    """
    if fwhm_nm <= 0:
        raise ValueError(f"FWHM must be positive, got {fwhm_nm}")
    if peak_intensity < 0:
        raise ValueError(f"Peak intensity cannot be negative, got {peak_intensity}")

    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    intensity = peak_intensity * np.exp(-((wavelength_grid_nm - peak_wavelength_nm) ** 2) / (2.0 * sigma ** 2))
    return intensity


def get_emission_summary(eg_eV: float) -> Dict[str, Any]:
    """Provide a complete optical emission summary package for a given bandgap.

    Args:
        eg_eV: Quantum dot bandgap in eV.

    Returns:
        Dictionary with wavelength, color category, RGB, hex code, and visible flag.
    """
    wavelength = float(bandgap_to_wavelength(eg_eV))
    category = wavelength_to_color_category(wavelength)
    r, g, b, hex_code = wavelength_to_rgb(wavelength)
    is_visible = 380.0 <= wavelength <= 750.0

    return {
        "bandgap_eV": eg_eV,
        "wavelength_nm": wavelength,
        "color_category": category,
        "rgb": (r, g, b),
        "hex_color": hex_code,
        "is_visible": is_visible
    }
