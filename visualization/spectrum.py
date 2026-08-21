"""Photoluminescence emission spectrum visualization with dynamic color mapping.
"""

from typing import Dict, Any
import numpy as np
import plotly.graph_objects as go
from physics.emission import (
    gaussian_spectrum,
    wavelength_to_rgb,
    wavelength_to_color_category
)


def plot_emission_spectrum(
    peak_wavelength_nm: float,
    fwhm_nm: float = 25.0,
    peak_intensity: float = 1.0,
    window_span_nm: float = 180.0,
    num_points: int = 400
) -> go.Figure:
    """Generate interactive Plotly Gaussian photoluminescence emission spectrum.

    Args:
        peak_wavelength_nm: Peak emission wavelength lambda_0 in nm.
        fwhm_nm: Spectral linewidth Full Width at Half Maximum in nm.
        peak_intensity: Peak luminescence intensity.
        window_span_nm: Total span around the peak to display.
        num_points: Number of wavelength points.

    Returns:
        Plotly Figure.
    """
    w_min = max(200.0, peak_wavelength_nm - window_span_nm / 2.0)
    w_max = peak_wavelength_nm + window_span_nm / 2.0
    w_grid = np.linspace(w_min, w_max, num_points)

    intensity = gaussian_spectrum(w_grid, peak_wavelength_nm, fwhm_nm, peak_intensity)
    r, g, b, hex_color = wavelength_to_rgb(peak_wavelength_nm)
    category = wavelength_to_color_category(peak_wavelength_nm)

    fig = go.Figure()

    # Shaded curve with fill
    fig.add_trace(go.Scatter(
        x=w_grid,
        y=intensity,
        mode="lines",
        name=f"PL Emission (λ₀ = {peak_wavelength_nm:.1f} nm)",
        line=dict(color=hex_color, width=3.0),
        fill="tozeroy",
        fillcolor=f"rgba({r}, {g}, {b}, 0.35)",
        hovertemplate="<b>Wavelength</b>: %{x:.1f} nm<br><b>Intensity</b>: %{y:.3f}<extra></extra>"
    ))

    # Vertical line at peak wavelength
    fig.add_vline(
        x=peak_wavelength_nm,
        line_dash="dash",
        line_color="#ffffff",
        line_width=1.5,
        annotation_text=f"λ₀ = {peak_wavelength_nm:.1f} nm ({category})",
        annotation_position="top",
        annotation_font=dict(color="#ffffff", size=12)
    )

    # Visible spectrum range highlight if in window
    if w_min < 750 and w_max > 380:
        fig.add_vrect(
            x0=max(w_min, 380),
            x1=min(w_max, 750),
            fillcolor="rgba(255, 255, 255, 0.03)",
            layer="below",
            line_width=0,
            annotation_text="Visible Range",
            annotation_position="bottom right"
        )

    fig.update_layout(
        title=f"<b>Photoluminescence Emission Spectrum: {category} (λ₀ = {peak_wavelength_nm:.1f} nm, FWHM = {fwhm_nm:.1f} nm)</b>",
        xaxis_title="<b>Wavelength λ (nm)</b>",
        yaxis_title="<b>Normalized Intensity (a.u.)</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=50, r=40, t=60, b=50),
        yaxis=dict(range=[0, peak_intensity * 1.15]),
        hovermode="closest"
    )
    return fig
