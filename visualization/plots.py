"""Plotly visualization modules for Bandgap vs Radius, Wavelength vs Radius, Material Comparison, and Alloy engineering.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import plotly.graph_objects as go
from physics.brus_model import calculate_bandgap
from physics.emission import bandgap_to_wavelength, wavelength_to_color_category, wavelength_to_rgb
from physics.alloy import interpolate_alloy_properties, compute_alloy_grid_bandgap


def plot_bandgap_vs_radius(
    material_params: Dict[str, Any],
    selected_radius_nm: float,
    radius_min_nm: float = 1.0,
    radius_max_nm: float = 10.0,
    num_points: int = 150
) -> go.Figure:
    """Generate interactive plot of Quantum Dot Bandgap (eV) vs Radius (nm) using Brus equation.

    Highlights the bulk bandgap asymptote and the currently selected radius.
    """
    r_arr = np.linspace(radius_min_nm, radius_max_nm, num_points)
    eg_bulk = float(material_params["bulk_bandgap_eV"])
    me = float(material_params["electron_effective_mass"])
    mh = float(material_params["hole_effective_mass"])
    eps = float(material_params["relative_dielectric_constant"])
    mat_name = material_params.get("formula", material_params.get("material_name", "Material"))

    eg_arr = calculate_bandgap(r_arr, eg_bulk, me, mh, eps)
    cur_data = calculate_bandgap(selected_radius_nm, eg_bulk, me, mh, eps)
    cur_eg = cur_data["qd_bandgap_eV"]

    fig = go.Figure()

    # Bulk bandgap baseline
    fig.add_trace(go.Scatter(
        x=[radius_min_nm, radius_max_nm],
        y=[eg_bulk, eg_bulk],
        mode="lines",
        name=f"Bulk Eg ({eg_bulk:.2f} eV)",
        line=dict(color="rgba(255, 255, 255, 0.4)", dash="dash", width=1.5)
    ))

    # EMA validity warning boundary line
    fig.add_vline(
        x=1.5,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text="EMA Breakdown (<1.5 nm)",
        annotation_position="top right"
    )

    # Brus bandgap curve
    fig.add_trace(go.Scatter(
        x=r_arr,
        y=eg_arr,
        mode="lines",
        name=f"{mat_name} (Brus EMA)",
        line=dict(color="#38bdf8", width=3.0),
        hovertemplate="<b>Radius</b>: %{x:.2f} nm<br><b>Bandgap</b>: %{y:.3f} eV<extra></extra>"
    ))

    # Selected radius marker
    fig.add_trace(go.Scatter(
        x=[selected_radius_nm],
        y=[cur_eg],
        mode="markers+text",
        name=f"Selected (R={selected_radius_nm:.2f} nm)",
        marker=dict(size=14, color="#ef4444", symbol="diamond", line=dict(width=2, color="#ffffff")),
        text=[f"  {cur_eg:.2f} eV"],
        textposition="top right",
        textfont=dict(color="#ef4444", size=13),
        hovertemplate=f"<b>Selected Point</b><br>R = {selected_radius_nm:.2f} nm<br>Eg = {cur_eg:.3f} eV<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>Quantum Confinement: Bandgap vs. Radius ({mat_name})</b>",
        xaxis_title="<b>Quantum Dot Radius R (nm)</b>",
        yaxis_title="<b>Bandgap Eg (eV)</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=50, r=40, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest"
    )
    return fig


def plot_wavelength_vs_radius(
    material_params: Dict[str, Any],
    selected_radius_nm: float,
    radius_min_nm: float = 1.0,
    radius_max_nm: float = 10.0,
    num_points: int = 150
) -> go.Figure:
    """Generate interactive plot of Emission Wavelength (nm) vs Radius (nm)."""
    r_arr = np.linspace(radius_min_nm, radius_max_nm, num_points)
    eg_bulk = float(material_params["bulk_bandgap_eV"])
    me = float(material_params["electron_effective_mass"])
    mh = float(material_params["hole_effective_mass"])
    eps = float(material_params["relative_dielectric_constant"])
    mat_name = material_params.get("formula", material_params.get("material_name", "Material"))

    eg_arr = calculate_bandgap(r_arr, eg_bulk, me, mh, eps)
    wl_arr = bandgap_to_wavelength(eg_arr)

    cur_data = calculate_bandgap(selected_radius_nm, eg_bulk, me, mh, eps)
    cur_wl = float(bandgap_to_wavelength(cur_data["qd_bandgap_eV"]))
    _, _, _, hex_color = wavelength_to_rgb(cur_wl)

    fig = go.Figure()

    # Visible spectrum background shading
    fig.add_hrect(
        y0=380, y1=750,
        fillcolor="rgba(56, 189, 248, 0.08)",
        layer="below",
        line_width=0,
        annotation_text="Visible Spectrum (380 - 750 nm)",
        annotation_position="bottom right"
    )

    # Wavelength curve
    fig.add_trace(go.Scatter(
        x=r_arr,
        y=wl_arr,
        mode="lines",
        name=f"{mat_name} Emission",
        line=dict(color="#a855f7", width=3.0),
        hovertemplate="<b>Radius</b>: %{x:.2f} nm<br><b>Wavelength</b>: %{y:.1f} nm<extra></extra>"
    ))

    # Selected radius marker
    fig.add_trace(go.Scatter(
        x=[selected_radius_nm],
        y=[cur_wl],
        mode="markers+text",
        name=f"Selected ({cur_wl:.1f} nm)",
        marker=dict(size=14, color=hex_color, symbol="circle", line=dict(width=2, color="#ffffff")),
        text=[f"  {cur_wl:.1f} nm"],
        textposition="top right",
        textfont=dict(color="#ffffff", size=13),
        hovertemplate=f"<b>Selected Point</b><br>R = {selected_radius_nm:.2f} nm<br>λ = {cur_wl:.1f} nm<br>Category: {wavelength_to_color_category(cur_wl)}<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>Emission Wavelength vs. Radius ({mat_name})</b>",
        xaxis_title="<b>Quantum Dot Radius R (nm)</b>",
        yaxis_title="<b>Emission Wavelength λ (nm)</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=50, r=40, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest"
    )
    return fig


def plot_material_comparison(
    materials_dict: Dict[str, Dict[str, Any]],
    selected_material_keys: List[str],
    radius_min_nm: float = 1.0,
    radius_max_nm: float = 10.0,
    plot_type: str = "bandgap"
) -> go.Figure:
    """Generate comparative overlay curves across multiple selected semiconductor materials.

    Args:
        materials_dict: Full materials dictionary.
        selected_material_keys: List of selected material keys (e.g. ['CdSe', 'CdS', 'InP', 'PbS']).
        radius_min_nm: Minimum radius.
        radius_max_nm: Maximum radius.
        plot_type: 'bandgap' or 'wavelength'.

    Returns:
        Plotly Figure.
    """
    r_arr = np.linspace(radius_min_nm, radius_max_nm, 120)
    fig = go.Figure()

    colors = ["#38bdf8", "#ec4899", "#10b981", "#f59e0b", "#a855f7", "#06b6d4", "#f43f5e", "#84cc16", "#eab308"]

    for i, key in enumerate(selected_material_keys):
        if key not in materials_dict:
            continue
        props = materials_dict[key]
        color = colors[i % len(colors)]
        mat_name = props.get("formula", key)

        eg_arr = calculate_bandgap(
            r_arr,
            float(props["bulk_bandgap_eV"]),
            float(props["electron_effective_mass"]),
            float(props["hole_effective_mass"]),
            float(props["relative_dielectric_constant"])
        )

        if plot_type == "bandgap":
            fig.add_trace(go.Scatter(
                x=r_arr,
                y=eg_arr,
                mode="lines",
                name=f"{mat_name} (Bulk: {props['bulk_bandgap_eV']:.2f} eV)",
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{mat_name}</b><br>Radius: %{{x:.2f}} nm<br>Bandgap: %{{y:.3f}} eV<extra></extra>"
            ))
        else:
            wl_arr = bandgap_to_wavelength(eg_arr)
            fig.add_trace(go.Scatter(
                x=r_arr,
                y=wl_arr,
                mode="lines",
                name=f"{mat_name}",
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{mat_name}</b><br>Radius: %{{x:.2f}} nm<br>Wavelength: %{{y:.1f}} nm<extra></extra>"
            ))

    if plot_type == "wavelength":
        fig.add_hrect(
            y0=380, y1=750,
            fillcolor="rgba(56, 189, 248, 0.08)",
            layer="below",
            line_width=0,
            annotation_text="Visible Range",
            annotation_position="bottom right"
        )

    title_text = "<b>Material Comparison: Bandgap vs. Radius</b>" if plot_type == "bandgap" else "<b>Material Comparison: Emission Wavelength vs. Radius</b>"
    y_text = "<b>Quantum Dot Bandgap (eV)</b>" if plot_type == "bandgap" else "<b>Emission Wavelength (nm)</b>"

    fig.update_layout(
        title=title_text,
        xaxis_title="<b>Quantum Dot Radius R (nm)</b>",
        yaxis_title=y_text,
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=50, r=40, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    return fig


def plot_alloy_composition_curves(
    mat_a_props: Dict[str, Any],
    mat_b_props: Dict[str, Any],
    bowing_parameter_eV: float,
    current_x: float = 0.5
) -> go.Figure:
    """Plot bulk bandgap vs alloy composition x showing optical bowing."""
    x_arr = np.linspace(0.0, 1.0, 101)
    eg_a = float(mat_a_props["bulk_bandgap_eV"])
    eg_b = float(mat_b_props["bulk_bandgap_eV"])
    name_a = mat_a_props.get("formula", "A")
    name_b = mat_b_props.get("formula", "B")

    eg_bowed = x_arr * eg_a + (1.0 - x_arr) * eg_b - bowing_parameter_eV * x_arr * (1.0 - x_arr)
    eg_linear = x_arr * eg_a + (1.0 - x_arr) * eg_b

    cur_eg = current_x * eg_a + (1.0 - current_x) * eg_b - bowing_parameter_eV * current_x * (1.0 - current_x)

    fig = go.Figure()

    # Linear Vegard's law baseline (zero bowing)
    fig.add_trace(go.Scatter(
        x=x_arr,
        y=eg_linear,
        mode="lines",
        name="Linear (b = 0 eV)",
        line=dict(color="rgba(255, 255, 255, 0.3)", dash="dash")
    ))

    # Bowed alloy curve
    fig.add_trace(go.Scatter(
        x=x_arr,
        y=eg_bowed,
        mode="lines",
        name=f"Bowed Eg (b = {bowing_parameter_eV:.2f} eV)",
        line=dict(color="#10b981", width=3.0),
        hovertemplate="<b>Composition x</b>: %{x:.2f}<br><b>Bulk Eg</b>: %{y:.3f} eV<extra></extra>"
    ))

    # Current selected composition point
    fig.add_trace(go.Scatter(
        x=[current_x],
        y=[cur_eg],
        mode="markers+text",
        name=f"Selected (x={current_x:.2f})",
        marker=dict(size=14, color="#f59e0b", symbol="circle", line=dict(width=2, color="#ffffff")),
        text=[f"  {cur_eg:.3f} eV"],
        textposition="top center",
        textfont=dict(color="#f59e0b", size=13),
        hovertemplate=f"<b>Selected Composition</b><br>x = {current_x:.2f}<br>Eg = {cur_eg:.3f} eV<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>Ternary Alloy Bandgap vs. Composition: {name_a}(x) {name_b}(1-x)</b>",
        xaxis_title=f"<b>Composition Fraction x ({name_a})</b>",
        yaxis_title="<b>Bulk Bandgap Eg (eV)</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=50, r=40, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_alloy_heatmap(
    mat_a_props: Dict[str, Any],
    mat_b_props: Dict[str, Any],
    bowing_parameter_eV: float,
    radius_min_nm: float = 1.0,
    radius_max_nm: float = 8.0,
    num_grid: int = 50
) -> go.Figure:
    """Generate 2D Contour Heatmap: X = Radius (nm), Y = Composition x, Color = Bandgap (eV)."""
    r_arr = np.linspace(radius_min_nm, radius_max_nm, num_grid)
    x_arr = np.linspace(0.0, 1.0, num_grid)
    name_a = mat_a_props.get("formula", "A")
    name_b = mat_b_props.get("formula", "B")

    z_matrix = compute_alloy_grid_bandgap(mat_a_props, mat_b_props, r_arr, x_arr, bowing_parameter_eV)

    fig = go.Figure(data=go.Contour(
        z=z_matrix,
        x=r_arr,
        y=x_arr,
        colorscale="Viridis",
        colorbar=dict(title=dict(text="<b>Eg (eV)</b>", side="right")),
        contours=dict(
            coloring="heatmap",
            showlabels=True,
            labelfont=dict(size=11, color="white")
        ),
        hovertemplate="<b>Radius</b>: %{x:.2f} nm<br><b>Composition x</b>: %{y:.2f}<br><b>Bandgap</b>: %{z:.3f} eV<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>2D Bandgap Mapping: Radius vs. Alloy Composition ({name_a}_{name_b})</b>",
        xaxis_title="<b>Quantum Dot Radius R (nm)</b>",
        yaxis_title=f"<b>Composition Fraction x ({name_a})</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=50, r=40, t=60, b=50)
    )
    return fig
