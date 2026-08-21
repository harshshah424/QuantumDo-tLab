"""Interactive visualization of electron and hole radial wavefunctions and probability densities.
"""

from typing import Dict, Any
import numpy as np
import plotly.graph_objects as go


def plot_probability_densities(
    sim_results: Dict[str, Any],
    plot_type: str = "density"  # "density" for |u(r)|^2, "wavefunction" for u(r)
) -> go.Figure:
    """Generate interactive Plotly plot of electron and hole radial probability density |u(r)|^2.

    Args:
        sim_results: Results dict from solve_core_shell_system().
        plot_type: "density" for |u|^2 or "wavefunction" for u(r).

    Returns:
        Plotly Figure.
    """
    r_grid = sim_results["r_grid_nm"]
    rc = sim_results["r_core_nm"]
    rt = sim_results["r_total_nm"]

    ue = sim_results["u_e"]
    uh = sim_results["u_h"]
    pe = sim_results["prob_density_e"]
    ph = sim_results["prob_density_h"]

    e_loc = sim_results["electron_localization"]
    h_loc = sim_results["hole_localization"]

    y_e = pe if plot_type == "density" else ue
    y_h = ph if plot_type == "density" else uh
    y_label = "<b>Radial Probability Density |u(r)|² (nm⁻¹)</b>" if plot_type == "density" else "<b>Reduced Wavefunction u(r) (nm⁻¹/²)</b>"
    title_suffix = "Probability Densities |u(r)|²" if plot_type == "density" else "Reduced Wavefunctions u(r)"

    fig = go.Figure()

    # Region highlights
    # Core
    fig.add_vrect(
        x0=0, x1=rc,
        fillcolor="rgba(56, 189, 248, 0.10)",
        layer="below",
        line_width=1,
        line_color="rgba(56, 189, 248, 0.3)",
        annotation_text=f"<b>Core</b> (R = {rc:.2f} nm)",
        annotation_position="top left",
        annotation_font=dict(color="#38bdf8", size=11)
    )

    # Shell
    if rt > rc:
        fig.add_vrect(
            x0=rc, x1=rt,
            fillcolor="rgba(168, 85, 247, 0.10)",
            layer="below",
            line_width=1,
            line_color="rgba(168, 85, 247, 0.3)",
            annotation_text=f"<b>Shell</b> (t = {rt-rc:.2f} nm)",
            annotation_position="top",
            annotation_font=dict(color="#a855f7", size=11)
        )

    # Outer region
    fig.add_vrect(
        x0=rt, x1=r_grid[-1],
        fillcolor="rgba(100, 116, 139, 0.05)",
        layer="below",
        line_width=0,
        annotation_text="Matrix / Barrier",
        annotation_position="top right",
        annotation_font=dict(color="#94a3b8", size=10)
    )

    # Electron trace
    fig.add_trace(go.Scatter(
        x=r_grid,
        y=y_e,
        mode="lines",
        name=f"Electron 1S (Core: {e_loc['core_percent']:.1f}%, Shell: {e_loc['shell_percent']:.1f}%)",
        line=dict(color="#22d3ee", width=3.0),
        fill="tozeroy" if plot_type == "density" else None,
        fillcolor="rgba(34, 211, 238, 0.20)" if plot_type == "density" else None,
        hovertemplate="<b>Radius</b>: %{x:.2f} nm<br><b>Electron</b>: %{y:.4f}<extra></extra>"
    ))

    # Hole trace
    fig.add_trace(go.Scatter(
        x=r_grid,
        y=y_h,
        mode="lines",
        name=f"Hole 1S (Core: {h_loc['core_percent']:.1f}%, Shell: {h_loc['shell_percent']:.1f}%)",
        line=dict(color="#fb923c", width=3.0),
        fill="tozeroy" if plot_type == "density" else None,
        fillcolor="rgba(251, 146, 60, 0.20)" if plot_type == "density" else None,
        hovertemplate="<b>Radius</b>: %{x:.2f} nm<br><b>Hole</b>: %{y:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>Numerical Carrier {title_suffix}</b>",
        xaxis_title="<b>Radial Coordinate r (nm)</b>",
        yaxis_title=y_label,
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=50, r=40, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    return fig
