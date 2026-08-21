"""Interactive energy band alignment diagram for Core/Shell quantum dots.
"""

from typing import Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
from physics.core_shell import CoreShellStructure


def plot_core_shell_band_diagram(
    core_shell_struct: CoreShellStructure,
    sim_results: Optional[Dict[str, Any]] = None,
    r_max_display_nm: Optional[float] = None
) -> go.Figure:
    """Generate an interactive Plotly energy band diagram showing Core, Shell, Conduction/Valence bands, and quantized energy levels.

    Args:
        core_shell_struct: CoreShellStructure instance.
        sim_results: Optional dictionary from numerical Schrödinger solver.
        r_max_display_nm: Maximum radial extent to plot.

    Returns:
        Plotly Figure.
    """
    rc = core_shell_struct.r_core
    rt = core_shell_struct.r_total
    rmax = r_max_display_nm if r_max_display_nm is not None else max(rt * 1.5, rt + 1.5)

    # Reference core band edges (vacuum referenced)
    cbm_core = float(core_shell_struct.core_params.get("conduction_band_edge_eV", -4.30))
    vbm_core = float(core_shell_struct.core_params.get("valence_band_edge_eV", -6.04))

    delta_ec = core_shell_struct.delta_ec
    delta_ev = core_shell_struct.delta_ev
    outer_bar = core_shell_struct.outer_barrier

    # Shell band edges
    cbm_shell = cbm_core + delta_ec
    vbm_shell = vbm_core - delta_ev

    # Outer barrier band edges
    cbm_outer = max(cbm_shell, cbm_core) + outer_bar
    vbm_outer = min(vbm_shell, vbm_core) - outer_bar

    # Construct step profiles for plotting
    r_steps = [0.0, rc, rc, rt, rt, rmax]
    ec_steps = [cbm_core, cbm_core, cbm_shell, cbm_shell, cbm_outer, cbm_outer]
    ev_steps = [vbm_core, vbm_core, vbm_shell, vbm_shell, vbm_outer, vbm_outer]

    fig = go.Figure()

    # Region shading
    # Core region
    fig.add_vrect(
        x0=0, x1=rc,
        fillcolor="rgba(56, 189, 248, 0.12)",
        layer="below",
        line_width=1,
        line_color="rgba(56, 189, 248, 0.3)",
        annotation_text=f"<b>Core</b> ({core_shell_struct.core_material})<br>R = {rc:.2f} nm",
        annotation_position="top left",
        annotation_font=dict(color="#38bdf8", size=11)
    )

    # Shell region
    if core_shell_struct.t_shell > 0:
        fig.add_vrect(
            x0=rc, x1=rt,
            fillcolor="rgba(168, 85, 247, 0.12)",
            layer="below",
            line_width=1,
            line_color="rgba(168, 85, 247, 0.3)",
            annotation_text=f"<b>Shell</b> ({core_shell_struct.shell_material})<br>t = {core_shell_struct.t_shell:.2f} nm",
            annotation_position="top right",
            annotation_font=dict(color="#a855f7", size=11)
        )

    # Outer matrix region
    fig.add_vrect(
        x0=rt, x1=rmax,
        fillcolor="rgba(100, 116, 139, 0.08)",
        layer="below",
        line_width=0,
        annotation_text="Matrix / Ligands",
        annotation_position="top right",
        annotation_font=dict(color="#94a3b8", size=10)
    )

    # Conduction Band Edge (CB)
    fig.add_trace(go.Scatter(
        x=r_steps,
        y=ec_steps,
        mode="lines",
        name="Conduction Band Edge (Ec)",
        line=dict(color="#38bdf8", width=3.5),
        hovertemplate="<b>Radius</b>: %{x:.2f} nm<br><b>Ec</b>: %{y:.2f} eV<extra></extra>"
    ))

    # Valence Band Edge (VB)
    fig.add_trace(go.Scatter(
        x=r_steps,
        y=ev_steps,
        mode="lines",
        name="Valence Band Edge (Ev)",
        line=dict(color="#f43f5e", width=3.5),
        hovertemplate="<b>Radius</b>: %{x:.2f} nm<br><b>Ev</b>: %{y:.2f} eV<extra></extra>"
    ))

    # If simulation eigenvalues are present, display quantized carrier levels
    if sim_results is not None:
        e_e = sim_results["electron_ground_energy_eV"]
        e_h = sim_results["hole_ground_energy_eV"]
        abs_e_e = cbm_core + e_e
        abs_e_h = vbm_core - e_h

        # Electron ground level
        fig.add_trace(go.Scatter(
            x=[0, rt],
            y=[abs_e_e, abs_e_e],
            mode="lines",
            name=f"Electron State Ee (+{e_e:.3f} eV)",
            line=dict(color="#22d3ee", width=2.5, dash="dashdot"),
            hovertemplate=f"<b>Electron Level</b>: {abs_e_e:.3f} eV (Confinement: +{e_e:.3f} eV)<extra></extra>"
        ))

        # Hole ground level
        fig.add_trace(go.Scatter(
            x=[0, rt],
            y=[abs_e_h, abs_e_h],
            mode="lines",
            name=f"Hole State Eh (-{e_h:.3f} eV)",
            line=dict(color="#fb923c", width=2.5, dash="dashdot"),
            hovertemplate=f"<b>Hole Level</b>: {abs_e_h:.3f} eV (Confinement: +{e_h:.3f} eV)<extra></extra>"
        ))

    # Band offset annotations
    fig.add_annotation(
        x=rc,
        y=(cbm_core + cbm_shell) / 2.0,
        text=f"ΔEc = {delta_ec:+.2f} eV",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#38bdf8",
        ax=-40,
        ay=0,
        font=dict(color="#38bdf8", size=11),
        bgcolor="rgba(15, 23, 42, 0.85)"
    )

    fig.add_annotation(
        x=rc,
        y=(vbm_core + vbm_shell) / 2.0,
        text=f"ΔEv = {delta_ev:+.2f} eV",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#f43f5e",
        ax=-40,
        ay=0,
        font=dict(color="#f43f5e", size=11),
        bgcolor="rgba(15, 23, 42, 0.85)"
    )

    fig.update_layout(
        title=f"<b>Core/Shell Energy Band Alignment: {core_shell_struct.core_material} / {core_shell_struct.shell_material}</b>",
        xaxis_title="<b>Radial Coordinate r (nm)</b>",
        yaxis_title="<b>Energy (eV, Vacuum-Referenced)</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=50, r=40, t=60, b=50),
        xaxis=dict(range=[0, rmax]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
