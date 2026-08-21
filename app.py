"""QuantumDotLab: Interactive Quantum Dot Bandgap and Optical Properties Simulator.

Main Streamlit Application Dashboard.
"""

import json
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

from physics.constants import EMA_MIN_RADIUS_NM, HC_EV_NM
from physics.brus_model import (
    calculate_bandgap,
    calculate_exciton_bohr_radius,
    classify_confinement_regime
)
from physics.emission import (
    bandgap_to_wavelength,
    wavelength_to_color_category,
    wavelength_to_rgb,
    get_emission_summary
)
from physics.alloy import interpolate_alloy_properties
from physics.band_alignment import classify_band_alignment
from physics.core_shell import CoreShellStructure
from physics.schrodinger_solver import solve_core_shell_system
from physics.strain import calculate_hydrostatic_strain_shift

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


# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="QuantumDotLab | QD Simulator",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for scientific styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #38bdf8;
        margin-top: 4px;
    }
    .color-swatch-badge {
        display: inline-block;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
        border: 1px solid #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_materials_database() -> Dict[str, Any]:
    """Load the semiconductor materials database from JSON file."""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(curr_dir, "data", "materials.json")
    if not os.path.exists(json_path):
        st.error(f"Materials database not found at {json_path}")
        return {"materials": {}, "default_core_shell_pairs": {}, "default_alloys": {}}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


db = load_materials_database()
materials_dict = db.get("materials", {})
default_core_shell_pairs = db.get("default_core_shell_pairs", {})
default_alloys = db.get("default_alloys", {})

# --- Sidebar Controls ---
st.sidebar.markdown("## ⚙️ Simulator Controls")

# Model Fidelity Mode Selector
sim_mode = st.sidebar.selectbox(
    "Simulation Fidelity Mode",
    [
        "Mode 1: Fast Analytical Mode (Brus EMA)",
        "Mode 2: Core/Shell Numerical Mode (Schrödinger Solver)",
        "Mode 3: Advanced / Experimental Mode (Strain & Corrections)"
    ],
    index=0,
    help="Select the physics fidelity level."
)

st.sidebar.markdown("---")

# Material Selection
mat_keys = list(materials_dict.keys())
default_mat_idx = mat_keys.index("CdSe") if "CdSe" in mat_keys else 0

if "Mode 2" in sim_mode or "Mode 3" in sim_mode:
    st.sidebar.markdown("### ⚛️ Core/Shell Materials")
    core_mat_key = st.sidebar.selectbox("Core Material", mat_keys, index=default_mat_idx, key="core_mat_select")
    shell_default_idx = mat_keys.index("ZnS") if "ZnS" in mat_keys else (1 % len(mat_keys))
    shell_mat_key = st.sidebar.selectbox("Shell Material", mat_keys, index=shell_default_idx, key="shell_mat_select")

    core_radius = st.sidebar.slider("Core Radius R_core (nm)", min_value=0.8, max_value=8.0, value=2.2, step=0.1)
    shell_thickness = st.sidebar.slider("Shell Thickness t_shell (nm)", min_value=0.0, max_value=4.0, value=1.0, step=0.1)
    active_mat_key = core_mat_key
    active_radius = core_radius
else:
    st.sidebar.markdown("### 🔬 Material & Geometry")
    active_mat_key = st.sidebar.selectbox("Semiconductor Material", mat_keys, index=default_mat_idx)
    active_radius = st.sidebar.slider("Quantum Dot Radius R (nm)", min_value=0.8, max_value=10.0, value=2.2, step=0.1)

# Active Material Parameters
mat_props = materials_dict[active_mat_key]

# Allow custom overrides in expander
with st.sidebar.expander("🛠️ Advanced Material Parameters"):
    st.caption("Customize bulk properties if calibrating against specific synthesis literature:")
    override_eg = st.number_input("Bulk Bandgap (eV)", value=float(mat_props["bulk_bandgap_eV"]), step=0.05, format="%.3f")
    override_me = st.number_input("Electron Mass (m*/m0)", value=float(mat_props["electron_effective_mass"]), step=0.01, format="%.3f")
    override_mh = st.number_input("Hole Mass (m*/m0)", value=float(mat_props["hole_effective_mass"]), step=0.02, format="%.3f")
    override_eps = st.number_input("Dielectric Const (eps_r)", value=float(mat_props["relative_dielectric_constant"]), step=0.2, format="%.2f")

    active_params = dict(mat_props)
    active_params["bulk_bandgap_eV"] = override_eg
    active_params["electron_effective_mass"] = override_me
    active_params["hole_effective_mass"] = override_mh
    active_params["relative_dielectric_constant"] = override_eps

# --- Core Calculations ---
brus_res = calculate_bandgap(
    radius_nm=active_radius,
    bulk_bandgap_eV=active_params["bulk_bandgap_eV"],
    electron_effective_mass=active_params["electron_effective_mass"],
    hole_effective_mass=active_params["hole_effective_mass"],
    relative_dielectric_constant=active_params["relative_dielectric_constant"]
)

# Core/Shell / Numerical calculations if in Mode 2 or 3
num_res = None
core_shell_struct = None
strain_res = None

if "Mode 2" in sim_mode or "Mode 3" in sim_mode:
    shell_props = materials_dict[shell_mat_key]

    # Pre-populate band offsets
    cs_pair_key = f"{core_mat_key}/{shell_mat_key}"
    default_offsets = default_core_shell_pairs.get(cs_pair_key, {})
    default_ec = default_offsets.get("delta_ec_eV", float(shell_props["conduction_band_edge_eV"]) - float(active_params["conduction_band_edge_eV"]))
    default_ev = default_offsets.get("delta_ev_eV", float(active_params["valence_band_edge_eV"]) - float(shell_props["valence_band_edge_eV"]))

    with st.sidebar.expander("⚡ Core/Shell Band Offsets & Alignment"):
        custom_ec = st.number_input("Conduction Offset ΔEc (eV)", value=float(default_ec), step=0.05, format="%.2f",
                                    help="Positive: Shell CB is higher than Core CB (well for electrons).")
        custom_ev = st.number_input("Valence Offset ΔEv (eV)", value=float(default_ev), step=0.05, format="%.2f",
                                    help="Positive: Shell VB is lower than Core VB (well for holes).")
        custom_barrier = st.number_input("Outer Matrix Barrier (eV)", value=3.5, step=0.5, format="%.1f")

    core_shell_struct = CoreShellStructure(
        core_material=core_mat_key,
        shell_material=shell_mat_key,
        core_radius_nm=core_radius,
        shell_thickness_nm=shell_thickness,
        core_params=active_params,
        shell_params=shell_props,
        delta_ec_eV=custom_ec,
        delta_ev_eV=custom_ev,
        outer_barrier_eV=custom_barrier
    )

    # Solve radial Schrödinger system
    num_res = solve_core_shell_system(core_shell_struct, num_grid_points=350)

    # Strain calculation
    strain_enabled = "Mode 3" in sim_mode
    strain_res = calculate_hydrostatic_strain_shift(
        core_radius_nm=core_radius,
        shell_thickness_nm=shell_thickness,
        a_core_angstrom=float(active_params.get("lattice_constant_angstrom", 4.30)),
        a_shell_angstrom=float(shell_props.get("lattice_constant_angstrom", 5.41)),
        deformation_potential_eV=float(active_params.get("hydrostatic_deformation_pot_eV", -3.0)),
        enabled=strain_enabled
    )

    # Active gap in Mode 2 / 3
    final_qd_gap = num_res["qd_bandgap_eV"] + (strain_res["strain_shift_eV"] if strain_enabled else 0.0)
else:
    final_qd_gap = brus_res["qd_bandgap_eV"]

emission_info = get_emission_summary(final_qd_gap)

# --- Header Section ---
st.markdown('<div class="main-title">QuantumDotLab</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive Quantum Dot Bandgap, Confinement, Core/Shell, and Optical Properties Simulator</div>', unsafe_allow_html=True)

# Top KPIs Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Bulk Bandgap</div>
        <div class="metric-value">{active_params['bulk_bandgap_eV']:.3f} eV</div>
        <div class="metric-sub">{active_mat_key} (Unconfined)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    shift_val = (final_qd_gap - active_params['bulk_bandgap_eV']) * 1000.0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">QD Bandgap</div>
        <div class="metric-value">{final_qd_gap:.3f} eV</div>
        <div class="metric-sub">Shift: +{shift_val:.1f} meV</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Emission Peak (λ)</div>
        <div class="metric-value">{emission_info['wavelength_nm']:.1f} nm</div>
        <div class="metric-sub">{emission_info['color_category']}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    hex_c = emission_info['hex_color']
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Visible Color</div>
        <div class="metric-value"><span class="color-swatch-badge" style="background-color: {hex_c};"></span>{emission_info['color_category']}</div>
        <div class="metric-sub">Hex: {hex_c.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    bohr_r = brus_res["exciton_bohr_radius_nm"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Exciton Bohr Radius</div>
        <div class="metric-value">{bohr_r:.2f} nm</div>
        <div class="metric-sub">{brus_res['confinement_regime']}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# EMA Warning banner if radius is very small
if active_radius < EMA_MIN_RADIUS_NM:
    st.warning(
        f"⚠️ **EMA Validity Warning**: Selected radius ($R = {active_radius:.2f}\\text{{ nm}}$) is below the "
        f"effective-mass threshold (${EMA_MIN_RADIUS_NM}\\text{{ nm}}$). Simple parabolic effective mass over-predicts "
        "the blue-shift in sub-nanometer clusters due to atomistic reconstruction and band non-parabolicity."
    )

# --- Tabbed Main Dashboard ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🔬 Bandgap & Radius",
    "🌈 Emission Spectrum",
    "⚖️ Material Comparison",
    "🔀 Alloy Engineering",
    "⚛️ Core/Shell Band Alignment",
    "🌊 Carrier Wavefunctions",
    "📐 Strain & Advanced Effects",
    "📚 Scientific Foundations"
])

# ==========================================
# TAB 1: Bandgap & Radius Analysis
# ==========================================
with tab1:
    st.markdown("### 🔬 Quantum Confinement Scaling")
    st.caption("Explore how shrinking the quantum dot radius blue-shifts the electronic bandgap via kinetic confinement vs Coulomb attraction.")

    c1, c2 = st.columns([2, 1])
    with c1:
        fig_eg = plot_bandgap_vs_radius(active_params, active_radius, radius_min_nm=0.8, radius_max_nm=10.0)
        st.plotly_chart(fig_eg, use_container_width=True)

    with c2:
        st.markdown("#### 📊 Energy Breakdown (Brus Model)")
        st.markdown(f"""
        - **Bulk Bandgap ($E_{{g,\\text{{bulk}}}}$)**: `{brus_res['bulk_bandgap_eV']:.3f} eV`
        - **Kinetic Confinement ($E_{{\\text{{conf}}}}$)**: `+{brus_res['confinement_energy_eV']:.3f} eV`
          - Scaling: $\\propto 1/R^2$ (dominant at small $R$)
        - **Coulomb Attraction ($E_{{\\text{{Coulomb}}}}$)**: `-{brus_res['coulomb_energy_eV']:.3f} eV`
          - Scaling: $\\propto 1/R$ (weak perturbation)
        - **Net Quantum Shift**: `+{brus_res['net_shift_eV']:.3f} eV`
        - **Calculated QD Bandgap ($E_{{g,\\text{{QD}}}}$)**: `{brus_res['qd_bandgap_eV']:.3f} eV`
        """)

        st.info(f"**Confinement Regime**: {brus_res['confinement_regime']}\n\n{brus_res['confinement_description']}")

    st.markdown("---")
    st.markdown("### 📈 Emission Wavelength vs. Radius")
    fig_wl = plot_wavelength_vs_radius(active_params, active_radius, radius_min_nm=0.8, radius_max_nm=10.0)
    st.plotly_chart(fig_wl, use_container_width=True)

# ==========================================
# TAB 2: Photoluminescence Emission Spectrum
# ==========================================
with tab2:
    st.markdown("### 🌈 Photoluminescence (PL) Emission Spectrum")
    st.caption("Phenomenological Gaussian emission profile centered at the quantum dot optical transition wavelength.")

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.markdown("#### 🎛️ Spectral Parameters")
        fwhm_slider = st.slider("Emission Linewidth / FWHM (nm)", min_value=10.0, max_value=60.0, value=25.0, step=1.0,
                                help="Colloidal QD samples typically exhibit 20-35 nm FWHM due to size distribution and phonon broadening.")
        intensity_slider = st.slider("Relative Peak Intensity (I₀)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

        st.markdown(f"""
        - **Peak Wavelength ($\\lambda_0$)**: `{emission_info['wavelength_nm']:.1f} nm`
        - **Photon Energy ($h\\nu$)**: `{final_qd_gap:.3f} eV`
        - **Color Band**: `{emission_info['color_category']}`
        - **Standard Deviation ($\\sigma$)**: `{fwhm_slider / 2.35482:.2f} nm`
        """)

        st.caption("📌 *Note: This spectrum is phenomenological and models ensemble line-shape broadening, not ab-initio transition dipole moments.*")

    with col_s2:
        fig_spec = plot_emission_spectrum(
            peak_wavelength_nm=emission_info['wavelength_nm'],
            fwhm_nm=fwhm_slider,
            peak_intensity=intensity_slider
        )
        st.plotly_chart(fig_spec, use_container_width=True)

# ==========================================
# TAB 3: Material Comparison
# ==========================================
with tab3:
    st.markdown("### ⚖️ Multi-Material Comparison")
    st.caption("Compare quantum confinement trajectories across standard II-VI, IV-VI, and III-V semiconductor materials.")

    selected_comp_materials = st.multiselect(
        "Select Materials to Compare",
        mat_keys,
        default=["CdSe", "CdS", "InP", "PbS"] if all(k in mat_keys for k in ["CdSe", "CdS", "InP", "PbS"]) else mat_keys[:4]
    )

    if selected_comp_materials:
        tab_comp_eg, tab_comp_wl = st.tabs(["Bandgap vs. Radius", "Wavelength vs. Radius"])
        with tab_comp_eg:
            fig_comp_eg = plot_material_comparison(materials_dict, selected_comp_materials, plot_type="bandgap")
            st.plotly_chart(fig_comp_eg, use_container_width=True)

        with tab_comp_wl:
            fig_comp_wl = plot_material_comparison(materials_dict, selected_comp_materials, plot_type="wavelength")
            st.plotly_chart(fig_comp_wl, use_container_width=True)

        # Comparison Table
        st.markdown("#### 📋 Material Parameter Reference Table")
        table_rows = []
        for k in selected_comp_materials:
            p = materials_dict[k]
            bg_data = calculate_bandgap(
                active_radius,
                float(p["bulk_bandgap_eV"]),
                float(p["electron_effective_mass"]),
                float(p["hole_effective_mass"]),
                float(p["relative_dielectric_constant"])
            )
            wl_val = bandgap_to_wavelength(bg_data["qd_bandgap_eV"])
            table_rows.append({
                "Material": p.get("formula", k),
                "Crystal Structure": p.get("crystal_structure", "ZB"),
                "Bulk Eg (eV)": f"{p['bulk_bandgap_eV']:.2f}",
                "me* / m0": f"{p['electron_effective_mass']:.3f}",
                "mh* / m0": f"{p['hole_effective_mass']:.2f}",
                "Dielectric (eps_r)": f"{p['relative_dielectric_constant']:.1f}",
                f"QD Eg @ {active_radius:.1f}nm (eV)": f"{bg_data['qd_bandgap_eV']:.3f}",
                f"λ @ {active_radius:.1f}nm (nm)": f"{wl_val:.1f}",
                "Color Band": wavelength_to_color_category(wl_val)
            })

        df_comp = pd.DataFrame(table_rows)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

# ==========================================
# TAB 4: Alloy Composition Engineering
# ==========================================
with tab4:
    st.markdown("### 🔀 Ternary Alloy Bandgap Engineering")
    st.caption("Interpolate bandgap, effective masses, and optical bowing across ternary semiconductor alloys: $A_x B_{1-x}$.")

    c_al1, c_al2 = st.columns([1, 2])
    with c_al1:
        st.markdown("#### ⚙️ Alloy Selection")
        # Preset alloy selector or custom
        alloy_preset_keys = list(default_alloys.keys())
        chosen_preset = st.selectbox("Preset Alloy Pair", ["Custom Pair"] + alloy_preset_keys)

        if chosen_preset != "Custom Pair":
            preset_data = default_alloys[chosen_preset]
            mat_a_sel = preset_data["matA"]
            mat_b_sel = preset_data["matB"]
            default_b = preset_data["bowing_parameter_eV"]
        else:
            mat_a_sel = "CdSe"
            mat_b_sel = "CdS"
            default_b = 0.30

        mat_a_key = st.selectbox("Material A (x = 1 limit)", mat_keys, index=mat_keys.index(mat_a_sel) if mat_a_sel in mat_keys else 0)
        mat_b_key = st.selectbox("Material B (x = 0 limit)", mat_keys, index=mat_keys.index(mat_b_sel) if mat_b_sel in mat_keys else 1)

        x_comp = st.slider(f"Composition Fraction x ({mat_a_key})", min_value=0.0, max_value=1.0, value=0.5, step=0.02)
        bowing_b = st.number_input("Optical Bowing Parameter b (eV)", value=float(default_b), step=0.05, format="%.3f",
                                  help="Empirical bowing parameter capturing structural/electronic disorder.")

        alloy_result = interpolate_alloy_properties(
            materials_dict[mat_a_key],
            materials_dict[mat_b_key],
            x_composition=x_comp,
            bowing_parameter_eV=bowing_b
        )

        st.markdown(f"""
        - **Alloy Formula**: `{alloy_result['formula']}`
        - **Interpolated Bulk Eg**: `{alloy_result['bulk_bandgap_eV']:.3f} eV`
        - **Effective Mass me\***: `{alloy_result['electron_effective_mass']:.3f} m0`
        - **Effective Mass mh\***: `{alloy_result['hole_effective_mass']:.3f} m0`
        - **Lattice Constant**: `{alloy_result['lattice_constant_angstrom']:.3f} Å`
        """)

    with c_al2:
        fig_alloy_curve = plot_alloy_composition_curves(
            materials_dict[mat_a_key],
            materials_dict[mat_b_key],
            bowing_parameter_eV=bowing_b,
            current_x=x_comp
        )
        st.plotly_chart(fig_alloy_curve, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🗺️ 2D Bandgap Mapping: Radius vs. Alloy Composition")
    fig_heatmap = plot_alloy_heatmap(
        materials_dict[mat_a_key],
        materials_dict[mat_b_key],
        bowing_parameter_eV=bowing_b,
        radius_min_nm=1.0,
        radius_max_nm=8.0
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ==========================================
# TAB 5: Core/Shell Band Alignment
# ==========================================
with tab5:
    st.markdown("### ⚛️ Core/Shell Quantum Dot Heterostructures")
    st.caption("Analyze conduction and valence band edge alignments, band offsets, and heterostructure classification.")

    if core_shell_struct is None:
        # Fallback if in Mode 1
        cs_core = active_mat_key
        cs_shell = "ZnS" if active_mat_key != "ZnS" else "CdS"
        core_shell_struct = CoreShellStructure(
            core_material=cs_core,
            shell_material=cs_shell,
            core_radius_nm=active_radius,
            shell_thickness_nm=1.0,
            core_params=active_params,
            shell_params=materials_dict[cs_shell]
        )

    alignment_info = classify_band_alignment(core_shell_struct.delta_ec, core_shell_struct.delta_ev)

    col_align1, col_align2 = st.columns([1, 2])
    with col_align1:
        st.markdown("#### 🏷️ Alignment Classification")
        st.markdown(f"### `{alignment_info['alignment_type']}`")
        st.info(alignment_info["description"])

        st.markdown(f"""
        - **Core Material**: `{core_shell_struct.core_material}` ($R_c = {core_shell_struct.r_core:.2f}\\text{{ nm}}$)
        - **Shell Material**: `{core_shell_struct.shell_material}` ($t_s = {core_shell_struct.t_shell:.2f}\\text{{ nm}}$)
        - **Conduction Offset ($\\Delta E_c$)**: `{core_shell_struct.delta_ec:+.2f} eV`
        - **Valence Offset ($\\Delta E_v$)**: `{core_shell_struct.delta_ev:+.2f} eV`
        - **Electron Ground Location**: `{alignment_info['electron_localization']}`
        - **Hole Ground Location**: `{alignment_info['hole_localization']}`
        """)

    with col_align2:
        fig_band_diag = plot_core_shell_band_diagram(core_shell_struct, sim_results=num_res)
        st.plotly_chart(fig_band_diag, use_container_width=True)

# ==========================================
# TAB 6: Carrier Wavefunctions & Localization
# ==========================================
with tab6:
    st.markdown("### 🌊 Numerical Radial Schrödinger Solver & Wavefunctions")
    st.caption("Radial envelope wavefunctions $u_e(r), u_h(r)$ and carrier probability densities $|u(r)|^2$ solved with BenDaniel-Duke boundary conditions.")

    if num_res is None:
        st.info("💡 Switch to **Mode 2: Core/Shell Numerical Mode** in the sidebar to configure numerical parameters.")
        # Generate on the fly
        if core_shell_struct is None:
            core_shell_struct = CoreShellStructure(
                core_material="CdSe", shell_material="ZnS", core_radius_nm=2.2, shell_thickness_nm=1.0,
                core_params=materials_dict["CdSe"], shell_params=materials_dict["ZnS"]
            )
        num_res = solve_core_shell_system(core_shell_struct, num_grid_points=350)

    e_loc = num_res["electron_localization"]
    h_loc = num_res["hole_localization"]

    col_wf_m1, col_wf_m2, col_wf_m3, col_wf_m4 = st.columns(4)
    with col_wf_m1:
        st.metric("Electron Ground Ee", f"+{num_res['electron_ground_energy_eV']:.3f} eV")
    with col_wf_m2:
        st.metric("Hole Ground Eh", f"+{num_res['hole_ground_energy_eV']:.3f} eV")
    with col_wf_m3:
        st.metric("Coulomb Binding Eb", f"-{num_res['coulomb_binding_eV']:.3f} eV")
    with col_wf_m4:
        st.metric("Numerical Optical Gap", f"{num_res['qd_bandgap_eV']:.3f} eV")

    st.markdown("#### 🎯 Spatial Probability Densities")
    fig_wf = plot_probability_densities(num_res, plot_type="density")
    st.plotly_chart(fig_wf, use_container_width=True)

    col_loc1, col_loc2 = st.columns(2)
    with col_loc1:
        st.markdown("##### 🔵 Electron Spatial Localization")
        st.progress(min(1.0, e_loc["core_percent"] / 100.0), text=f"Core: {e_loc['core_percent']:.1f}%")
        st.progress(min(1.0, e_loc["shell_percent"] / 100.0), text=f"Shell: {e_loc['shell_percent']:.1f}%")
        st.progress(min(1.0, e_loc["outer_percent"] / 100.0), text=f"Outer Matrix: {e_loc['outer_percent']:.1f}%")

    with col_loc2:
        st.markdown("##### 🟠 Hole Spatial Localization")
        st.progress(min(1.0, h_loc["core_percent"] / 100.0), text=f"Core: {h_loc['core_percent']:.1f}%")
        st.progress(min(1.0, h_loc["shell_percent"] / 100.0), text=f"Shell: {h_loc['shell_percent']:.1f}%")
        st.progress(min(1.0, h_loc["outer_percent"] / 100.0), text=f"Outer Matrix: {h_loc['outer_percent']:.1f}%")

# ==========================================
# TAB 7: Strain & Advanced Effects
# ==========================================
with tab7:
    st.markdown("### 📐 Interfacial Strain & Advanced Model Effects")
    st.caption("Continuum elasticity strain calculation and deformation potential shifts for lattice-mismatched core/shell heterostructures.")

    if strain_res is None:
        strain_res = calculate_hydrostatic_strain_shift(
            core_radius_nm=active_radius,
            shell_thickness_nm=1.0,
            a_core_angstrom=float(active_params.get("lattice_constant_angstrom", 4.30)),
            a_shell_angstrom=float(materials_dict.get("ZnS", {}).get("lattice_constant_angstrom", 5.41)),
            deformation_potential_eV=float(active_params.get("hydrostatic_deformation_pot_eV", -3.0)),
            enabled=True
        )

    col_str1, col_str2 = st.columns([1, 1])
    with col_str1:
        st.markdown("#### 🔬 Lattice Mismatch Analysis")
        st.markdown(f"""
        - **Lattice Mismatch ($\\eta$)**: `{strain_res['lattice_mismatch_percent']:+.2f}%`
        - **Strain Type**: `{strain_res.get('strain_nature', 'N/A')}`
        - **Volumetric Core Strain**: `{strain_res.get('volumetric_strain', 0.0):.4f}`
        - **Hydrostatic Shift ($\\Delta E_g^{{\\text{{strain}}}}$)**: `{strain_res.get('strain_shift_eV', 0.0):+.3f} eV`
        """)

        st.info(strain_res.get("notes", ""))

    with col_str2:
        st.markdown("#### 🧪 Model Fidelity Comparison")
        unstrained_gap = num_res["qd_bandgap_eV"] if num_res is not None else brus_res["qd_bandgap_eV"]
        strained_gap = unstrained_gap + strain_res.get("strain_shift_eV", 0.0)

        df_fid = pd.DataFrame([
            {"Model Level": "Level 1: Fast Brus Analytical", "Predicted Gap (eV)": f"{brus_res['qd_bandgap_eV']:.3f}", "Wavelength (nm)": f"{bandgap_to_wavelength(brus_res['qd_bandgap_eV']):.1f}"},
            {"Model Level": "Level 2: Numerical Schrödinger (Unstrained)", "Predicted Gap (eV)": f"{unstrained_gap:.3f}", "Wavelength (nm)": f"{bandgap_to_wavelength(unstrained_gap):.1f}"},
            {"Model Level": "Level 3: Numerical Schrödinger + Strain Correction", "Predicted Gap (eV)": f"{strained_gap:.3f}", "Wavelength (nm)": f"{bandgap_to_wavelength(strained_gap):.1f}"},
        ])
        st.dataframe(df_fid, use_container_width=True, hide_index=True)

# ==========================================
# TAB 8: Scientific Foundations & Assumptions
# ==========================================
with tab8:
    st.markdown("### 📚 Scientific Foundations & Theoretical Limitations")
    st.caption("Transparent documentation of physical approximations, boundary conditions, and model validity domains.")

    with st.expander("1. Effective Mass Approximation (EMA) & Parabolic Bands", expanded=True):
        st.markdown("""
        - **Approximation**: Electrons and holes near band extrema are treated as free quasiparticles with isotropic bulk effective masses ($m_e^*, m_h^*$).
        - **Validity**: Highly accurate for $R \\ge 1.5\\text{ nm}$ in direct-gap zinc-blende and wurtzite semiconductors (CdSe, CdS, ZnS, InP, GaAs).
        - **Breakdown**: Below $\\sim 1.5\\text{ nm}$, atomistic effects, surface reconstructive relaxation, and non-parabolic band mixing become significant.
        """)

    with st.expander("2. Brus Equation Formulation & $1/R^2$ vs $1/R$ Scaling"):
        st.markdown("""
        The Brus equation for a spherical nanocrystal in the infinite-well limit:
        $$E_{g,\\text{QD}}(R) = E_{g,\\text{bulk}} + \\frac{\\hbar^2 \\pi^2}{2 R^2}\\left(\\frac{1}{m_e^* m_0} + \\frac{1}{m_h^* m_0}\\right) - \\frac{1.8 e^2}{4\\pi \\varepsilon_0 \\varepsilon_r R}$$
        - **Kinetic Term ($\\propto 1/R^2$)**: Particle-in-a-box kinetic quantum confinement (blue shift).
        - **Coulomb Term ($\\propto 1/R$)**: Attractive electron-hole Coulomb interaction screened by static permittivity $\\varepsilon_r$ (red shift).
        """)

    with st.expander("3. Confinement Regimes ($R$ vs $a_B$)"):
        st.markdown("""
        - **Strong Confinement ($R < a_B$)**: Electron and hole are independently confined; kinetic confinement energy dominates Coulomb energy.
        - **Intermediate Confinement ($R \\approx a_B$)**: Correlated electron-hole motion.
        - **Weak Confinement ($R > a_B$)**: Exciton center-of-mass is confined as a quasi-particle; bandgap approaches bulk limit.
        """)

    with st.expander("4. BenDaniel-Duke Boundary Conditions for Hetero-Interfaces"):
        st.markdown("""
        At an interface between core and shell where effective mass changes discontinuously ($m_{\\text{core}}^* \\neq m_{\\text{shell}}^*$):
        - Continuity of wavefunction: $u_{\\text{core}}(R_c) = u_{\\text{shell}}(R_c)$
        - Continuity of probability current: $\\frac{1}{m_{\\text{core}}^*} \\frac{du_{\\text{core}}}{dr} = \\frac{1}{m_{\\text{shell}}^*} \\frac{du_{\\text{shell}}}{dr}$
        - The symmetrized tridiagonal Hamiltonian automatically preserves self-adjointness and probability conservation across interfaces.
        """)

    with st.expander("5. Photoluminescence Spectrum: Phenomenological Gaussian Model"):
        st.markdown("""
        - The emission spectrum generated in this simulator uses a Gaussian lineshape:
        $$I(\\lambda) = I_0 \\exp\\left( -\\frac{(\\lambda - \\lambda_0)^2}{2\\sigma^2} \\right)$$
        - In experimental colloidal quantum dots, spectral linewidth (FWHM $\\approx 20-40\\text{ nm}$) arises from **ensemble size dispersion** ($\sim 5-10\\%$ polydispersity) and **homogeneous longitudinal optical (LO) phonon coupling**.
        """)

st.markdown("---")
st.markdown("<center><small style='color: #64748b;'>QuantumDotLab | Scientific Quantum Dot Simulator | Developed with Streamlit, NumPy, SciPy & Plotly</small></center>", unsafe_allow_html=True)
