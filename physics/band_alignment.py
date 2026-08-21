"""Band alignment analysis, band offset calculations, and heterostructure classification for Core/Shell QDs.
"""

from typing import Dict, Any, Tuple


def calculate_natural_band_offsets(
    core_cbm_eV: float,
    core_vbm_eV: float,
    shell_cbm_eV: float,
    shell_vbm_eV: float
) -> Tuple[float, float]:
    """Calculate conduction and valence band offsets using vacuum-referenced band edges.

    Conventions:
        Delta_Ec = CBM(shell) - CBM(core)
            > 0: Shell presents a barrier to electrons in core (core is potential well for e-)
            < 0: Shell CB is lower; electrons prefer shell

        Delta_Ev = VBM(core) - VBM(shell)
            > 0: Shell presents a barrier to holes in core (core is potential well for h+)
            < 0: Shell VB is higher; holes prefer shell

    Args:
        core_cbm_eV: Core conduction band minimum (vacuum-referenced, e.g. -4.3 eV).
        core_vbm_eV: Core valence band maximum (vacuum-referenced, e.g. -6.04 eV).
        shell_cbm_eV: Shell conduction band minimum (vacuum-referenced, e.g. -3.4 eV).
        shell_vbm_eV: Shell valence band maximum (vacuum-referenced, e.g. -7.05 eV).

    Returns:
        Tuple of (delta_ec_eV, delta_ev_eV).
    """
    delta_ec = shell_cbm_eV - core_cbm_eV
    delta_ev = core_vbm_eV - shell_vbm_eV
    return float(delta_ec), float(delta_ev)


def classify_band_alignment(
    delta_ec_eV: float,
    delta_ev_eV: float,
    quasi_threshold_eV: float = 0.20
) -> Dict[str, Any]:
    """Classify the core/shell heterostructure into Type I, Quasi-Type II, or Type II alignment.

    Args:
        delta_ec_eV: Conduction band offset in eV.
        delta_ev_eV: Valence band offset in eV.
        quasi_threshold_eV: Threshold below which a band offset is considered weak/quasi-delocalized.

    Returns:
        Dictionary with alignment_type, electron_location, hole_location, and physical description.
    """
    ec = float(delta_ec_eV)
    ev = float(delta_ev_eV)

    if ec > 0 and ev > 0:
        # Straddling gap
        if ec <= quasi_threshold_eV and ev > quasi_threshold_eV:
            alignment = "Quasi-Type II (Electron Delocalized)"
            e_loc = "Delocalized (Core + Shell)"
            h_loc = "Confined to Core"
            desc = (
                f"Conduction band offset (ΔEc = {ec:.2f} eV) is small (< {quasi_threshold_eV} eV), allowing electrons "
                f"to easily tunnel and delocalize across both core and shell, while holes remain deeply confined in the core (ΔEv = {ev:.2f} eV)."
            )
        elif ev <= quasi_threshold_eV and ec > quasi_threshold_eV:
            alignment = "Quasi-Type II (Hole Delocalized)"
            e_loc = "Confined to Core"
            h_loc = "Delocalized (Core + Shell)"
            desc = (
                f"Valence band offset (ΔEv = {ev:.2f} eV) is small (< {quasi_threshold_eV} eV), allowing holes to delocalize, "
                f"while electrons remain strongly confined in the core (ΔEc = {ec:.2f} eV)."
            )
        else:
            alignment = "Type I (Straddling)"
            e_loc = "Confined to Core"
            h_loc = "Confined to Core"
            desc = (
                f"Both conduction band (ΔEc = {ec:.2f} eV) and valence band (ΔEv = {ev:.2f} eV) offsets are positive. "
                "Both electrons and holes are energetically confined inside the core, providing high photoluminescence quantum yield and surface passivation."
            )
    elif ec < 0 and ev > 0:
        # Staggered: Shell CB is lower, Core VB is higher
        alignment = "Type II (Staggered - Electron in Shell, Hole in Core)"
        e_loc = "Confined to Shell"
        h_loc = "Confined to Core"
        desc = (
            f"Staggered band alignment (ΔEc = {ec:.2f} eV < 0, ΔEv = {ev:.2f} eV > 0). "
            "Electrons migrate to the shell while holes remain in the core. Spatially separates carriers, yielding longer radiative lifetimes and red-shifted interband transitions."
        )
    elif ec > 0 and ev < 0:
        # Staggered: Core CB is lower, Shell VB is higher
        alignment = "Type II (Staggered - Electron in Core, Hole in Shell)"
        e_loc = "Confined to Core"
        h_loc = "Confined to Shell"
        desc = (
            f"Staggered band alignment (ΔEc = {ec:.2f} eV > 0, ΔEv = {ev:.2f} eV < 0). "
            "Electrons stay confined to the core while holes migrate to the shell."
        )
    else:
        # Broken gap (Type III)
        alignment = "Type III (Broken Gap)"
        e_loc = "Shell"
        h_loc = "Shell"
        desc = f"Broken gap alignment (ΔEc = {ec:.2f} eV, ΔEv = {ev:.2f} eV). Band edges do not overlap."

    return {
        "alignment_type": alignment,
        "delta_ec_eV": ec,
        "delta_ev_eV": ev,
        "electron_localization": e_loc,
        "hole_localization": h_loc,
        "description": desc
    }
