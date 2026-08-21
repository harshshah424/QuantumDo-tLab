"""Core/Shell quantum dot structure definition and radial potential/mass profiles.
"""

from typing import Dict, Any, Tuple
import numpy as np


class CoreShellStructure:
    """Represents a spherically symmetric Core/Shell semiconductor heterostructure."""

    def __init__(
        self,
        core_material: str,
        shell_material: str,
        core_radius_nm: float,
        shell_thickness_nm: float,
        core_params: Dict[str, Any],
        shell_params: Dict[str, Any],
        delta_ec_eV: float | None = None,
        delta_ev_eV: float | None = None,
        outer_barrier_eV: float = 3.5
    ):
        """Initialize Core/Shell structure.

        Args:
            core_material: Name or formula of core material.
            shell_material: Name or formula of shell material.
            core_radius_nm: Core radius R_c in nm (> 0).
            shell_thickness_nm: Shell thickness t_s in nm (>= 0).
            core_params: Material dictionary for core.
            shell_params: Material dictionary for shell.
            delta_ec_eV: Conduction band offset in eV (if None, calculated from CBM).
            delta_ev_eV: Valence band offset in eV (if None, calculated from VBM).
            outer_barrier_eV: Potential barrier height of surrounding matrix/ligands in eV.
        """
        if core_radius_nm <= 0:
            raise ValueError(f"Core radius must be positive, got {core_radius_nm}")
        if shell_thickness_nm < 0:
            raise ValueError(f"Shell thickness cannot be negative, got {shell_thickness_nm}")

        self.core_material = core_material
        self.shell_material = shell_material
        self.r_core = float(core_radius_nm)
        self.t_shell = float(shell_thickness_nm)
        self.r_total = self.r_core + self.t_shell
        self.core_params = core_params
        self.shell_params = shell_params
        self.outer_barrier = float(outer_barrier_eV)

        # Determine band offsets
        if delta_ec_eV is not None and delta_ev_eV is not None:
            self.delta_ec = float(delta_ec_eV)
            self.delta_ev = float(delta_ev_eV)
        else:
            cbm_core = float(core_params.get("conduction_band_edge_eV", -4.30))
            vbm_core = float(core_params.get("valence_band_edge_eV", -6.04))
            cbm_shell = float(shell_params.get("conduction_band_edge_eV", -3.40))
            vbm_shell = float(shell_params.get("valence_band_edge_eV", -7.05))
            self.delta_ec = cbm_shell - cbm_core
            self.delta_ev = vbm_core - vbm_shell

    def generate_radial_profiles(
        self,
        r_grid_nm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate position-dependent potential and effective mass profiles across the radial grid.

        Args:
            r_grid_nm: 1D numpy array of radial coordinate points in nm.

        Returns:
            Tuple of:
                V_e(r): Electron potential profile [eV]
                V_h(r): Hole potential profile [eV]
                m_e(r): Electron effective mass profile [m0]
                m_h(r): Hole effective mass profile [m0]
                eps_r(r): Dielectric constant profile
        """
        r = np.asarray(r_grid_nm, dtype=float)
        n = len(r)

        ve = np.zeros(n, dtype=float)
        vh = np.zeros(n, dtype=float)
        me = np.zeros(n, dtype=float)
        mh = np.zeros(n, dtype=float)
        eps = np.zeros(n, dtype=float)

        me_core = float(self.core_params["electron_effective_mass"])
        mh_core = float(self.core_params["hole_effective_mass"])
        eps_core = float(self.core_params["relative_dielectric_constant"])

        me_shell = float(self.shell_params["electron_effective_mass"])
        mh_shell = float(self.shell_params["hole_effective_mass"])
        eps_shell = float(self.shell_params["relative_dielectric_constant"])

        # Outer medium / ligand environment parameters
        me_outer = 1.0
        mh_outer = 1.0
        eps_outer = 2.0

        for i, ri in enumerate(r):
            if ri <= self.r_core:
                # Core region: well bottom at 0 eV
                ve[i] = 0.0
                vh[i] = 0.0
                me[i] = me_core
                mh[i] = mh_core
                eps[i] = eps_core
            elif ri <= self.r_total:
                # Shell region
                ve[i] = self.delta_ec
                vh[i] = self.delta_ev
                me[i] = me_shell
                mh[i] = mh_shell
                eps[i] = eps_shell
            else:
                # Outer barrier region (ligands / solvent / vacuum)
                ve[i] = max(self.delta_ec, 0.0) + self.outer_barrier
                vh[i] = max(self.delta_ev, 0.0) + self.outer_barrier
                me[i] = me_outer
                mh[i] = mh_outer
                eps[i] = eps_outer

        return ve, vh, me, mh, eps
