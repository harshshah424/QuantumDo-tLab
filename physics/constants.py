"""Physical constants and standard unit conversion factors for QuantumDotLab.

All values are in standard SI units or clearly specified eV/nm equivalents.
Derived from CODATA recommended values.
"""

# Fundamental constants (SI units)
HBAR_SI = 1.054571817e-34  # Reduced Planck constant (J*s)
M0_SI = 9.1093837015e-31   # Electron rest mass (kg)
Q_E = 1.602176634e-19      # Elementary charge (C)
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
SPEED_OF_LIGHT = 2.99792458e8  # Speed of light (m/s)

# Conversion factors and useful dimensional constants
PLANCK_CONSTANT_EV_S = 4.135667696e-15  # Planck constant (eV*s)
HBAR_EV_S = 6.582119569e-16            # Reduced Planck constant (eV*s)
HC_EV_NM = 1239.841984                 # hc product in eV * nm (~1240 eV*nm)
BOHR_RADIUS_NM = 0.05291772109         # Atomic Bohr radius a_0 (nm)

# Kinetic energy prefactor: hbar^2 / (2 * m0) in eV * nm^2
# (1.054571817e-34)^2 / (2 * 9.1093837015e-31 * 1.602176634e-19) * 1e18 = 0.03809982 eV*nm^2
HBAR2_OVER_2M0_EV_NM2 = 0.03809982

# Coulomb prefactor: e^2 / (4 * pi * epsilon_0) in eV * nm
# e / (4 * pi * epsilon_0) * 1e9 = 1.4399645 eV*nm
COULOMB_PREFACTOR_EV_NM = 1.4399645

# EMA validity limit guideline (nm)
EMA_MIN_RADIUS_NM = 1.5
