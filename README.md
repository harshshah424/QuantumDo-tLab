# QuantumDotLab ⚛️
### Interactive Quantum Dot Bandgap, Confinement, and Optical Properties Simulator

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/pytest-23%20passed-brightgreen.svg)]()

**QuantumDotLab** is a modular computational physics simulator and interactive scientific dashboard designed to model, analyze, and visualize the electronic and optical properties of semiconductor quantum dots (QDs) and core/shell heterostructures.

---

## 🌟 Key Features

- **Multi-Fidelity Physical Modeling**:
  - **Mode 1: Fast Analytical Mode (Brus Equation)**: Real-time slider interactivity with kinetic confinement and Coulomb interaction terms.
  - **Mode 2: Core/Shell Numerical Mode (1D Radial Schrödinger Solver)**: Finite-difference solver with position-dependent effective masses and **BenDaniel-Duke boundary conditions** at material interfaces.
  - **Mode 3: Advanced / Experimental Mode (Strain & Corrections)**: Interfacial lattice mismatch calculations and continuum elasticity hydrostatic deformation shifts.
- **Comprehensive Material Database**:
  - 12+ standard semiconductors across II-VI, IV-VI, and III-V families: **CdSe, CdS, CdTe, ZnS, ZnSe, ZnTe, PbS, PbSe, PbTe, InP, InAs, GaAs, InSb**.
  - Documented bulk parameters ($E_{g,\text{bulk}}$, $m_e^*$, $m_h^*$, $\varepsilon_r$, $a$, CBM, VBM, deformation potentials) with literature citations.
- **Ternary Alloy Engineering**:
  - Vegard's law interpolation with quadratic **optical bowing parameters** ($E_g(x) = x E_A + (1-x)E_B - b x(1-x)$).
  - 2D contour mapping across (Radius $\times$ Composition).
- **Core/Shell Heterostructure Analysis**:
  - Natural band offsets ($\Delta E_c, \Delta E_v$) and automated classification into **Type I**, **Quasi-Type II**, and **Type II** alignments.
  - Interactive spatial energy band diagrams with quantized electron and hole energy levels.
- **Wavefunction & Carrier Localization**:
  - Computes electron and hole 1S ground states ($E_e, E_h$) and radial probability densities ($|u_e(r)|^2, |u_h(r)|^2$).
  - Quantitative spatial localization percentages (% in Core vs % in Shell vs % in Outer Barrier).
  - Two-particle Coulomb binding energy overlap integral.
- **Optical Emission & Color Synthesis**:
  - Bandgap to wavelength conversion ($\lambda = hc / E_g$).
  - Visible color category classification (UV, Violet, Blue, Green, Yellow, Orange, Red, IR) and Dan Bruton CIE color approximation for live UI swatches.
  - Phenomenological Gaussian photoluminescence (PL) emission spectrum with tunable linewidth (FWHM) and peak intensity.

---

## 📐 Mathematical Models & Physics

### 1. The Brus Equation (Effective Mass Approximation)
For a spherical nanocrystal of radius $R$ in the strong/intermediate confinement regime:

$$E_{g,\text{QD}}(R) = E_{g,\text{bulk}} + \frac{\hbar^2 \pi^2}{2 R^2}\left(\frac{1}{m_e^* m_0} + \frac{1}{m_h^* m_0}\right) - \frac{1.8 e^2}{4 \pi \varepsilon_0 \varepsilon_r R}$$

- **Kinetic Confinement Shift ($\propto 1/R^2$)**: Particle-in-a-box energy penalty for spatial localization (blue shift).
- **Coulomb Attraction Shift ($\propto 1/R$)**: Attractive electron-hole interaction screened by dielectric permittivity $\varepsilon_r$ (red shift).

### 2. Exciton Bohr Radius & Confinement Regimes
The bulk exciton Bohr radius $a_B$ defines the confinement length scale:

$$a_B = \varepsilon_r \frac{m_0}{\mu} a_0 \quad \text{where} \quad \frac{1}{\mu} = \frac{1}{m_e^*} + \frac{1}{m_h^*}$$

- **Strong Confinement ($R < a_B$)**: Electron and hole are confined independently; kinetic term dominates.
- **Intermediate Confinement ($R \approx a_B$)**: Correlated carrier motion.
- **Weak Confinement ($R > a_B$)**: Exciton center-of-mass is quantized; bulk-like gap.

### 3. Radial Schrödinger Equation with BenDaniel-Duke Matching
For multi-region core/shell heterostructures with position-dependent mass $m^*(r)$ and step potential $V(r)$:

$$\left[ -\frac{\hbar^2}{2 m_0} \frac{d}{dr}\left(\frac{1}{m^*(r)} \frac{d}{dr}\right) + V(r) \right] u(r) = E u(r)$$

Where $u(r) = r \cdot R(r)$ is the reduced radial wavefunction ($u(0) = 0, u(R_{\text{max}}) = 0$).

**Symmetrized Finite-Difference Tridiagonal Hamiltonian**:
$$H_{i,i+1} = H_{i+1,i} = -\frac{\hbar^2}{2 m_0 \Delta r^2} \frac{1}{m^*_{i+1/2}}$$
$$H_{i,i} = \frac{\hbar^2}{2 m_0 \Delta r^2}\left(\frac{1}{m^*_{i+1/2}} + \frac{1}{m^*_{i-1/2}}\right) + V_i$$

### 4. Two-Particle Coulomb Overlap Integral
$$E_{\text{coulomb}} = \frac{e^2}{4\pi\varepsilon_0} \int_0^{R_{\text{max}}} \int_0^{R_{\text{max}}} \frac{|u_e(r_e)|^2 |u_h(r_h)|^2}{\varepsilon_r \max(r_e, r_h)} \, dr_e \, dr_h$$

### 5. Ternary Alloy Bowing Model
$$E_g(A_x B_{1-x}) = x E_g(A) + (1-x) E_g(B) - b \cdot x(1-x)$$

### 6. Lattice Mismatch & Hydrostatic Strain
$$\eta = \frac{a_{\text{shell}} - a_{\text{core}}}{a_{\text{core}}}$$
$$\Delta E_g^{\text{strain}} = a_{cv} \cdot 3 \epsilon_{\text{hydro}} \approx a_{cv} \cdot \eta \left(1 - \frac{R_{\text{core}}^3}{R_{\text{total}}^3}\right)$$

---

## 🏗️ Project Architecture

```
QuantumDotLab/
├── app.py                     # Interactive Streamlit dashboard
├── requirements.txt           # Dependency specifications
├── README.md                  # Scientific & technical documentation
│
├── data/
│   └── materials.json         # Material database with documented references
│
├── physics/
│   ├── __init__.py
│   ├── constants.py           # CODATA physical constants & unit conversions
│   ├── brus_model.py          # Analytical Brus model & confinement regimes
│   ├── emission.py            # Optical emission, wavelength, and color maps
│   ├── alloy.py               # Vegard's law interpolation & optical bowing
│   ├── band_alignment.py      # Band offsets & Type I/II/Quasi classification
│   ├── core_shell.py          # Core/shell geometry & piecewise radial profiles
│   ├── schrodinger_solver.py  # 1D Radial Schrödinger finite-difference solver
│   └── strain.py              # Interfacial lattice mismatch & hydrostatic strain
│
├── visualization/
│   ├── __init__.py
│   ├── plots.py               # Eg vs R, Wavelength vs R, 2D Alloy Heatmap
│   ├── band_diagram.py        # Core/shell energy band diagram & offsets
│   ├── spectrum.py            # Gaussian PL emission spectrum with color fills
│   └── wavefunctions.py       # Radial probability density |u(r)|² plots
│
└── tests/
    ├── __init__.py
    ├── test_brus_model.py     # Brus scaling, limits, and Bohr radius tests
    ├── test_emission.py       # Wavelength, color categories, and Gaussian tests
    ├── test_alloy.py          # Alloy endpoints and bowing tests
    ├── test_core_shell.py     # Band offsets and strain tests
    └── test_solver.py         # Solver eigenvalues, normalization, and convergence
```

---

## 🚀 Installation & Running

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Simulator
```bash
streamlit run app.py
```
The simulator will open in your default browser at `http://localhost:8501`.

### 3. Run Automated Tests
```bash
pytest -v
```

---

## 🧪 Benchmark Simulations

| System | Core Radius $R_c$ | Shell Thickness $t_s$ | Predicted $E_g$ | Emission $\lambda$ | Alignment Type |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CdSe (Bare QD)** | 1.80 nm | 0.00 nm | 2.30 eV | 539 nm (Green) | N/A |
| **CdSe / ZnS** | 2.20 nm | 1.00 nm | 2.18 eV | 569 nm (Yellow-Green) | Type I (Straddling) |
| **CdSe / CdS** | 2.20 nm | 1.50 nm | 2.08 eV | 596 nm (Orange) | Quasi-Type II |
| **CdTe / CdSe** | 2.50 nm | 1.20 nm | 1.42 eV | 873 nm (Near-IR) | Type II (Staggered) |
| **InP / ZnS** | 1.60 nm | 1.00 nm | 2.35 eV | 528 nm (Green) | Type I (Cd-Free) |
| **PbS (Bare QD)** | 2.50 nm | 0.00 nm | 1.04 eV | 1192 nm (SWIR) | N/A |

---

## 🔬 Scientific Limitations & Validity

1. **Effective Mass Approximation (EMA)**: Assumes single isotropic parabolic bands; breaks down below $R \sim 1.5\text{ nm}$ where atomistic tight-binding or DFT is required.
2. **Dielectric Confinement (Image Charge Effect)**: Strong dielectric mismatch between QD and organic ligands ($\varepsilon_{\text{QD}} \gg \varepsilon_{\text{matrix}}$) can induce small self-energy shifts ($\sim 0.05-0.1\text{ eV}$).
3. **Colloidal Polydispersity**: Real synthesized quantum dots possess a size distribution ($\sim 5-10\%$), which broadens the experimental photoluminescence peak.

---

## 📖 References & Citations

1. **L. E. Brus**, *"Electronic wave functions in semiconductor clusters: experiment and theory,"* *J. Phys. Chem.* **90**, 2555 (1986).
2. **Al. L. Efros and A. L. Efros**, *"Interband absorption of light in a semiconductor sphere,"* *Sov. Phys. Semicond.* **16**, 772 (1982).
3. **Y. Kayanuma**, *"Quantum-size effects of interacting electrons and holes in semiconductor microcrystals with different band gaps,"* *Phys. Rev. B* **38**, 9797 (1988).
4. **I. Vurgaftman, J. R. Meyer, L. R. Ram-Mohan**, *"Band parameters for III–V compound semiconductors and their alloys,"* *J. Appl. Phys.* **89**, 5815 (2001).
5. **O. Madelung (ed.)**, *Semiconductors: Data Handbook*, 3rd ed., Springer (2004).
6. **C. G. Van de Walle**, *"Band lineups and deformation potentials in the model-solid theory,"* *Phys. Rev. B* **39**, 1871 (1989).
7. **P. Reiss, M. Protière, L. Li**, *"Core/shell semiconductor nanocrystals,"* *Small* **5**, 154 (2009).
