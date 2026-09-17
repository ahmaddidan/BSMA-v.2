<h1 align="center">BMKG Strong Motion Analyzer (BSMA v2.0.0)</h1>
<p align="center"><b>Computational Platform for Accelerograph Signal Processing, Seismic Kinematics, & Response Spectra</b></p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="https://strong-motion.streamlit.app/"><img src="https://img.shields.io/badge/Streamlit-Live_Cloud_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit App" /></a>
  <a href="https://docs.obspy.org/"><img src="https://img.shields.io/badge/ObsPy-Seismology_Framework-4A90E2?style=for-the-badge" alt="ObsPy" /></a>
  <a href="https://scipy.org/"><img src="https://img.shields.io/badge/SciPy-DSP_%26_Solvers-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy" /></a>
  <a href="https://plotly.com/"><img src="https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly" /></a>
  <a href="https://github.com/ahmaddidan/BSMA-v.2"><img src="https://img.shields.io/badge/Status-Release_v2.0.0-059669?style=for-the-badge" alt="Status" /></a>
</p>

<p align="center">
  <b>Language:</b> <b>English</b> | <a href="README.id.md">Bahasa Indonesia</a>
</p>

---

### Project Deliverables & Documentation Center

Access the live cloud platform and download the official publication-grade technical documentation:

* **Interactive Cloud Web Application**: [https://strong-motion.streamlit.app/](https://strong-motion.streamlit.app/)
* **📘 Operational Technical Manual (Visual User Guide)**:
  * **English Edition**: [outputs/BSMA_Panduan_Teknis_Operasional_EN.pdf](outputs/BSMA_Panduan_Teknis_Operasional_EN.pdf)
  * **Indonesian Edition**: [outputs/BSMA_Panduan_Teknis_Operasional_ID.pdf](outputs/BSMA_Panduan_Teknis_Operasional_ID.pdf)
* **📑 Scientific & Engineering Guidebook (Theoretical Foundations & Algorithms)**:
  * **English Edition**: [outputs/BSMA_Scientific_Guidebook_EN.pdf](outputs/BSMA_Scientific_Guidebook_EN.pdf)
  * **Indonesian Edition**: [outputs/BSMA_Scientific_Guidebook_ID.pdf](outputs/BSMA_Scientific_Guidebook_ID.pdf)
* **GitHub Source Code Repository**: [https://github.com/ahmaddidan/BSMA-v.2](https://github.com/ahmaddidan/BSMA-v.2)

---

## Executive Summary & Project Overview

**BMKG Strong Motion Analyzer (BSMA v2.0.0)** is an accelerograph signal analysis software suite independently developed as part of an undergraduate **Internship Program (Kerja Praktik)** by a student of **Geophysical Engineering, Faculty of Industrial Technology, Institut Teknologi Sumatera (ITERA)** at the **Sleman Geophysical Station Class I, Meteorology, Climatology, and Geophysical Agency (BMKG) D.I. Yogyakarta** (conducted from July 20 to August 20, 2026). This software is an independent academic project and does not constitute an official operational system or commercial software of BMKG.

The platform provides an automated, end-to-end processing pipeline for strong ground motion records—spanning raw count ingestion, StationXML instrument response deconvolution, multi-tier data quality control (QC), digital signal processing (DSP), numerical kinematic integration (PGA, PGV, PGD), instrumental Modified Mercalli Intensity (MMI) estimation based on GMICE formulations by Worden et al. (2012), and 5% damped elastic Single-Degree-of-Freedom (SDOF) Pseudo-Spectral Acceleration (PSA) response spectra modeling using both the **Nigam–Jennings (1969)** analytical recursive filter and the **Newmark-Beta (1959)** implicit integration solver, alongside standard structural design reference overlays (**SNI 1726:2019**).

---

## Signal Processing Pipeline Architecture

The BSMA architecture enforces a rigorous, sequential computational pipeline designed for full data provenance tracking and scientific reproducibility:

```text
┌────────────────────────────────────────────────────────┐
│ 1. INGESTION & DATA INTEGRITY                          │
│    MiniSEED / SAC Reader + StationXML PAZ              │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ 2. QUALITY CONTROL (QC) SCREENING                      │
│    • Quality Score (0–100)                             │
│    • Status: PASS (≥70) | WARNING (50–69) | FAIL (<50) │
│    • Fatal Override: Clipping & ADC Saturation -> FAIL │
│    • Diagnostic Flags: Clipping, Flatline, Spikes, SNR │
│    • 6 Diagnostic Classes (Class 1–6)                  │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ 3. INSTRUMENT RESPONSE CORRECTION                      │
│    Deconvolution: Counts -> Physical Acc (m/s²)        │
│    (Physical Acceleration Bypass if pre-calibrated)    │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ 4. DIGITAL SIGNAL PROCESSING (DSP)                     │
│    • Baseline Detrending (Mean & Polynomial / Linear)  │
│    • Cosine Tapering 5% (Tukey Window)                 │
│    • Forward-Backward Butterworth Bandpass Filtering   │
│      Bandpass 0.10–25.0 Hz, f_max ≤ 0.40 f_s           │
│      Adaptive SNR low-frequency floor (0.20–0.40 Hz)   │
└───────────────────────────┬────────────────────────────┘
                            │
               [ Corrected Acceleration a(t) ]
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ 5a. KINEMATIC INTEGRATION     │ │ 5b. SDOF RESPONSE SPECTRUM    │
│  • Cumulative Trapezoidal     │ │  • Solvers: Nigam–Jennings    │
│    Rule: a(t) -> v(t) -> d(t) │ │    (1969) & Newmark-β (1959)  │
│  • Strict Acceleration-only   │ │  • 5% Damping Pseudo-Spectral │
│    detrending pre-integration │ │    Acceleration: PSA = ω²·Sd  │
│  • No post-integration drift  │ │  • SDOF Solver Numerical      │
│    forcing on v(t) or d(t)    │ │    Cross-Validation Benchmark │
│  • PGD represents transient   │ │  • SNI 1726:2019 Design Code  │
│    dynamic peak displacement  │ │    Reference Spectrum Overlay │
└───────────────┬───────────────┘ └───────────────┬───────────────┘
                │                                 │
                ▼                                 │
┌───────────────────────────────┐                 │
│ 6. PARAMETERS & GMICE MMI     │                 │
│  • Peaks: PGA, PGV, PGD       │                 │
│  • Energy: Arias Ia, Husid    │                 │
│  • Duration: D5-95, D5-75     │                 │
│  • MMI: Worden et al. (2012)  │                 │
│    on Max Horizontal Comp.    │                 │
└───────────────┬───────────────┘                 │
                │                                 │
                └───────────────┬─────────────────┘
                                ▼
┌────────────────────────────────────────────────────────┐
│ 7. REPORT GENERATION & DATA EXPORT                     │
│    • Technical PDF Report (Single & Multi-Station)     │
│    • Tabular CSV Kinematics Summary                    │
│    • Discrete Spectral Response Matrix CSV             │
│    • Full Export ZIP Archive Package                   │
└────────────────────────────────────────────────────────┘
```

### Detailed Pipeline Methodology

1. **Ingestion & Metadata Validation**:
   * Supports triaxial **MiniSEED (`.mseed`)** (FDSN standard) and **SAC (`.sac`)** (IRIS standard) waveform formats.
   * Automated temporal synchronization for 3 orthogonal channels (Z, N, E / U-D, N-S, E-W) based on absolute UTC timestamps.
   * Instrument transfer function deconvolution using **StationXML (`.xml`)** metadata (poles-zeros / PAZ and stage gain deconvolution via ObsPy).
   * Automatic *Physical Acceleration Bypass* when input records are already calibrated in physical acceleration units (m/s² or Gal).

2. **Quality Control (QC) & Signal Integrity Screening**:
   * **Quality Score ($Q \in [0, 100]$)**: Quantitative cleanliness index computed from calibrated physical deductions:
     $$Q = \max\left(0, \min\left(100, 100 - \sum_{i} P_i\right)\right)$$
     with penalty deductions: `WARNING` = $-15\text{ pts}$, `ERROR` = $-40\text{ pts}$, and `CRITICAL` anomalies acting as a fatal override ($Q = 0$).
   * **Three-Tier Operational Status**:
     * **`QC PASS`** ($Q \ge 70$): Clean, high-fidelity signals suitable for engineering analysis and response spectra computation.
     * **`QC WARNING`** ($50 \le Q < 70$): Non-fatal anomalies detected (isolated spikes, marginal pre-event SNR, or mild baseline tilt). Usable with engineering caution.
     * **`QC FAIL`** ($Q < 50$ or Fatal Override): Sensor clipping, ADC saturation, flatlines, or corrupted data channels. Clipped records are strictly invalidated for peak ground motion analysis because PGA is truncated and MMI estimation becomes artificially underestimated.
   * **Six-Class Diagnostic Taxonomy (Class 1–6)**: Standardized physical classification across UI, PDF exporter, and batch processing:
     1. *Class 1 (Pre-Event SNR Gate)*: Ratio of event RMS to pre-event noise RMS ($\text{SNR} < 10\text{ dB} \to \text{Warning}$, $< 3\text{ dB} \to \text{Error}$).
     2. *Class 2 (Sensor Clipping & ADC Saturation)*: Amplitudes exceeding $98\%$ full-scale or flat plateaus $\ge 0.02\text{ s}$ ($\to \text{Critical Fatal Override}, Q=0$).
     3. *Class 3 (Impulsive Spikes)*: Modified Z-Score outlier detection based on Median Absolute Deviation ($M_i = 0.6745 \cdot |a_i - \text{median}(a)| / \text{MAD} \ge 6.0$).
     4. *Class 4 (Dead Channel / Flatline)*: Zero variance or consecutive identical samples $\ge 1.0\text{ s}$ ($\to \text{Critical Fatal Override}, Q=0$).
     5. *Class 5 (Pre-event Window Duration)*: Noise baseline window duration $< 5.0\text{ s}$ ($\to \text{Warning}$).
     6. *Class 6 (Baseline DC Offset & Drift)*: Initial mean $> 2\%$ PGA or polynomial drift $> 5\%$ ($\to \text{Warning}$).

3. **Digital Signal Processing (DSP)**:
   * **Baseline Detrending**: Mean subtraction and linear/polynomial trend removal to eliminate initial DC baseline offset (Boore & Bommer, 2005).
   * **5% Cosine Tapering (Tukey Window)**: Smooths amplitude discontinuities at both record boundaries to mitigate spectral leakage during Fourier and filter operations (Harris, 1978).
   * **Zero-Phase 4th-Order Butterworth Bandpass Filtering**: Forward-backward two-pass filtering (`scipy.signal.sosfiltfilt`) that guarantees zero net phase lag ($\Delta \phi = 0$). The squared magnitude transfer function ($|H(f)|^2$) achieves an effective 48 dB/octave attenuation rate in the stopband (equivalent to an 8th-order slope) while preserving the intrinsic numerical stability and pole structure of the 4th-order prototype.
   * **Nyquist Safeguard Limit**: $f_{\max} \le 0.80 f_{\mathrm{Nyquist}} = 0.40 f_s$ to prevent high-frequency aliasing and filter distortion.
   * **Adaptive Low-Frequency SNR Floor**: Highpass corner $f_{\min}$ is automatically elevated to $0.20\text{ Hz}$ or $0.40\text{ Hz}$ when pre-event SNR is marginal ($< 20\text{ dB}$ or $< 10\text{ dB}$) to suppress low-frequency integration drift.

4. **Kinematic Integration & Baseline Policy**:
   * Sequential numerical integration from acceleration $a(t)$ to velocity $v(t)$, and velocity to displacement $d(t)$ using the cumulative trapezoidal rule.
   * **Strict Baseline Policy**: Baseline detrending is strictly applied to acceleration *prior* to integration. In accordance with rigorous kinematic calculus, BSMA deliberately avoids artificial polynomial drift fitting on $v(t)$ or $d(t)$ post-integration, preserving strict derivative consistency ($a = \dot{v} = \ddot{d}$).
   * **PGD Interpretation**: PGD represents the **transient dynamic peak displacement** within the passband of the bandpass filter, not the permanent tectonic static offset (*fling-step*), which seismologically requires specialized baseline correction or high-rate GNSS instrumentation.

5. **Kinematic Parameters & Instrumental MMI**:
   * Peak Ground Motion Extraction: **PGA**, **PGV**, **PGD**, and the structural indicator ratio **$V_{\max}/A_{\max}$**.
   * Energy and Duration Metrics: Cumulative **Arias Intensity ($I_a$)** and Significant Duration intervals (**$D_{5-95}$** and **$D_{5-75}$**).
   * Instrumental MMI estimation using the **Worden et al. (2012)** USGS ShakeMap GMICE formulation evaluated on the **Maximum Horizontal Component (Max-H)** (excluding vertical Z/U), featuring continuous transition from PGA dominance to PGV dominance at higher shaking levels ($I_{\text{MMI}} \ge 5.0$).

6. **SDOF Response Spectra & Structural Design Standards**:
   * Elastic 5% critically damped Single-Degree-of-Freedom (SDOF) **Pseudo-Spectral Acceleration (PSA)** across period range $T = 0.01 - 10.0$ seconds.
   * Dual numerical solver engines:
     * **Nigam–Jennings (1969)**: Recursive exact analytical state-transition solver assuming piecewise linear ground acceleration.
     * **Newmark-Beta (1959)**: Implicit time integration solver ($\gamma = 1/2, \beta = 1/4$, average acceleration scheme).
   * **SDOF Solver Cross-Validation Benchmark**: Interactive quantitative comparison calculating maximum relative deviation ($< 0.08\%$), mean relative deviation ($< 0.02\%$), and RMS difference ($< 1.5 \times 10^{-4}\text{ g}$) across 100 period points.
   * **SNI 1726:2019 Design Code Overlay**: Direct comparison against Indonesian building code design spectra ($S_{DS}, S_{D1}, T_0, T_s, T_L$) as an elastic demand benchmark.

7. **Technical Reporting & Multi-Format Export**:
   * Publication-grade PDF engineering reports (single-station and multi-station batch).
   * Comprehensive tabular CSV kinematics summaries.
   * Discrete spectral response matrix CSV exports ($T$ vs $S_a$).
   * Standalone ZIP archive bundling all processed data products.

---

## 7 Interactive Analysis Tabs

The BSMA graphical interface is built with Streamlit and Plotly, organized into 7 functional tabs:

| Tab | Name | Scope & Capabilities |
| :---: | :--- | :--- |
| **1** | **Summary** | Executive dashboard featuring station metadata (latitude, longitude, elevation), triaxial waveform preview (40–50% viewport height), peak kinematic metrics of the strongest channel, QC status badge, instrumental MMI, and an audit trail panel. |
| **2** | **Waveforms** | Interactive 3-component time series for complete kinematics: Acceleration ($a$), Velocity ($v$), and Displacement ($d$), complete with automated P-wave arrival markers and peak amplitude annotations. |
| **3** | **Quality Control** | Signal integrity diagnostics: numerical Quality Score (0–100), validation badges (PASS/WARNING/FAIL), polar radar chart of noise metrics, Power Spectral Density (PSD) curves, SNR estimates (dB), and diagnostic anomaly flags. |
| **4** | **Strong Motion** | Advanced energy diagnostics: cumulative Arias Intensity curves (*Husid Plot*), Cumulative Absolute Velocity (CAV, with EPRI 1988 screening threshold $\ge 0.16\text{ g}\cdot\text{s}$), Significant Duration intervals ($D_{5-95}$ and $D_{5-75}$), and $V_{\max}/A_{\max}$ kinematic ratio. |
| **5** | **Intensity** | Objective instrumental shaking interpretation via Worden et al. (2012) GMICE regressions derived from maximum horizontal (Max-H) kinematics, with perceived shaking and potential damage metrics. |
| **6** | **Spectrum** | Elastic 5% damped Pseudo-Spectral Acceleration ($S_a$) response curves across $T = 0.01 - 10.0$ s. Includes design code reference overlay (**SNI 1726:2019**) and the numerical **SDOF Solver Cross-Validation Benchmark** (Nigam-Jennings vs. Newmark-Beta). |
| **7** | **Report** | Technical PDF report preview and download, kinematic parameter CSV tables, discrete spectral response matrix CSVs, and complete ZIP package export. |

---

## Installation & Local Setup Guide

### System Requirements
* **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
* **Python Environment**: Version **3.10** through **3.13** (primary test environment: Python 3.10 / 3.13)
* **Core Dependencies**: Streamlit, ObsPy, NumPy, SciPy, Pandas, Matplotlib, Plotly, FPDF, PyMuPDF
* **RAM**: Minimum 4 GB (8 GB recommended for batch multi-station workflows)

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ahmaddidan/BSMA-v.2.git
   cd BSMA-v.2
   ```

2. **Create and Activate a Virtual Environment:**
   * On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * On Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the BSMA Streamlit Application:**
   ```bash
   streamlit run app.py
   ```
   The browser will automatically open at `http://localhost:8501`.

5. **(Optional) Run the Automated Test Suite:**
   ```bash
   pytest tests/
   ```

---

## Repository Directory Structure

```text
Project BSMA/
├── .streamlit/                      # Visual theme configurations (light/dark) & Streamlit server settings
├── core/                            # Pure scientific computational core (Standard Python + NumPy/SciPy)
│   ├── interfaces/                  # Abstract preprocessor and solver interface contracts
│   ├── io/                          # Waveform parsers (MiniSEED, SAC) & StationXML deconvolution
│   ├── preprocessing/               # Detrending, cosine tapering, Butterworth filtering, QC screening
│   ├── processing/                  # Numerical integration, kinematic parameters (PGA/PGV/PGD, Arias, MMI)
│   ├── sdof/                        # SDOF elastic response solvers (Nigam-Jennings 1969 & Newmark-Beta 1959)
│   ├── types/                       # Dataclass schemas for context, waveforms, and processing states
│   ├── orchestrator.py              # Central signal processing pipeline orchestrator
│   └── pipeline.py                  # End-to-end sequential processing pipeline
├── services/                        # Application service layer
│   ├── analysis_service.py          # Single-station processing and adaptive filter recommendation
│   ├── batch_service.py             # Multi-station batch processing service
│   └── export_service.py            # CSV tabular data, spectral matrix, and ZIP archive export service
├── utils/                           # Supporting utility modules
│   ├── exceptions.py                # Structured scientific error codes and custom exceptions
│   ├── exporter.py                  # Tabular data serialization utilities
│   ├── logger.py                    # Provenance tracking and execution audit logging system
│   └── pdf_exporter.py              # Technical engineering PDF report generator
├── scripts/                         # Automation & document generator scripts
│   ├── generate_guidebook.py        # Automated generator for Scientific Guidebook (ID & EN)
│   └── generate_operational_guidebook.py # Automated generator for Operational Manual (ID & EN)
├── tests/                           # Comprehensive automated test suite (pytest - 83 test cases)
│   ├── test_analysis_service.py     # Single-station analysis service integration tests
│   ├── test_integration.py          # Analytical synthetic integration verification (exact sine wave)
│   ├── test_level4_real_data.py     # Real multi-station operational BMKG earthquake record validation
│   ├── test_mmi.py                  # Worden et al. (2012) GMICE formulation tests
│   ├── test_parameters.py           # Kinematics and energy parameter extraction tests
│   ├── test_qc.py                   # Three-tier Quality Control system tests
│   ├── test_reference_benchmark.py  # End-to-end reference dataset regression benchmarks
│   └── test_response_spectrum.py    # SDOF solvers & Nigam-Jennings vs. Newmark-Beta benchmark tests
├── outputs/                         # Documentation and export outputs
│   ├── BSMA_Panduan_Teknis_Operasional_EN.pdf # Operational Technical Manual (English - 18 Pages)
│   ├── BSMA_Panduan_Teknis_Operasional_ID.pdf # Operational Technical Manual (Indonesian - 18 Pages)
│   ├── BSMA_Scientific_Guidebook_EN.pdf       # Scientific & Algorithmic Guidebook (English - 17 Pages)
│   └── BSMA_Scientific_Guidebook_ID.pdf       # Scientific & Algorithmic Guidebook (Indonesian - 17 Pages)
├── docs/images/                     # High-resolution UI screenshots for documentation previews
├── app.py                           # Streamlit graphical user interface entry point (GUI)
├── assets/                          # Official institution branding & visual identity assets
├── requirements.txt                 # Python dependency specifications with version bounds
├── README.md                        # Comprehensive project documentation (English)
└── README.id.md                     # Comprehensive project documentation (Bahasa Indonesia)
```

---

## References & Scientific Bibliography

1. **Arias, A.** (1970). A measure of earthquake intensity. In R. J. Hansen (Ed.), *Seismic design for nuclear power plants* (pp. 438–483). Cambridge: MIT Press.
2. **Badan Standardisasi Nasional.** (2019). *SNI 1726:2019: Tata cara perencanaan ketahanan gempa untuk struktur bangunan gedung dan non gedung*. Jakarta: Badan Standardisasi Nasional.
3. **Beyreuther, M., Barsch, R., Krischer, L., Megies, T., Behr, Y., & Wassermann, J.** (2010). ObsPy: A Python toolbox for seismology. *Seismological Research Letters*, 81(3), 530–533. [https://doi.org/10.1785/gssrl.81.3.530](https://doi.org/10.1785/gssrl.81.3.530)
4. **Boore, D. M., & Bommer, J. J.** (2005). Processing of strong-motion accelerograms: Needs, options and consequences. *Soil Dynamics and Earthquake Engineering*, 25(2), 93–115. [https://doi.org/10.1016/j.soildyn.2004.10.007](https://doi.org/10.1016/j.soildyn.2004.10.007)
5. **Electric Power Research Institute (EPRI).** (1988). *Standardization of the Cumulative Absolute Velocity (CAV) parameter and its application to nuclear power plant seismic design*. EPRI NP-5930. Palo Alto, CA: EPRI.
6. **Harris, F. J.** (1978). On the use of windows for harmonic analysis with the discrete Fourier transform. *Proceedings of the IEEE*, 66(1), 51–83. [https://doi.org/10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837)
7. **Newmark, N. M.** (1959). A method of computation for structural dynamics. *Journal of the Engineering Mechanics Division, ASCE*, 85(3), 67–94. [https://doi.org/10.1061/JMCEA3.0000098](https://doi.org/10.1061/JMCEA3.0000098)
8. **Nigam, N. C., & Jennings, P. C.** (1969). Calculation of response spectra from strong-motion earthquake records. *Bulletin of the Seismological Society of America*, 59(2), 909–922. [https://doi.org/10.1785/BSSA0590020909](https://doi.org/10.1785/BSSA0590020909)
9. **Trifunac, M. D., & Brady, A. G.** (1975). A study on the duration of strong earthquake ground motion. *Bulletin of the Seismological Society of America*, 65(3), 581–626. [https://doi.org/10.1785/BSSA0650030581](https://doi.org/10.1785/BSSA0650030581)
10. **Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., & van der Walt, S. J.** (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272. [https://doi.org/10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)
11. **Worden, C. B., Gerstenberger, M. C., Rhoades, D. A., & Wald, D. J.** (2012). Probabilistic relationships between ground-motion parameters and MMI. *Bulletin of the Seismological Society of America*, 102(1), 204–221. [https://doi.org/10.1785/0120110156](https://doi.org/10.1785/0120110156)

---

## Developer & Institutional Affiliation

* **Author**: Ahmad Didane Setyawan Putra
* **Student ID (NIM)**: 123120094
* **Department / Faculty**: Geophysical Engineering, Faculty of Industrial Technology
* **University**: Institut Teknologi Sumatera (ITERA)
* **Contact Email**: [ahmad.123120094@student.itera.ac.id](mailto:ahmad.123120094@student.itera.ac.id)
* **GitHub Repository**: [https://github.com/ahmaddidan/BSMA-v.2](https://github.com/ahmaddidan/BSMA-v.2)
* **Host Institution**: Sleman Geophysical Station Class I, Meteorology, Climatology, and Geophysical Agency (BMKG) D.I. Yogyakarta (Internship Period: July 20 – August 20, 2026)

---

<div align="center">
  <sub>Ahmad Didane Setyawan Putra © 2026 · Academic Internship Project of Geophysical Engineering ITERA at Sleman Geophysical Station BMKG</sub>
</div>
