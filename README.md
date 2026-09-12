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

### Application Access & Project Deliverables

The interactive web application, official technical guidebook, and source code repository can be accessed via the following links:

* **Interactive Cloud Web Application**: [https://strong-motion.streamlit.app/](https://strong-motion.streamlit.app/)
* **User Guidebook & Technical Reference (PDF)**: [outputs/BSMA_User_Guidebook.pdf](outputs/BSMA_User_Guidebook.pdf)
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
   * **Quality Score (0–100)**: Quantitative cleanliness index computed from calibrated physical penalties.
   * **Three-Tier Operational Status**:
     * **`QC PASS`** ($\ge 70$): Clean, high-fidelity signals suitable for engineering analysis and response spectra computation.
     * **`QC WARNING`** ($50-69$): Non-fatal anomalies detected (isolated spikes, marginal pre-event SNR, or mild baseline tilt).
     * **`QC FAIL`** ($< 50$ or Fatal Override): Sensor clipping, ADC saturation, flatlines, or corrupted data channels. Clipped records are strictly invalidated for peak ground motion analysis because PGA is truncated and MMI estimation becomes artificially underestimated.
   * **Six-Class Diagnostic Taxonomy (Class 1–6)**: Standardized physical classification presented across the UI and export reports.

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
   * **SDOF Solver Cross-Validation Benchmark**: Interactive quantitative comparison calculating maximum relative deviation, mean relative deviation, and RMS difference between solvers.
   * **SNI 1726:2019 Design Code Overlay**: Direct comparison against Indonesian building code design spectra ($S_{DS}, S_{D1}, T_0, T_s$) as an engineering reference overlay.

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
| **4** | **Strong Motion** | Advanced energy diagnostics: cumulative Arias Intensity curves (*Husid Plot*), Significant Duration intervals ($D_{5-95}$ and $D_{5-75}$), $V_{\max}/A_{\max}$ ratio, and effective peak acceleration. |
| **5** | **Intensity** | Instrumental MMI classification (Worden et al., 2012) derived from maximum horizontal PGA and PGV, paired with perceived shaking descriptions and potential structural damage assessments. |
| **6** | **Spectrum** | Elastic 5% damped Pseudo-Spectral Acceleration ($S_a$) response curves across $T = 0.01 - 10.0$ s. Includes design code reference overlay (**SNI 1726:2019**) and the numerical **SDOF Solver Cross-Validation Benchmark** (Nigam-Jennings vs. Newmark-Beta). |
| **7** | **Report** | Technical PDF report preview and download, kinematic parameter CSV tables, discrete spectral response matrix CSVs, and complete ZIP package export. |

---

## Advanced Features

* **Batch Multi-Station Processing**: Process dozens of accelerograph station files simultaneously in one click, generating a unified regional comparison matrix sortable by highest PGA or MMI.
* **SDOF Solver Cross-Validation Benchmark**: Interactive computational verification tool that computes maximum, mean, and RMS relative discrepancies between Nigam-Jennings and Newmark-Beta solvers directly on active records.
* **Dual-Theme Switcher (Light Mode & Dark Mode)**: One-click toggle in the top-right header to seamlessly switch between light mode (formal reporting and bright environments) and dark mode (night observation in low-light seismic monitoring rooms).
* **Provenance Logging & Audit Trail**: Every signal processing step, filter parameter, library version, execution timestamp, and StationXML response status is logged transparently into export reports for rigorous scientific reproducibility.

---

## Mathematical Formulations

The following core mathematical formulas are implemented within the BSMA computational engine:

### 1. Nyquist Frequency Upper Limit

$$
f_{\max} \le 0.80 \times f_{\mathrm{Nyquist}} = 0.40 \times f_s \quad [\text{Hz}]
$$

### 2. Signal-to-Noise Ratio (SNR)

$$
\text{SNR} = 20 \log_{10}\left( \frac{\mathrm{RMS}_{\mathrm{signal}}}{\mathrm{RMS}_{\mathrm{noise}}} \right) \quad [\text{dB}]
$$

### 3. Peak Ground Motion Kinematics

$$
\text{PGA} = \max_{t} |a(t)| \quad [\text{Gal or cm/s}^2]
$$

$$
\text{PGV} = \max_{t} |v(t)| = \max_{t} \left| \int_0^t a(\tau) \, d\tau \right| \quad [\text{cm/s}]
$$

$$
\text{PGD} = \max_{t} |d(t)| = \max_{t} \left| \int_0^t v(\tau) \, d\tau \right| \quad [\text{cm}]
$$

### 4. Cumulative Arias Intensity ($I_a$)

$$
I_a = \frac{\pi}{2g} \int_0^{t_{\max}} [a(t)]^2 \, dt \quad [\text{m/s}]
$$

### 5. Significant Duration ($D_{5-95}$)

$$
D_{5-95} = t_{95} - t_{5} \quad [\text{seconds}]
$$

*Note: Defined by the time interval between 5% and 95% of total accumulated Arias Intensity on the Husid curve.*

### 6. SDOF 5% Damped Elastic Pseudo-Spectral Acceleration (PSA)

$$
\text{PSA}(T, \xi) = \omega^2 S_d(T, \xi) = \omega^2 \max_{t} |u(t)| \quad [g \text{ or m/s}^2]
$$

*Note: $\omega = \frac{2\pi}{T}$ denotes the oscillator undamped natural circular frequency, $S_d(T, \xi)$ is the maximum relative displacement spectrum, and $\xi = 0.05$ (5% critical damping ratio standard in earthquake engineering).*

### 7. Instrumental MMI Relationships (Worden et al., 2012)

Evaluated standardly on the **Maximum Horizontal Component (Max-H)**:

**PGA-Based Formulation (PGA in Gal):**

$$
\text{MMI}_{\text{PGA}} = \begin{cases}
1.78 + 1.55 \log_{10}(\text{PGA}), & \log_{10}(\text{PGA}) \le 1.57 \\
-1.60 + 3.70 \log_{10}(\text{PGA}), & \log_{10}(\text{PGA}) > 1.57
\end{cases}
$$

**PGV-Based Formulation (PGV in cm/s):**

$$
\text{MMI}_{\text{PGV}} = \begin{cases}
3.78 + 2.99 \log_{10}(\text{PGV}), & \log_{10}(\text{PGV}) \le 0.53 \\
2.40 + 4.96 \log_{10}(\text{PGV}), & \log_{10}(\text{PGV}) > 0.53
\end{cases}
$$

At moderate-to-severe ground shaking ($I_{\text{MMI}} \ge 5.0$), the PGV formulation automatically governs the final intensity value in adherence with USGS ShakeMap standards.

---

## Installation & Local Setup Guide

### System Requirements
* **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
* **Python Environment**: Version **3.10** through **3.13** (primary test environment: Python 3.13.2)
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

## Scientific Validation & Automated Test Suite

BSMA includes a comprehensive `pytest` test suite (83 unit tests, reference regression benchmarks, and real earthquake validation tests):
* **Synthetic Analytical Integration ([tests/test_integration.py](tests/test_integration.py))**: Verifies cumulative trapezoidal integration against exact sinusoidal solutions $a(t) = A\sin(\omega t)$ with relative error tolerances $< 0.1\%$.
* **Reference Dataset Regression Benchmark ([tests/test_reference_benchmark.py](tests/test_reference_benchmark.py))**: End-to-end tests on canonical synthetic waveforms ensuring deterministic convergence of peak kinematics (PGA, PGV, PGD), Arias energy ($I_a$), duration $D_{5-95}$, MMI (Worden et al., 2012), and PSA response spectra.
* **SDOF Solver Cross-Validation ([tests/test_response_spectrum.py](tests/test_response_spectrum.py))**: Verifies consistency between Nigam-Jennings (1969) and Newmark-Beta (1959) formulations (mean relative discrepancy $< 5\%$) and proves the high-frequency rigid limit anchor $\lim_{T \to 0} \text{PSA}(T) = \text{PGA}$.
* **MMI & ShakeMap Invariance ([tests/test_mmi.py](tests/test_mmi.py))**: Validates Worden et al. (2012) branching, monotonicity, PGV dominance at strong shaking, and finite number safety.
* **Quality Control Screening ([tests/test_qc.py](tests/test_qc.py))**: Validates threshold logic (PASS $\ge 70$, WARNING $50-69$, FAIL $< 50$) and mandatory fatal disqualification for sensor clipping and ADC saturation.
* **Real Operational BMKG Records Validation ([tests/test_level4_real_data.py](tests/test_level4_real_data.py))**: End-to-end multi-station validation on real BMKG earthquake records (PPJR, PCJI, PRJI) confirming 100% parameter agreement with operational standards.

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
├── scripts/                         # Automation & utility scripts
│   └── generate_guidebook.py        # Automated generator for the official User Guidebook
├── tests/                           # Comprehensive automated test suite (pytest)
│   ├── test_analysis_service.py     # Single-station analysis service integration tests
│   ├── test_integration.py          # Analytical synthetic integration verification (exact sine wave)
│   ├── test_level4_real_data.py     # Real multi-station operational BMKG earthquake record validation
│   ├── test_mmi.py                  # Worden et al. (2012) GMICE formulation tests
│   ├── test_parameters.py           # Kinematics and energy parameter extraction tests
│   ├── test_qc.py                   # Three-tier Quality Control system tests
│   ├── test_reference_benchmark.py  # End-to-end reference dataset regression benchmarks
│   └── test_response_spectrum.py    # SDOF solvers & Nigam-Jennings vs. Newmark-Beta benchmark tests
├── outputs/                         # Documentation and export outputs
│   └── BSMA_User_Guidebook.pdf      # Official User Guidebook & Technical Reference
├── app.py                           # Streamlit graphical user interface entry point (GUI)
├── assets/                          # Official institution branding & visual identity assets
│   ├── Logo_BMKG_Icon.png           # Transparent BMKG logo icon with solid white interior
│   ├── Logo_ITERA_Icon.png          # Transparent ITERA golden diamond logo icon
│   ├── Logo_ITERA.png               # High-resolution ITERA official crest
│   └── Logo_Judul.png               # BMKG Sleman original banner crest
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
5. **Harris, F. J.** (1978). On the use of windows for harmonic analysis with the discrete Fourier transform. *Proceedings of the IEEE*, 66(1), 51–83. [https://doi.org/10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837)
6. **Newmark, N. M.** (1959). A method of computation for structural dynamics. *Journal of the Engineering Mechanics Division, ASCE*, 85(3), 67–94. [https://doi.org/10.1061/JMCEA3.0000098](https://doi.org/10.1061/JMCEA3.0000098)
7. **Nigam, N. C., & Jennings, P. C.** (1969). Calculation of response spectra from strong-motion earthquake records. *Bulletin of the Seismological Society of America*, 59(2), 909–922. [https://doi.org/10.1785/BSSA0590020909](https://doi.org/10.1785/BSSA0590020909)
8. **Trifunac, M. D., & Brady, A. G.** (1975). A study on the duration of strong earthquake ground motion. *Bulletin of the Seismological Society of America*, 65(3), 581–626. [https://doi.org/10.1785/BSSA0650030581](https://doi.org/10.1785/BSSA0650030581)
9. **Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., & van der Walt, S. J.** (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272. [https://doi.org/10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)
10. **Worden, C. B., Gerstenberger, M. C., Rhoades, D. A., & Wald, D. J.** (2012). Probabilistic relationships between ground-motion parameters and MMI. *Bulletin of the Seismological Society of America*, 102(1), 204–221. [https://doi.org/10.1785/0120110156](https://doi.org/10.1785/0120110156)

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
