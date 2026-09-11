# BSMA Processing Methodology & Scientific Validation

## 1. Scope and Input Declaration

Each processing run in the BMKG Strong Motion Analyzer (BSMA v2.0.0) is declared under one of two operational modes:
1. **Raw Instrument Counts**: Requires matching StationXML metadata with valid poles-zeros (PAZ) and stage gain to deconvolve the instrument transfer function into physical acceleration (m/s²).
2. **Processed Physical Acceleration**: Requires an operator-declared unit (m/s², cm/s² / Gal, or g). BSMA does not infer physical calibration or unit solely from filename extensions or labels (e.g., `BP4`).

Internal calculations are strictly executed in SI units (m/s², m/s, m), and scaled to engineering units (Gal, cm/s, cm, g) only at the presentation and export layers.

---

## 2. End-to-End Processing Pipeline

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. INGESTION   │ ──> │  2. QC SCREEN   │ ──> │  3. DECONV/DSP  │
│ MiniSEED / SAC  │     │ Score 0-100     │     │ Detrend, Taper, │
│ + StationXML    │     │ Flags & Status  │     │ 4th-order BP    │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                    [ Corrected Acceleration a(t) ]
                                                         │
                                    ┌────────────────────┴───────────────────┐
                                    ▼                                        ▼
                         ┌───────────────────────┐               ┌───────────────────────┐
                         │ 4a. KINEMATICS & INT. │               │ 4b. SDOF SPECTRUM     │
                         │ • a(t) -> v(t) -> d(t)│               │ • Nigam-Jennings /    │
                         │ • PGA, PGV, PGD       │               │   Newmark-Beta (5%)   │
                         │ • Arias Ia, D5-95     │               │ • Cross-Solver Bench  │
                         │ • Worden 2012 MMI     │               │ • SNI 1726:2019 Design│
                         │   (Max-H component)   │               │   Reference Overlay   │
                         └──────────┬────────────┘               └───────────┬───────────┘
                                    │                                        │
                                    └────────────────────┬───────────────────┘
                                                         ▼
                                             ┌───────────────────────┐
                                             │ 5. REPORT & EXPORT    │
                                             │ • PDF Report / CSV    │
                                             │ • Full ZIP Archive    │
                                             └───────────────────────┘
```

### Stage 1: Quality Control (QC) & Signal Integrity Screening
Waveform data are audited prior to irreversible transformation:
- **3-Tier Operational Status**:
  - **`QC PASS`** (Score ≥ 70): Nominal signal integrity suitable for engineering design spectra and ground-motion modeling.
  - **`QC WARNING`** (Score 50–69): Non-fatal anomaly present (e.g., isolated spike, marginal SNR, minor baseline drift); requires filter corner and waveform review.
  - **`QC FAIL`** (Score < 50 or Fatal Anomaly): Triggered by sensor clipping, ADC saturation, flatline (dead channel), or telemetry gaps. Records with clipping/saturation are flagged as invalid for peak kinematics because PGA is truncated and MMI estimation will be underestimated.
- **Diagnostic Metrics**: Independent boolean flags (`clipping`, `adc_saturation`, `spikes`, `flatline`, `baseline_anomaly`, `low_snr`, `missing_data`) and continuous signal-to-noise ratio SNR = 20 log10(RMS_signal / RMS_noise).

### Stage 2: Instrument Response Deconvolution
- For raw digitizer counts, the instrument transfer function is deconvolved using ObsPy's inverse filtering engine (`simulate` with water-level regulation, default 60 dB).
- For pre-calibrated physical data, deconvolution is bypassed.

### Stage 3: Digital Signal Processing (DSP)
- **Baseline Detrending**: Mean removal and polynomial/linear trend correction to eliminate initial DC offset.
- **Cosine Tapering**: 5% Tukey window applied to both record edges to prevent Gibbs phenomenon and spectral leakage.
- **Forward-Backward Butterworth Filtering**: 4th-order forward-backward filter (`scipy.signal.sosfiltfilt` using Second-Order Sections). Passing the time series in both directions applies the squared magnitude response ($|H(f)|^2$), achieving an effective 48 dB/octave stopband attenuation slope (equivalent steepness to an 8th-order filter) with zero net phase distortion and preserving the stability of the 4th-order prototype.
- **Corner Frequencies**:
  - Baseline default band: 0.10 – 25.0 Hz.
  - High-frequency upper bound: f_max ≤ 0.80 f_Nyquist = 0.40 f_s to avoid near-Nyquist numerical artifacts.
  - Adaptive low-frequency floor: automatically elevated to 0.20 Hz or 0.40 Hz when record SNR is low (< 20 dB or < 10 dB) to suppress long-period drift.

### Stage 4: Stepwise Numerical Integration & Scientific Baseline Policy
- Numerical integration uses the cumulative trapezoidal rule:
  Velocity: v(t) = ∫ a(τ) dτ
  Displacement: d(t) = ∫ v(τ) dτ
- **Scientific Baseline Policy**: Baseline correction (linear detrending and mean removal) is applied strictly to acceleration prior to integration. To preserve the physical kinematic derivative relationship (a = dv/dt = d²x/dt²), no artificial post-integration detrending is applied to velocity or displacement.
- **Scientific Limitation**: PGD computed via bandpass-filtered integration represents **transient dynamic peak displacement**, not static tectonic fling-step or permanent ground deformation (which would require specialized non-linear baseline corrections or high-rate GNSS data).

### Stage 5: Kinematics, Energy, and Intensity Metrics
- **PGA, PGV, PGD**: Extracted as the absolute peak value max |x(t)| across the analyzed time window.
- **Arias Intensity (Ia)**:
  Ia = (π / 2g) ∫ [a(t)]² dt  [m/s]
- **Significant Duration (D5-95)**: Time interval between 5% and 95% of cumulative Husid Arias energy (t_95% - t_5%).
- **Instrumental Intensity (MMI)**: Evaluated using Worden et al. (2012) GMICE relations based strictly on the **Maximum Horizontal Component** (max(PGA_H), max(PGV_H)), excluding vertical Z/U channels:
  - For log10(PGA) ≤ 1.57: MMI = 1.78 + 1.55 log10(PGA)
  - For log10(PGA) > 1.57: MMI = -1.60 + 3.70 log10(PGA)
  - For log10(PGV) <= 0.53: MMI = 3.78 + 2.99 log10(PGV)
  - For log10(PGV) > 0.53: MMI = 2.40 + 4.96 log10(PGV)
  For strong shaking (MMI ≥ 5.0), PGV provides dominant physical correlation with structural damage according to USGS ShakeMap guidelines.

### Stage 6: SDOF Response Spectrum & Design Spectrum Overlay
- **Elastic SDOF Formulation**: Pseudo-Spectral Acceleration (PSA = ω² max |u(t)|) for critical damping ratio ξ = 5% over periods T = 0.01 – 10.0 s.
- **Solvers**:
  1. **Nigam & Jennings (1969)**: Closed-form recurrence solution for piecewise-linear ground acceleration input.
  2. **Newmark-Beta (1959)**: Implicit step-by-step numerical integration (γ = 1/2, β = 1/4, average acceleration).
- **Built-in SDOF Benchmark**: Automated cross-validation computing maximum relative error (≤ 5%), mean relative difference, and RMS concordance across the period spectrum.
- **Design Spectrum Comparison**: Overlay against Indonesian standard **SNI 1726:2019** design response spectra (S_DS, S_D1, T0, Ts) as an engineering design reference overlay for structural safety comparison, rather than an empirical quantity extracted from the waveform.

---

## 3. Automated Verification & Benchmark Testing

BSMA maintains a comprehensive automated test suite (`pytest tests/`) validating scientific computations:
- **Synthetic Sinusoidal Integration**: Verifies analytical accuracy of cumulative trapezoid integration against a(t) = A sin(ωt), v(t) = (A/ω)(1 - cos ωt), d(t) = (A/ω)t - (A/ω²)sin(ωt) with < 0.1% error.
- **Rigid Limit Asymptote**: Confirms lim_{T → 0} PSA(T) = PGA within < 0.5% numerical tolerance.
- **Damping Decay**: Confirms free-vibration logarithmic decrement concordance across extreme damping ratios (ξ ∈ [0.0, 0.99]).
- **SDOF Cross-Solver Concordance**: Verifies < 5% mean relative error between Nigam-Jennings and Newmark-Beta.
- **Worden MMI Calibration**: Unit tests verifying piecewise branch continuity, monotonicity, PGV dominance at high intensity, and non-finite handling.

---

## 4. References

1. **Arias, A.** (1970). A measure of earthquake intensity. In R. J. Hansen (Ed.), *Seismic design for nuclear power plants* (pp. 438–483). MIT Press.
2. **Badan Standardisasi Nasional.** (2019). *SNI 1726:2019: Tata cara perencanaan ketahanan gempa untuk struktur bangunan gedung dan non gedung*. BSN.
3. **Beyreuther, M., et al.** (2010). ObsPy: A Python toolbox for seismology. *Seismological Research Letters*, 81(3), 530–533. https://doi.org/10.1785/gssrl.81.3.530
4. **Newmark, N. M.** (1959). A method of computation for structural dynamics. *Journal of the Engineering Mechanics Division, ASCE*, 85(3), 67–94. https://doi.org/10.1061/JMCEA3.0000098
5. **Nigam, N. C., & Jennings, P. C.** (1969). Calculation of response spectra from strong-motion earthquake records. *Bulletin of the Seismological Society of America*, 59(2), 909–922. https://doi.org/10.1785/BSSA0590020909
6. **Trifunac, M. D., & Brady, A. G.** (1975). A study on the duration of strong earthquake ground motion. *Bulletin of the Seismological Society of America*, 65(3), 581–626. https://doi.org/10.1785/BSSA0650030581
7. **Virtanen, P., et al.** (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2
8. **Worden, C. B., Gerstenberger, M. C., Rhoades, D. A., & Wald, D. J.** (2012). Probabilistic relationships between ground-motion parameters and MMI. *Bulletin of the Seismological Society of America*, 102(1), 204–221. https://doi.org/10.1785/0120110156
