<h1 align="center">BMKG Strong Motion Analyzer (BSMA v2.0.0)</h1>
<p align="center"><b>Platform Komputasi Sinyal Akselerograf, Kinematika Seismik, & Spektrum Respons</b></p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="https://strong-motion.streamlit.app/"><img src="https://img.shields.io/badge/Streamlit-Live_Cloud_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit App" /></a>
  <a href="https://docs.obspy.org/"><img src="https://img.shields.io/badge/ObsPy-Seismology_Framework-4A90E2?style=for-the-badge" alt="ObsPy" /></a>
  <a href="https://scipy.org/"><img src="https://img.shields.io/badge/SciPy-DSP_%26_Solvers-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy" /></a>
  <a href="https://plotly.com/"><img src="https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly" /></a>
  <a href="https://github.com/ahmaddidan/BSMA-v.2"><img src="https://img.shields.io/badge/Status-Release_v2.0.0-059669?style=for-the-badge" alt="Status" /></a>
</p>

<p align="center">
  <b>Bahasa:</b> <a href="README.md">English</a> | <b>Bahasa Indonesia</b>
</p>

---

### Pusat Dokumentasi & Luaran Resmi Proyek

Akses aplikasi web interaktif dan unduh dokumen resmi teknis serta ilmiah yang telah diterbitkan:

* **Aplikasi Web Cloud Interaktif (Daring)**: [https://strong-motion.streamlit.app/](https://strong-motion.streamlit.app/)
* **📘 Panduan Teknis Operasional (Visual User Manual)**:
  * **Edisi Bahasa Indonesia**: [outputs/BSMA_Panduan_Teknis_Operasional_ID.pdf](outputs/BSMA_Panduan_Teknis_Operasional_ID.pdf)
  * **Edisi Bahasa Inggris**: [outputs/BSMA_Panduan_Teknis_Operasional_EN.pdf](outputs/BSMA_Panduan_Teknis_Operasional_EN.pdf)
* **📑 Buku Panduan Ilmiah (Landasan Teori, Formulasi, & Algoritma)**:
  * **Edisi Bahasa Indonesia**: [outputs/BSMA_Scientific_Guidebook_ID.pdf](outputs/BSMA_Scientific_Guidebook_ID.pdf)
  * **Edisi Bahasa Inggris**: [outputs/BSMA_Scientific_Guidebook_EN.pdf](outputs/BSMA_Scientific_Guidebook_EN.pdf)
* **Repositori Kode Sumber GitHub**: [https://github.com/ahmaddidan/BSMA-v.2](https://github.com/ahmaddidan/BSMA-v.2)

---

## Ringkasan Eksekutif & Informasi Proyek

**BMKG Strong Motion Analyzer (BSMA v2.0.0)** merupakan perangkat lunak analisis sinyal akselerograf yang dirancang dan dikembangkan secara mandiri dalam rangka pelaksanaan kegiatan **Kerja Praktik (KP)** mahasiswa **Program Studi Teknik Geofisika, Fakultas Teknologi Industri, Institut Teknologi Sumatera (ITERA)** di **Stasiun Geofisika Kelas I Sleman, Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) D.I. Yogyakarta** (periode 20 Juli – 20 Agustus 2026). Perangkat lunak ini merupakan karya akademik independen dan bukan merupakan sistem operasional atau produk komersial BMKG.

Aplikasi ini mengotomatisasi alur kerja analisis rekaman getaran tanah kuat (*strong ground motion*) secara terpadu—mulai dari pembacaan data mentah (*raw counts*), dekonvolusi respons instrumen StationXML, evaluasi integritas kualitas data (Quality Control), pemrosesan sinyal digital (DSP), integrasi kinematika puncak (PGA, PGV, PGD), estimasi intensitas instrumental Skala MMI berbasis perumusan GMICE Worden et al. (2012), hingga pemodelan Spektrum Respons *Pseudo-Spectral Acceleration* (PSA) elastis redaman 5% dengan opsi solver analitik **Nigam–Jennings (1969)** dan integrasi implisit **Newmark-Beta (1959)** yang dapat diperbandingkan dengan standar ketahanan gempa **SNI 1726:2019**.

---

## Arsitektur Pipeline Pemrosesan Sinyal

Sistem BSMA mengimplementasikan pipeline komputasi sekuensial yang ketat guna menjamin transparansi (*provenance tracking*) dan reproduksibilitas ilmiah (*scientific reproducibility*):

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

### Rincian Metodologi Pipeline

1. **Ingesti Data & Validasi Metadata**:
   * Mendukung format gelombang triaksial standar **MiniSEED (`.mseed`)** (FDSN) dan **SAC (`.sac`)** (IRIS).
   * Sinkronisasi waktu otomatis untuk 3 kanal ortogonal (Z, N, E / U-D, N-S, E-W) berbasis penanda waktu absolut UTC.
   * Dekonvolusi fungsi transfer instrumen menggunakan metadata **StationXML (`.xml`)** (kutub-nol / PAZ dan tahapan penguatan melalui ObsPy).
   * Fitur otomatis *Physical Acceleration Bypass* ketika rekaman masukan telah dikalibrasi ke satuan percepatan fisik (m/s² atau Gal).

2. **Kendali Mutu (Quality Control) & Penapisan Integritas Sinyal**:
   * **Skor Mutu ($Q \in [0, 100]$)**: Indeks kebersihan kuantitatif dari penalti fisik terkalibrasi:
     $$Q = \max\left(0, \min\left(100, 100 - \sum_{i} P_i\right)\right)$$
     dengan bobot penalti: `WARNING` = $-15\text{ poin}$, `ERROR` = $-40\text{ poin}$, dan anomali `CRITICAL` memicu pembatalan fatal ($Q = 0$).
   * **Tiga Tingkat Status Operasional**:
     * **`QC PASS`** ($Q \ge 70$): Sinyal bersih dan layak untuk analisis rekayasa serta komputasi spektrum respons.
     * **`QC WARNING`** ($50 \le Q < 70$): Terdeteksi anomali non-fatal (derau latar marginal, lonjakan terisolasi, atau tilt baseline ringan).
     * **`QC FAIL`** ($Q < 50$ atau Fatal Override): Sinyal terpotong (*clipping* ADC), kanal mati (*flatline*), atau berkas korup. Data terpotong didiskualifikasi mutlak karena PGA terpangkas dan estimasi MMI menjadi palsu.

3. **Pemrosesan Sinyal Digital (DSP)**:
   * **Koreksi Garis Dasar (Detrending)**: Pengurangan nilai rata-rata (*mean removal*) dan tren linier/polinomial untuk meniadakan offset garis dasar awal (Boore & Bommer, 2005).
   * **Penjendelaan Kosinus 5% (Tukey Window)**: Meredam diskontinuitas amplitudo pada batas awal dan akhir rekaman guna mencegah kebocoran spektral (*spectral leakage*).
   * **Filter Lolos Pita Butterworth Orde-4 Zero-Phase**: Penapisan maju-mundur dua arah (`scipy.signal.sosfiltfilt`) yang menjamin pergeseran fase nol ($\Delta \phi = 0$) dengan kemiringan atenuasi efektif 48 dB/oktaf pada stopband.
   * **Batas Pengaman Shannon-Nyquist**: $f_{\max} \le 0.40 f_s$ untuk mencegah aliasing dan ketidakstabilan numerik.

4. **Integrasi Kinematika & Kebijakan Garis Dasar**:
   * Integrasi numerik sekuensial percepatan $a(t)$ ke kecepatan $v(t)$, dan kecepatan ke perpindahan $d(t)$ menggunakan aturan trapesium kumulatif.
   * **Kebijakan Garis Dasar Ketat**: Koreksi baseline hanya diterapkan pada percepatan *sebelum* integrasi. Sesuai prinsip kalkulus kinematik, BSMA menghindari pemaksaan garis dasar polinomial buatan pada kecepatan atau perpindahan pascaintegrasi guna mempertahankan konsistensi turunan ($a = \dot{v} = \ddot{d}$).
   * **Interpretasi PGD**: PGD mencerminkan **perpindahan puncak dinamis transien** dalam batas lolos pita filter, bukan deformasi tektonik permanen (*fling-step*).

5. **Parameter Kinematika & MMI Instrumental**:
   * Ekstraksi parameter puncak: **PGA**, **PGV**, **PGD**, dan rasio dinamis **$V_{\max}/A_{\max}$**.
   * Metrik energi dan durasi: **Intensitas Arias ($I_a$)** dan durasi signifikan (**$D_{5-95}$** dan **$D_{5-75}$**).
   * Estimasi intensitas MMI instrumental menggunakan regresi GMICE **Worden et al. (2012)** pada **Komponen Horizontal Maksimum (Max-H)** dengan transisi kontinu dari dominansi PGA ke PGV pada guncangan kuat ($I_{\text{MMI}} \ge 5.0$).

6. **Spektrum Respons SDOF & Standar Desain Bangunan**:
   * Spektrum Pseudo-Spectral Acceleration (PSA) elastis redaman 5% pada rentang periode $T = 0.01 - 10.0$ detik.
   * Dua mesin solver numerik:
     * **Nigam–Jennings (1969)**: Metode analitik eksak matriks transisi dengan eksitasi sepotong-sepotong linier.
     * **Newmark-Beta (1959)**: Metode integrasi waktu implisit percepatan rata-rata konstan ($\gamma = 1/2, \beta = 1/4$).
   * **Benchmark Validasi Silang Solver SDOF**: Evaluasi komparatif langsung yang membuktikan selisih relatif $< 0.08\%$ dan perbedaan RMS $< 1.5 \times 10^{-4}\text{ g}$ pada 100 titik periode.
   * **Overlay Spektrum Desain SNI 1726:2019**: Perbandingan terhadap batas kapasitas desain gempa untuk kelas situs Tanah Lunak (SE), Tanah Sedang (SD), dan Tanah Keras (SC).

---

## 7 Tab Analisis Interaktif

Antarmuka grafis BSMA dibangun berbasis Streamlit dan Plotly yang terbagi ke dalam 7 tab fungsional:

| Tab | Nama | Ruang Lingkup & Kemampuan Analisis |
| :---: | :--- | :--- |
| **1** | **Summary** | Dasbor eksekutif yang merangkum metadata stasiun, pratinjau bentuk gelombang 3-komponen, triad metrik kinematika kanal terkuat, status QC, estimasi MMI, dan panel jejak audit. |
| **2** | **Waveforms** | Deret waktu interaktif 3 komponen untuk kinematika lengkap: Percepatan ($a$), Kecepatan ($v$), dan Perpindahan ($d$), lengkap dengan penanda kedatangan gelombang P dan anotasi puncak. |
| **3** | **Quality Control** | Diagnostik integritas sinyal: Skor Mutu kuantitatif (0–100), lencana kelayakan (PASS/WARNING/FAIL), diagram polar radar kendali mutu, kurva PSD, estimasi SNR (dB), dan flag anomali diagnostik. |
| **4** | **Strong Motion** | Karakterisasi energi getaran: Kurva akumulasi energi Arias (*Husid Plot*), *Cumulative Absolute Velocity* (CAV, ambang batas EPRI 1988 $\ge 0.16\text{ g}\cdot\text{s}$), durasi signifikan ($D_{5-95}$ dan $D_{5-75}$), serta rasio kinematik $V_{\max}/A_{\max}$. |
| **5** | **Intensity** | Interpretasi guncangan instrumental objektif melalui regresi GMICE Worden et al. (2012) berbasis komponen horizontal maksimum (Max-H) disertai deskripsi persepsi dan potensi kerusakan BMKG. |
| **6** | **Spectrum** | Spektrum respons elastis PSA redaman 5% pada rentang $T = 0.01 - 10.0$ detik. Dilengkapi overlay spektrum desain gempa **SNI 1726:2019** dan panel **Benchmark Validasi Silang Solver SDOF** (Nigam-Jennings vs. Newmark-Beta). |
| **7** | **Report** | Pratinjau dan pengunduhan laporan teknik PDF resmi 2-halaman, tabel CSV parameter kinematika, matriks spektral CSV, dan paket arsip ZIP terpadu. |

---

## Panduan Instalasi & Pengoperasian Lokal

### Kebutuhan Sistem
* **Sistem Operasi**: Windows 10/11, macOS, atau Linux (Ubuntu 20.04+)
* **Lingkungan Python**: Versi **3.10** hingga **3.13**
* **Pustaka Utama**: Streamlit, ObsPy, NumPy, SciPy, Pandas, Matplotlib, Plotly, FPDF, PyMuPDF
* **Kapasitas RAM**: Minimal 4 GB (disarankan 8 GB untuk pemrosesan batch massal)

### Langkah-Langkah Instalasi

1. **Kloning Repositori:**
   ```bash
   git clone https://github.com/ahmaddidan/BSMA-v.2.git
   cd BSMA-v.2
   ```

2. **Buat dan Aktifkan Virtual Environment:**
   * Di Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * Di Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Pasang Dependensi:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi BSMA:**
   ```bash
   streamlit run app.py
   ```
   Peramban web akan otomatis membuka alamat `http://localhost:8501`.

5. **(Opsional) Eksekusi Pengujian Otomatis:**
   ```bash
   pytest tests/
   ```

---

## Struktur Direktori Repositori

```text
Project BSMA/
├── .streamlit/                      # Konfigurasi tema visual (light/dark) & server Streamlit
├── core/                            # Komputasi ilmiah murni (Standard Python + NumPy/SciPy)
│   ├── interfaces/                  # Kontrak antarmuka abstrak preprocessor dan solver
│   ├── io/                          # Parser gelombang (MiniSEED, SAC) & dekonvolusi StationXML
│   ├── preprocessing/               # Detrending, tapering, filter Butterworth, screening QC
│   ├── processing/                  # Integrasi kinematika, parameter gempa (PGA/PGV/PGD, Arias, MMI)
│   ├── sdof/                        # Solver dinamika struktur SDOF (Nigam-Jennings 1969 & Newmark-Beta 1959)
│   ├── types/                       # Skema dataclass untuk konteks, sinyal gelombang, dan state pemrosesan
│   ├── orchestrator.py              # Orkestrator pusat pipeline pemrosesan sinyal
│   └── pipeline.py                  # Pipeline pemrosesan terpadu sekuensial
├── services/                        # Lapisan layanan aplikasi
│   ├── analysis_service.py          # Layanan analisis stasiun tunggal dan rekomendasi adaptif filter
│   ├── batch_service.py             # Layanan pemrosesan batch multi-stasiun
│   └── export_service.py            # Layanan ekspor CSV tabel, matriks spektral, dan arsip ZIP
├── utils/                           # Modul utilitas pendukung
│   ├── exceptions.py                # Kode galat terstruktur dan eksepsi kustom
│   ├── exporter.py                  # Serialisasi data tabel dan format luaran
│   ├── logger.py                    # Pencatatan jejak audit dan log eksekusi ilmiah
│   └── pdf_exporter.py              # Generator laporan teknik rekayasa berformat PDF
├── scripts/                         # Skrip otomasi & generator dokumen
│   ├── generate_guidebook.py        # Generator Buku Panduan Ilmiah (ID & EN)
│   └── generate_operational_guidebook.py # Generator Panduan Teknis Operasional (ID & EN)
├── tests/                           # Pengujian perangkat lunak otomatis (pytest - 83 test cases)
│   ├── test_analysis_service.py     # Pengujian integrasi layanan analisis stasiun tunggal
│   ├── test_integration.py          # Verifikasi integrasi analitik gelombang sintetis
│   ├── test_level4_real_data.py     # Validasi data rekaman gempa operasional riil BMKG
│   ├── test_mmi.py                  # Pengujian formulasi GMICE Worden et al. (2012)
│   ├── test_parameters.py           # Pengujian ekstraksi parameter kinematika dan energi
│   ├── test_qc.py                   # Pengujian sistem tiga-tingkat Kendali Mutu sinyal
│   ├── test_reference_benchmark.py  # Benchmark regresi dataset acuan end-to-end
│   └── test_response_spectrum.py    # Benchmark solver SDOF Nigam-Jennings vs. Newmark-Beta
├── outputs/                         # Berkas luaran dokumen resmi
│   ├── BSMA_Panduan_Teknis_Operasional_EN.pdf # Panduan Teknis Operasional (Bahasa Inggris - 18 Halaman)
│   ├── BSMA_Panduan_Teknis_Operasional_ID.pdf # Panduan Teknis Operasional (Bahasa Indonesia - 18 Halaman)
│   ├── BSMA_Scientific_Guidebook_EN.pdf       # Buku Panduan Ilmiah & Algoritma (Bahasa Inggris - 17 Halaman)
│   └── BSMA_Scientific_Guidebook_ID.pdf       # Buku Panduan Ilmiah & Algoritma (Bahasa Indonesia - 17 Halaman)
├── docs/images/                     # Tangkapan layar antarmuka resolusi tinggi untuk pratinjau dokumen
├── app.py                           # Titik masuk utama antarmuka pengguna Streamlit (GUI)
├── assets/                          # Aset visual resmi kelembagaan (logo BMKG & ITERA)
├── requirements.txt                 # Spesifikasi dependensi Python dengan pembatasan versi
├── README.md                        # Dokumentasi repositori komprehensif (Bahasa Inggris)
└── README.id.md                     # Dokumentasi repositori komprehensif (Bahasa Indonesia)
```

---

## Daftar Pustaka Ilmiah

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

## Profil Pengembang & Afiliasi Kelembagaan

* **Penyusun / Pengembang**: Ahmad Didane Setyawan Putra
* **Nomor Induk Mahasiswa (NIM)**: 123120094
* **Program Studi / Fakultas**: Teknik Geofisika, Fakultas Teknologi Industri
* **Perguruan Tinggi**: Institut Teknologi Sumatera (ITERA)
* **Kontak Surel Resmi**: [ahmad.123120094@student.itera.ac.id](mailto:ahmad.123120094@student.itera.ac.id)
* **Tautan Repositori GitHub**: [https://github.com/ahmaddidan/BSMA-v.2](https://github.com/ahmaddidan/BSMA-v.2)
* **Instansi Mitra Kerja Praktik**: Stasiun Geofisika Kelas I Sleman, Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) D.I. Yogyakarta (Periode Kerja Praktik: 20 Juli – 20 Agustus 2026)

---

<div align="center">
  <sub>Ahmad Didane Setyawan Putra © 2026 · Karya Akademik Kerja Praktik Mahasiswa Teknik Geofisika ITERA di BMKG Stasiun Geofisika Sleman</sub>
</div>
