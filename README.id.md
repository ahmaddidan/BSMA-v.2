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

### Akses Aplikasi & Luaran Proyek

Aplikasi interaktif, berkas panduan teknis resmi, dan repositori kode sumber dapat diakses melalui tautan berikut:

* **Aplikasi Cloud Web (Daring)**: [https://strong-motion.streamlit.app/](https://strong-motion.streamlit.app/)
* **Buku Panduan Pengguna (PDF)**: [outputs/BSMA_User_Guidebook.pdf](outputs/BSMA_User_Guidebook.pdf)
* **Repositori Kode Sumber GitHub**: [https://github.com/ahmaddidan/BSMA-v.2](https://github.com/ahmaddidan/BSMA-v.2)

---

## Ringkasan Eksekutif & Informasi Proyek

**BMKG Strong Motion Analyzer (BSMA v2.0.0)** merupakan perangkat lunak analisis sinyal akselerograf yang dirancang dan dikembangkan secara mandiri dalam rangka pelaksanaan kegiatan **Kerja Praktik (KP)** mahasiswa **Program Studi Teknik Geofisika, Fakultas Teknik Industri, Institut Teknologi Sumatera (ITERA)** di **Stasiun Geofisika Kelas I Sleman, Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) D.I. Yogyakarta** (periode 20 Juli – 20 Agustus 2026). Perangkat lunak ini merupakan karya akademik independen dan bukan merupakan sistem operasional atau produk komersial BMKG.

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

### Rincian Metodologi Tahapan

1. **Ingestion & Validasi Metadata**:
   * Mendukung berkas triaksial **MiniSEED (`.mseed`)** standar FDSN dan **SAC (`.sac`)** standar IRIS.
   * Sinkronisasi otomatis 3-komponen ortogonal (Z, N, E / U-D, N-S, E-W) berbasis penanda waktu UTC absolut.
   * Dekonvolusi fungsi transfer instrumen (*transfer function*) berbasis berkas **StationXML (`.xml`)** (analisis *poles-zeros* / PAZ dan *stage gain*).
   * Mode *Physical Acceleration Bypass* diaktifkan otomatis bila masukan telah berdimensi percepatan fisik (m/s² atau Gal).

2. **Quality Control (QC) & Integritas Sinyal**:
   * **Skor Kualitas (Quality Score, 0–100)**: Ukuran kuantitatif kebersihan sinyal berdasarkan penalti anomali fisik.
   * **Status Validasi Operasional**:
     * **`QC PASS`** ($\ge 70$): Kualitas sinyal nominal, andal untuk analisis rekayasa dan spektrum respons.
     * **`QC WARNING`** ($50-69$): Terdapat anomali non-fatal (lonjakan terisolasi, SNR marginal, pergeseran baseline minor).
     * **`QC FAIL`** ($< 50$ atau Anomali Fatal): Mengalami sensor *clipping*, saturasi ADC, *flatline* (sensor mati), atau *data gap*. Data dengan *clipping* ditandai tidak valid untuk analisis puncak karena PGA terpotong dan estimasi MMI *underestimate*.
   * **Taksonomi 6 Kategori Diagnostik (*Class 1* – *Class 6*)**: Klasifikasi fisik komparatif untuk penyajian antarmuka dan laporan.

3. **Digital Signal Processing (DSP)**:
   * **Penghilangan Tren (*Detrending*)**: Koreksi *mean offset* dan tren polinomial linier untuk mengoreksi pergeseran DC awal (Boore & Bommer, 2005).
   * **Jendela Kosinus Tukey 5% (*Tapering*)**: Menghaluskan diskontinuitas amplitudo di kedua ujung rekaman untuk mereduksi kebocoran spektral (*spectral leakage*) (Harris, 1978).
   * **Filtering Butterworth Orde-4 Maju-Mundur (*Forward-Backward Filtering*)**: Diterapkan melalui pemrosesan dua arah (`scipy.signal.sosfiltfilt`) untuk mengeliminasi distorsi fase tanpa pergeseran waktu neto (*zero net phase lag*). Kuadrat fungsi transfer magnitudo ($|H(f)|^2$) memberikan laju atenuasi efektif 48 dB/oktaf pada pita henti (setara kemiringan orde-8), dengan tetap mempertahankan kestabilan dan sifat pole prototipe orde-4 aslinya.
   * **Batas Atas Nyquist 80%**: $f_{\max} \le 0.80 f_{\mathrm{Nyquist}} = 0.40 f_s$ guna mencegah timbulnya artefak numerik frekuensi tinggi.
   * **Floor Adaptif SNR Frekuensi Rendah**: Cutoff $f_{\min}$ dinaikkan secara adaptif ke $0.20\text{ Hz}$ atau $0.40\text{ Hz}$ saat rekaman memiliki rasio sinyal-derau rendah ($< 20\text{ dB}$ atau $< 10\text{ dB}$) guna menekan drift periode panjang.

4. **Integrasi Kinematika & Kebijakan Garis Dasar (Baseline Policy)**:
   * Integrasi numerik bertahap dari percepatan $a(t)$ ke kecepatan $v(t)$, dilanjutkan ke perpindahan $d(t)$ menggunakan aturan trapesium kumulatif (*cumulative trapezoidal rule*).
   * **Kebijakan Ilmiah Garis Dasar**: Koreksi garis dasar (*detrending*) diaplikasikan secara ketat hanya pada deret waktu percepatan *sebelum* integrasi. Sesuai kaidah kalkulus kinematika murni, BSMA secara sengaja **tidak melakukan modifikasi detrending pasca-integrasi** pada kecepatan atau perpindahan guna menjaga konsistensi derivatif fisik ($a = \dot{v} = \ddot{d}$).
   * **Catatan Metodologis PGD**: Nilai PGD yang dihasilkan merepresentasikan **perpindahan puncak dinamik transien** dalam batas pita frekuensi filter, bukan deformasi tektonik statis permanen (*static fling-step*), yang secara seismologis membutuhkan koreksi garis dasar nonlinear khusus atau data GPS laju-tinggi (*high-rate GNSS*).

5. **Parameter Kinematika Seismik & Intensitas Instrumental MMI**:
   * Ekstraksi parameter puncak absolut: **PGA**, **PGV**, **PGD**, dan rasio **$V_{\max}/A_{\max}$**.
   * Perhitungan energi seismik kumulatif **Intensitas Arias ($I_a$)** dan interval waktu **Durasi Signifikan ($D_{5-95}$)**.
   * Estimasi intensitas instrumental Skala MMI berbasis perumusan empiris GMICE **Worden et al. (2012)** USGS ShakeMap yang dihitung secara baku dari **Komponen Horizontal Maksimum (Max-H)** (mengecualikan kanal vertikal Z/U), dengan transisi dominansi PGV pada guncangan kuat ($I_{\text{MMI}} \ge 5.0$).

6. **Spektrum Respons SDOF & Standar Desain SNI 1726:2019**:
   * Komputasi kurva **Pseudo-Spectral Acceleration (PSA)** osilator elastis SDOF dengan redaman kritis 5% ($\xi = 0.05$) pada rentang periode $T = 0.01 - 10.0$ detik.
   * Pilihan algoritma solver: formulasi rekursif analitik **Nigam & Jennings (1969)** (*piecewise-linear ground acceleration state-transition*) atau integrasi implisit **Newmark-Beta (1959)** ($\gamma = 1/2, \beta = 1/4$, *average acceleration*).
   * Fitur **SDOF Solver Cross-Validation & Numerical Benchmark**: Evaluasi kuantitatif deviasi relatif maksimal, deviasi rata-rata, dan selisih RMS antara kedua solver.
   * Perbandingan langsung (*overlay*) terhadap kurva spektrum desain **SNI 1726:2019** ($S_{DS}, S_{D1}, T_0, T_s$) sebagai acuan standar rekayasa kegempaan (*design code reference overlay*), bukan sebagai besaran yang diestimasi langsung dari rekaman sinyal.

7. **Pelaporan Teknis & Ekspor Multi-Format**:
   * Dokumen laporan teknis PDF komprehensif individual dan batch.
   * Lembar data tabular CSV ringkasan parameter kinematika.
   * Lembar data matriks CSV spektrum respons diskret ($T$ vs $S_a$).
   * Paket arsip terkompresi ZIP untuk pemrosesan banyak stasiun (*batch processing*).

---

## Eksplorasi 7 Tab Fitur Analisis Interaktif

Antarmuka BSMA v2.0.0 dibangun di atas pustaka interaktif Streamlit dan visualisasi saintifik Plotly, terbagi ke dalam 7 tab tematik:

| Tab | Nama Tab | Deskripsi & Kapabilitas Fungsional |
| :---: | :--- | :--- |
| **1** | **Summary** | Dasbor eksekutif menyajikan kartu metadata stasiun (lintang, bujur, elevasi), pratinjau seismogram 3-kanal (40–50% tinggi layar), metrik kinematika puncak komponen terkuat, status QC, estimasi MMI, dan panel jejak audit (*audit trail*). |
| **2** | **Waveforms** | Visualisasi interaktif riwayat waktu 3-komponen untuk kinematika lengkap: Percepatan ($a$), Kecepatan ($v$), dan Perpindahan ($d$). Dilengkapi penanda waktu tiba otomatis (*pick markers*) dan titik amplitudo puncak absolut. |
| **3** | **Quality Control** | Diagnostik integritas rekaman: skor numerik (0–100), status validasi (PASS/WARNING/FAIL), visualisasi radar metrik derau latar, kurva *Power Spectral Density* (PSD), estimasi SNR (dB), dan bendera deteksi anomali (*clipping*, *spikes*, *flatline*). |
| **4** | **Strong Motion** | Analisis energi guncangan mendalam: kurva akumulasi Intensitas Arias (*Husid Plot*), interval Durasi Signifikan ($D_{5-95}$ dan $D_{5-75}$), rasio $V_{\max}/A_{\max}$, dan percepatan efektif (*effective peak acceleration*). |
| **5** | **Intensity** | Klasifikasi tingkat guncangan Skala MMI instrumental (Worden et al., 2012) berbasis PGA dan PGV horizontal maksimum. Menyajikan kartu deskripsi dampak fisik guncangan (*Perceived Shaking*) dan potensi kerusakan struktural (*Potential Damage*). |
| **6** | **Spectrum** | Kurva Spektrum Respons Pseudo-Percepatan ($S_a$) elastis redaman 5% untuk ketiga kanal pada rentang periode $T = 0.01 - 10.0$ s. Mendukung perbandingan langsung (*overlay*) terhadap kurva spektrum desain **SNI 1726:2019**, serta panel verifikasi silang numerik (*SDOF Solver Benchmark: Nigam-Jennings vs Newmark-Beta*). |
| **7** | **Report** | Pratinjau dan pengunduhan dokumen laporan teknis PDF komprehensif, tabel CSV ringkasan parameter kinematika, lembar CSV matriks spektrum respons diskret, dan paket arsip ZIP. |

---

## Fitur Unggulan Lanjutan

* **Batch Processing Multi-Stasiun**: Memproses puluhan rekaman stasiun akselerograf secara simultan dalam satu klik, menghasilkan tabel matriks komparasi regional terpadu yang dapat diurutkan berdasarkan PGA tertinggi atau estimasi MMI.
* **SDOF Solver Cross-Validation Benchmark**: Alat uji presisi komputasi interaktif yang menghitung deviasi relatif maksimal, rata-rata, dan RMS antara Nigam-Jennings dan Newmark-Beta secara langsung pada rekaman aktif.
* **Dual-Theme Switcher (Light Mode & Dark Mode)**: Tombol saklar instan di pojok kanan atas aplikasi untuk beralih antara tema terang (kebutuhan pelaporan formal dan pencahayaan terang) dan tema gelap (pengamatan jangka panjang di ruang monitor seismologi redup).
* **Provenance Logging & Audit Trail**: Setiap tahapan pemrosesan dicatat secara transparan (parameter filter, versi pustaka numerik, waktu eksekusi, dan status StationXML) pada laporan akhir guna menjamin reproduksibilitas ilmiah (*scientific reproducibility*).

---

## Landasan Formulasi Matematis

Berikut adalah formulasi matematis baku yang diimplementasikan dalam mesin komputasi BSMA:

### 1. Batas Numerik Frekuensi Nyquist

$$
f_{\max} \le 0.80 \times f_{\mathrm{Nyquist}} = 0.40 \times f_s \quad [\text{Hz}]
$$

### 2. Rasio Sinyal terhadap Derau (Signal-to-Noise Ratio)

$$
\text{SNR} = 20 \log_{10}\left( \frac{\mathrm{RMS}_{\mathrm{signal}}}{\mathrm{RMS}_{\mathrm{noise}}} \right) \quad [\text{dB}]
$$

### 3. Kinematika Puncak Gerakan Tanah (Peak Ground Motion)

$$
\text{PGA} = \max_{t} |a(t)| \quad [\text{Gal atau cm/s}^2]
$$

$$
\text{PGV} = \max_{t} |v(t)| = \max_{t} \left| \int_0^t a(\tau) \, d\tau \right| \quad [\text{cm/s}]
$$

$$
\text{PGD} = \max_{t} |d(t)| = \max_{t} \left| \int_0^t v(\tau) \, d\tau \right| \quad [\text{cm}]
$$

### 4. Intensitas Arias Kumulatif ($I_a$)

$$
I_a = \frac{\pi}{2g} \int_0^{t_{\max}} [a(t)]^2 \, dt \quad [\text{m/s}]
$$

### 5. Durasi Signifikan ($D_{5-95}$)

$$
D_{5-95} = t_{95} - t_{5} \quad [\text{detik}]
$$

Keterangan: Dihitung dari selisih waktu antara pencapaian 5% hingga 95% integral akumulasi energi Husid.

### 6. Spektrum Respons Pseudo-Percepatan (PSA SDOF Redaman 5%)

$$
\text{PSA}(T, \xi) = \omega^2 S_d(T, \xi) = \omega^2 \max_{t} |u(t)| \quad [g \text{ atau m/s}^2]
$$

Keterangan: $\omega = \frac{2\pi}{T}$ adalah frekuensi sudut alami osilator SDOF, $S_d(T, \xi)$ adalah spektrum simpangan relatif maksimum, dan $\xi = 0.05$ (rasio redaman kritis 5% standar rekayasa gempa).

### 7. Hubungan Intensitas Instrumental Skala MMI (Worden et al., 2012)

Dihitung secara baku dari **Komponen Horizontal Maksimum (Max-H)**:

**Formulasi Berdasarkan PGA (Gal):**

$$
\text{MMI}_{\text{PGA}} = \begin{cases}
1.78 + 1.55 \log_{10}(\text{PGA}), & \log_{10}(\text{PGA}) \le 1.57 \\
-1.60 + 3.70 \log_{10}(\text{PGA}), & \log_{10}(\text{PGA}) > 1.57
\end{cases}
$$

**Formulasi Berdasarkan PGV (cm/s):**

$$
\text{MMI}_{\text{PGV}} = \begin{cases}
3.78 + 2.99 \log_{10}(\text{PGV}), & \log_{10}(\text{PGV}) \le 0.53 \\
2.40 + 4.96 \log_{10}(\text{PGV}), & \log_{10}(\text{PGV}) > 0.53
\end{cases}
$$

Pada intensitas guncangan kuat ($I_{\text{MMI}} \ge 5.0$), perumusan PGV mendominasi penentuan intensitas instrumental sesuai pedoman konvensi USGS ShakeMap.

---

## Panduan Instalasi & Menjalankan Aplikasi Secara Lokal

### Prasyarat Sistem
* **Sistem Operasi**: Windows 10/11, macOS, atau Linux (Ubuntu 20.04+)
* **Lingkungan Python**: Versi **3.10** hingga **3.13** (lingkungan pengujian primer: Python 3.13.2)
* **Pustaka Dependensi**: Streamlit, ObsPy, NumPy, SciPy, Pandas, Matplotlib, Plotly, FPDF, PyMuPDF
* **RAM**: Minimal 4 GB (disarankan 8 GB untuk batch processing)

### Langkah Instalasi

1. **Kloning Repositori:**
   ```bash
   git clone https://github.com/ahmaddidan/BSMA-v.2.git
   cd BSMA-v.2
   ```

2. **Buat dan Aktifkan Lingkungan Virtual (*Virtual Environment*):**
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

3. **Instal Seluruh Dependensi:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi BSMA Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan terbuka otomatis di peramban pada alamat `http://localhost:8501`.

5. **(Opsional) Menjalankan Test Suite Otomatis:**
   ```bash
   pytest tests/
   ```

---

## Validasi Ilmiah & Suite Pengujian Otomatis

BSMA dilengkapi dengan test suite otomatis berbasis `pytest` (83 butir pengujian unit, regresi referensi, dan validasi rekaman riil) yang mencakup:
* **Pengujian Sintetik Analitik ([tests/test_integration.py](tests/test_integration.py))**: Verifikasi integrasi trapesium kumulatif terhadap solusi analitik eksak gelombang sinusoidal $a(t) = A\sin(\omega t)$ dengan toleransi galat relatif $< 0.1\%$.
* **Benchmark Regresi Dataset Referensi ([tests/test_reference_benchmark.py](tests/test_reference_benchmark.py))**: Uji coba *end-to-end* sinyal gempa sintetik kanonikal untuk verifikasi konvergensi deterministik nilai puncak kinematika (PGA/PGV/PGD), energi Arias ($I_a$), durasi $D_{5-95}$, MMI Worden et al. (2012), dan spektrum respons PSA.
* **Uji Keselarasan Solver SDOF ([tests/test_response_spectrum.py](tests/test_response_spectrum.py))**: Verifikasi komparasi antara formulasi analitik rekursif Nigam-Jennings (1969) dan integrasi implisit Newmark-Beta (1959) dengan rata-rata deviasi relatif $< 5\%$, serta pembuktian batas kaku (*rigid limit anchor*) $\lim_{T \to 0} \text{PSA}(T) = \text{PGA}$.
* **Uji Intensitas MMI & ShakeMap ([tests/test_mmi.py](tests/test_mmi.py))**: Validasi percabangan formulasi Worden et al. (2012), sifat monotonitas kurva intensitas, dominansi PGV pada guncangan tinggi, dan penanganan nilai non-finit.
* **Uji Ambang Batas QC ([tests/test_qc.py](tests/test_qc.py))**: Pengujian ketepatan keputusan status kelayakan (PASS $\ge 70$, WARNING $50-69$, FAIL $< 50$), serta diskualifikasi otomatis pada sinyal yang mengalami *sensor clipping* atau *ADC saturation*.
* **Uji Validasi Rekaman Riil BMKG ([tests/test_level4_real_data.py](tests/test_level4_real_data.py))**: Pengujian *end-to-end* pada rekaman multi-stasiun gempa nyata BMKG (PPJR, PCJI, PRJI) untuk memverifikasi keselarasan penuh ekstraksi parameter akselerogram (PGA 100% konsisten terhadap standar operasional BMKG).

---

## Struktur Direktori Repositori

```text
Project BSMA/
├── .streamlit/                      # Konfigurasi tema visual (light/dark) & server Streamlit
├── core/                            # Modul logika sains murni (Python standard + NumPy/SciPy)
│   ├── interfaces/                  # Kontrak antarmuka abstrak preprocessor & solver
│   ├── io/                          # Modul parser waveform (MiniSEED, SAC) & StationXML
│   ├── preprocessing/               # Detrending, cosine tapering, Butterworth filtering, QC
│   ├── processing/                  # Integrasi numerik, parameter kinematika (PGA/PGV/PGD, Arias, MMI)
│   ├── sdof/                        # Solvers SDOF elastis (Nigam-Jennings 1969 & Newmark-Beta 1959)
│   ├── types/                       # Definisi dataclass context, waveform, & processing state
│   ├── orchestrator.py              # Koordinator orkestrasi pemrosesan sinyal
│   └── pipeline.py                  # Pipeline sekuensial end-to-end
├── services/                        # Lapisan layanan aplikasi
│   ├── analysis_service.py          # Layanan pemrosesan stasiun tunggal & rekomendasi filter
│   ├── batch_service.py             # Layanan pemrosesan multi-stasiun (batch mode)
│   └── export_service.py            # Layanan ekspor CSV, matriks spektrum, & arsip ZIP
├── utils/                           # Utilitas pendukung
│   ├── exceptions.py                # Penanganan eksepsi & kode kesalahan ilmiah terstruktur
│   ├── exporter.py                  # Utilitas serialisasi data tabular
│   ├── logger.py                    # Sistem pencatatan log jejak audit (provenance tracking)
│   └── pdf_exporter.py              # Generator dokumen laporan teknis PDF komprehensif
├── scripts/                         # Skrip utilitas & otomatisasi
│   └── generate_guidebook.py        # Skrip penyusun Buku Panduan Pengguna
├── tests/                           # Pengujian unit & benchmark regresi otomatis (pytest)
│   ├── test_analysis_service.py     # Pengujian integrasi service stasiun tunggal
│   ├── test_integration.py          # Pengujian integrasi analitik sintetik (sinusoidal exact)
│   ├── test_level4_real_data.py     # Pengujian validasi rekaman riil multi-stasiun BMKG
│   ├── test_mmi.py                  # Pengujian formulasi GMICE Worden et al. (2012)
│   ├── test_parameters.py           # Pengujian ekstraksi parameter kinematika & energi
│   ├── test_qc.py                   # Pengujian sistem Quality Control 3-tingkat terharmonisasi
│   ├── test_reference_benchmark.py  # Pengujian regresi dataset referensi end-to-end
│   └── test_response_spectrum.py    # Pengujian solver SDOF & benchmark silang NJ vs Newmark
├── outputs/                         # Direktori luaran dokumen & laporan
│   └── BSMA_User_Guidebook.pdf      # Buku Panduan Pengguna & Referensi Teknis
├── app.py                           # Titik masuk utama antarmuka pengguna Streamlit (GUI)
├── assets/                          # Aset identitas visual & logo resmi instansi
│   ├── Logo_BMKG_Icon.png           # Ikon logo BMKG transparan dengan interior putih solid
│   ├── Logo_ITERA_Icon.png          # Ikon logo ITERA transparan
│   ├── Logo_ITERA.png               # Logo resmi ITERA resolusi tinggi
│   └── Logo_Judul.png               # Logo banner asli BMKG Stasiun Geofisika Sleman
├── requirements.txt                 # Daftar dependensi pustaka Python dengan batas versi
├── README.md                        # Dokumentasi komprehensif proyek (English)
└── README.id.md                     # Dokumentasi komprehensif proyek (Bahasa Indonesia)
```

---

## Daftar Pustaka & Rujukan Ilmiah

1. **Arias, A.** (1970). A measure of earthquake intensity. Dalam R. J. Hansen (Ed.), *Seismic design for nuclear power plants* (hlm. 438–483). Cambridge: MIT Press.
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

## Profil Pengembang & Lembaga Pelaksana

* **Nama Pengembang**: Ahmad Didane Setyawan Putra
* **NIM**: 123120094
* **Program Studi / Fakultas**: Teknik Geofisika, Fakultas Teknik Industri
* **Perguruan Tinggi**: Institut Teknologi Sumatera (ITERA)
* **Surel Kontak**: [ahmad.123120094@student.itera.ac.id](mailto:ahmad.123120094@student.itera.ac.id)
* **Repositori GitHub**: [https://github.com/ahmaddidan/BSMA-v.2](https://github.com/ahmaddidan/BSMA-v.2)
* **Instansi Pelaksanaan KP**: Stasiun Geofisika Kelas I Sleman, Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) D.I. Yogyakarta (Periode: 20 Juli 2026 – 20 Agustus 2026)

---

<div align="center">
  <sub>Ahmad Didane Setyawan Putra © 2026 · Karya Kerja Praktik Mahasiswa Teknik Geofisika ITERA di Stasiun Geofisika Sleman BMKG</sub>
</div>
