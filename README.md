# 🏛️ BMKG Strong Motion Analyzer (BSMA v2.0.0)
**Platform Komputasi Terpadu Sinyal Akselerograf, Kinematika Seismik, & Spektrum Respons Desain SNI 1726:2019**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Cloud_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://strong-motion.streamlit.app/)
[![ObsPy](https://img.shields.io/badge/ObsPy-Seismology_Framework-4A90E2?style=for-the-badge)](https://docs.obspy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-DSP_%26_Solvers-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Status](https://img.shields.io/badge/Status-Release_v2.0.0-059669?style=for-the-badge)](https://github.com/ahmaddidan/BSMA-v.2)

---

### 🌐 Akses Cepat & Luaran Proyek
* 🚀 **Aplikasi Cloud Web (Daring Tanpa Instalasi)**: [https://strong-motion.streamlit.app/](https://strong-motion.streamlit.app/)
* 📄 **Buku Panduan Pengguna Resmi (PDF 11 Halaman)**: [outputs/BSMA_User_Guidebook.pdf](outputs/BSMA_User_Guidebook.pdf)
* 🐙 **Repositori Resmi GitHub**: [https://github.com/ahmaddidan/BSMA-v.2](https://github.com/ahmaddidan/BSMA-v.2)

---

## 📌 Ringkasan Eksekutif & Informasi Proyek

**BMKG Strong Motion Analyzer (BSMA v2.0.0)** merupakan perangkat lunak analisis sinyal akselerograf mutakhir yang dirancang dan dikembangkan secara mandiri dalam rangka pelaksanaan kegiatan **Kerja Praktik** mahasiswa **Program Studi Teknik Geofisika, Fakultas Teknik Industri, Institut Teknologi Sumatera (ITERA)** di **Stasiun Geofisika Kelas I Sleman, Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) D.I. Yogyakarta** (periode 20 Juli – 20 Agustus 2026).

Aplikasi ini mengotomatisasi alur kerja analisis data getaran tanah kuat (*strong ground motion*) secara *end-to-end*—mulai dari pembacaan data mentah (*raw counts*), dekonvolusi respons instrumen StationXML, pemrosesan sinyal digital (DSP), ekstraksi kinematika puncak (PGA, PGV, PGD), penentuan intensitas instrumental Skala MMI berbasis GMICE Worden et al. (2012), hingga perhitungan Spektrum Respons *Pseudo-Spectral Acceleration* (PSA) elastis redaman 5% dengan opsi solver analitik **Nigam–Jennings (1969)** dan numerik implisit **Newmark (1959)** yang terhubung langsung dengan acuan standar desain ketahanan gempa **SNI 1726:2019**.

---

## 🔬 Arsitektur Pipeline Pemrosesan Sinyal 5-Tahap

BSMA menerapkan pipeline pemrosesan sekuensial yang ketat guna menjamin akurasi dan reproduksibilitas ilmiah:

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. INGESTION   │ ──> │  2. PREPROCESS  │ ──> │ 3. INTEGRATION  │ ──> │  4. KINEMATICS  │ ──> │  5. REPORTING   │
│ MiniSEED / SAC  │     │ Detrend, Taper  │     │ Acc -> Vel ->   │     │ PGA, PGV, Arias │     │ PDF Resmi, CSV, │
│ + StationXML    │     │ Butterworth 4-P │     │ Disp Mitigation │     │ SDOF PSA (5%)   │     │ ShakeMap MMI    │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Tahap 1: Ingestion & Metadata Validation**
   * Mendukung berkas triaksial **MiniSEED (`.mseed`)** standar FDSN dan **SAC (`.sac`)** standar IRIS.
   * Sinkronisasi otomatis 3-komponen ortogonal (Z, N, E / U-D, N-S, E-W) berdasarkan penanda waktu UTC absolut.
   * Koreksi respons instrumen melalui dekonvolusi fungsi transfer (*transfer function*) berbasis berkas **StationXML (`.xml`)** (analisis *poles-zeros* / PAZ & *stage gain*).
   * Deteksi otomatis: bila data masukan telah berdimensi percepatan fisik (m/s² atau Gal), sistem otomatis mengaktifkan mode *Physical Acceleration Bypass*.

2. **Tahap 2: Digital Signal Processing (DSP)**
   * **Penghilangan Tren (*Detrending*)**: Koreksi *mean offset* dan tren polinomial derajat tinggi guna menghilangkan efek instrumental geser.
   * **Jendela Kosinus Tukey 5% (*Tapering*)**: Menghaluskan diskontinuitas amplitudo di kedua ujung rekaman untuk mencegah kebocoran spektral (*spectral leakage*).
   * **Penapis Butterworth Orde-4 Dua Arah (*Zero-Phase Filtering*)**: Eksekusi penapisan maju-mundur (*forward-backward*) yang menghasilkan penapis efektif orde-8 (kemiringan *roll-off* 48 dB/oktaf) dengan distorsi pergeseran fase **nol mutlak** (*zero-phase lag*).
   * **Batas Numerik Nyquist 80%**: Menerapkan ambang atas pengaman $f_{\max} \le 0.80 f_{\mathrm{Nyquist}} = 0.40 f_s$ guna mencegah timbulnya artefak frekuensi tinggi.

3. **Tahap 3: Stepwise Numerical Integration & Baseline Drift Mitigation**
   * Integrasi numerik bertahap dari percepatan $a(t)$ menjadi kecepatan $v(t)$, dilanjutkan integrasi ke perpindahan $d(t)$.
   * Dilengkapi mitigasi residual *baseline drift* frekuensi rendah pasca-integrasi untuk menjaga kestabilan kurva perpindahan dinamik transien.

4. **Tahap 4: Parameter Kinematika Seismik & Spektrum Respons SDOF**
   * Ekstraksi parameter puncak: **PGA**, **PGV**, **PGD**, dan rasio **$V_{\max}/A_{\max}$**.
   * Perhitungan energi seismik kumulatif **Intensitas Arias ($I_a$)** dan **Durasi Signifikan ($D_{5-95}$)**.
   * Komputasi kurva **Pseudo-Spectral Acceleration (PSA)** osilator elastis SDOF dengan redaman kritis 5% ($\xi = 0.05$) pada rentang periode $T = 0.01 - 10.0$ detik.
   * Pilihan algoritma solver: formulasi rekursif analitik **Nigam & Jennings (1969)** (representasi *piecewise-linear*) atau integrasi implisit **Newmark-Beta (1959)** ($\gamma = 1/2, \beta = 1/4$).
   * Dilengkapi modul otomatis **SDOF Solver Cross-Validation & Numerical Benchmark** yang memverifikasi residual deviasi relatif antara kedua solver pada spektrum percepatan.

5. **Tahap 5: Quality Control (QC), ShakeMap MMI, & Multi-Format Reporting**
   * Sistem evaluasi kualitas rekaman terharmonisasi 3-tingkat: **QC PASS** ($\ge 70$), **QC WARNING** ($50-69$), dan **QC FAIL** ($< 50$ atau anomali fatal sensor *clipping* / saturasi ADC / *flatline*), terpisah tegas dari evaluasi skor numerik (0–100) dan kelompok diagnostik fisik sinyal (*Class 1* hingga *Class 6*).
   * Estimasi tingkat guncangan instrumental **Skala MMI** berbasis perumusan empiris GMICE **Worden et al. (2012)** USGS ShakeMap yang dihitung secara baku dari **Komponen Horizontal Maksimum** (Max-H) dengan transisi dominansi PGV pada guncangan kuat ($I_{\text{MMI}} \ge 5.0$).
   * Ekspor laporan teknis resmi **PDF komprehensif siap cetak**, lembar **CSV ringkasan parameter kinematika**, lembar **CSV matriks spektrum respons diskret**, serta paket arsip terkompresi **ZIP**.

---

## 🖥️ Eksplorasi 7 Tab Fitur Analisis Interaktif

Antarmuka BSMA v2.0.0 dibangun di atas pustaka interaktif modern Streamlit dan visualisasi saintifik Plotly, terbagi ke dalam 7 tab tematik:

| Tab | Nama Tab | Deskripsi & Kapabilitas Fungsional |
| :---: | :--- | :--- |
| **1** | **Summary** | Dasbor eksekutif menyajikan kartu metadata stasiun (koordinat lintang, bujur, elevasi), pratinjau seismogram 3-kanal (40–50% tinggi layar), metrik kinematika puncak komponen terkuat, status QC, estimasi MMI, dan panel jejak audit (*audit trail*). |
| **2** | **Waveforms** | Visualisasi interaktif riwayat waktu 3-komponen untuk kinematika lengkap: Percepatan ($a$), Kecepatan ($v$), dan Perpindahan ($d$). Dilengkapi penanda waktu tiba otomatis (*pick markers*) dan titik amplitudo puncak absolut. |
| **3** | **Quality Control** | Diagnostik integritas rekaman: skor numerik (0–100), visualisasi radar metrik derau latar, kurva *Power Spectral Density* (PSD), estimasi SNR (dB), dan bendera deteksi anomali (*clipping*, *spikes*, *flatline*). |
| **4** | **Strong Motion** | Analisis energi guncangan mendalam: kurva akumulasi Intensitas Arias (*Husid Plot*), interval Durasi Signifikan ($D_{5-95}$), rasio $V_{\max}/A_{\max}$, dan percepatan efektif (*effective peak acceleration*). |
| **5** | **Intensity** | Klasifikasi tingkat guncangan Skala MMI instrumental (Worden et al., 2012) berbasis PGA dan PGV. Menyajikan kartu deskripsi dampak fisik guncangan (*Perceived Shaking*) dan potensi kerusakan struktural (*Potential Damage*). |
| **6** | **Spectrum** | Kurva Spektrum Respons Pseudo-Percepatan ($S_a$) elastis redaman 5% untuk ketiga kanal pada rentang periode $T = 0.01 - 10.0$ s. Mendukung perbandingan langsung (*overlay*) terhadap kurva spektrum desain **SNI 1726:2019**, serta panel verifikasi silang numerik (*SDOF Solver Benchmark: Nigam-Jennings vs Newmark-Beta*). |
| **7** | **Report** | Pratinjau dan pengunduhan dokumen laporan resmi PDF terstandarisasi BMKG, tabel CSV ringkasan parameter kinematika, lembar CSV matriks spektrum respons diskret, dan paket arsip ZIP. |

---

## 🚀 Fitur Unggulan Lanjutan

* ⚡ **Batch Processing Multi-Stasiun**: Mampu memproses puluhan rekaman stasiun akselerograf secara simultan dalam satu klik, menghasilkan tabel matriks komparasi regional terpadu yang dapat diurutkan berdasarkan PGA tertinggi atau estimasi MMI.
* 🔬 **SDOF Solver Cross-Validation Benchmark**: Alat uji presisi komputasi interaktif yang menghitung deviasi relatif maksimal, rata-rata, dan RMS antara Nigam-Jennings dan Newmark-Beta.
* 🌓 **Dual-Theme Switcher (Light Mode & Dark Mode)**: Tombol saklar instan di pojok kanan atas aplikasi untuk beralih antara tema terang (kebutuhan pelaporan formal dan pencahayaan terang) dan tema gelap (pengamatan jangka panjang di ruang monitor seismologi redup).
* 🛡️ **Provenance Logging & Audit Trail**: Setiap tahapan pemrosesan dicatat secara transparan (parameter filter, versi pustaka numerik, waktu eksekusi, dan status StationXML) pada laporan akhir guna menjamin reproduksibilitas ilmiah (*scientific reproducibility*).

---

## 📐 Landasan Formulasi Matematis

| Parameter | Persamaan Matematis | Satuan | Acuan Standar |
| :--- | :--- | :---: | :--- |
| **Batas Nyquist** | $f_{\max} \le 0.80 \times f_{\mathrm{Nyquist}} = 0.40 \times f_s$ | $\text{Hz}$ | Nyquist (1928), Shannon (1949) |
| **Signal-to-Noise** | $\text{SNR} = 20 \log_{10}\left( \frac{\mathrm{RMS_{signal}}}{\mathrm{RMS_{noise}}} \right)$ | $\text{dB}$ | Konvensi FDSN / PEER |
| **PGA** | $\text{PGA} = \max \vert a(t) \vert$ | $\text{Gal}$ | Seismologi Rekayasa Baku |
| **PGV** | $\text{PGV} = \max \vert v(t) \vert = \max \left\vert \int_0^t a(\tau) d\tau \right\vert$ | $\text{cm/s}$ | Kalkulus Integral Tentu Sinyal |
| **PGD** | $\text{PGD} = \max \vert d(t) \vert = \max \left\vert \int_0^t v(\tau) d\tau \right\vert$ | $\text{cm}$ | Integrasi Ganda Transien (Deformasi Dinamik) |
| **Intensitas Arias** | $I_a = \frac{\pi}{2g} \int_0^{t_{\max}} [a(t)]^2 dt$ | $\text{m/s}$ | Arias (1970) |
| **Durasi Signifikan** | $D_{5-95} = t_{95} - t_{5}$ | $\text{detik}$ | Trifunac & Brady (1975) (akumulasi $I_a$ 5% – 95%) |
| **Spektrum Respons** | $\text{PSA}(T, \xi) = \omega^2 S_d(T, \xi) = \omega^2 \max \vert u(t) \vert$ | $\text{g}$ | Nigam-Jennings (1969), Newmark (1959), SNI 1726:2019 |
| **GMICE MMI** | $\text{MMI}_{\text{PGA}} = 1.78 + 1.55 \log_{10}(\text{PGA}) \quad (\log_{10}\text{PGA} \le 1.57)$<br>$\text{MMI}_{\text{PGA}} = -1.60 + 3.70 \log_{10}(\text{PGA}) \quad (\log_{10}\text{PGA} > 1.57)$<br>$\text{MMI}_{\text{PGV}} = 3.78 + 2.89 \log_{10}(\text{PGV}) \quad (\log_{10}\text{PGV} \le 0.53)$<br>$\text{MMI}_{\text{PGV}} = 2.40 + 4.00 \log_{10}(\text{PGV}) \quad (\log_{10}\text{PGV} > 0.53)$ | Skala I–IX+ | Worden et al. (2012), USGS ShakeMap (Komponen Horizontal Maksimum) |

---

## 💻 Panduan Instalasi & Menjalankan Aplikasi Secara Lokal

### Prasyarat Sistem
* **Sistem Operasi**: Windows 10/11, macOS, atau Linux (Ubuntu 20.04+)
* **Python**: Versi **3.10** atau lebih baru
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
   Aplikasi akan otomatis terbuka di peramban pada alamat `http://localhost:8501`.

5. **(Opsional) Mengompilasi Ulang Buku Panduan PDF Resmi (11 Halaman):**
   ```bash
   python scripts/generate_guidebook.py
   ```
   Berkas PDF akan diperbarui di `outputs/BSMA_User_Guidebook.pdf`.

6. **(Opsional) Menjalankan Test Suite Otomatis:**
   ```bash
   pytest tests/
   ```

---

## 📂 Struktur Direktori Repositori

```text
Project BSMA/
├── .streamlit/               # Konfigurasi tema visual (light/dark) & server Streamlit
├── core/                     # Modul logika sains murni (Python standard + NumPy/SciPy)
│   ├── interfaces/           # Kontrak antarmuka abstrak
│   ├── io/                   # Modul parser waveform (MiniSEED, SAC) & StationXML
│   ├── preprocessing/        # Modul detrending, cosine tapering, & penapisan Butterworth
│   ├── processing/           # Integrasi numerik, ekstraksi kinematika (PGA/PGV/PGD, Arias)
│   ├── sdof/                 # Solvers SDOF elastis (Nigam-Jennings analitik & Newmark numerik)
│   ├── types/                # Definisi dataclass parameter & hasil analisis
│   ├── orchestrator.py       # Koordinator orkestrasi pemrosesan
│   └── pipeline.py           # Pipeline sekuensial end-to-end
├── services/                 # Lapisan layanan pemrosesan aplikasi
│   ├── analysis_service.py   # Layanan pemrosesan data stasiun tunggal
│   ├── batch_service.py      # Layanan pemrosesan multi-stasiun (batch mode)
│   └── export_service.py     # Layanan ekspor laporan CSV, matriks spektrum, & arsip ZIP
├── utils/                    # Utilitas pendukung
│   ├── exceptions.py         # Penanganan eksepsi kustom
│   ├── exporter.py           # Utilitas serialisasi data tabular
│   ├── logger.py             # Sistem pencatatan log (provenance tracking)
│   └── pdf_exporter.py       # Mesin pembuat laporan teknis resmi PDF
├── scripts/                  # Skrip utilitas & otomatisasi
│   └── generate_guidebook.py # Skrip penyusun Buku Panduan Pengguna resmi (11 halaman)
├── tests/                    # Pengujian unit otomatis (pytest)
│   ├── test_analysis_service.py
│   └── test_parameters.py
├── outputs/                  # Direktori luaran dokumen & laporan
│   └── BSMA_User_Guidebook.pdf # Buku Panduan Pengguna & Referensi Teknis Resmi
├── app.py                    # Titik masuk utama antarmuka pengguna Streamlit (GUI)
├── requirements.txt          # Daftar dependensi pustaka Python
├── Logo_Judul.png            # Aset logo resmi BMKG & universitas
└── README.md                 # Dokumentasi komprehensif proyek
```

---

## 📚 Daftar Pustaka & Rujukan Ilmiah

Format sitasi disusun berdasarkan kaidah **APA Style 7th Edition**:

1. **Arias, A.** (1970). A measure of earthquake intensity. Dalam R. J. Hansen (Ed.), *Seismic design for nuclear power plants* (hlm. 438–483). MIT Press.
2. **Badan Standardisasi Nasional.** (2019). *SNI 1726:2019: Tata cara perencanaan ketahanan gempa untuk struktur bangunan gedung dan non gedung*. BSN.
3. **Beyreuther, M., Barsch, R., Krischer, L., Megies, T., Behr, Y., & Wassermann, J.** (2010). ObsPy: A Python toolbox for seismology. *Seismological Research Letters*, 81(3), 530–533. [https://doi.org/10.1785/gssrl.81.3.530](https://doi.org/10.1785/gssrl.81.3.530)
4. **Newmark, N. M.** (1959). A method of computation for structural dynamics. *Journal of the Engineering Mechanics Division, ASCE*, 85(3), 67–94. [https://doi.org/10.1061/JMCEA3.0000098](https://doi.org/10.1061/JMCEA3.0000098)
5. **Nigam, N. C., & Jennings, P. C.** (1969). Calculation of response spectra from strong-motion earthquake records. *Bulletin of the Seismological Society of America*, 59(2), 909–922. [https://doi.org/10.1785/BSSA0590020909](https://doi.org/10.1785/BSSA0590020909)
6. **Trifunac, M. D., & Brady, A. G.** (1975). A study on the duration of strong earthquake ground motion. *Bulletin of the Seismological Society of America*, 65(3), 581–626. [https://doi.org/10.1785/BSSA0650030581](https://doi.org/10.1785/BSSA0650030581)
7. **Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., & van der Walt, S. J.** (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272. [https://doi.org/10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)
8. **Worden, C. B., Gerstenberger, M. C., Rhoades, D. A., & Wald, D. J.** (2012). Probabilistic relationships between ground-motion parameters and MMI. *Bulletin of the Seismological Society of America*, 102(1), 204–221. [https://doi.org/10.1785/0120110156](https://doi.org/10.1785/0120110156)

---

## 👨‍💻 Profil Pengembang & Lembaga Pelaksana

* **Nama Pengembang**: Ahmad Didane Setyawan Putra
* **NIM**: 123120094
* **Program Studi / Fakultas**: Teknik Geofisika, Fakultas Teknik Industri
* **Perguruan Tinggi**: Institut Teknologi Sumatera (ITERA)
* **Surel Resmi**: [ahmad.123120094@student.itera.ac.id](mailto:ahmad.123120094@student.itera.ac.id)
* **Repositori GitHub**: [https://github.com/ahmaddidan/BSMA-v.2](https://github.com/ahmaddidan/BSMA-v.2)
* **Instansi Pelaksanaan KP**: Stasiun Geofisika Kelas I Sleman, Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) D.I. Yogyakarta (Periode: 20 Juli 2026 – 20 Agustus 2026)

---

<div align="center">
  <sub>Didan Putra © 2026 · Dikembangkan dengan standar integritas keilmuan Geofisika Komputasional & Rekayasa Kegempaan BMKG–ITERA</sub>
</div>
