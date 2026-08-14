# 🌍 Strong-Motion Analytical Software
**Advanced Earthquake Ground Motion Processing & Spectral Analysis**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?style=flat-square&logo=streamlit)
![Geophysics](https://img.shields.io/badge/Domain-Seismology_%26_Earthquake_Engineering-00599C?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

> **Pernyataan Arsitektur Teknis:**  
> Perangkat lunak ini merupakan *blueprint* fungsional untuk pemrosesan *ground motion* tingkat lanjut. Dibangun dengan kaidah Seismologi dan Geofisika komputasional, sistem ini dirancang menggunakan algoritma deterministik untuk mengekstraksi parameter gerak tanah secara mutlak dan presisi: **PGA, PGV, dan PGD**. Sistem juga dilengkapi mesin komputasi untuk evaluasi spektral tingkat lanjut termasuk *Response Spectrum*, *Arias Intensity*, *Fourier Amplitude Spectrum* (FAS), serta *Husid Plot*.

---

## ✨ Fitur Utama (Core Features)

### 📊 1. Manajemen Data & Auto-Sinkronisasi MSEED
*   **3-Component Auto-Merge:** Mampu menelan data mentah `Mini-SEED` (MSEED) standar IRIS/FDSN secara terpisah (Vertikal, North, East) dan mengintegrasikannya ke dalam satu entitas stasiun tersinkronisasi berdasarkan waktu absolut UTC. Mengeliminasi total risiko *phase-shift*.
*   **Deteksi Cerdas:** Otomatis mendeteksi apakah sinyal masih berupa *Raw Counts* digitizer atau telah terkalibrasi menjadi unit percepatan.
*   **Fleksibilitas Satuan:** Konversi absolut ke berbagai satuan teknik: `Gal (cm/s²)`, `m/s²`, atau fraksi gravitasi bumi `g`.

### 🎛️ 2. Digital Signal Processing & Quality Control
*   **Zero-Phase Butterworth Filter:** Modul pemfilteran dinamis (*forward-backward*) untuk menekan *noise* seismik frekuensi rendah tanpa mendistorsi posisi waktu puncak gelombang (*peak arrival time*).
*   **Automated QC Audit:** Menugaskan "Skor QC" secara algoritmik (Grade A hingga D) dengan mengevaluasi *Signal-to-Noise Ratio* (SNR), *clipping* instrumen, dan *baseline drift* pasca-integrasi.

### 📈 3. Komponen Analitik Spektral & Visualisasi
Aplikasi menyediakan *dashboard* interaktif resolusi tinggi untuk parameter *ground motion*:
*   **Waveform 3 Komponen:** Representasi domain waktu untuk Percepatan, Kecepatan, dan Perpindahan secara terpadu.
*   **FAS (Fourier Amplitude Spectrum):** Transformasi FFT untuk identifikasi dominansi frekuensi energi gempa.
*   **Response Spectrum (PSA/PSV/SD):** Analisis respons dinamik struktur SDOF menggunakan integrasi *Newmark-Beta* (redaman 5%).
*   **Husid Plot & Arias Intensity:** Kalkulasi integral energi seismik kumulatif untuk menentukan *Significant Duration* ($D_{5-95}$).

### 🗺️ 4. Geospasial, SIG BMKG & Benchmark
*   **Klasifikasi Otomatis SIG BMKG:** Otomatis memetakan nilai PGA ke dalam Skala Intensitas Gempa (SIG) BMKG dan ekuivalensi *Modified Mercalli Intensity* (MMI) dengan indikator warna spasial.
*   **Metadata Hiposenter:** Komputasi presisi Jarak Episentral ($R_{epi}$) dan Jarak Hiposentral ($R_{hypo}$) via metode *Haversine*.
*   **Benchmark Overlay:** Validasi atenuasi lokal dengan membandingkan spektrum observasi terhadap kurva model empiris *Ground Motion Prediction Equations* (GMPE).

### 🚀 5. Batch Export System (Automasi Ekspor)
Sistem memiliki *Data Pipeline* terintegrasi untuk mengekspor analisis stasiun tunggal maupun multi-stasiun secara instan:
1.  **Automated PDF Report:** Merangkai seluruh *metadata*, skor QC, parameter *ground motion*, dan grafik *lossless* ke dalam satu laporan teknis siap cetak.
2.  **CSV Time-Series:** Ekspor *database* tabular untuk rekayasa *Machine Learning* atau pemodelan spasial GIS.
3.  **JSON Serialization:** Ekspor metadata spasial dan vektor spektral menjadi objek terstruktur API untuk integrasi *Web Service*.

---

## 💻 Instalasi (Installation)

Pastikan Anda telah menginstal **Python 3.8+** di sistem Anda. Ikuti langkah berikut untuk menjalankan aplikasi secara lokal:

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/username/strong-motion-app.git
   cd strong-motion-app
   ```

2. **Buat Virtual Environment (Opsional namun direkomendasikan):**
   ```bash
   # Membuat virtual environment
   python -m venv venv

   # Aktivasi untuk Linux/Mac
   source venv/bin/activate  

   # Aktivasi untuk Windows
   venv\Scripts\activate     
   ```

3. **Instal dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```

---

## 🧭 Panduan Singkat (Quick Start)

1. Pilih mode ruang kerja (*Workspace*): **Single Station Review** atau **Multi-Station Analysis**.
2. Unggah data gempabumi Anda (mendukung `MSEED`, `AT2`, `TXT`, atau `CSV`).
3. Sesuaikan parameter *Bandpass Filter* ($f_{low}$ dan $f_{high}$) untuk membersihkan sinyal dari *noise*.
4. Masukkan *metadata* gempa (Magnitudo, Kedalaman, Koordinat) untuk menghitung jarak episentral.
5. Lakukan analisis grafik (FAS, *Response Spectrum*, *Husid Plot*) secara interaktif.
6. Klik tombol **Export** untuk mengunduh laporan PDF, CSV, atau JSON.

---

## 👨‍💻 Pengembang (Author)

Proyek ini didesain dan dikembangkan secara independen sebagai tugas magang pembuatan platform pemrosesan Geofisika terapan oleh:

* **Nama:** Ahmad Didane Setyawan Putra
* **NIM:** 123120094
* **Program Studi:** Teknik Geofisika
* **Instansi:** Institut Teknologi Sumatera (ITERA)

*Dedicated to advancing computational seismology and earthquake engineering analysis.*
