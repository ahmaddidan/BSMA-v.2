"""
BMKG Strong Motion Analyzer (BSMA)
app.py - Final Professional Production-Grade Dashboard (Optimized UI & Logo Styling)

Author: Ahmad Didane
"""
import streamlit as st
from pathlib import Path
import obspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import io

from core.types.context import ProcessingContext, TraceMetadata
from core.pipeline import PipelineOrchestrator
from core.preprocessing.filter import ButterworthFilterPlugin, FilterConfig
from core.preprocessing.baseline import BaselineRemovalPlugin, BaselineConfig
from core.processing.kinematics import KinematicIntegrationPlugin
from core.processing.parameters import ParameterExtractionPlugin, ParameterConfig
from core.processing.spectra import NigamJenningsSpectraPlugin, SpectraConfig
from core.processing.advanced_analysis import compute_husid_and_duration, compute_fas
from visualization.plots import BSMAVisualizer
from core.io.pdf_exporter import generate_station_pdf

st.set_page_config(page_title="BMKG Strong Motion Analyzer", page_icon="📈", layout="wide")

def estimate_sig_bmkg(pga_gal: float) -> str:
    if pga_gal < 2.9:
        return "I - TIDAK DIRASAKAN (< 2.9 gal)"
    elif pga_gal < 89.0:
        return "II - DIRASAKAN (2.9 - 88 gal)"
    elif pga_gal < 167.0:
        return "III - KERUSAKAN RINGAN (89 - 167 gal)"
    elif pga_gal < 564.0:
        return "IV - KERUSAKAN SEDANG (168 - 564 gal)"
    else:
        return "V - KERUSAKAN BERAT (> 564 gal)"

def get_dynamic_pipeline(freq_min, freq_max, freq_type, damping):
    return PipelineOrchestrator(plugins=[
        ButterworthFilterPlugin(config=FilterConfig(type=freq_type, freq_min=freq_min, freq_max=freq_max, corners=4, zerophase=True)),
        BaselineRemovalPlugin(config=BaselineConfig(method="linear")),
        KinematicIntegrationPlugin(),
        ParameterExtractionPlugin(config=ParameterConfig(gravity_constant=981.0)),
        NigamJenningsSpectraPlugin(config=SpectraConfig(damping=damping))
    ])

@st.cache_data
def process_single_station(file_path_str: str, freq_min, freq_max, freq_type, damping):
    st_data = obspy.read(file_path_str)
    tr = st_data[0]
    data_raw = tr.data.astype(np.float64) * 100.0  # Konversi ke Gal

    metadata = TraceMetadata(
        network=tr.stats.network, station=tr.stats.station,
        location=tr.stats.location, channel=tr.stats.channel,
        sampling_rate=tr.stats.sampling_rate, starttime=tr.stats.starttime.timestamp
    )

    pipeline = get_dynamic_pipeline(freq_min, freq_max, freq_type, damping)
    context = ProcessingContext(data=data_raw, metadata=metadata)
    final_context = pipeline.run(context)

    time_vec = np.arange(len(final_context.data)) / metadata.sampling_rate
    husid, t_5, t_95, d_5_95 = compute_husid_and_duration(final_context.data, metadata.sampling_rate)
    freqs, fas = compute_fas(final_context.data, metadata.sampling_rate)

    return final_context, metadata, time_vec, husid, t_5, t_95, d_5_95, freqs, fas

def main():
    # CSS Kustom untuk merapikan logo container (latar putih bersih agar logo selalu terlihat jelas di mode apa pun)
    # Serta menyeragamkan tinggi kotak metrik agar sejajar sempurna
    st.markdown(
        """
        <style>
        button[title="View fullscreen"] {
            visibility: hidden;
        }
        .logo-card {
            background-color: #FFFFFF;
            padding: 8px 10px;
            border-radius: 10px;
            display: inline-flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col_logo, col_title = st.columns([0.07, 0.93], vertical_alignment="center")
    with col_logo:
        logo_path = Path("bmkg_logo.png")
        if logo_path.exists():
            st.markdown('<div class="logo-card">', unsafe_allow_html=True)
            st.image(str(logo_path), width=45)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("🏛️") 
    with col_title:
        st.title("BMKG Strong Motion Analyzer (BSMA)")

    data_dir = Path("Data/mseed")
    data_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path("output/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # SIDEBAR PROFESIONAL & BERURUTAN
    # =========================================================================
    st.sidebar.header("1. Manajemen Data")
    uploaded_files = st.sidebar.file_uploader("Unggah file .mseed / .sac", type=["mseed", "sac"], accept_multiple_files=True)

    if uploaded_files:
        for uf in uploaded_files:
            sp = data_dir / uf.name
            if not sp.exists():
                with open(sp, "wb") as f:
                    f.write(uf.getbuffer())

    mseed_files = sorted(list(data_dir.glob("*.mseed")) + list(data_dir.glob("*.sac")))
    if not mseed_files:
        st.warning("Belum ada file data. Silakan unggah file melalui panel samping.")
        return

    st.sidebar.info(f"Total Stasiun Aktif: **{len(mseed_files)} file**")

    st.sidebar.markdown("---")
    st.sidebar.header("2. Objek Analisis")
    file_options = {f.name: str(f) for f in mseed_files}
    selected_filename = st.sidebar.selectbox("Pilih File Stasiun", list(file_options.keys()))
    selected_path = file_options[selected_filename]

    st.sidebar.markdown("---")
    freq_min = 0.1
    freq_max = 25.0
    filter_type = "bandpass"
    damping_ratio = 0.05

    with st.sidebar.expander("Konfigurasi Filter"):
        filter_type = st.selectbox("Tipe Filter", ["bandpass", "lowpass", "highpass", "bandstop"], index=0)
        if filter_type in ["bandpass", "highpass", "bandstop"]:
            freq_min = st.slider("Frekuensi Min (Hz)", 0.01, 2.0, 0.1, 0.01)
        if filter_type in ["bandpass", "highpass", "bandstop"]:
            freq_max = st.slider("Frekuensi Max (Hz)", 5.0, 50.0, 25.0, 1.0)
        damping_ratio = st.slider("Rasio Redaman", 0.01, 0.20, 0.05, 0.01)

    st.sidebar.markdown("---")
    st.sidebar.header("3. Navigasi Utama")
    app_mode = st.sidebar.radio(
        "Pilih Mode Tampilan",
        [
            "Analisis Stasiun Tunggal", 
            "Komparasi Multi-Stasiun", 
            "Tabel Rekapitulasi Batch", 
            "Ekspor Arsip ZIP Massal"
        ]
    )

    # ==========================================
    # MENU 1: ANALISIS STASIUN TUNGGAL
    # ==========================================
    if app_mode == "Analisis Stasiun Tunggal":
        with st.spinner("Memproses stasiun aktif..."):
            final_context, metadata, time_vec, husid, t_5, t_95, d_5_95, freqs, fas = process_single_station(
                selected_path, freq_min, freq_max, filter_type, damping_ratio
            )

        st.header(f"Hasil Analisis: Stasiun {metadata.station} ({metadata.channel})")

        st.markdown(
            f"""
            <div style="background-color: #1E1E24; padding: 12px 18px; border-radius: 8px; border: 1px solid #3A3B45; margin-bottom: 20px;">
                <span style="color: #979A9A; font-size: 14px;">📌 <b>Metadata Stasiun:</b> Network: <b>{metadata.network}</b> &nbsp;|&nbsp; Station: <b>{metadata.station}</b> &nbsp;|&nbsp; Channel: <b>{metadata.channel}</b> &nbsp;|&nbsp; Sampling Rate: <b>{metadata.sampling_rate} Hz</b> &nbsp;|&nbsp; Durasi (D<sub>5-95</sub>): <b>{d_5_95:.2f} detik</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )

        if final_context.parameters is not None:
            p = final_context.parameters
            sig_status = estimate_sig_bmkg(p.pga)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Peak Acceleration (PGA)", f"{p.pga:.2f} cm/s²")
            c2.metric("Peak Velocity (PGV)", f"{p.pgv:.4f} cm/s")
            c3.metric("Peak Displacement (PGD)", f"{p.pgd:.4f} cm")
            c4.metric("Arias Intensity", f"{p.arias_intensity:.4f} m/s")
            
            c5.markdown(
                f"""
                <div style="background-color: #262730; padding: 10px; border-radius: 6px; border: 1px solid #41424C; height: 82px; display: flex; flex-direction: column; justify-content: center;">
                    <span style="color: #979A9A; font-size: 11px; font-weight: 600; margin-bottom: 4px; text-transform: uppercase;">Klasifikasi SIG BMKG</span>
                    <span style="color: #FFFFFF; font-size: 13px; font-weight: bold; line-height: 1.2;">{sig_status}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Kinematik", "Spektrum Respons", "Husid Plot", "FAS", "Ekspor PDF"
        ])

        with tab1:
            st.subheader("Grafik Percepatan, Kecepatan, & Perpindahan")
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axs[0].plot(time_vec, final_context.data, color='black', lw=0.8); axs[0].set_ylabel("Percepatan\n(cm/s²)"); axs[0].grid(True, ls='--')
            if final_context.velocity is not None:
                axs[1].plot(time_vec, final_context.velocity, color='blue', lw=0.8); axs[1].set_ylabel("Kecepatan\n(cm/s)"); axs[1].grid(True, ls='--')
            if final_context.displacement is not None:
                axs[2].plot(time_vec, final_context.displacement, color='red', lw=0.8); axs[2].set_ylabel("Perpindahan\n(cm)"); axs[2].set_xlabel("Waktu (detik)"); axs[2].grid(True, ls='--')
            plt.tight_layout(); st.pyplot(fig)

        with tab2:
            st.subheader(f"Kurva Pseudo-Spectral Acceleration ({int(damping_ratio*100)}% Damping)")
            if final_context.spectra is not None:
                fig_s, ax_s = plt.subplots(figsize=(10, 5))
                ax_s.plot(final_context.spectra.periods, final_context.spectra.psa, color='purple', lw=1.8)
                ax_s.set_xscale('log'); ax_s.set_xlabel("Periode (detik)"); ax_s.set_ylabel("PSA (cm/s²)"); ax_s.grid(True, which="both", ls='--')
                plt.tight_layout(); st.pyplot(fig_s)

        with tab3:
            st.subheader(f"Husid Plot & Durasi Signifikan (D5-95: {d_5_95:.2f}s)")
            fig_h, ax_h = plt.subplots(figsize=(10, 5))
            ax_h.plot(time_vec, husid * 100, color='darkgreen', lw=1.8, label="Husid Curve")
            ax_h.axvline(t_5, color='orange', ls='--', label=f"5% Energy ({t_5:.2f}s)")
            ax_h.axvline(t_95, color='red', ls='--', label=f"95% Energy ({t_95:.2f}s)")
            ax_h.set_xlabel("Waktu (detik)"); ax_h.set_ylabel("Energi Kumulatif (%)"); ax_h.grid(True, ls='--'); ax_h.legend()
            plt.tight_layout(); st.pyplot(fig_h)

        with tab4:
            st.subheader("Fourier Amplitude Spectra (FAS)")
            fig_f, ax_f = plt.subplots(figsize=(10, 5))
            ax_f.plot(freqs, fas, color='crimson', lw=1.2)
            ax_f.set_xscale('log'); ax_f.set_yscale('log'); ax_f.set_xlabel("Frekuensi (Hz)"); ax_f.set_ylabel("Amplitudo (cm/s)"); ax_f.grid(True, which="both", ls='--')
            plt.tight_layout(); st.pyplot(fig_f)

        with tab5:
            st.subheader("Cetak Laporan Resmi (PDF)")
            if st.button("Generate Laporan PDF Stasiun Ini"):
                ts_img = temp_dir / "ts.png"
                spec_img = temp_dir / "spec.png"
                hus_img = temp_dir / "hus.png"
                
                BSMAVisualizer.plot_time_series(final_context, save_path=str(ts_img))
                if final_context.spectra:
                    BSMAVisualizer.plot_response_spectra(final_context, save_path=str(spec_img))
                BSMAVisualizer.plot_husid(time_vec, husid, t_5, t_95, d_5_95, save_path=str(hus_img))
                
                plots_dict = {
                    "Analisis Kinematik": str(ts_img),
                    "Spektrum Respons (PSA)": str(spec_img),
                    "Husid Plot & Durasi Signifikan": str(hus_img)
                }
                
                pdf_output_path = temp_dir / f"Laporan_BSMA_{metadata.station}_{metadata.channel}.pdf"
                generate_station_pdf(metadata, final_context.parameters, sig_status, plots_dict, str(pdf_output_path))
                
                with open(pdf_output_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Unduh Dokumen PDF",
                        data=pdf_file,
                        file_name=f"Laporan_BSMA_{metadata.station}_{metadata.channel}.pdf",
                        mime="application/pdf"
                    )
                st.success("Laporan PDF berhasil dibuat!")

    # ==========================================
    # MENU 2: KOMPARASI MULTI-STASIUN
    # ==========================================
    elif app_mode == "Komparasi Multi-Stasiun":
        st.header("Komparasi & Pemilihan Multi-Stasiun")
        st.markdown("Pilih stasiun-stasiun tertentu di bawah ini untuk membandingkan kurva spektrum respons dan parameter ringkasannya secara bersamaan.")
        
        select_all = st.checkbox("Pilih / Centang Semua Stasiun", value=False)
        default_selection = list(file_options.keys()) if select_all else list(file_options.keys())[:min(3, len(file_options))]

        selected_stations = st.multiselect(
            "Pilih Stasiun untuk Dibandingkan",
            options=list(file_options.keys()),
            default=default_selection
        )

        if selected_stations:
            if st.button("Jalankan Komparasi Stasiun Terpilih"):
                with st.spinner("Memproses stasiun terpilih..."):
                    multi_contexts = []
                    multi_summaries = []
                    for stn_name in selected_stations:
                        path_str = file_options[stn_name]
                        try:
                            fc, meta, _, _, _, _, _, _, _ = process_single_station(path_str, freq_min, freq_max, filter_type, damping_ratio)
                            multi_contexts.append(fc)
                            if fc.parameters is not None:
                                p = fc.parameters
                                multi_summaries.append({
                                    "Station": meta.station,
                                    "Channel": meta.channel,
                                    "PGA (cm/s²)": round(p.pga, 4),
                                    "PGV (cm/s)": round(p.pgv, 4),
                                    "PGD (cm)": round(p.pgd, 4),
                                    "Arias Intensity (m/s)": round(p.arias_intensity, 4),
                                    "SIG BMKG": estimate_sig_bmkg(p.pga)
                                })
                        except Exception:
                            continue
                    st.session_state["multi_contexts"] = multi_contexts
                    
                    df_multi = pd.DataFrame(multi_summaries)
                    if not df_multi.empty:
                        df_multi.index = range(1, len(df_multi) + 1)
                    st.session_state["multi_summaries"] = df_multi

            if "multi_contexts" in st.session_state and st.session_state["multi_contexts"]:
                st.subheader("Grafik Gabungan Spektrum Respons (PSA)")
                fig_mc, ax_mc = plt.subplots(figsize=(12, 6))
                for ctx in st.session_state["multi_contexts"]:
                    if ctx.spectra is not None:
                        ax_mc.plot(ctx.spectra.periods, ctx.spectra.psa, lw=1.8, label=f"{ctx.metadata.station} ({ctx.metadata.channel})")
                ax_mc.set_xscale('log')
                ax_mc.set_xlabel("Periode (detik)")
                ax_mc.set_ylabel("Pseudo-Spectral Acceleration (cm/s²)")
                ax_mc.grid(True, which="both", ls='--')
                ax_mc.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_mc)

                st.subheader("Tabel Ringkasan Stasiun Terpilih")
                st.dataframe(st.session_state["multi_summaries"], use_container_width=True)
        else:
            st.info("Silakan pilih minimal satu stasiun pada kotak di atas.")

    # ==========================================
    # MENU 3: TABEL REKAPITULASI BATCH
    # ==========================================
    elif app_mode == "Tabel Rekapitulasi Batch":
        st.header("Tabel Rekapitulasi Seluruh Stasiun")
        st.markdown("Daftar lengkap parameter rekayasa gempa untuk seluruh stasiun yang diunggah.")
        
        with st.spinner("Memproses seluruh stasiun..."):
            summaries = []
            for fp in mseed_files:
                try:
                    fc, meta, _, _, _, _, _, _, _ = process_single_station(str(fp), freq_min, freq_max, filter_type, damping_ratio)
                    if fc.parameters is not None:
                        p = fc.parameters
                        summaries.append({
                            "Network": meta.network,
                            "Station": meta.station,
                            "Channel": meta.channel,
                            "PGA (cm/s²)": round(p.pga, 4),
                            "PGV (cm/s)": round(p.pgv, 4),
                            "PGD (cm)": round(p.pgd, 4),
                            "Arias Intensity (m/s)": round(p.arias_intensity, 4),
                            "SIG BMKG": estimate_sig_bmkg(p.pga)
                        })
                except Exception:
                    continue
            summary_df = pd.DataFrame(summaries)
            if not summary_df.empty:
                summary_df.index = range(1, len(summary_df) + 1)

        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True)
            csv_data = summary_df.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="📥 Unduh Tabel sebagai CSV",
                data=csv_data,
                file_name="summary_parameters.csv",
                mime="text/csv"
            )
        else:
            st.info("Belum ada data rekapitulasi.")

    # ==========================================
    # MENU 4: EKSPOR ARSIP ZIP MASSAL
    # ==========================================
    elif app_mode == "Ekspor Arsip ZIP Massal":
        st.header("Ekspor Seluruh Hasil Analisis (.zip)")
        st.markdown("Kemas seluruh plot grafik stasiun dan file rekapitulasi CSV ke dalam satu arsip terpadu.")
        
        if st.button("Buat Arsip ZIP Sekarang"):
            with st.spinner("Menyiapkan file ZIP..."):
                zip_buffer = io.BytesIO()
                summaries = []
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for fp in mseed_files:
                        try:
                            ctx, meta, _, _, _, _, _, _, _ = process_single_station(str(fp), freq_min, freq_max, filter_type, damping_ratio)
                            stn = meta.station
                            chan = meta.channel
                            
                            ts_p = temp_dir / f"{stn}_{chan}_ts.png"
                            spec_p = temp_dir / f"{stn}_{chan}_spec.png"
                            
                            BSMAVisualizer.plot_time_series(ctx, save_path=str(ts_p))
                            if ctx.spectra:
                                BSMAVisualizer.plot_response_spectra(ctx, save_path=str(spec_p))
                                
                            if ts_p.exists():
                                zip_file.write(ts_p, arcname=f"plots/{stn}_{chan}_timeseries.png")
                            if spec_p.exists():
                                zip_file.write(spec_p, arcname=f"plots/{stn}_{chan}_spectra.png")
                                
                            if ctx.parameters is not None:
                                summaries.append({
                                    "Network": meta.network, 
                                    "Station": meta.station, 
                                    "Channel": meta.channel,
                                    "PGA (cm/s²)": round(p.pga, 4),
                                    "PGD (cm)": round(p.pgd, 4),
                                    "SIG BMKG": estimate_sig_bmkg(p.pga)
                                })
                        except Exception:
                            continue
                            
                    if summaries:
                        df_zip = pd.DataFrame(summaries)
                        csv_bytes = df_zip.to_csv(index=False).encode('utf-8')
                        zip_file.writestr("summary_parameters.csv", csv_bytes)
                        
                zip_buffer.seek(0)
                st.download_button(
                    label="📥 Unduh ZIP Berisi Seluruh Hasil",
                    data=zip_buffer,
                    file_name="BSMA_Batch_Results.zip",
                    mime="application/zip"
                )
                st.success("Arsip ZIP berhasil disiapkan!")

if __name__ == "__main__":
    main()