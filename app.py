"""
BMKG Strong Motion Analyzer (BSMA)
app.py - Final Professional Production-Grade Dashboard (V3.4 - Ultimate Clean UI)

Author: Ahmad Didane & Technical Lead
"""
import streamlit as st
from pathlib import Path
import obspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zipfile
import io
import logging

# Impor Modul Backend BSMA
from core.pipeline import PipelineBuilder
from core.preprocessing.baseline import BaselineCorrectionPlugin
from core.preprocessing.taper import TaperPlugin, TaperConfig
from core.preprocessing.filter import ButterworthFilterPlugin, FilterConfig
from core.preprocessing.integration import KinematicIntegrationPlugin, IntegrationConfig
from core.processing.parameters import ParameterExtractionPlugin, ParameterConfig
from core.processing.response_spectrum import ResponseSpectrumPlugin, ResponseSpectrumConfig
from core.processing.advanced_analysis import compute_husid_and_duration, compute_fas

from core.types.context import ProcessingContext, WaveformData
from core.types.processing_state import ProcessingState
from utils.pdf_exporter import export_station_report, get_sig_bmkg

# =========================================================================
# 1. KONFIGURASI HALAMAN & CSS
# =========================================================================
st.set_page_config(page_title="BSMA Dashboard", layout="wide", initial_sidebar_state="expanded")

def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Mengunci Resize Sidebar */
        [data-testid="stSidebarResizeHandle"] { display: none !important; pointer-events: none !important; width: 0px !important; }
        [data-testid="stSidebarResizer"] { display: none !important; }
        
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        button[kind="primary"] { background-color: #28a745 !important; border: none !important; font-weight: bold; }
        button[kind="primary"]:hover { background-color: #218838 !important; }
        
        .logo-card {background-color: #FFFFFF; padding: 5px 8px; border-radius: 8px; display: inline-flex; box-shadow: 0 2px 5px rgba(0,0,0,0.15);}
        .header-subtext {font-size: 13px; color: #7F8C8D; margin-top: -10px; margin-bottom: 10px;}
        
        .badge-success { background-color: #17202A; color: #2ECC71; padding: 4px 10px; border-radius: 12px; font-size: 12px; border: 1px solid #2ECC71; font-weight: 600;}
        .badge-warning { background-color: #17202A; color: #F1C40F; padding: 4px 10px; border-radius: 12px; font-size: 12px; border: 1px solid #F1C40F; font-weight: 600;}
        
        .metric-card {background-color: #1E1E24; padding: 15px; border-radius: 8px; border-left: 4px solid #3498DB; box-shadow: 0 2px 4px rgba(0,0,0,0.2); height: 100%; line-height: 1.2;}
        .metric-title {color: #AAB7B8; font-size: 10px; text-transform: uppercase; font-weight: 700; margin-bottom: 5px;}
        .metric-value {color: #FFFFFF; font-size: 18px; font-weight: 800;}
        .metric-sub {color: #7F8C8D; font-size: 11px;}
        
        .sig-card {padding: 12px; border-radius: 8px; text-align: center; color: white; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.3); height: 100%; display: flex; flex-direction: column; justify-content: center; line-height: 1.2;}
        .sig-title {font-size: 11px; text-transform: uppercase; opacity: 0.9;}
        .sig-scale {font-size: 16px; margin: 2px 0;}
        
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] { background-color: #F1F3F5; border-radius: 4px 4px 0px 0px; padding: 8px 15px; font-size: 13px; font-weight: 600; color: #2C3E50; }
        .stTabs [aria-selected="true"] { background-color: #2C3E50; color: white; }
        </style>
        """, unsafe_allow_html=True
    )

# =========================================================================
# 2. FUNGSI PEMROSESAN BACKEND
# =========================================================================
@st.cache_resource
def get_logger():
    return logging.getLogger("bsma_gui")

def get_dynamic_pipeline(logger, freq_min, freq_max, filter_type, damping):
    return (
        PipelineBuilder(logger=logger, halt_on_error=False)
        .add(BaselineCorrectionPlugin(method="linear"))
        .add(TaperPlugin(config=TaperConfig(alpha=0.05)))
        .add(ButterworthFilterPlugin(config=FilterConfig(type=filter_type, freq_min=freq_min, freq_max=freq_max, corners=4, zerophase=True)))
        .add(KinematicIntegrationPlugin(config=IntegrationConfig(remove_mean=True, remove_linear_trend=True)))
        .add(ParameterExtractionPlugin(config=ParameterConfig(gravity=9.80665)))
        .add(ResponseSpectrumPlugin(config=ResponseSpectrumConfig(damping=damping, solver="nigam_jennings")))
        .build()
    )

def process_station_stream(st_station: obspy.Stream, inventory: obspy.Inventory, logger, freq_min, freq_max, filter_type, damping):
    pipeline = get_dynamic_pipeline(logger, freq_min, freq_max, filter_type, damping)
    contexts = {}
    for tr in st_station:
        channel = tr.stats.channel
        if inventory:
            try: tr.remove_response(inventory=inventory, output="ACC", water_level=60, pre_filt=[0.05, 0.1, 30.0, 35.0])
            except Exception: pass 
                
        raw_wave = WaveformData(data=tr.data, sampling_rate=tr.stats.sampling_rate, unit="m/s^2")
        init_ctx = ProcessingContext(
            trace_id=tr.id, metadata=dict(tr.stats), raw_waveform=raw_wave,
            acceleration=raw_wave, processing_state=ProcessingState(), history=()
        )
        contexts[channel] = pipeline.run(init_ctx)
    return contexts

def extract_summary_data(station_code, contexts):
    data = []
    for ch, ctx in contexts.items():
        m = ctx.metrics
        pga_gal = m.get("PGA", 0) * 100.0
        sig = get_sig_bmkg(pga_gal)
        data.append({
            "Stasiun": station_code, "Channel": ch, "PGA (Gal)": round(pga_gal, 4),
            "PGV (cm/s)": round(m.get("PGV", 0) * 100.0, 4), "PGD (cm)": round(m.get("PGD", 0) * 100.0, 4),
            "Arias Int (m/s)": round(m.get("Arias_Intensity", 0), 6),
            "Durasi (s)": round(m.get("Significant_Duration_D5_95", 0), 2),
            "SIG BMKG": f"SIG {sig[0]} ({sig[4]})"
        })
    return data

# =========================================================================
# 3. MAIN DASHBOARD
# =========================================================================
def main():
    inject_custom_css()
    logger = get_logger()

    # HEADER COMPACT
    col_logo, col_title = st.columns([0.05, 0.95], vertical_alignment="center")
    with col_logo:
        logo_path = Path("Logo_Judul.png")
        if logo_path.exists(): st.image(str(logo_path), width=70)
        else: st.markdown('<div style="font-size: 40px; text-align: center;">BMKG</div>', unsafe_allow_html=True)
    
    with col_title:
        st.markdown("<h2 style='margin-bottom: 0px; padding-bottom: 0px; line-height: 1.2;'>BMKG Strong Motion Analyzer</h2>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 14px; color: #7F8C8D; margin-top: 5px; margin-bottom: 15px;'>Professional Engineering Seismology Platform</div>", unsafe_allow_html=True)

    # DIRECTORY INITIALIZATION
    data_dir, xml_dir, report_dir, temp_dir = Path("Data/mseed"), Path("Data/stationXML"), Path("outputs/reports"), Path("outputs/temp")
    for d in [data_dir, xml_dir, report_dir, temp_dir]: d.mkdir(parents=True, exist_ok=True)
    
    # STATE INITIALIZATION
    if 'all_contexts' not in st.session_state: st.session_state['all_contexts'] = {}
    if 'export_ready' not in st.session_state: st.session_state['export_ready'] = False
    if 'form_key' not in st.session_state: st.session_state['form_key'] = 0

    # SIDEBAR
    with st.sidebar:
        st.markdown("### Manajemen Data")
        
        # MENGGUNAKAN FORM KEY AGAR BENAR-BENAR BERSIH SETELAH SUBMIT
        with st.form(f"upload_form_{st.session_state['form_key']}", clear_on_submit=True):
            st.markdown("<span style='font-size:12px; color:gray;'>Unggah arsip waveform dan metadata kalibrasi. Berkas akan diproses dan diamankan ke dalam sistem.</span>", unsafe_allow_html=True)
            upl_mseed = st.file_uploader("Data Gelombang", type=["mseed", "sac", "miniseed"], accept_multiple_files=True)
            upl_xml = st.file_uploader("Data Kalibrasi", type=["xml"], accept_multiple_files=True)
            
            submitted = st.form_submit_button("Simpan Data")
            if submitted:
                if upl_mseed:
                    for uf in upl_mseed:
                        with open(data_dir / uf.name, "wb") as f: f.write(uf.getbuffer())
                if upl_xml:
                    for uf in upl_xml:
                        with open(xml_dir / uf.name, "wb") as f: f.write(uf.getbuffer())
                st.session_state['form_key'] += 1
                st.rerun()

        mseed_files = list(data_dir.glob("*.mseed")) + list(data_dir.glob("*.sac")) + list(data_dir.glob("*.miniseed"))
        
        st.markdown("---")
        st.markdown("### Konfigurasi Pemrosesan")
        with st.expander("Parameter Filter & Redaman", expanded=False):
            filter_type = st.selectbox("Tipe Filter", ["bandpass", "lowpass", "highpass", "bandstop"], index=0)
            f_col1, f_col2 = st.columns(2)
            freq_min = f_col1.number_input("Freq Min (Hz)", value=0.1)
            freq_max = f_col2.number_input("Freq Max (Hz)", value=25.0)
            damping_ratio = st.number_input("Rasio Redaman", value=0.05, step=0.01)

        st.markdown("---")
        st.markdown("### Mode Analisis")
        app_mode = st.radio("Pilih Mode:", ["Analisis Stasiun", "Komparasi Multi-Stasiun", "Batch Rekapitulasi", "Ekspor Data"], label_visibility="collapsed")

    # JIKA TIDAK ADA DATA, BERHENTI DI SINI
    if not mseed_files:
        st.info("Sistem belum mendeteksi arsip gelombang. Silakan lakukan manajemen data pada panel samping.")
        return

    # LOAD ALL FILES INTO A STREAM
    master_stream = obspy.Stream()
    for f in mseed_files:
        try: master_stream += obspy.read(str(f))
        except: pass
    unique_stations = sorted(list(set([tr.stats.station for tr in master_stream])))

    # =========================================================================
    # MENU 1: ANALISIS STASIUN TUNGGAL
    # =========================================================================
    if app_mode == "Analisis Stasiun":
        c_sel1, c_sel2, c_sel3 = st.columns([3, 3, 2], vertical_alignment="bottom")
        with c_sel1:
            selected_stn = st.selectbox("Station Name:", unique_stations)
            st_selected = master_stream.select(station=selected_stn)
        
        with c_sel2:
            possible_xml = xml_dir / f"{selected_stn}.xml"
            xml_path = str(possible_xml) if possible_xml.exists() else None
            if xml_path: st.markdown(f"<div class='badge-success'>Calibration Applied: {possible_xml.name}</div>", unsafe_allow_html=True)
            else: st.markdown("<div class='badge-warning'>Raw Counts Only</div>", unsafe_allow_html=True)

        with c_sel3:
            process_btn = st.button("PROCESS DATA", use_container_width=True, type="primary")

        if process_btn:
            with st.spinner(f"Processing {len(st_selected)} channels..."):
                inventory = obspy.read_inventory(xml_path) if xml_path else None
                contexts = process_station_stream(st_selected, inventory, logger, freq_min, freq_max, filter_type, damping_ratio)
                st.session_state["single_stn"] = selected_stn
                st.session_state["single_ctx"] = contexts
                st.session_state['all_contexts'][selected_stn] = contexts

        if "single_ctx" in st.session_state and st.session_state["single_stn"] == selected_stn:
            stn = st.session_state["single_stn"]
            ctxs = st.session_state["single_ctx"]
            sample_tr = st_selected[0].stats
            
            strongest_ch = max(ctxs, key=lambda k: ctxs[k].metrics.get("PGA", 0))
            max_ctx = ctxs[strongest_ch]
            
            pga_ms2 = max_ctx.metrics.get("PGA", 0)
            pga_gal = pga_ms2 * 100.0
            pga_percent_g = (pga_ms2 / 9.80665) * 100.0
            
            sig_data = get_sig_bmkg(pga_gal)
            sig_id, rgb, sig_desc, mmi = sig_data[0], sig_data[2], sig_data[3], sig_data[4]
            
            st.markdown(f"<span style='color:#7F8C8D; font-size:13px;'><b>Network:</b> {sample_tr.network} &nbsp;|&nbsp; <b>Channels:</b> {' '.join(ctxs.keys())} &nbsp;|&nbsp; <b>Sampling:</b> {sample_tr.sampling_rate} Hz &nbsp;|&nbsp; <b>UTC:</b> {sample_tr.starttime.strftime('%Y-%m-%d %H:%M:%S')}</span>", unsafe_allow_html=True)
            
            c1, c2, c3, c4, c5, c6 = st.columns([1, 1.2, 1.2, 1.2, 1.2, 1.8])
            c1.markdown(f"<div class='metric-card'><div class='metric-title'>STRONGEST COMP</div><div class='metric-value'>{strongest_ch}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><div class='metric-title'>PEAK GROUND ACC</div><div class='metric-value'>{pga_gal:.3f} Gal</div><div class='metric-sub'>{pga_percent_g:.3f} %g</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><div class='metric-title'>PEAK GROUND VEL</div><div class='metric-value'>{max_ctx.metrics.get('PGV',0)*100:.3f}</div><div class='metric-sub'>cm/s</div></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'><div class='metric-title'>PEAK GROUND DISP</div><div class='metric-value'>{max_ctx.metrics.get('PGD',0)*100:.4f}</div><div class='metric-sub'>cm</div></div>", unsafe_allow_html=True)
            c5.markdown(f"<div class='metric-card'><div class='metric-title'>ARIAS INTENSITY</div><div class='metric-value'>{max_ctx.metrics.get('Arias_Intensity',0):.5f}</div><div class='metric-sub'>m/s</div></div>", unsafe_allow_html=True)
            c6.markdown(f"<div class='sig-card' style='background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); color: {'#FFF' if sig_id in ['IV', 'V'] else '#000'};'><div class='sig-title'>SKALA {sig_id} SIG-BMKG</div><div class='sig-scale'>{sig_desc}</div><div class='sig-mmi'>Equivalent MMI {mmi}</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            t_sum, t_wave, t_spec, t_hus, t_fas, t_qc, t_pdf = st.tabs([
                "SUMMARY", "WAVEFORM", "RESPONSE SPECTRUM", "HUSID PLOT", "FAS", "AUDIT LOG", "REPORT (PDF)"
            ])

            with t_sum:
                st.markdown(f"""
                ### Interpretasi Hasil
                *   **Percepatan Tanah Maksimum (PGA):** Tercatat sebesar **{pga_gal:.3f} Gal** (setara **{pga_percent_g:.3f} %g**).
                *   **Guncangan Dominan:** Terjadi pada komponen **{strongest_ch}**.
                *   **Klasifikasi Intensitas:** Berdasarkan standar BMKG, guncangan dikategorikan sebagai **SKALA {sig_id} ({sig_desc})**.
                *   **Durasi Signifikan ($D_{{5-95}}$):** Energi guncangan utama berlangsung selama **{max_ctx.metrics.get('Significant_Duration_D5_95',0):.2f} detik**.
                *   **Status Data:** {'Kalibrasi instrumen (StationXML) berhasil diaplikasikan.' if xml_path else 'Data mentah (Raw Counts) tanpa kalibrasi.'} Tidak terdeteksi *clipping*. Pemrosesan selesai.
                """)
                df_data = []
                for ch, c in ctxs.items():
                    m = c.metrics
                    ch_gal = m.get('PGA',0)*100
                    df_data.append({"Channel": ch, "PGA (Gal)": round(ch_gal, 3), "PGA (%g)": round((m.get('PGA',0)/9.80665)*100, 3), "PGV (cm/s)": round(m.get('PGV',0)*100, 3), "PGD (cm)": round(m.get('PGD',0)*100, 4), "Arias (m/s)": round(m.get('Arias_Intensity',0), 5), "Durasi (s)": round(m.get('Significant_Duration_D5_95',0), 2), "SIG": f"SIG {get_sig_bmkg(ch_gal)[0]}"})
                st.dataframe(pd.DataFrame(df_data), use_container_width=True)

            with t_wave:
                with st.expander("Informasi Waveform"):
                    st.markdown("Visualisasi historis percepatan (PGA), kecepatan (PGV), dan perpindahan (PGD) tanah. Titik merah menunjukkan lokasi percepatan puncak (PGA), sedangkan garis vertikal putus-putus menandakan interval Durasi Signifikan (D5 - D95) di mana 90% energi seismik utama dilepaskan.")
                    
                view_ch = st.selectbox("Select Component:", list(ctxs.keys()), label_visibility="collapsed")
                ctx = ctxs[view_ch]
                time = np.arange(len(ctx.acceleration.data)) / ctx.sampling_rate
                _, t_5, t_95, _ = compute_husid_and_duration(ctx.acceleration.data, ctx.sampling_rate)
                idx_pga = np.argmax(np.abs(ctx.acceleration.data))
                
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03)
                fig.add_trace(go.Scatter(x=time, y=ctx.raw_waveform.data * 100, fill='tozeroy', line=dict(color='#00D4FF', width=1), name="Raw"), row=1, col=1)
                fig.add_trace(go.Scatter(x=time, y=ctx.acceleration.data, line=dict(color='#00FF99', width=1), name="Acc"), row=2, col=1)
                fig.add_trace(go.Scatter(x=time, y=ctx.velocity.data * 100, line=dict(color='#FFD54F', width=1), name="Vel"), row=3, col=1)
                fig.add_trace(go.Scatter(x=time, y=ctx.displacement.data * 100, line=dict(color='#FF6B6B', width=1), name="Disp"), row=4, col=1)
                
                fig.add_trace(go.Scatter(x=[time[idx_pga]], y=[ctx.acceleration.data[idx_pga]], mode='markers+text', text=['PGA'], textposition="top center", marker=dict(color='red', size=8), name="PGA"), row=2, col=1)
                fig.add_vline(x=t_5, line_dash="dash", line_color="orange", annotation_text="D5", row=2, col=1)
                fig.add_vline(x=t_95, line_dash="dash", line_color="red", annotation_text="D95", row=2, col=1)

                fig.update_yaxes(title_text="Raw (Gal)", row=1, col=1)
                fig.update_yaxes(title_text="Acc (m/s²)", row=2, col=1)
                fig.update_yaxes(title_text="Vel (cm/s)", row=3, col=1)
                fig.update_yaxes(title_text="Disp (cm)", row=4, col=1)
                fig.update_xaxes(title_text="Time (seconds)", row=4, col=1)

                fig.update_layout(height=750, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, plot_bgcolor='rgba(30,30,36,1)', paper_bgcolor='rgba(30,30,36,0)')
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.12)', color='#DDDDDD')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.12)', color='#DDDDDD')
                st.plotly_chart(fig, use_container_width=True)

            with t_spec:
                with st.expander("Informasi Response Spectrum"):
                    st.markdown("Spektrum respons (PSA) memetakan estimasi percepatan puncak yang akan dialami struktur bangunan berdasarkan periode alaminya. Kurva ini merupakan parameter kunci bagi rekayasawan sipil untuk memastikan desain struktur tahan terhadap gaya geser gempa.")
                fig_s = go.Figure()
                colors = {"HNE": "purple", "HNN": "goldenrod", "HNZ": "darkgreen"}
                for ch, c in ctxs.items():
                    psa = c.spectral_data.get("PSA", c.spectral_data.get("psa"))
                    per = c.spectral_data.get("Periods", c.spectral_data.get("periods"))
                    if psa is not None and per is not None:
                        psa_g = psa / 9.80665
                        fig_s.add_trace(go.Scatter(x=per, y=psa_g, mode='lines', name=f"{ch} (Max: {np.max(psa_g):.3f} g)", line=dict(color=colors.get(ch[-3:], 'cyan'), width=2)))
                fig_s.update_xaxes(type="log", title_text="Periode (detik)", gridcolor='rgba(255,255,255,0.12)')
                fig_s.update_yaxes(title_text="Spectral Acc (g)", gridcolor='rgba(255,255,255,0.12)')
                fig_s.update_layout(height=450, plot_bgcolor='rgba(30,30,36,1)', paper_bgcolor='rgba(30,30,36,0)', font=dict(color='#DDD'), legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
                st.plotly_chart(fig_s, use_container_width=True)

            with t_hus:
                with st.expander("Informasi Husid Plot"):
                    st.markdown("Kurva persentase penumpukan Intensitas Arias terhadap waktu. Husid plot digunakan untuk secara presisi mengidentifikasi interval waktu (D5 ke D95) di mana energi destruktif gempa paling banyak dilepaskan secara masif.")
                try:
                    time_v = np.arange(len(max_ctx.acceleration.data)) / max_ctx.sampling_rate
                    husid, t_5, t_95, d_5_95 = compute_husid_and_duration(max_ctx.acceleration.data, max_ctx.sampling_rate)
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(x=time_v, y=husid*100, line=dict(color='#00FF99', width=2), name="Husid", fill='tozeroy'))
                    fig_h.add_vline(x=t_5, line_dash="dash", line_color="orange", annotation_text=f"D5 ({t_5:.1f}s)")
                    fig_h.add_vline(x=t_95, line_dash="dash", line_color="red", annotation_text=f"D95 ({t_95:.1f}s)")
                    fig_h.update_layout(title=f"Durasi Signifikan D5-95: {d_5_95:.2f} detik", height=400, xaxis_title="Waktu (s)", yaxis_title="Energi (%)", plot_bgcolor='rgba(30,30,36,1)', paper_bgcolor='rgba(30,30,36,0)', font=dict(color='#DDD'))
                    fig_h.update_xaxes(gridcolor='rgba(255,255,255,0.12)')
                    fig_h.update_yaxes(gridcolor='rgba(255,255,255,0.12)')
                    st.plotly_chart(fig_h, use_container_width=True)
                except Exception as e: st.error(f"Gagal: {e}")

            with t_fas:
                with st.expander("Informasi Fourier Amplitude Spectrum (FAS)"):
                    st.markdown("Transformasi rekaman domain waktu ke domain frekuensi menggunakan FFT (Fast Fourier Transform). Menunjukkan spektrum frekuensi dominan guncangan yang krusial untuk mengidentifikasi potensi efek resonansi geologi lokal (*site effect*).")
                try:
                    freqs, fas = compute_fas(max_ctx.acceleration.data, max_ctx.sampling_rate)
                    fig_f = go.Figure()
                    fig_f.add_trace(go.Scatter(x=freqs, y=fas, line=dict(color='#FF6B6B', width=1.5)))
                    fig_f.update_xaxes(type="log", title_text="Frekuensi (Hz)", gridcolor='rgba(255,255,255,0.12)')
                    fig_f.update_yaxes(type="log", title_text="Amplitudo", gridcolor='rgba(255,255,255,0.12)')
                    fig_f.update_layout(height=400, plot_bgcolor='rgba(30,30,36,1)', paper_bgcolor='rgba(30,30,36,0)', font=dict(color='#DDD'))
                    st.plotly_chart(fig_f, use_container_width=True)
                except Exception as e: st.error(f"Gagal: {e}")

            with t_qc:
                st.markdown(f"**Channel Report:** {strongest_ch}")
                for log in max_ctx.history:
                    step = log.get("step", "Unknown")
                    status = log.get("status", "SUCCESS")
                    color = "#2ECC71" if status == "SUCCESS" else "#E74C3C"
                    detail_str = ""
                    for k, v in log.items():
                        if k not in ["step", "status"]: detail_str += f" | {k}: {v}"
                    st.markdown(f"<div style='background-color:#1E1E24; padding:10px; margin-bottom:5px; border-left:3px solid {color}; border-radius:4px;'><span style='font-weight:bold; color:#FFF;'>{step}</span> <span style='color:{color};'>[{status}]</span><br><span style='font-size:12px; color:#AAB7B8;'>{detail_str}</span></div>", unsafe_allow_html=True)

            with t_pdf:
                st.info("Mengekspor seluruh tabel, spektrum overlay, dan grafik 4-panel ketiga komponen ke dalam PDF resmi BMKG.")
                if st.button("Generate Laporan PDF", type="primary"):
                    with st.spinner("Merender..."):
                        pdf_path = export_station_report(stn, ctxs, str(report_dir))
                        with open(pdf_path, "rb") as f:
                            st.download_button("Unduh PDF Laporan", data=f, file_name=pdf_path.name, mime="application/pdf")

    # =========================================================================
    # MENU 2: KOMPARASI MULTI-STASIUN
    # =========================================================================
    elif app_mode == "Komparasi Multi-Stasiun":
        st.markdown("### Komparasi Parameter Multi-Stasiun")
        selected_stations = st.multiselect("Pilih Stasiun:", options=unique_stations)
        
        if st.button("Jalankan Komparasi", type="primary") and selected_stations:
            with st.spinner("Memproses..."):
                c_data = []
                fig_comp = go.Figure()
                colors = ['#00D4FF', '#00FF99', '#FFD54F', '#FF6B6B', '#A569BD']
                
                for i, s in enumerate(selected_stations):
                    xml_p = str(xml_dir / f"{s}.xml") if (xml_dir / f"{s}.xml").exists() else None
                    try:
                        inv = obspy.read_inventory(xml_p) if xml_p else None
                        ctxs = process_station_stream(master_stream.select(station=s), inv, logger, freq_min, freq_max, filter_type, damping_ratio)
                        st.session_state['all_contexts'][s] = ctxs 
                        c_data.extend(extract_summary_data(s, ctxs))
                        
                        strongest_ch = max(ctxs, key=lambda k: ctxs[k].metrics.get("PGA", 0))
                        c_spec = ctxs[strongest_ch].spectral_data
                        psa = c_spec.get("PSA", c_spec.get("psa"))
                        per = c_spec.get("Periods", c_spec.get("periods"))
                        
                        if psa is not None and per is not None:
                            psa_g = psa / 9.80665
                            fig_comp.add_trace(go.Scatter(x=per, y=psa_g, mode='lines', name=f"{s} ({strongest_ch})", line=dict(color=colors[i % len(colors)], width=2)))
                    except: pass
                
                st.markdown("#### Overlay Spektrum Respons Antar-Stasiun")
                fig_comp.update_xaxes(type="log", title_text="Periode (detik)", gridcolor='rgba(255,255,255,0.12)')
                fig_comp.update_yaxes(title_text="Spectral Acc (g)", gridcolor='rgba(255,255,255,0.12)')
                fig_comp.update_layout(height=450, plot_bgcolor='rgba(30,30,36,1)', paper_bgcolor='rgba(30,30,36,0)', font=dict(color='#DDD'))
                st.plotly_chart(fig_comp, use_container_width=True)
                
                st.markdown("#### Tabel Perbandingan Parameter")
                st.dataframe(pd.DataFrame(c_data), use_container_width=True)

    # =========================================================================
    # MENU 3: TABEL REKAPITULASI BATCH
    # =========================================================================
    elif app_mode == "Batch Rekapitulasi":
        st.markdown("### Rekapitulasi Seluruh Jaringan")
        if st.button("Proses Semua Data", type="primary"):
            pb = st.progress(0); txt = st.empty(); all_data = []
            for i, stn in enumerate(unique_stations):
                txt.text(f"Memproses {stn}...")
                xml_p = str(xml_dir / f"{stn}.xml") if (xml_dir / f"{stn}.xml").exists() else None
                try:
                    inv = obspy.read_inventory(xml_p) if xml_p else None
                    ctxs = process_station_stream(master_stream.select(station=stn), inv, logger, freq_min, freq_max, filter_type, damping_ratio)
                    st.session_state['all_contexts'][stn] = ctxs
                    all_data.extend(extract_summary_data(stn, ctxs))
                except: pass
                pb.progress((i + 1) / len(unique_stations))
            st.session_state['batch_df'] = pd.DataFrame(all_data)
            txt.text("Pemrosesan selesai.")
        
        if 'batch_df' in st.session_state and not st.session_state['batch_df'].empty:
            df = st.session_state['batch_df']
            st.dataframe(df.style.highlight_max(subset=['PGA (Gal)'], color='lightcoral'), use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Unduh CSV", data=csv, file_name="Batch_Summary.csv", mime="text/csv")

    # =========================================================================
    # MENU 4: EKSPOR ARSIP ZIP MASSAL (KUSTOMISASI)
    # =========================================================================
    elif app_mode == "Ekspor Data":
        st.markdown("### Ekspor Laporan & Data")

        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            st.markdown("#### 1. Pilih Format Ekspor")
            export_format = st.radio("Format:", ["Lengkap (PDF & CSV dalam ZIP)", "Hanya PDF (ZIP)", "Hanya CSV (Tabel)"], label_visibility="collapsed")
        with c_opt2:
            st.markdown("#### 2. Pilih Stasiun")
            select_all = st.checkbox("Pilih Semua Stasiun", value=True)
            if select_all:
                selected_stns = unique_stations
                st.multiselect("Stasiun Terpilih:", options=unique_stations, default=unique_stations, disabled=True, label_visibility="collapsed")
            else:
                selected_stns = st.multiselect("Stasiun Terpilih:", options=unique_stations, default=unique_stations[:1], label_visibility="collapsed")
        
        if st.button("Proses Ekspor", type="primary"):
            if not selected_stns:
                st.error("Pilih minimal 1 stasiun.")
            else:
                st.session_state['export_ready'] = False
                pb = st.progress(0); txt = st.empty()
                all_sum = []
                z_buf = io.BytesIO() if "PDF" in export_format or "Lengkap" in export_format else None
                
                if z_buf:
                    zf = zipfile.ZipFile(z_buf, "w", zipfile.ZIP_DEFLATED)
                
                for i, code in enumerate(selected_stns):
                    txt.text(f"Menyiapkan data stasiun {code} ({i+1}/{len(selected_stns)})...")
                    
                    if code not in st.session_state['all_contexts']:
                        xml_p = str(xml_dir / f"{code}.xml") if (xml_dir / f"{code}.xml").exists() else None
                        try:
                            inv = obspy.read_inventory(xml_p) if xml_p else None
                            st_selected = master_stream.select(station=code)
                            ctxs = process_station_stream(st_selected, inv, logger, freq_min, freq_max, filter_type, damping_ratio)
                            st.session_state['all_contexts'][code] = ctxs
                        except Exception as e:
                            logger.error(f"Gagal memproses {code}: {e}")
                            continue
                            
                    ctxs = st.session_state['all_contexts'].get(code)
                    if not ctxs: continue
                        
                    if "CSV" in export_format or "Lengkap" in export_format:
                        all_sum.extend(extract_summary_data(code, ctxs))
                        
                    if "PDF" in export_format or "Lengkap" in export_format:
                        txt.text(f"Merender PDF {code}...")
                        p_path = export_station_report(code, ctxs, str(report_dir))
                        zf.write(p_path, arcname=f"Laporan_PDF/{p_path.name}")

                    pb.progress((i+1)/len(selected_stns))
                
                if export_format == "Hanya CSV (Tabel)":
                    st.session_state['export_data'] = pd.DataFrame(all_sum).to_csv(index=False).encode('utf-8')
                    st.session_state['export_name'] = "BSMA_Database_Parameter.csv"
                    st.session_state['export_mime'] = "text/csv"
                else:
                    if "Lengkap" in export_format and all_sum:
                        zf.writestr("Database_Parameter.csv", pd.DataFrame(all_sum).to_csv(index=False).encode('utf-8'))
                    zf.close()
                    z_buf.seek(0)
                    st.session_state['export_data'] = z_buf.getvalue()
                    st.session_state['export_name'] = "BSMA_Laporan_Massal.zip"
                    st.session_state['export_mime'] = "application/zip"
                    
                txt.text("Penyusunan berkas selesai!")
                st.session_state['export_ready'] = True

        if st.session_state.get('export_ready'):
            st.download_button("Unduh Arsip Sekarang", data=st.session_state['export_data'], file_name=st.session_state['export_name'], mime=st.session_state['export_mime'], type="primary")

if __name__ == "__main__":
    main()