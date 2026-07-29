"""
BMKG Strong Motion Analyzer (BSMA)
visualization/plots.py
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from core.types.context import ProcessingContext

class BSMAVisualizer:
    
    @staticmethod
    def _get_time_vector(context: ProcessingContext) -> np.ndarray:
        n_samples = len(context.data)
        fs = context.metadata.sampling_rate
        return np.arange(n_samples) / fs

    @staticmethod
    def plot_time_series(context: ProcessingContext, save_path: str = None):
        time = BSMAVisualizer._get_time_vector(context)

        fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        meta = context.metadata
        fig.suptitle(f"Analisis Kinematik: Stasiun {meta.station} ({meta.channel})", fontsize=14, fontweight='bold')

        # Subplot Percepatan
        axs[0].plot(time, context.data, color='black', linewidth=0.8)
        axs[0].set_ylabel("Percepatan\n(cm/s²)")
        axs[0].grid(True, linestyle='--', alpha=0.6)

        # Subplot Kecepatan
        if context.velocity is not None:
            axs[1].plot(time, context.velocity, color='blue', linewidth=0.8)
            axs[1].set_ylabel("Kecepatan\n(cm/s)")
            axs[1].grid(True, linestyle='--', alpha=0.6)

        # Subplot Perpindahan
        if context.displacement is not None:
            axs[2].plot(time, context.displacement, color='red', linewidth=0.8)
            axs[2].set_ylabel("Perpindahan\n(cm)")
            axs[2].set_xlabel("Waktu (detik)")
            axs[2].grid(True, linestyle='--', alpha=0.6)

        if context.parameters is not None:
            p = context.parameters
            info_text = (
                f"--- RINGKASAN PARAMETER ---\n"
                f"PGA : {p.pga:.2f} cm/s²\n"
                f"PGV : {p.pgv:.4f} cm/s\n"
                f"PGD : {p.pgd:.4f} cm\n"
                f"Arias Intensity : {p.arias_intensity:.4f} m/s"
            )
            axs[0].text(0.98, 0.92, info_text, transform=axs[0].transAxes,
                        fontsize=9, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3, edgecolor='orange'))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_response_spectra(context: ProcessingContext, save_path: str = None):
        if context.spectra is None:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        meta = context.metadata
        
        # Ambil nilai damping dengan aman menggunakan getattr (default 0.05 / 5%)
        damping_val = getattr(context.spectra, 'damping', 0.05)
        
        ax.plot(context.spectra.periods, context.spectra.psa, color='purple', linewidth=1.8, label=f"Damping {damping_val*100:.0f}%")
        ax.set_xscale('log')
        ax.set_title(f"Spektrum Respons (PSA): Stasiun {meta.station} ({meta.channel})", fontsize=13)
        ax.set_xlabel("Periode (detik)")
        ax.set_ylabel("Pseudo-Spectral Acceleration (cm/s²)")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_combined_spectra(contexts: List[ProcessingContext], save_path: str = None):
        fig, ax = plt.subplots(figsize=(12, 7))
        for ctx in contexts:
            if ctx.spectra is not None:
                label = f"{ctx.metadata.station} ({ctx.metadata.channel})"
                ax.plot(ctx.spectra.periods, ctx.spectra.psa, linewidth=1.5, label=label)
        ax.set_xscale('log')
        ax.set_title("Perbandingan Spektrum Respons (PSA) Seluruh Stasiun", fontsize=14, fontweight='bold')
        ax.set_xlabel("Periode (detik)")
        ax.set_ylabel("Pseudo-Spectral Acceleration (cm/s²)")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_multi_component_comparison(contexts: List[ProcessingContext], station_name: str, save_path: str = None):
        fig, axs = plt.subplots(len(contexts), 1, figsize=(12, 3 * len(contexts)), sharex=True)
        if len(contexts) == 1:
            axs = [axs]
        fig.suptitle(f"Perbandingan Komponen Gelombang: Stasiun {station_name}", fontsize=14, fontweight='bold')

        for idx, ctx in enumerate(contexts):
            time = BSMAVisualizer._get_time_vector(ctx)
            axs[idx].plot(time, ctx.data, color='black', linewidth=0.7, label=f"Channel: {ctx.metadata.channel}")
            axs[idx].set_ylabel("Percepatan\n(cm/s²)")
            axs[idx].grid(True, linestyle='--', alpha=0.5)
            axs[idx].legend(loc='upper left')

            if ctx.parameters is not None:
                p = ctx.parameters
                param_str = f"PGA: {p.pga:.2f} cm/s² | PGV: {p.pgv:.4f} cm/s | PGD: {p.pgd:.4f} cm"
                axs[idx].text(0.98, 0.88, param_str, transform=axs[idx].transAxes,
                              fontsize=9, verticalalignment='top', horizontalalignment='right',
                              bbox=dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

        axs[-1].set_xlabel("Waktu (detik)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_husid(time: np.ndarray, husid: np.ndarray, t_5: float, t_95: float, d_5_95: float, save_path: str = None):
        """Membuat grafik Husid Plot & Durasi Signifikan."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time, husid * 100, color='darkgreen', linewidth=1.8, label="Husid Curve (Normalized Energy)")
        ax.axvline(t_5, color='orange', linestyle='--', label=f"5% Energy (t = {t_5:.2f}s)")
        ax.axvline(t_95, color='red', linestyle='--', label=f"95% Energy (t = {t_95:.2f}s)")
        
        ax.set_title(f"Husid Plot & Durasi Signifikan ($D_{{5-95}}$ = {d_5_95:.2f} detik)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Waktu (detik)")
        ax.set_ylabel("Energi Kumulatif (%)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_fas(freqs: np.ndarray, fas: np.ndarray, save_path: str = None):
        """Membuat grafik Fourier Amplitude Spectra (FAS)."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(freqs, fas, color='crimson', linewidth=1.2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("Fourier Amplitude Spectra (FAS)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Frekuensi (Hz)")
        ax.set_ylabel("Amplitudo (cm/s)")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()