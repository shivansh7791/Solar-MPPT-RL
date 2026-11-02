"""Photovoltaic Characteristic Curve Visualization Module.

Generates publication-quality scientific plots for I-V and P-V curves,
environmental sensitivity analysis, and Maximum Power Point (MPP) tracking
behavior using Matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless and script execution
import matplotlib.pyplot as plt
import numpy as np

from app.pv.pv_characteristics import PVCharacteristicAnalyzer, PVCharacteristicCurve
from app.utils.logger import get_logger
from config.default_config import get_default_config

logger = get_logger("pv_visualization")

# Distinct, accessible color sequences for parametric sweeps
IRRADIANCE_COLORS = ["#2b5c8f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
TEMPERATURE_COLORS = ["#1f77b4", "#00a86b", "#e67e22", "#d35400", "#c0392b"]


def plot_iv_curves_irradiance(
    curves: Sequence[PVCharacteristicCurve],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate combined I-V characteristic curves under varying irradiance."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    for i, c in enumerate(curves):
        color = IRRADIANCE_COLORS[i % len(IRRADIANCE_COLORS)]
        label = f"{c.irradiance:.0f} W/m² (Isc={c.i_sc_analytical:.2f}A, Voc={c.v_oc_analytical:.1f}V)"
        ax.plot(c.voltage, c.current, label=label, color=color, linewidth=2.0)

    ax.set_title("PV Module I-V Characteristics under Varying Irradiance (T = 25°C)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Terminal Voltage $V_{pv}$ [V]", fontsize=11)
    ax.set_ylabel("Terminal Current $I_{pv}$ [A]", fontsize=11)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Solar Irradiance $G$", fontsize=9.5, title_fontsize=10, loc="lower left")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved I-V irradiance plot to: {save_path}")
    return fig


def plot_pv_curves_irradiance(
    curves: Sequence[PVCharacteristicCurve],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate combined P-V characteristic curves with marked MPPs under varying irradiance."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    for i, c in enumerate(curves):
        color = IRRADIANCE_COLORS[i % len(IRRADIANCE_COLORS)]
        label = f"{c.irradiance:.0f} W/m² (Pmp={c.p_mp_analytical:.1f} W)"
        ax.plot(c.voltage, c.power, label=label, color=color, linewidth=2.0)
        # Mark Maximum Power Point
        ax.plot(
            c.v_mp_analytical,
            c.p_mp_analytical,
            marker="o",
            markersize=7,
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=5,
        )

    ax.set_title("PV Module P-V Characteristics under Varying Irradiance (T = 25°C)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Terminal Voltage $V_{pv}$ [V]", fontsize=11)
    ax.set_ylabel("Output Power $P_{pv}$ [W]", fontsize=11)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Solar Irradiance $G$ (○ = MPP)", fontsize=9.5, title_fontsize=10, loc="upper left")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved P-V irradiance plot to: {save_path}")
    return fig


def plot_iv_curves_temperature(
    curves: Sequence[PVCharacteristicCurve],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate combined I-V characteristic curves under varying cell temperature."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    for i, c in enumerate(curves):
        color = TEMPERATURE_COLORS[i % len(TEMPERATURE_COLORS)]
        label = f"{c.temperature:.0f}°C (Voc={c.v_oc_analytical:.2f}V, Isc={c.i_sc_analytical:.2f}A)"
        ax.plot(c.voltage, c.current, label=label, color=color, linewidth=2.0)

    ax.set_title("PV Module I-V Characteristics under Varying Temperature (G = 1000 W/m²)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Terminal Voltage $V_{pv}$ [V]", fontsize=11)
    ax.set_ylabel("Terminal Current $I_{pv}$ [A]", fontsize=11)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Cell Temperature $T_{cell}$", fontsize=9.5, title_fontsize=10, loc="lower left")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved I-V temperature plot to: {save_path}")
    return fig


def plot_pv_curves_temperature(
    curves: Sequence[PVCharacteristicCurve],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate combined P-V characteristic curves with marked MPPs under varying temperature."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    for i, c in enumerate(curves):
        color = TEMPERATURE_COLORS[i % len(TEMPERATURE_COLORS)]
        label = f"{c.temperature:.0f}°C (Pmp={c.p_mp_analytical:.1f} W, Vmp={c.v_mp_analytical:.1f} V)"
        ax.plot(c.voltage, c.power, label=label, color=color, linewidth=2.0)
        # Mark Maximum Power Point
        ax.plot(
            c.v_mp_analytical,
            c.p_mp_analytical,
            marker="s",
            markersize=7,
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=5,
        )

    ax.set_title("PV Module P-V Characteristics under Varying Temperature (G = 1000 W/m²)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Terminal Voltage $V_{pv}$ [V]", fontsize=11)
    ax.set_ylabel("Output Power $P_{pv}$ [W]", fontsize=11)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Cell Temperature $T_{cell}$ (□ = MPP)", fontsize=9.5, title_fontsize=10, loc="upper right")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved P-V temperature plot to: {save_path}")
    return fig


def plot_mpp_power_vs_irradiance(
    curves: Sequence[PVCharacteristicCurve],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot Maximum Power (Pmp) as a function of incident irradiance."""
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)

    g_vals = [c.irradiance for c in curves]
    p_vals = [c.p_mp_analytical for c in curves]

    ax.plot(g_vals, p_vals, marker="o", color="#1f77b4", linewidth=2.2, markersize=6, label="Maximum Power $P_{mp}$")

    # Annotate points
    for g, p in zip(g_vals, p_vals):
        ax.annotate(f"{p:.1f} W", (g, p), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    ax.set_title("Maximum Power Point Output vs. Solar Irradiance (T = 25°C)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Solar Irradiance $G$ [W/m²]", fontsize=11)
    ax.set_ylabel("Maximum Power $P_{mp}$ [W]", fontsize=11)
    ax.set_ylim(bottom=0.0, top=max(p_vals) * 1.15)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved MPP power vs irradiance plot to: {save_path}")
    return fig


def plot_mpp_power_vs_temperature(
    curves: Sequence[PVCharacteristicCurve],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot Maximum Power (Pmp) as a function of cell temperature."""
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)

    t_vals = [c.temperature for c in curves]
    p_vals = [c.p_mp_analytical for c in curves]

    ax.plot(t_vals, p_vals, marker="s", color="#d62728", linewidth=2.2, markersize=6, label="Maximum Power $P_{mp}$")

    # Annotate points
    for t, p in zip(t_vals, p_vals):
        ax.annotate(f"{p:.1f} W", (t, p), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    ax.set_title("Maximum Power Point Output vs. Cell Temperature (G = 1000 W/m²)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Cell Temperature $T_{cell}$ [°C]", fontsize=11)
    ax.set_ylabel("Maximum Power $P_{mp}$ [W]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved MPP power vs temperature plot to: {save_path}")
    return fig


def plot_mpp_voltage_vs_irradiance(
    curves: Sequence[PVCharacteristicCurve],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot Maximum Power Point Voltage (Vmp) and Voc as a function of incident irradiance."""
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)

    g_vals = [c.irradiance for c in curves]
    vmp_vals = [c.v_mp_analytical for c in curves]
    voc_vals = [c.v_oc_analytical for c in curves]

    ax.plot(g_vals, voc_vals, marker="^", color="#2ca02c", linewidth=2.0, markersize=6, label="Open-Circuit Voltage $V_{oc}$")
    ax.plot(g_vals, vmp_vals, marker="o", color="#1f77b4", linewidth=2.0, markersize=6, label="MPP Voltage $V_{mp}$")

    for g, v in zip(g_vals, vmp_vals):
        ax.annotate(f"{v:.1f} V", (g, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5)

    ax.set_title("MPP Voltage and Open-Circuit Voltage vs. Solar Irradiance (T = 25°C)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Solar Irradiance $G$ [W/m²]", fontsize=11)
    ax.set_ylabel("Voltage [V]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved MPP voltage vs irradiance plot to: {save_path}")
    return fig


def plot_mpp_voltage_vs_temperature(
    curves: Sequence[PVCharacteristicCurve],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot Maximum Power Point Voltage (Vmp) and Voc as a function of cell temperature."""
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)

    t_vals = [c.temperature for c in curves]
    vmp_vals = [c.v_mp_analytical for c in curves]
    voc_vals = [c.v_oc_analytical for c in curves]

    ax.plot(t_vals, voc_vals, marker="^", color="#2ca02c", linewidth=2.0, markersize=6, label="Open-Circuit Voltage $V_{oc}$")
    ax.plot(t_vals, vmp_vals, marker="s", color="#e67e22", linewidth=2.0, markersize=6, label="MPP Voltage $V_{mp}$")

    for t, v in zip(t_vals, vmp_vals):
        ax.annotate(f"{v:.1f} V", (t, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)

    ax.set_title("MPP Voltage and Open-Circuit Voltage vs. Cell Temperature (G = 1000 W/m²)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Cell Temperature $T_{cell}$ [°C]", fontsize=11)
    ax.set_ylabel("Voltage [V]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved MPP voltage vs temperature plot to: {save_path}")
    return fig


def generate_all_phase_2_plots(
    output_dir: Optional[Path] = None,
    analyzer: Optional[PVCharacteristicAnalyzer] = None,
) -> List[Path]:
    """Execute complete Phase 2 visualization suite and save publication-quality plots.

    Parameters
    ----------
    output_dir : Optional[Path], optional
        Directory where figures will be saved. If None, resolves to results/graphs/.
    analyzer : Optional[PVCharacteristicAnalyzer], optional
        Analyzer instance. If None, a new analyzer with default model is created.

    Returns
    -------
    List[Path]
        List of filesystem paths for all generated figures.
    """
    if output_dir is None:
        config = get_default_config()
        output_dir = config.paths.graphs_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = analyzer if analyzer is not None else PVCharacteristicAnalyzer()

    # 1. Irradiance variations: 200, 400, 600, 800, 1000 W/m^2 at 25°C
    irradiances = [200.0, 400.0, 600.0, 800.0, 1000.0]
    irr_curves = analyzer.analyze_irradiance_variation(irradiances, temperature=25.0)

    # 2. Temperature variations: 15, 25, 35, 45, 55°C at 1000 W/m^2
    temperatures = [15.0, 25.0, 35.0, 45.0, 55.0]
    temp_curves = analyzer.analyze_temperature_variation(temperatures, irradiance=1000.0)

    generated_paths: List[Path] = []

    p1 = output_dir / "iv_curves_irradiance.png"
    plot_iv_curves_irradiance(irr_curves, save_path=p1)
    plt.close()
    generated_paths.append(p1)

    p2 = output_dir / "pv_curves_irradiance.png"
    plot_pv_curves_irradiance(irr_curves, save_path=p2)
    plt.close()
    generated_paths.append(p2)

    p3 = output_dir / "iv_curves_temperature.png"
    plot_iv_curves_temperature(temp_curves, save_path=p3)
    plt.close()
    generated_paths.append(p3)

    p4 = output_dir / "pv_curves_temperature.png"
    plot_pv_curves_temperature(temp_curves, save_path=p4)
    plt.close()
    generated_paths.append(p4)

    p5 = output_dir / "mpp_power_vs_irradiance.png"
    plot_mpp_power_vs_irradiance(irr_curves, save_path=p5)
    plt.close()
    generated_paths.append(p5)

    p6 = output_dir / "mpp_power_vs_temperature.png"
    plot_mpp_power_vs_temperature(temp_curves, save_path=p6)
    plt.close()
    generated_paths.append(p6)

    p7 = output_dir / "mpp_voltage_vs_irradiance.png"
    plot_mpp_voltage_vs_irradiance(irr_curves, save_path=p7)
    plt.close()
    generated_paths.append(p7)

    p8 = output_dir / "mpp_voltage_vs_temperature.png"
    plot_mpp_voltage_vs_temperature(temp_curves, save_path=p8)
    plt.close()
    generated_paths.append(p8)

    logger.info(f"Successfully generated all {len(generated_paths)} Phase 2 figures in {output_dir}")
    return generated_paths
