"""Publication-Quality Visualizations for Phase 4 Perturb & Observe (P&O) MPPT Controller.

Generates 10 academic figures saved at 300 DPI:
1. pv_voltage_vs_time.png
2. pv_current_vs_time.png
3. pv_power_vs_time.png
4. mpp_reference_vs_actual.png
5. duty_cycle_vs_time.png
6. irradiance_vs_time.png
7. temperature_vs_time.png
8. tracking_error_vs_time.png
9. perturbation_step_comparison.png
10. po_controller_multi_panel.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger("controller_visualization")

# Publication styling defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "figure.titlesize": 13,
})


def plot_pv_voltage(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot PV terminal voltage vs. time alongside analytical V_mpp reference."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    time_s = df["time"]
    ax.plot(time_s, df["v_pv"], color="#1f77b4", linewidth=1.8, label="PV Voltage $V_{pv}$")
    if "v_mpp" in df.columns:
        ax.plot(time_s, df["v_mpp"], color="#d62728", linestyle="--", linewidth=1.5, label="Reference $V_{mp}$")

    ax.set_title("PV Terminal Voltage vs. Time", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [s]", fontsize=10.5)
    ax.set_ylabel("Voltage $V_{pv}$ [V]", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved PV voltage plot: {save_path}")
    return fig


def plot_pv_current(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot PV array current vs. time alongside analytical I_mpp reference."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    time_s = df["time"]
    ax.plot(time_s, df["i_pv"], color="#2ca02c", linewidth=1.8, label="PV Current $I_{pv}$")
    if "i_mpp" in df.columns:
        ax.plot(time_s, df["i_mpp"], color="#ff7f0e", linestyle="--", linewidth=1.5, label="Reference $I_{mp}$")

    ax.set_title("PV Output Current vs. Time", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [s]", fontsize=10.5)
    ax.set_ylabel("Current $I_{pv}$ [A]", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved PV current plot: {save_path}")
    return fig


def plot_pv_power(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot PV output electrical power vs. time."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    time_s = df["time"]
    ax.plot(time_s, df["p_pv"], color="#9467bd", linewidth=1.8, label="Extracted Power $P_{pv}$")
    if "p_mpp" in df.columns:
        ax.plot(time_s, df["p_mpp"], color="#d62728", linestyle="--", linewidth=1.5, label="Maximum Available $P_{mp}$")

    ax.set_title("PV Output Power Tracking vs. Time", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [s]", fontsize=10.5)
    ax.set_ylabel("Power $P_{pv}$ [W]", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved PV power plot: {save_path}")
    return fig


def plot_mpp_reference_vs_actual(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot direct comparison overlay of reference MPP power vs. actual tracked PV power."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    time_s = df["time"]
    p_ideal = df["p_mpp"] if "p_mpp" in df.columns else df["p_pv"]
    ax.plot(time_s, p_ideal, color="#1f77b4", linewidth=2.0, label="Reference $P_{MPP}(G, T)$")
    ax.plot(time_s, df["p_pv"], color="#ff7f0e", linewidth=1.5, linestyle="-", label="Actual Tracked $P_{actual}$")

    ax.set_title("MPP Reference vs. Actual Tracked Power", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [s]", fontsize=10.5)
    ax.set_ylabel("Power [W]", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved MPP reference vs actual plot: {save_path}")
    return fig


def plot_duty_cycle(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot controller commanded duty cycle D vs. time."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    time_s = df["time"]
    ax.step(time_s, df["duty_cycle"], color="#e377c2", linewidth=1.8, where="post", label="Commanded Duty $D$")
    ax.axhline(0.3939, color="#7f7f7f", linestyle=":", linewidth=1.2, label=r"Theoretical $D^* \approx 0.3939$ (STC)")

    ax.set_title("P&O Commanded Duty Cycle vs. Time", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [s]", fontsize=10.5)
    ax.set_ylabel("Duty Cycle $D$", fontsize=10.5)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved duty cycle plot: {save_path}")
    return fig


def plot_irradiance(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot solar irradiance profile vs. time."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=300)
    time_s = df["time"]
    irrad = df["irradiance"] if "irradiance" in df.columns else np.full_like(time_s, 1000.0)
    ax.step(time_s, irrad, color="#bcbd22", linewidth=2.0, where="post", label="Solar Irradiance $G$")

    ax.set_title("Incident Solar Irradiance Profile", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [s]", fontsize=10.5)
    ax.set_ylabel(r"Irradiance $G$ [W/m$^2$]", fontsize=10.5)
    ax.set_ylim(0.0, 1200.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved irradiance plot: {save_path}")
    return fig


def plot_temperature(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot cell temperature profile vs. time."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=300)
    time_s = df["time"]
    temp = df["cell_temperature"] if "cell_temperature" in df.columns else np.full_like(time_s, 25.0)
    ax.step(time_s, temp, color="#17becf", linewidth=2.0, where="post", label="Cell Junction Temp $T_{cell}$")

    ax.set_title("PV Cell Operating Temperature Profile", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [s]", fontsize=10.5)
    ax.set_ylabel("Temperature $T$ [°C]", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved temperature plot: {save_path}")
    return fig


def plot_tracking_error(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot instantaneous MPPT tracking error (P_mpp - P_actual) vs. time."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    time_s = df["time"]
    error = df["tracking_error"] if "tracking_error" in df.columns else (df["p_mpp"] - df["p_pv"])
    ax.plot(time_s, error, color="#d62728", linewidth=1.5, label=r"Tracking Error $e_P = P_{MPP} - P_{actual}$")
    ax.axhline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.0)

    ax.set_title("Instantaneous MPPT Tracking Error vs. Time", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [s]", fontsize=10.5)
    ax.set_ylabel("Tracking Error [W]", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved tracking error plot: {save_path}")
    return fig


def plot_perturbation_step_comparison(
    comparison_dfs: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Compare tracking response and steady-state ripple across different perturbation steps delta_D."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=True, dpi=300)

    colors = {"0.001": "#1f77b4", "0.002": "#2ca02c", "0.005": "#d62728"}

    for label, df in comparison_dfs.items():
        time_s = df["time"]
        c = colors.get(label, None)
        ax1.plot(time_s, df["p_pv"], label=rf"$\delta_D = {label}$", color=c, linewidth=1.5)
        ax2.step(time_s, df["duty_cycle"], label=rf"$\delta_D = {label}$", color=c, linewidth=1.4, where="post")

    if "p_mpp" in next(iter(comparison_dfs.values())).columns:
        first_df = next(iter(comparison_dfs.values()))
        ax1.plot(first_df["time"], first_df["p_mpp"], color="#333333", linestyle="--", linewidth=1.2, label="Reference $P_{MPP}$")

    ax1.set_ylabel("PV Power $P_{pv}$ [W]", fontsize=10.5)
    ax1.set_title("MPPT Performance Comparison Across Perturbation Step Sizes", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9.0)

    ax2.set_xlabel("Time $t$ [s]", fontsize=10.5)
    ax2.set_ylabel("Duty Cycle $D$", fontsize=10.5)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9.0)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved step size comparison plot: {save_path}")
    return fig


def plot_po_controller_multi_panel(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Generate comprehensive 4-panel overview of closed-loop P&O MPPT controller operation."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, dpi=300)
    time_s = df["time"]

    # Panel 1: Environmental Conditions (Irradiance & Temperature)
    ax1_temp = axes[0].twinx()
    l1 = axes[0].step(time_s, df["irradiance"], label="Irradiance $G$", color="#bcbd22", linewidth=1.8, where="post")
    l2 = ax1_temp.step(time_s, df["cell_temperature"], label="Cell Temp $T$", color="#17becf", linewidth=1.6, linestyle="--", where="post")
    axes[0].set_ylabel(r"Irradiance [W/m$^2$]", fontsize=10)
    ax1_temp.set_ylabel("Temperature [°C]", fontsize=10)
    axes[0].set_title("Closed-Loop Perturb & Observe (P&O) MPPT System Dynamics", fontsize=13, fontweight="bold", pad=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    lines = l1 + l2
    axes[0].legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9.0)

    # Panel 2: PV Power vs MPP Reference
    axes[1].plot(time_s, df["p_mpp"], label="Available $P_{MPP}$", color="#333333", linestyle="--", linewidth=1.5)
    axes[1].plot(time_s, df["p_pv"], label="Tracked $P_{pv}$", color="#2ca02c", linewidth=1.8)
    axes[1].set_ylabel("Power [W]", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="lower right", fontsize=9.0)

    # Panel 3: Terminal Voltages (PV & Converter Output)
    axes[2].plot(time_s, df["v_pv"], label="PV Voltage $V_{pv}$", color="#1f77b4", linewidth=1.8)
    if "v_mpp" in df.columns:
        axes[2].plot(time_s, df["v_mpp"], label="Reference $V_{mp}$", color="#d62728", linestyle=":", linewidth=1.5)
    axes[2].plot(time_s, df["v_out"], label="Load Voltage $V_{out}$", color="#8c564b", linewidth=1.6)
    axes[2].set_ylabel("Voltage [V]", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="lower right", fontsize=9.0)

    # Panel 4: Commanded Duty Cycle & Conversion Efficiency
    ax4_eff = axes[3].twinx()
    l3 = axes[3].step(time_s, df["duty_cycle"], label="Duty Cycle $D$", color="#e377c2", linewidth=1.8, where="post")
    l4 = ax4_eff.plot(time_s, df["efficiency"], label=r"Converter Eff $\eta_{conv}$", color="#9467bd", linewidth=1.5)
    axes[3].set_xlabel("Simulation Time $t$ [s]", fontsize=10.5)
    axes[3].set_ylabel("Duty Cycle $D$", fontsize=10)
    ax4_eff.set_ylabel(r"Efficiency $\eta_{conv}$ [%]", fontsize=10)
    axes[3].set_ylim(0.0, 1.0)
    ax4_eff.set_ylim(90.0, 100.0)
    axes[3].grid(True, linestyle="--", alpha=0.6)
    lines4 = l3 + l4
    axes[3].legend(lines4, [l.get_label() for l in lines4], loc="lower right", fontsize=9.0)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved comprehensive multi-panel plot: {save_path}")
    return fig


def generate_all_phase_4_plots(
    df: pd.DataFrame,
    step_comparison_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Generate all 10 required Phase 4 visualization figures.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation timeseries dataset.
    step_comparison_dfs : Optional[Dict[str, pd.DataFrame]], optional
        Dictionary of DataFrames for perturbation step comparison (e.g. {'0.001': df1, ...}).
    output_dir : Optional[Path], optional
        Target directory to save figures. If None, defaults to results/graphs/phase_4.

    Returns
    -------
    Dict[str, Path]
        Mapping of plot names to saved file paths.
    """
    if output_dir is None:
        from config.default_config import get_default_config
        output_dir = get_default_config().paths.graphs_dir / "phase_4"

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Path] = {}

    plots = [
        ("pv_voltage_vs_time.png", plot_pv_voltage, (df,)),
        ("pv_current_vs_time.png", plot_pv_current, (df,)),
        ("pv_power_vs_time.png", plot_pv_power, (df,)),
        ("mpp_reference_vs_actual.png", plot_mpp_reference_vs_actual, (df,)),
        ("duty_cycle_vs_time.png", plot_duty_cycle, (df,)),
        ("irradiance_vs_time.png", plot_irradiance, (df,)),
        ("temperature_vs_time.png", plot_temperature, (df,)),
        ("tracking_error_vs_time.png", plot_tracking_error, (df,)),
        ("po_controller_multi_panel.png", plot_po_controller_multi_panel, (df,)),
    ]

    for fname, func, args in plots:
        fpath = output_dir / fname
        fig = func(*args, save_path=fpath)
        plt.close(fig)
        generated[fname] = fpath

    if step_comparison_dfs is not None:
        fpath = output_dir / "perturbation_step_comparison.png"
        fig = plot_perturbation_step_comparison(step_comparison_dfs, save_path=fpath)
        plt.close(fig)
        generated["perturbation_step_comparison.png"] = fpath

    logger.info(f"Successfully generated {len(generated)} Phase 4 figures in {output_dir}")
    return generated
