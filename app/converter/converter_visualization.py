"""DC-DC Boost Converter Simulation Visualization Module.

Generates publication-quality scientific plots of converter state transients,
electrical powers, conversion efficiency, and duty cycle dynamics using Matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import pandas as pd

from app.utils.logger import get_logger
from config.default_config import get_default_config

logger = get_logger("converter_visualization")


def plot_pv_voltage(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot PV terminal voltage vs. time."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.plot(df["time"] * 1e3, df["v_pv"], color="#1f77b4", linewidth=2.0, label="$V_{pv}$ (Terminal Voltage)")
    ax.set_title("PV Input Voltage Response vs. Time", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [ms]", fontsize=10.5)
    ax.set_ylabel("PV Voltage $V_{pv}$ [V]", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved PV voltage plot: {save_path}")
    return fig


def plot_inductor_current(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot inductor current vs. time."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.plot(df["time"] * 1e3, df["i_l"], color="#d62728", linewidth=2.0, label="$i_L$ (Inductor Current)")
    ax.set_title("Inductor Current Dynamic Response vs. Time", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [ms]", fontsize=10.5)
    ax.set_ylabel("Inductor Current $i_L$ [A]", fontsize=10.5)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved inductor current plot: {save_path}")
    return fig


def plot_output_voltage(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot converter output voltage vs. time."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.plot(df["time"] * 1e3, df["v_out"], color="#2ca02c", linewidth=2.0, label="$V_{out}$ (Load Voltage)")
    ax.set_title("Converter Output Voltage Response vs. Time", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [ms]", fontsize=10.5)
    ax.set_ylabel("Output Voltage $V_{out}$ [V]", fontsize=10.5)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved output voltage plot: {save_path}")
    return fig


def plot_duty_cycle(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot converter duty cycle vs. time."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.step(df["time"] * 1e3, df["duty_cycle"], color="#ff7f0e", linewidth=2.2, where="post", label="Duty Cycle $D$")
    ax.set_title("Converter Duty Cycle Actuation vs. Time", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [ms]", fontsize=10.5)
    ax.set_ylabel("Duty Cycle $D$ [Unitless]", fontsize=10.5)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved duty cycle plot: {save_path}")
    return fig


def plot_input_power(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot PV input power vs. time."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.plot(df["time"] * 1e3, df["p_pv"], color="#9467bd", linewidth=2.0, label="$P_{pv}$ (Input Power)")
    ax.set_title("PV Generated Power vs. Time", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [ms]", fontsize=10.5)
    ax.set_ylabel("Input Power $P_{pv}$ [W]", fontsize=10.5)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved input power plot: {save_path}")
    return fig


def plot_output_power(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot converter output power vs. time."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.plot(df["time"] * 1e3, df["p_out"], color="#8c564b", linewidth=2.0, label="$P_{out}$ (Load Power)")
    ax.set_title("Converter Output Delivered Power vs. Time", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [ms]", fontsize=10.5)
    ax.set_ylabel("Output Power $P_{out}$ [W]", fontsize=10.5)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved output power plot: {save_path}")
    return fig


def plot_converter_efficiency(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot converter power conversion efficiency vs. time."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax.plot(df["time"] * 1e3, df["efficiency"], color="#17becf", linewidth=2.0, label=r"Conversion Efficiency $\eta$")
    ax.set_title("Boost Converter Efficiency vs. Time", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time $t$ [ms]", fontsize=10.5)
    ax.set_ylabel(r"Efficiency $\eta$ [%]", fontsize=10.5)
    ax.set_ylim(90.0, 100.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved converter efficiency plot: {save_path}")
    return fig


def plot_transient_multi_panel(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Generate comprehensive 4-panel transient overview of boost converter operation."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, dpi=300)

    time_ms = df["time"] * 1e3

    # Panel 1: Voltages
    axes[0].plot(time_ms, df["v_pv"], label="PV Voltage $V_{pv}$", color="#1f77b4", linewidth=1.8)
    axes[0].plot(time_ms, df["v_out"], label="Output Voltage $V_{out}$", color="#2ca02c", linewidth=1.8)
    axes[0].set_ylabel("Voltage [V]", fontsize=10)
    axes[0].set_title("DC-DC Boost Converter Transient Step Response", fontsize=13, fontweight="bold", pad=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", fontsize=9.5)

    # Panel 2: Currents
    axes[1].plot(time_ms, df["i_l"], label="Inductor Current $i_L$", color="#d62728", linewidth=1.8)
    axes[1].plot(time_ms, df["i_pv"], label="PV Current $I_{pv}$", color="#ff7f0e", linewidth=1.5, linestyle="--")
    axes[1].plot(time_ms, df["i_out"], label="Load Current $I_{out}$", color="#17becf", linewidth=1.5)
    axes[1].set_ylabel("Current [A]", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=9.5)

    # Panel 3: Powers
    axes[2].plot(time_ms, df["p_pv"], label="Input Power $P_{pv}$", color="#9467bd", linewidth=1.8)
    axes[2].plot(time_ms, df["p_out"], label="Output Power $P_{out}$", color="#8c564b", linewidth=1.8)
    axes[2].set_ylabel("Power [W]", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right", fontsize=9.5)

    # Panel 4: Duty Cycle & Efficiency
    ax4_eff = axes[3].twinx()
    l1 = axes[3].step(time_ms, df["duty_cycle"], label="Duty Cycle $D$", color="#e377c2", linewidth=2.0, where="post")
    l2 = ax4_eff.plot(time_ms, df["efficiency"], label=r"Efficiency $\eta$", color="#bcbd22", linewidth=1.8)
    axes[3].set_xlabel("Simulation Time $t$ [ms]", fontsize=10.5)
    axes[3].set_ylabel("Duty Cycle $D$", fontsize=10)
    ax4_eff.set_ylabel("Efficiency [%]", fontsize=10)
    axes[3].set_ylim(0.0, 1.0)
    ax4_eff.set_ylim(90.0, 100.0)
    axes[3].grid(True, linestyle="--", alpha=0.6)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    axes[3].legend(lines, labels, loc="lower right", fontsize=9.5)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved comprehensive multi-panel transient plot: {save_path}")
    return fig


def generate_all_phase_3_plots(
    simulation_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Generate and save all Phase 3 converter figures.

    Parameters
    ----------
    simulation_df : pd.DataFrame
        Simulation results DataFrame.
    output_dir : Optional[Path], optional
        Destination directory. If None, resolves to results/graphs/phase_3/.

    Returns
    -------
    List[Path]
        List of generated figure file paths.
    """
    if output_dir is None:
        config = get_default_config()
        output_dir = config.paths.graphs_dir / "phase_3"

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: List[Path] = []

    p1 = output_dir / "pv_voltage_vs_time.png"
    plot_pv_voltage(simulation_df, save_path=p1)
    plt.close()
    generated_paths.append(p1)

    p2 = output_dir / "inductor_current_vs_time.png"
    plot_inductor_current(simulation_df, save_path=p2)
    plt.close()
    generated_paths.append(p2)

    p3 = output_dir / "output_voltage_vs_time.png"
    plot_output_voltage(simulation_df, save_path=p3)
    plt.close()
    generated_paths.append(p3)

    p4 = output_dir / "duty_cycle_vs_time.png"
    plot_duty_cycle(simulation_df, save_path=p4)
    plt.close()
    generated_paths.append(p4)

    p5 = output_dir / "input_power_vs_time.png"
    plot_input_power(simulation_df, save_path=p5)
    plt.close()
    generated_paths.append(p5)

    p6 = output_dir / "output_power_vs_time.png"
    plot_output_power(simulation_df, save_path=p6)
    plt.close()
    generated_paths.append(p6)

    p7 = output_dir / "converter_efficiency_vs_time.png"
    plot_converter_efficiency(simulation_df, save_path=p7)
    plt.close()
    generated_paths.append(p7)

    p8 = output_dir / "converter_transient_multi_panel.png"
    plot_transient_multi_panel(simulation_df, save_path=p8)
    plt.close()
    generated_paths.append(p8)

    logger.info(f"Successfully generated all {len(generated_paths)} Phase 3 figures in {output_dir}")
    return generated_paths
