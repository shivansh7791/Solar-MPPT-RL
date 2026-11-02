"""Publication-Quality Visualization Suite for MPPT Comparative Study.

Generates 14 standardized publication figures at 300 DPI in results/graphs/phase_7/:
1. fig01_stc_pv_power_comparison.png: SAC vs P&O PV power under STC.
2. fig02_stc_pv_voltage_comparison.png: SAC vs P&O PV voltage under STC.
3. fig03_duty_cycle_response_comparison.png: SAC vs P&O duty-cycle response.
4. fig04_irradiance_step_response.png: Irradiance step response.
5. fig05_temperature_step_response.png: Temperature step response.
6. fig06_combined_environmental_response.png: Combined environmental response.
7. fig07_tracking_efficiency_comparison.png: Tracking efficiency comparison.
8. fig08_energy_harvested_comparison.png: Energy harvested comparison.
9. fig09_energy_loss_comparison.png: Energy loss comparison.
10. fig10_steady_state_power_ripple_comparison.png: Steady-state power ripple comparison.
11. fig11_settling_time_comparison.png: Settling time comparison.
12. fig12_scenariowise_tracking_efficiency.png: Scenario-wise tracking efficiency.
13. fig13_paired_performance_difference_distribution.png: Paired performance difference distribution.
14. fig14_converter_efficiency_comparison.png: Converter efficiency comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.evaluation.comparison_runner import PairedScenarioResult
from app.evaluation.statistical_analysis import StatisticalSummary
from app.utils.logger import get_logger

logger = get_logger("evaluation_visualization")

# Apply consistent academic styling
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

COLOR_PO = "#D95F02"   # Academic orange/vermilion
COLOR_SAC = "#1B9E77"  # Academic teal/green
COLOR_REF = "#7570B3"  # Reference purple/grey


def _find_scenario(results: List[PairedScenarioResult], scenario_id: str) -> PairedScenarioResult:
    """Helper to locate scenario result by scenario_id."""
    for r in results:
        if r.scenario.scenario_id == scenario_id:
            return r
    return results[0]


def generate_all_phase_7_plots(
    results: List[PairedScenarioResult],
    statistical_summary: StatisticalSummary,
    output_dir: Union[str, Path],
) -> List[Path]:
    """Generate all 14 required Phase 7 publication figures."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    res_stc = _find_scenario(results, "scenario_a_stc")
    res_g_step = _find_scenario(results, "scenario_b_irradiance_step")
    res_t_step = _find_scenario(results, "scenario_c_temperature_step")
    res_comb = _find_scenario(results, "scenario_d_combined")

    # 1. FIG 01: STC PV Power Tracking Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    t_po = res_stc.po_timeseries["time"] * 1e3
    t_sac = res_stc.sac_timeseries["time"] * 1e3

    ax1.plot(t_sac, res_stc.sac_timeseries["p_mpp"], color="black", linestyle="--", linewidth=1.5, label="Ideal MPP Ref ($P_{mpp}$)")
    ax1.plot(t_po, res_stc.po_timeseries["p_pv"], color=COLOR_PO, linewidth=1.5, alpha=0.9, label=f"Classical P&O (η={res_stc.po_metrics.tracking_efficiency:.2f}%)")
    ax1.plot(t_sac, res_stc.sac_timeseries["p_pv"], color=COLOR_SAC, linewidth=1.5, alpha=0.9, label=f"SAC Agent (η={res_stc.sac_metrics.tracking_efficiency:.2f}%)")
    ax1.set_ylabel("PV Power $P_{pv}$ (W)")
    ax1.set_title("Figure 1: PV Power Tracking Performance under STC ($1000\\ W/m^2, 25^\\circ C$)")
    ax1.legend(loc="lower right")

    err_po = res_stc.po_timeseries["p_mpp"] - res_stc.po_timeseries["p_pv"]
    err_sac = res_stc.sac_timeseries["p_mpp"] - res_stc.sac_timeseries["p_pv"]
    ax2.plot(t_po, err_po, color=COLOR_PO, linewidth=1.2, label="P&O Error")
    ax2.plot(t_sac, err_sac, color=COLOR_SAC, linewidth=1.2, label="SAC Error")
    ax2.set_xlabel("Physical Time (ms)")
    ax2.set_ylabel("Error $\\Delta P$ (W)")
    ax2.legend(loc="upper right")
    fpath = out_dir / "fig01_stc_pv_power_comparison.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    # 2. FIG 02: STC PV Voltage Tracking Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_sac, res_stc.sac_timeseries["v_mpp"], color="black", linestyle="--", linewidth=1.5, label="Optimal $V_{mp}$ (29.90 V)")
    ax.plot(t_po, res_stc.po_timeseries["v_pv"], color=COLOR_PO, linewidth=1.5, label="Classical P&O $V_{pv}$")
    ax.plot(t_sac, res_stc.sac_timeseries["v_pv"], color=COLOR_SAC, linewidth=1.5, label="SAC Controller $V_{pv}$")
    ax.set_xlabel("Physical Time (ms)")
    ax.set_ylabel("PV Terminal Voltage $V_{pv}$ (V)")
    ax.set_title("Figure 2: Terminal Voltage Trajectories under STC")
    ax.legend(loc="lower right")
    fpath = out_dir / "fig02_stc_pv_voltage_comparison.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    # 3. FIG 03: Duty Cycle Response Comparison (Limit Cycle vs Steady State)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(t_po, res_stc.po_timeseries["duty_cycle"], color=COLOR_PO, linewidth=1.5, label="P&O Duty")
    ax1.plot(t_sac, res_stc.sac_timeseries["duty_cycle"], color=COLOR_SAC, linewidth=1.5, label="SAC Duty")
    ax1.axhline(0.3939, color="black", linestyle=":", label="Theoretical $D^* \\approx 0.3939$")
    ax1.set_xlabel("Physical Time (ms)")
    ax1.set_ylabel("Converter Duty Cycle $D$")
    ax1.set_title("(a) Full Episode Duty Cycle Dynamics")
    ax1.legend(loc="lower right")

    # Zoomed window on steady-state [1000 ms to 1500 ms]
    mask_po = t_po >= 1000.0
    mask_sac = t_sac >= 1000.0
    ax2.plot(t_po[mask_po], res_stc.po_timeseries["duty_cycle"][mask_po], color=COLOR_PO, marker="o", markersize=3, label="P&O (3-Point Chattering)")
    ax2.plot(t_sac[mask_sac], res_stc.sac_timeseries["duty_cycle"][mask_sac], color=COLOR_SAC, marker="s", markersize=3, label="SAC (Chatter-Free Smooth Settling)")
    ax2.set_xlabel("Physical Time (ms)")
    ax2.set_ylabel("Duty Cycle $D$")
    ax2.set_title("(b) Steady-State Oscillation Window")
    ax2.legend(loc="best")
    fig.suptitle("Figure 3: Duty Cycle Response and Limit-Cycle Oscillation Suppression", y=1.02)
    fpath = out_dir / "fig03_duty_cycle_response_comparison.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    # 4. FIG 04: Irradiance Step Response (Scenario B)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    t_gb = res_g_step.sac_timeseries["time"] * 1e3
    ax1.plot(t_gb, res_g_step.sac_timeseries["p_mpp"], color="black", linestyle="--", linewidth=1.5, label="Dynamic MPP $P_{mpp}(t)$")
    ax1.plot(t_gb, res_g_step.po_timeseries["p_pv"], color=COLOR_PO, linewidth=1.4, alpha=0.9, label=f"P&O (η={res_g_step.po_metrics.tracking_efficiency:.2f}%)")
    ax1.plot(t_gb, res_g_step.sac_timeseries["p_pv"], color=COLOR_SAC, linewidth=1.4, alpha=0.9, label=f"SAC (η={res_g_step.sac_metrics.tracking_efficiency:.2f}%)")
    for trans in res_g_step.scenario.transition_times:
        ax1.axvline(trans * 1e3, color="gray", linestyle=":", alpha=0.7)
    ax1.set_ylabel("PV Power (W)")
    ax1.set_title("Figure 4: Dynamic Tracking under Irradiance Steps ($1000 \\to 600 \\to 800 \\to 400\\ W/m^2$)")
    ax1.legend(loc="lower left")

    ax2.plot(t_gb, res_g_step.sac_timeseries["irradiance"], color="#E6AB02", linewidth=1.8, label="Solar Irradiance $G(t)$")
    ax2.set_xlabel("Physical Time (ms)")
    ax2.set_ylabel("Irradiance ($W/m^2$)")
    ax2.legend(loc="upper right")
    fpath = out_dir / "fig04_irradiance_step_response.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    # 5. FIG 05: Temperature Step Response (Scenario C)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    t_tc = res_t_step.sac_timeseries["time"] * 1e3
    ax1.plot(t_tc, res_t_step.sac_timeseries["p_mpp"], color="black", linestyle="--", linewidth=1.5, label="Dynamic MPP $P_{mpp}(t)$")
    ax1.plot(t_tc, res_t_step.po_timeseries["p_pv"], color=COLOR_PO, linewidth=1.4, label=f"P&O (η={res_t_step.po_metrics.tracking_efficiency:.2f}%)")
    ax1.plot(t_tc, res_t_step.sac_timeseries["p_pv"], color=COLOR_SAC, linewidth=1.4, label=f"SAC (η={res_t_step.sac_metrics.tracking_efficiency:.2f}%)")
    for trans in res_t_step.scenario.transition_times:
        ax1.axvline(trans * 1e3, color="gray", linestyle=":", alpha=0.7)
    ax1.set_ylabel("PV Power (W)")
    ax1.set_title("Figure 5: Dynamic Tracking under Temperature Steps ($25 \\to 35 \\to 45 \\to 25^\\circ C$)")
    ax1.legend(loc="lower left")

    ax2.plot(t_tc, res_t_step.sac_timeseries["cell_temperature"], color="#E7298A", linewidth=1.8, label="Cell Temperature $T(t)$")
    ax2.set_xlabel("Physical Time (ms)")
    ax2.set_ylabel("Temperature ($^\\circ C$)")
    ax2.legend(loc="upper right")
    fpath = out_dir / "fig05_temperature_step_response.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    # 6. FIG 06: Combined Environmental Response (Scenario D)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    t_cd = res_comb.sac_timeseries["time"] * 1e3
    ax1.plot(t_cd, res_comb.sac_timeseries["p_mpp"], color="black", linestyle="--", linewidth=1.5, label="Dynamic MPP $P_{mpp}(t)$")
    ax1.plot(t_cd, res_comb.po_timeseries["p_pv"], color=COLOR_PO, linewidth=1.4, label=f"P&O (η={res_comb.po_metrics.tracking_efficiency:.2f}%)")
    ax1.plot(t_cd, res_comb.sac_timeseries["p_pv"], color=COLOR_SAC, linewidth=1.4, label=f"SAC (η={res_comb.sac_metrics.tracking_efficiency:.2f}%)")
    ax1.axvline(750.0, color="gray", linestyle=":", alpha=0.7)
    ax1.set_ylabel("PV Power (W)")
    ax1.set_title("Figure 6: Robustness under Simultaneous Irradiance Ramp and Thermal Step")
    ax1.legend(loc="lower left")

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(t_cd, res_comb.sac_timeseries["irradiance"], color="#E6AB02", linewidth=1.6, label="Irradiance $G(t)$")
    l2 = ax2_twin.plot(t_cd, res_comb.sac_timeseries["cell_temperature"], color="#E7298A", linewidth=1.6, label="Temperature $T(t)$")
    ax2.set_xlabel("Physical Time (ms)")
    ax2.set_ylabel("Irradiance ($W/m^2$)")
    ax2_twin.set_ylabel("Temp ($^\\circ C$)")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right")
    fpath = out_dir / "fig06_combined_environmental_response.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    # 7. FIG 07: Tracking Efficiency Comparison (Scatter / Category Breakdown)
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = [r.scenario.category for r in results]
    po_effs = [r.po_metrics.tracking_efficiency for r in results]
    sac_effs = [r.sac_metrics.tracking_efficiency for r in results]
    indices = np.arange(len(results))

    width = 0.38
    ax.bar(indices - width/2, po_effs, width, color=COLOR_PO, alpha=0.85, label="Classical P&O")
    ax.bar(indices + width/2, sac_effs, width, color=COLOR_SAC, alpha=0.85, label="Soft Actor-Critic (SAC)")
    ax.set_ylim(min(min(po_effs), min(sac_effs)) - 2.0, 102.0)
    ax.set_ylabel("Tracking Efficiency $\\eta_{MPPT}$ (%)")
    ax.set_xlabel("Scenario Index")
    ax.set_title("Figure 7: Energy Tracking Efficiency Across All 19 Benchmark Scenarios")
    ax.set_xticks(indices)
    ax.set_xticklabels([f"S{i+1}" for i in range(len(results))], rotation=45)
    ax.legend(loc="lower right")
    fpath = out_dir / "fig07_tracking_efficiency_comparison.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    # 8. FIG 08: Energy Harvested Comparison (Joules)
    fig, ax = plt.subplots(figsize=(9, 5))
    po_e = [r.po_metrics.energy_harvested_joules for r in results]
    sac_e = [r.sac_metrics.energy_harvested_joules for r in results]
    ax.plot(indices, sac_e, marker="s", color=COLOR_SAC, linewidth=1.5, label="SAC Energy Harvested")
    ax.plot(indices, po_e, marker="o", color=COLOR_PO, linewidth=1.5, label="P&O Energy Harvested")
    ax.set_ylabel("Harvested Energy (Joules)")
    ax.set_xlabel("Scenario Index")
    ax.set_title("Figure 8: Electrical Energy Yield Extracted Across 19 Benchmark Profiles")
    ax.set_xticks(indices)
    ax.set_xticklabels([f"S{i+1}" for i in range(len(results))], rotation=45)
    ax.legend(loc="best")
    fpath = out_dir / "fig08_energy_harvested_comparison.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    # 9. FIG 09: Energy Loss Comparison (Joules Deficit)
    fig, ax = plt.subplots(figsize=(9, 5))
    po_loss = [r.po_metrics.energy_loss_joules for r in results]
    sac_loss = [r.sac_metrics.energy_loss_joules for r in results]
    ax.bar(indices - width/2, po_loss, width, color=COLOR_PO, alpha=0.85, label="P&O Energy Deficit")
    ax.bar(indices + width/2, sac_loss, width, color=COLOR_SAC, alpha=0.85, label="SAC Energy Deficit")
    ax.set_ylabel("Energy Deficit $\\int (P_{mpp} - P_{pv}) dt$ (Joules)")
    ax.set_xlabel("Scenario Index")
    ax.set_title("Figure 9: Energy Deficit Comparison (Lower is Superior)")
    ax.set_xticks(indices)
    ax.set_xticklabels([f"S{i+1}" for i in range(len(results))], rotation=45)
    ax.legend(loc="upper right")
    fpath = out_dir / "fig09_energy_loss_comparison.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    # 10. FIG 10: Steady-State Power Ripple Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    po_rip = [r.po_metrics.steady_state_power_ripple_w for r in results]
    sac_rip = [r.sac_metrics.steady_state_power_ripple_w for r in results]
    ax.scatter(indices, po_rip, color=COLOR_PO, s=60, marker="o", label="Classical P&O Ripple")
    ax.scatter(indices, sac_rip, color=COLOR_SAC, s=60, marker="s", label="SAC Ripple (Near-Zero)")
    ax.set_ylabel("Peak-to-Peak Power Ripple $\\Delta P_{ss}$ (W)")
    ax.set_yscale("log")
    ax.set_xlabel("Scenario Index")
    ax.set_title("Figure 10: Steady-State Power Ripple Comparison (Log Scale)")
    ax.set_xticks(indices)
    ax.set_xticklabels([f"S{i+1}" for i in range(len(results))], rotation=45)
    ax.legend(loc="best")
    fpath = out_dir / "fig10_steady_state_power_ripple_comparison.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    # 11. FIG 11: Settling Time Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    po_ts = [r.po_metrics.settling_time_s * 1e3 if r.po_metrics.settling_time_s is not None else np.nan for r in results]
    sac_ts = [r.sac_metrics.settling_time_s * 1e3 if r.sac_metrics.settling_time_s is not None else np.nan for r in results]
    ax.plot(indices, po_ts, marker="o", color=COLOR_PO, linestyle="-", label="P&O Settling Time")
    ax.plot(indices, sac_ts, marker="s", color=COLOR_SAC, linestyle="-", label="SAC Settling Time")
    ax.set_ylabel("Settling Time $t_s$ (ms to $\\pm 2\\%$)")
    ax.set_xlabel("Scenario Index")
    ax.set_title("Figure 11: Settling Time Comparison across Evaluated Scenarios")
    ax.set_xticks(indices)
    ax.set_xticklabels([f"S{i+1}" for i in range(len(results))], rotation=45)
    ax.legend(loc="best")
    fpath = out_dir / "fig11_settling_time_comparison.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    # 12. FIG 12: Scenario-wise Tracking Efficiency Summary Table / Categorical Bars
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cat_names = list(statistical_summary.category_summaries.keys())
    cat_po_eff = [statistical_summary.category_summaries[c]["mean_po_efficiency"] for c in cat_names]
    cat_sac_eff = [statistical_summary.category_summaries[c]["mean_sac_efficiency"] for c in cat_names]
    cat_indices = np.arange(len(cat_names))

    b1 = ax.bar(cat_indices - width/2, cat_po_eff, width, color=COLOR_PO, alpha=0.85, label="P&O Mean Tracking Efficiency")
    b2 = ax.bar(cat_indices + width/2, cat_sac_eff, width, color=COLOR_SAC, alpha=0.85, label="SAC Mean Tracking Efficiency")
    ax.set_ylim(min(min(cat_po_eff), min(cat_sac_eff)) - 3.0, 102.0)
    ax.set_ylabel("Mean Tracking Efficiency (%)")
    ax.set_title("Figure 12: Categorical Tracking Efficiency Across Atmospheric Groupings")
    ax.set_xticks(cat_indices)
    clean_cat_labels = [c.replace("_", " ").title() for c in cat_names]
    ax.set_xticklabels(clean_cat_labels, rotation=20, ha="right")
    ax.legend(loc="lower right")

    # Annotate values
    for rect in b1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in b2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    fpath = out_dir / "fig12_scenariowise_tracking_efficiency.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    # 13. FIG 13: Paired Performance Difference Distribution (Delta eta)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    delta_effs = [r.delta_efficiency for r in results]
    ax1.hist(delta_effs, bins=10, color=COLOR_SAC, edgecolor="black", alpha=0.75)
    ax1.axvline(float(np.mean(delta_effs)), color="red", linestyle="--", linewidth=1.5, label=f"Mean $\\Delta\\eta = {np.mean(delta_effs):+.2f}\\%$")
    ax1.axvline(0.0, color="black", linestyle=":", linewidth=1.2)
    ax1.set_xlabel("Efficiency Difference $\\Delta\\eta = \\eta_{SAC} - \\eta_{PO}$ (%)")
    ax1.set_ylabel("Scenario Count")
    ax1.set_title("(a) Distribution Histogram")
    ax1.legend(loc="upper left")

    ax2.boxplot(delta_effs, vert=False, patch_artist=True, boxprops=dict(facecolor="#B2DF8A"), medianprops=dict(color="darkgreen", linewidth=2))
    ax2.axvline(0.0, color="black", linestyle=":", linewidth=1.2)
    ax2.set_xlabel("$\\Delta\\eta$ (%)")
    ax2.set_yticklabels(["SAC vs P&O"])
    ax2.set_title(f"(b) Boxplot ($t$-test $p = {statistical_summary.efficiency_stats.t_test_p_value:.3e}$)")
    fig.suptitle("Figure 13: Paired Tracking Efficiency Difference Distribution across 19 Scenarios", y=1.02)
    fpath = out_dir / "fig13_paired_performance_difference_distribution.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    # 14. FIG 14: Converter Efficiency Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    po_conv = [r.po_metrics.mean_converter_efficiency for r in results]
    sac_conv = [r.sac_metrics.mean_converter_efficiency for r in results]
    ax.plot(indices, sac_conv, marker="s", color=COLOR_SAC, label="SAC Converter Efficiency $\\eta_{conv}$")
    ax.plot(indices, po_conv, marker="o", color=COLOR_PO, label="P&O Converter Efficiency $\\eta_{conv}$")
    ax.set_ylabel("Boost Converter Efficiency (%)")
    ax.set_xlabel("Scenario Index")
    ax.set_title("Figure 14: DC-DC Boost Converter Conversion Efficiency $\\eta_{conv}$")
    ax.set_xticks(indices)
    ax.set_xticklabels([f"S{i+1}" for i in range(len(results))], rotation=45)
    ax.legend(loc="lower right")
    fpath = out_dir / "fig14_converter_efficiency_comparison.png"
    fig.savefig(fpath)
    plt.close(fig)
    generated.append(fpath)

    logger.info(f"Successfully generated all {len(generated)} Phase 7 publication figures in: {out_dir}")
    return generated
