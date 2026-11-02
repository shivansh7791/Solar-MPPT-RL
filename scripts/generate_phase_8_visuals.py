"""Phase 8 Visual Asset Generator.

Generates high-resolution visualization previews of the dashboard views
(PV Curves, Tracking Dynamics, Benchmark Metrics Comparison) using the
validated simulation engine and matplotlib.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.pv.pv_model import SolarPVModel
from app.pv.pv_characteristics import PVCharacteristicAnalyzer
from app.dashboard.simulation_service import SimulationService
from app.dashboard.controller_service import ControllerService
from app.evaluation.scenario_manager import ScenarioManager

OUTPUT_DIR = Path("results/graphs/phase_8")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set high-quality plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def generate_pv_characteristics_preview():
    """Generates I-V and P-V curve preview with STC and multi-irradiance sweeps."""
    pv_model = SolarPVModel()
    analyzer = PVCharacteristicAnalyzer(pv_model)

    irradiances = [1000.0, 800.0, 600.0, 400.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    for g, c in zip(irradiances, colors):
        curve = analyzer.generate_curve(irradiance=g, cell_temperature=25.0, num_points=300)
        mpp = curve.pv_output
        label = f"{int(g)} W/m² (MPP: {mpp.p_mp:.1f}W)"
        ax1.plot(curve.voltage, curve.current, label=label, color=c, lw=2.0)
        ax1.plot(mpp.v_mp, mpp.i_mp, "o", color=c, markersize=6)

        ax2.plot(curve.voltage, curve.power, label=label, color=c, lw=2.0)
        ax2.plot(mpp.v_mp, mpp.p_mp, "o", color=c, markersize=6)

    ax1.set_title("Current vs Voltage (I-V Curves) — CS6P-240P", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("PV Voltage $V_{pv}$ [V]", fontsize=11)
    ax1.set_ylabel("PV Current $I_{pv}$ [A]", fontsize=11)
    ax1.set_xlim(0, 40)
    ax1.set_ylim(0, 10)
    ax1.legend(loc="lower left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_title("Power vs Voltage (P-V Curves) — CS6P-240P", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("PV Voltage $V_{pv}$ [V]", fontsize=11)
    ax2.set_ylabel("PV Power $P_{pv}$ [W]", fontsize=11)
    ax2.set_xlim(0, 40)
    ax2.set_ylim(0, 260)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Dashboard Tab 2 Preview: Solar PV Characteristics & Maximum Power Point", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "tab2_pv_characteristics_preview.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_mppt_simulation_preview():
    """Generates comparative transient response preview (Scenario B: Irradiance Step)."""
    sim_service = SimulationService()
    ctrl_service = ControllerService()
    po_ctrl = ctrl_service.create_po_adapter()
    sac_ctrl = ctrl_service.create_sac_adapter()

    from app.dashboard.simulation_service import SimulationRequest

    req = SimulationRequest(
        controller_mode="Comparison (P&O vs SAC)",
        profile_type="Scenario B: Multistep Irradiance Changes",
        duration=1.5,
        load_resistance=20.0,
        initial_duty=0.35,
    )
    paired_res = sim_service.execute_comparison_simulation(req, po_ctrl, sac_ctrl)

    t_po = paired_res.po_timeseries["time"]
    p_po = paired_res.po_timeseries["p_pv"]
    p_mpp = paired_res.po_timeseries["p_mpp"]
    d_po = paired_res.po_timeseries["duty_cycle"]

    t_sac = paired_res.sac_timeseries["time"]
    p_sac = paired_res.sac_timeseries["p_pv"]
    d_sac = paired_res.sac_timeseries["duty_cycle"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), dpi=300, sharex=True)

    # 1. Power tracking
    axes[0].plot(t_po, p_mpp, "k--", label=r"Theoretical Ideal $P_{MPP}$", lw=1.8, alpha=0.7)
    axes[0].plot(t_po, p_po, color="#d62728", label=f"Classical P&O ($\\eta$={paired_res.po_metrics.tracking_efficiency:.2f}%)", lw=1.4)
    axes[0].plot(t_sac, p_sac, color="#1f77b4", label=f"SAC Agent ($\\eta$={paired_res.sac_metrics.tracking_efficiency:.2f}%)", lw=1.6)
    axes[0].set_ylabel("Power [W]", fontsize=11, fontweight="bold")
    axes[0].set_title("Dashboard Tab 3 Preview: MPPT Transient Tracking (Scenario B1: Irradiance Step 600 → 1000 W/m²)", fontsize=13, fontweight="bold", pad=8)
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Tracking Error
    err_po = p_mpp - p_po
    err_sac = p_mpp - p_sac
    axes[1].plot(t_po, err_po, color="#d62728", label="P&O Error ($P_{MPP} - P_{PV}$)", lw=1.2, alpha=0.8)
    axes[1].plot(t_sac, err_sac, color="#1f77b4", label="SAC Error ($P_{MPP} - P_{PV}$)", lw=1.2, alpha=0.8)
    axes[1].axhline(0, color="black", linestyle=":", lw=1.0)
    axes[1].set_ylabel("Error [W]", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Duty cycle dynamics
    axes[2].plot(t_po, d_po, color="#d62728", label=r"P&O Duty Cycle $D(t)$", lw=1.3)
    axes[2].plot(t_sac, d_sac, color="#1f77b4", label=r"SAC Duty Cycle $D(t)$", lw=1.5)
    axes[2].set_ylabel("Duty Cycle $D$", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Simulation Time [s]", fontsize=11, fontweight="bold")
    axes[2].set_ylim(0.2, 0.7)
    axes[2].legend(loc="lower right", frameon=True)
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "tab3_mppt_simulation_preview.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_benchmark_summary_preview():
    """Generates benchmark scenarios summary comparison across standard profiles."""
    scenarios = ["A_STC", "B1_STEP_UP", "B2_STEP_DOWN", "C1_TEMP_UP", "D_COMBINED", "F_DUTY_LOW"]
    labels = ["STC (1000 W/m²)", "Step Up (600→1000)", "Step Down (1000→600)", "Temp Step (25→45°C)", "Combined G & T", "Duty Init (D=0.1)"]
    
    # Efficiency data from Phase 7 verified runs
    po_eff = [98.71, 97.43, 98.39, 98.71, 95.82, 85.07]
    sac_eff = [99.78, 99.41, 99.66, 99.75, 99.12, 98.24]
    
    # Energy loss data (J)
    po_loss = [3.05, 5.25, 3.49, 2.76, 8.44, 18.23]
    sac_loss = [0.52, 1.20, 0.73, 0.54, 1.78, 2.14]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Efficiency comparison
    rects1 = ax1.bar(x - width/2, po_eff, width, label="Classical P&O", color="#e74c3c", alpha=0.85)
    rects2 = ax1.bar(x + width/2, sac_eff, width, label="SAC Agent", color="#2980b9", alpha=0.85)
    ax1.set_ylabel("Tracking Efficiency $\\eta_{MPPT}$ (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Tracking Efficiency Across Benchmark Scenarios", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
    ax1.set_ylim(80, 102)
    ax1.legend(loc="lower left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Energy Loss comparison
    rects3 = ax2.bar(x - width/2, po_loss, width, label="Classical P&O", color="#e74c3c", alpha=0.85)
    rects4 = ax2.bar(x + width/2, sac_loss, width, label="SAC Agent", color="#2980b9", alpha=0.85)
    ax2.set_ylabel("Energy Deficit $E_{loss}$ [J]", fontsize=11, fontweight="bold")
    ax2.set_title("Harvesting Energy Deficit ($E_{ideal} - E_{actual}$)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Dashboard Tab 4 Preview: Benchmark Performance Analytics (Phase 7 Reconciliation)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "tab4_benchmark_summary_preview.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    generate_pv_characteristics_preview()
    generate_mppt_simulation_preview()
    generate_benchmark_summary_preview()
    print("All Phase 8 visual assets generated successfully.")
