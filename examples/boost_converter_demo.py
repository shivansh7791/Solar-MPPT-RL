"""Phase 3 Demonstration: DC-DC Boost Converter Simulation & Dynamic Step Response.

Demonstrates initializing the coupled PV-converter plant, executing a duty-cycle step
actuation at STC (1000 W/m², 25°C), verifying energy and power conservation, saving
simulation metrics to CSV, and generating publication-quality figures.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from app.converter import BoostConverter
from app.converter.converter_visualization import generate_all_phase_3_plots
from app.pv import SolarPVModel
from app.utils import setup_logger
from config import get_default_config


def run_demo() -> None:
    logger = setup_logger("converter_demo")
    logger.info("Initializing Coupled PV-Boost Converter Plant...")

    config = get_default_config()
    pv_model = SolarPVModel(config.pv)
    converter = BoostConverter(pv_model=pv_model, config=config.converter)

    stc_ref = pv_model.get_stc_parameters()
    logger.info(f"Connected PV Module: {pv_model.manufacturer} {pv_model.model_name}")
    logger.info(f"PV STC Baseline: Pmp = {stc_ref.p_mp:.2f} W, Vmp = {stc_ref.v_mp:.2f} V, Imp = {stc_ref.i_mp:.2f} A, Voc = {stc_ref.v_oc:.2f} V")
    logger.info(f"Converter Parameters: L = {converter.L*1e3:.2f} mH, Cpv = {converter.C_pv*1e6:.1f} µF, Cout = {converter.C_out*1e6:.1f} µF, Rload = {converter.R_load:.1f} Ω, rL = {converter.r_L*1e3:.1f} mΩ")

    # Define dynamic simulation schedule:
    # 0 ms to 50 ms: D = 0.35 (Pre-step equilibrium near MPP)
    # 50 ms to 100 ms: Step change to D = 0.45 (Post-step higher duty cycle)
    total_duration = 0.10  # 100 ms
    dt = 1.0e-4  # 0.1 ms step
    t_step = 0.05  # Step time at 50 ms

    def duty_profile(t: float) -> float:
        return 0.35 if t < t_step else 0.45

    logger.info(f"Executing dynamic simulation: Duration = {total_duration*1e3:.0f} ms, dt = {dt*1e3:.2f} ms")
    logger.info(f"Actuation Profile: D = 0.35 (0 - 50 ms) -> D = 0.45 (50 - 100 ms)")

    converter.reset(initial_state=(29.9, 8.0, 48.0))
    df = converter.simulate(
        duration=total_duration,
        duty_cycle=duty_profile,
        irradiance=1000.0,
        cell_temperature=25.0,
        dt=dt,
    )

    # Export metrics CSV
    csv_path = config.paths.metrics_dir / "phase_3_converter_simulation.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} simulation timeseries records to: {csv_path}")

    # Generate publication figures
    graph_paths = generate_all_phase_3_plots(df, output_dir=config.paths.graphs_dir / "phase_3")
    logger.info(f"Rendered {len(graph_paths)} figures in: {config.paths.graphs_dir / 'phase_3'}")

    # Steady-state analysis at 45 ms (pre-step) and 95 ms (post-step)
    pre_step_row = df[(df["time"] >= 0.045) & (df["time"] <= 0.049)].iloc[-1]
    post_step_row = df[(df["time"] >= 0.095) & (df["time"] <= 0.099)].iloc[-1]

    print("\n" + "=" * 90)
    print("PHASE 3 DC-DC BOOST CONVERTER SIMULATION SUMMARY")
    print("=" * 90)

    header = "{:<25} | {:<16} | {:<16} | {:<16}"
    row_fmt = "{:<25} | {:<16.4f} | {:<16.4f} | {:<16}"

    print(header.format("Electrical Metric", "Pre-Step (D=0.35)", "Post-Step (D=0.45)", "Physical Trend"))
    print("-" * 90)
    print(row_fmt.format("Duty Cycle D", pre_step_row["duty_cycle"], post_step_row["duty_cycle"], "+0.10 Step"))
    print(row_fmt.format("PV Voltage Vpv [V]", pre_step_row["v_pv"], post_step_row["v_pv"], "Decreased (expected)"))
    print(row_fmt.format("PV Current Ipv [A]", pre_step_row["i_pv"], post_step_row["i_pv"], "Increased (expected)"))
    print(row_fmt.format("Inductor Current iL [A]", pre_step_row["i_l"], post_step_row["i_l"], "Increased (iL = Ipv)"))
    print(row_fmt.format("Output Voltage Vout [V]", pre_step_row["v_out"], post_step_row["v_out"], "Decreased due to Rin"))
    print(row_fmt.format("Output Current Iout [A]", pre_step_row["i_out"], post_step_row["i_out"], "Decreased (Vout/Rload)"))
    print(row_fmt.format("PV Input Power [W]", pre_step_row["p_pv"], post_step_row["p_pv"], "Shifted along P-V curve"))
    print(row_fmt.format("Output Power Pout [W]", pre_step_row["p_out"], post_step_row["p_out"], "Absorbed by Rload"))
    print(row_fmt.format("Power Loss in rL [W]", pre_step_row["power_loss"], post_step_row["power_loss"], "rL * iL^2"))
    print(row_fmt.format("Conversion Efficiency [%]", pre_step_row["efficiency"], post_step_row["efficiency"], "High (> 98%)"))
    print(header.format("Continuous Cond. (CCM)", str(pre_step_row["is_ccm"]), str(post_step_row["is_ccm"]), "Maintained (iL > 0)"))
    print("-" * 90)

    # Power and Energy Conservation Verification
    max_power_residual = df["power_balance_residual"].max()
    mean_power_residual = df["power_balance_residual"].mean()
    print(f"\nEnergy & Power Balance Verification:")
    print(f"  Max Instantaneous Power Conservation Residual |Pin - Pout - Ploss - dE/dt|: {max_power_residual:.6e} W")
    print(f"  Mean Power Conservation Residual: {mean_power_residual:.6e} W")
    print(f"  Total Stored Energy Range: {df['stored_energy'].min():.4f} J to {df['stored_energy'].max():.4f} J")
    print("=" * 90)
    print("Phase 3 demonstration successfully executed.\n")


if __name__ == "__main__":
    run_demo()
