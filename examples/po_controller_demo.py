"""Phase 4 Demonstration: Closed-Loop Perturb & Observe (P&O) MPPT Control.

Executes all 5 required Phase 4 engineering experiments:
1. Experiment 1: STC Baseline Tracking (1000 W/m^2, 25°C, 1.5s).
2. Experiment 2: Dynamic Irradiance Step Changes (1000 -> 600 -> 800 -> 400 W/m^2).
3. Experiment 3: Dynamic Temperature Step Changes (25 -> 35 -> 45 -> 25°C).
4. Experiment 4: Perturbation Step Size Comparison (delta_D in {0.001, 0.002, 0.005}).
5. Experiment 5: Initial Duty Cycle Sensitivity (D_0 in {0.20, 0.30, 0.50, 0.60}).

Saves simulation metrics CSVs in results/metrics/ and publication figures in results/graphs/phase_4/.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.controllers import PerturbAndObserveController
from app.controllers.controller_visualization import generate_all_phase_4_plots
from app.controllers.mppt_metrics import MPPTMetrics, calculate_mppt_metrics
from app.controllers.mppt_simulation import MPPTSimulator
from app.converter import BoostConverter
from app.pv import SolarPVModel
from app.utils.logger import setup_logger
from config import get_default_config


def run_experiment_1_stc(sim: MPPTSimulator, config) -> Tuple[pd.DataFrame, MPPTMetrics]:
    """Experiment 1: STC Baseline MPPT Tracking (G=1000 W/m^2, T=25°C)."""
    logger = setup_logger("po_demo")
    logger.info("Executing Experiment 1: STC Baseline MPPT Tracking...")

    # Start at D_0 = 0.35 to verify convergence to D* = 0.3939
    df = sim.run(duration=1.5, irradiance=1000.0, cell_temperature=25.0, initial_duty=0.35)
    metrics = calculate_mppt_metrics(df, pv_model=sim.pv_model)

    logger.info(f"  Tracking Efficiency: {metrics.tracking_efficiency:.2f} %")
    logger.info(f"  Settling Time (2% band): {metrics.settling_time:.4f} s")
    logger.info(f"  Steady-State Power: {metrics.steady_state_power_mean:.2f} +/- {metrics.steady_state_power_std:.3f} W (Ripple: {metrics.steady_state_power_ripple:.3f} W)")
    logger.info(f"  Final Duty Cycle: {metrics.final_duty_cycle:.4f} (Ideal D*: 0.3939)")
    return df, metrics


def run_experiment_2_irradiance(sim: MPPTSimulator) -> Tuple[pd.DataFrame, MPPTMetrics]:
    """Experiment 2: Dynamic Irradiance Step Changes."""
    logger = setup_logger("po_demo")
    logger.info("Executing Experiment 2: Dynamic Irradiance Variations (1000 -> 600 -> 800 -> 400 W/m^2)...")

    def g_profile(t: float) -> float:
        if t < 0.5:
            return 1000.0
        elif t < 1.0:
            return 600.0
        elif t < 1.5:
            return 800.0
        else:
            return 400.0

    df = sim.run(duration=2.0, irradiance=g_profile, cell_temperature=25.0, initial_duty=0.35)
    metrics = calculate_mppt_metrics(df, pv_model=sim.pv_model)
    logger.info(f"  Tracking Efficiency across dynamic irradiance: {metrics.tracking_efficiency:.2f} %")
    logger.info(f"  Cumulative Energy Loss: {metrics.cumulative_energy_loss:.2f} J")
    return df, metrics


def run_experiment_3_temperature(sim: MPPTSimulator) -> Tuple[pd.DataFrame, MPPTMetrics]:
    """Experiment 3: Dynamic Temperature Step Changes."""
    logger = setup_logger("po_demo")
    logger.info("Executing Experiment 3: Dynamic Temperature Variations (25 -> 35 -> 45 -> 25°C)...")

    def t_profile(t: float) -> float:
        if t < 0.5:
            return 25.0
        elif t < 1.0:
            return 35.0
        elif t < 1.5:
            return 45.0
        else:
            return 25.0

    df = sim.run(duration=2.0, irradiance=1000.0, cell_temperature=t_profile, initial_duty=0.35)
    metrics = calculate_mppt_metrics(df, pv_model=sim.pv_model)
    logger.info(f"  Tracking Efficiency across dynamic temperature: {metrics.tracking_efficiency:.2f} %")
    logger.info(f"  Final PV Voltage: {metrics.final_v_pv:.2f} V (Ref: {metrics.final_v_mpp:.2f} V)")
    return df, metrics


def run_experiment_4_step_sizes(config) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Experiment 4: Comparison across perturbation step sizes delta_D in {0.001, 0.002, 0.005}."""
    logger = setup_logger("po_demo")
    logger.info("Executing Experiment 4: Perturbation Step Size Comparison...")

    step_sizes = [0.001, 0.002, 0.005]
    comp_dfs: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, Any]] = []

    for delta_d in step_sizes:
        ctrl = PerturbAndObserveController(
            initial_duty=0.35,
            perturbation_step=delta_d,
            min_duty=0.05,
            max_duty=0.90,
        )
        sim = MPPTSimulator(controller=ctrl, config=config)
        df = sim.run(duration=1.5, irradiance=1000.0, cell_temperature=25.0, initial_duty=0.35)
        metrics = calculate_mppt_metrics(df, pv_model=sim.pv_model)

        label = str(delta_d)
        comp_dfs[label] = df

        summary_rows.append({
            "perturbation_step": delta_d,
            "tracking_efficiency_pct": round(metrics.tracking_efficiency, 3),
            "settling_time_s": round(metrics.settling_time, 4) if metrics.settling_time else None,
            "steady_state_power_mean_w": round(metrics.steady_state_power_mean, 3),
            "steady_state_power_std_w": round(metrics.steady_state_power_std, 4),
            "steady_state_power_ripple_w": round(metrics.steady_state_power_ripple, 4),
            "steady_state_duty_std": round(metrics.steady_state_duty_std, 5),
            "steady_state_duty_ripple": round(metrics.steady_state_duty_ripple, 4),
            "cumulative_energy_loss_j": round(metrics.cumulative_energy_loss, 3),
        })

        logger.info(
            f"  delta_D = {delta_d:.3f} -> Eff: {metrics.tracking_efficiency:.2f}%, "
            f"Settling: {metrics.settling_time}s, Ripple: {metrics.steady_state_power_ripple:.3f}W"
        )

    summary_df = pd.DataFrame(summary_rows)
    return comp_dfs, summary_df


def run_experiment_5_initial_duty(config) -> pd.DataFrame:
    """Experiment 5: Initial Duty Cycle Sensitivity (D_0 in {0.20, 0.30, 0.50, 0.60})."""
    logger = setup_logger("po_demo")
    logger.info("Executing Experiment 5: Initial Duty Cycle Sensitivity...")

    initial_duties = [0.20, 0.30, 0.50, 0.60]
    rows: List[Dict[str, Any]] = []

    for d0 in initial_duties:
        sim = MPPTSimulator(config=config)
        df = sim.run(duration=1.5, irradiance=1000.0, cell_temperature=25.0, initial_duty=d0)
        metrics = calculate_mppt_metrics(df, pv_model=sim.pv_model)

        rows.append({
            "initial_duty": d0,
            "converged": metrics.is_converged,
            "settling_time_s": round(metrics.settling_time, 4) if metrics.settling_time else None,
            "tracking_efficiency_pct": round(metrics.tracking_efficiency, 3),
            "final_duty_cycle": round(metrics.final_duty_cycle, 4),
            "final_power_w": round(metrics.final_p_pv, 2),
            "saturation_events": metrics.total_saturation_events,
        })
        logger.info(
            f"  D_0 = {d0:.2f} -> Settling: {metrics.settling_time}s, Final D: {metrics.final_duty_cycle:.4f}, "
            f"Final P: {metrics.final_p_pv:.2f}W, Converged: {metrics.is_converged}"
        )

    return pd.DataFrame(rows)


def main() -> int:
    """Execute all Phase 4 demonstrations and export verified artifacts."""
    logger = setup_logger("po_demo")
    logger.info("======================================================================")
    logger.info("Starting Phase 4 P&O MPPT Controller Demonstration & Evaluation")
    logger.info("======================================================================")

    config = get_default_config()
    sim = MPPTSimulator(config=config)

    # 1. Run Experiments
    df_stc, m_stc = run_experiment_1_stc(sim, config)
    df_irrad, m_irrad = run_experiment_2_irradiance(sim)
    df_temp, m_temp = run_experiment_3_temperature(sim)
    comp_dfs, step_comp_df = run_experiment_4_step_sizes(config)
    init_d_df = run_experiment_5_initial_duty(config)

    # 2. Save Simulation Timeseries CSV
    metrics_dir = config.paths.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)

    po_csv_path = metrics_dir / "phase_4_po_simulation.csv"
    df_stc.to_csv(po_csv_path, index=False)
    logger.info(f"Saved STC timeseries ({len(df_stc)} steps) to: {po_csv_path}")

    # 3. Save Step Size Comparison CSV
    step_csv_path = metrics_dir / "phase_4_step_size_comparison.csv"
    step_comp_df.to_csv(step_csv_path, index=False)
    logger.info(f"Saved perturbation comparison summary to: {step_csv_path}")

    # 4. Save Overall Experiment Summary CSV
    summary_rows = [
        {"experiment": "Exp 1: STC Baseline Tracking", "condition": "1000 W/m^2, 25°C", "tracking_efficiency_pct": round(m_stc.tracking_efficiency, 2), "settling_time_s": round(m_stc.settling_time, 4) if m_stc.settling_time else None, "steady_state_ripple_w": round(m_stc.steady_state_power_ripple, 3), "final_duty": round(m_stc.final_duty_cycle, 4)},
        {"experiment": "Exp 2: Dynamic Irradiance Steps", "condition": "1000->600->800->400 W/m^2", "tracking_efficiency_pct": round(m_irrad.tracking_efficiency, 2), "settling_time_s": round(m_irrad.settling_time, 4) if m_irrad.settling_time else None, "steady_state_ripple_w": round(m_irrad.steady_state_power_ripple, 3), "final_duty": round(m_irrad.final_duty_cycle, 4)},
        {"experiment": "Exp 3: Dynamic Temperature Steps", "condition": "25->35->45->25°C", "tracking_efficiency_pct": round(m_temp.tracking_efficiency, 2), "settling_time_s": round(m_temp.settling_time, 4) if m_temp.settling_time else None, "steady_state_ripple_w": round(m_temp.steady_state_power_ripple, 3), "final_duty": round(m_temp.final_duty_cycle, 4)},
    ]
    summary_df = pd.DataFrame(summary_rows)
    exp_summary_path = metrics_dir / "phase_4_experiment_summary.csv"
    summary_df.to_csv(exp_summary_path, index=False)
    logger.info(f"Saved experiment summary to: {exp_summary_path}")

    # 5. Generate Visualizations (use dynamic irradiance dataset for multi-panel overview)
    graphs_dir = config.paths.graphs_dir / "phase_4"
    generated_plots = generate_all_phase_4_plots(
        df=df_irrad,
        step_comparison_dfs=comp_dfs,
        output_dir=graphs_dir,
    )
    logger.info(f"Generated {len(generated_plots)} figures in: {graphs_dir}")

    logger.info("======================================================================")
    logger.info("Phase 4 P&O MPPT Demonstrations Completed Successfully.")
    logger.info("======================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
