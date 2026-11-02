"""Main entry point for RL-Based Solar MPPT Project.

Serves as the high-level orchestration interface. In Phase 0, it validates
project configuration, logging setup, and foundational environment readiness.
"""

from __future__ import annotations

import argparse
import sys

from app.utils.logger import get_logger, setup_logger
from config import get_default_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Design and Performance Analysis of an RL-Based MPPT Controller for Solar PV Systems"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Perform a foundational environment and configuration health check.",
    )
    parser.add_argument(
        "--test-pv",
        action="store_true",
        help="Execute single-diode PV model at STC and display electrical outputs.",
    )
    parser.add_argument(
        "--analyze-pv",
        action="store_true",
        help="Generate I-V/P-V characteristics, export metrics CSV, and generate publication plots.",
    )
    parser.add_argument(
        "--simulate-converter",
        action="store_true",
        help="Simulate DC-DC boost converter dynamic response to duty cycle step.",
    )
    parser.add_argument(
        "--simulate-po",
        action="store_true",
        help="Simulate closed-loop Perturb & Observe (P&O) MPPT controller under STC.",
    )
    parser.add_argument(
        "--validate-env",
        action="store_true",
        help="Validate custom Gymnasium Solar MPPT environment and run baseline benchmark.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute main application entry point."""
    args = parse_args()
    logger = setup_logger("mppt_main", level=args.log_level)

    logger.info("Initializing RL-Based Solar MPPT Project...")

    config = get_default_config()
    logger.info(f"Loaded Master Configuration with random seed: {config.simulation.seed}")
    logger.info(f"Configured PV Module: {config.pv.manufacturer} {config.pv.model_name}")
    logger.info(f"Default simulation time step dt: {config.simulation.time_step} s")
    logger.info(f"Root directory resolved to: {config.paths.root_dir}")

    if args.check_env:
        logger.info("Environment check completed successfully: All foundational subpackages operational.")

    if args.test_pv:
        from app.pv import SolarPVModel
        pv_model = SolarPVModel(config.pv)
        stc = pv_model.get_stc_parameters()
        logger.info(f"PV STC Evaluation ({config.pv.model_name}):")
        logger.info(f"  Pmp = {stc.p_mp:.3f} W (Ref: {config.pv.stc_p_mp:.3f} W)")
        logger.info(f"  Vmp = {stc.v_mp:.3f} V (Ref: {config.pv.stc_v_mp:.3f} V)")
        logger.info(f"  Imp = {stc.i_mp:.3f} A (Ref: {config.pv.stc_i_mp:.3f} A)")
        logger.info(f"  Voc = {stc.v_oc:.3f} V (Ref: {config.pv.stc_v_oc:.3f} V)")
        logger.info(f"  Isc = {stc.i_sc:.3f} A (Ref: {config.pv.stc_i_sc:.3f} A)")

    if args.analyze_pv:
        from app.pv import PVCharacteristicAnalyzer, SolarPVModel
        from app.pv.pv_visualization import generate_all_phase_2_plots

        logger.info("Executing Phase 2 PV Characteristic Analysis & Sweep...")
        analyzer = PVCharacteristicAnalyzer(SolarPVModel(config.pv))

        scenarios = [
            (200.0, 25.0),
            (400.0, 25.0),
            (600.0, 25.0),
            (800.0, 25.0),
            (1000.0, 25.0),
            (1000.0, 15.0),
            (1000.0, 35.0),
            (1000.0, 45.0),
            (1000.0, 55.0),
            (800.0, 30.0),
            (600.0, 35.0),
            (400.0, 45.0),
            (250.0, 10.0),
            (1000.0, 5.0),
        ]
        curves = analyzer.analyze_environmental_scenarios(scenarios)
        df = analyzer.curves_to_mpp_dataframe(curves)
        df = df.drop_duplicates(subset=["irradiance", "cell_temperature"]).sort_values(
            by=["irradiance", "cell_temperature"], ascending=[False, True]
        )

        csv_path = config.paths.metrics_dir / "phase_2_mpp_analysis.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} MPP analysis records to: {csv_path}")

        plot_paths = generate_all_phase_2_plots(
            output_dir=config.paths.graphs_dir, analyzer=analyzer
        )
        logger.info(f"Generated {len(plot_paths)} publication figures in: {config.paths.graphs_dir}")

    if args.simulate_converter:
        from app.converter import BoostConverter
        from app.converter.converter_visualization import generate_all_phase_3_plots

        logger.info("Executing Phase 3 DC-DC Boost Converter Simulation...")
        converter = BoostConverter()

        duty_profile = lambda t: 0.35 if t < 0.05 else 0.45
        df = converter.simulate(duration=0.10, duty_cycle=duty_profile, dt=converter.default_dt)

        csv_path = config.paths.metrics_dir / "phase_3_converter_simulation.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} simulation steps to: {csv_path}")

        # Compute steady states
        pre_idx = int(0.045 / converter.default_dt)
        post_idx = len(df) - 1
        s1 = df.iloc[pre_idx]
        s2 = df.iloc[post_idx]

        logger.info("Phase 3 Simulation Results:")
        logger.info(f"  Steady State 1 (D = {s1['duty_cycle']:.2f}):")
        logger.info(f"    V_pv = {s1['v_pv']:.2f} V | I_pv = {s1['i_pv']:.2f} A | P_pv = {s1['p_pv']:.2f} W")
        logger.info(f"    V_out = {s1['v_out']:.2f} V | I_out = {s1['i_out']:.2f} A | P_out = {s1['p_out']:.2f} W")
        logger.info(f"    Efficiency = {s1['efficiency']:.2f} % | CCM Mode: {s1['is_ccm']}")
        logger.info(f"  Steady State 2 (D = {s2['duty_cycle']:.2f}):")
        logger.info(f"    V_pv = {s2['v_pv']:.2f} V | I_pv = {s2['i_pv']:.2f} A | P_pv = {s2['p_pv']:.2f} W")
        logger.info(f"    V_out = {s2['v_out']:.2f} V | I_out = {s2['i_out']:.2f} A | P_out = {s2['p_out']:.2f} W")
        logger.info(f"    Efficiency = {s2['efficiency']:.2f} % | CCM Mode: {s2['is_ccm']}")

        plots_dir = config.paths.graphs_dir / "phase_3"
        plot_paths = generate_all_phase_3_plots(df, output_dir=plots_dir)
        logger.info(f"Generated {len(plot_paths)} converter transient plots in: {plots_dir}")

    if args.simulate_po:
        from app.controllers.controller_visualization import generate_all_phase_4_plots
        from app.controllers.mppt_metrics import calculate_mppt_metrics
        from app.controllers.mppt_simulation import MPPTSimulator

        logger.info("Executing Phase 4 Perturb & Observe (P&O) MPPT Simulation...")
        sim = MPPTSimulator(config=config)
        df = sim.run(duration=1.5, irradiance=1000.0, cell_temperature=25.0, initial_duty=0.35)
        metrics = calculate_mppt_metrics(df, pv_model=sim.pv_model)

        csv_path = config.paths.metrics_dir / "phase_4_po_simulation.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} MPPT simulation steps to: {csv_path}")

        logger.info("Phase 4 P&O MPPT Performance Metrics (STC):")
        logger.info(f"  Tracking Efficiency (eta_MPPT): {metrics.tracking_efficiency:.2f} %")
        logger.info(f"  Settling Time (2% band): {metrics.settling_time:.4f} s | Converged: {metrics.is_converged}")
        logger.info(f"  Extracted Power: {metrics.final_p_pv:.2f} W | Reference P_MPP: {metrics.final_p_mpp:.2f} W")
        logger.info(f"  Final Duty Cycle: {metrics.final_duty_cycle:.4f} | Theoretical D*: 0.3939")
        logger.info(
            f"  Steady-State Power: {metrics.steady_state_power_mean:.2f} W | "
            f"Std: {metrics.steady_state_power_std:.3f} W | Ripple: {metrics.steady_state_power_ripple:.3f} W"
        )
        logger.info(f"  Mean Converter Efficiency: {metrics.mean_converter_efficiency:.2f} %")
        logger.info(f"  MPPT Controller Updates: {metrics.total_controller_updates} | Saturation Events: {metrics.total_saturation_events}")

        plots_dir = config.paths.graphs_dir / "phase_4"
        plot_paths = generate_all_phase_4_plots(df, output_dir=plots_dir)
        logger.info(f"Generated {len(plot_paths)} P&O controller plots in: {plots_dir}")

    if args.validate_env:
        from examples.validate_rl_environment import main as run_env_validation

        logger.info("Executing Phase 5 Custom Gymnasium RL Environment Validation...")
        run_env_validation()

    logger.info("Operational status: PHASE 5 CUSTOM GYMNASIUM SOLAR MPPT ENVIRONMENT OPERATIONAL.")
    return 0




if __name__ == "__main__":
    sys.exit(main())
