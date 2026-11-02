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
        "--train-sac",
        action="store_true",
        help="Train Soft Actor-Critic (SAC) reinforcement learning MPPT agent.",
    )
    parser.add_argument(
        "--compare-mppt",
        action="store_true",
        help="Execute Phase 7 comparative performance study between Classical P&O and Trained SAC MPPT controllers.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps for SAC agent.",
    )
    parser.add_argument(
        "--rl-mode",
        type=str,
        default="realistic",
        choices=["realistic", "privileged"],
        help="SAC observation mode: realistic (8-dim) or privileged (10-dim with MPP reference).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run short 2,000 timestep SAC training smoke test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
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

    if args.train_sac:
        from app.rl.sac_config import SACConfig
        from app.rl.training import run_sac_training

        steps = 2_000 if args.smoke_test else (args.timesteps if args.timesteps is not None else 100_000)
        s_cfg = SACConfig(
            total_timesteps=steps,
            seed=args.seed,
            learning_starts=min(2_000, max(100, steps // 10)),
        )

        logger.info(f"Executing Phase 6 SAC Agent Training ({steps:,} steps, mode={args.rl_mode})...")
        summary = run_sac_training(
            sac_config=s_cfg,
            master_config=config,
            mode=args.rl_mode,
            seed=args.seed,
        )
        logger.info(
            f"Phase 6 SAC Training Complete: {summary['total_timesteps_completed']:,} steps in {summary['duration_seconds']}s "
            f"({summary['throughput_steps_per_second']} steps/s). Best Tracking Eff: {summary['best_tracking_efficiency_achieved']}%"
        )

    if args.compare_mppt:
        from pathlib import Path
        from app.evaluation.comparison_runner import ComparisonRunner
        from app.evaluation.controller_adapters import POControllerAdapter, SACControllerAdapter
        from app.evaluation.result_exporter import ResultExporter
        from app.evaluation.scenario_manager import ScenarioManager, create_standard_scenarios
        from app.evaluation.statistical_analysis import perform_statistical_analysis
        from app.evaluation.visualization import generate_all_phase_7_plots
        from app.rl.sac_agent import load_sac_agent

        logger.info("================================================================================")
        logger.info("STARTING PHASE 7: MPPT COMPARATIVE PERFORMANCE EVALUATION (SAC vs Classical P&O)")
        logger.info("================================================================================")

        model_path = config.paths.root_dir / "artifacts/rl/realistic/best_model/best_mppt_model.zip"
        if not model_path.is_file():
            model_path = config.paths.root_dir / "artifacts/rl/realistic/final_model/sac_mppt_final.zip"

        if not model_path.is_file():
            logger.error(f"Cannot run Phase 7: No trained realistic SAC model found at {model_path}")
            return 1

        logger.info(f"Loading trained realistic SAC model from: {model_path}")
        sac_model = load_sac_agent(model_path, device="cpu")

        po_adapter = POControllerAdapter(config=config.mppt.po)
        sac_adapter = SACControllerAdapter(model=sac_model)

        scenarios = create_standard_scenarios()
        manager = ScenarioManager(scenarios)
        logger.info(f"Initialized {len(manager)} standardized evaluation scenarios.")

        runner = ComparisonRunner(config=config)
        paired_results = runner.run_all_scenarios(manager, po_adapter, sac_adapter)

        logger.info("Performing formal statistical hypothesis testing...")
        stats_summary = perform_statistical_analysis(paired_results)

        metrics_out = config.paths.metrics_dir / "phase_7"
        exporter = ResultExporter(metrics_out)
        export_paths = exporter.export_all(manager, paired_results, stats_summary)

        graphs_out = config.paths.graphs_dir / "phase_7"
        plot_paths = generate_all_phase_7_plots(paired_results, stats_summary, graphs_out)

        logger.info("================================================================================")
        logger.info("PHASE 7 COMPARATIVE STUDY COMPLETE")
        logger.info("================================================================================")
        logger.info(f"Arithmetic Mean Scenario Efficiency: SAC = {stats_summary.overall_mean_sac_efficiency:.2f}% | P&O = {stats_summary.overall_mean_po_efficiency:.2f}% (Gain: {stats_summary.overall_mean_efficiency_gain:+.2f}%)")
        logger.info(f"Energy-Weighted Aggregate Efficiency: SAC = {stats_summary.overall_energy_weighted_sac_efficiency:.2f}% | P&O = {stats_summary.overall_energy_weighted_po_efficiency:.2f}% (Gain: {stats_summary.overall_energy_weighted_efficiency_gain:+.2f}%)")
        logger.info(f"Total Energy Extracted across 19 Scenarios: SAC = {stats_summary.total_sac_energy_harvested_joules:.1f} J | P&O = {stats_summary.total_po_energy_harvested_joules:.1f} J (Net Gain: {stats_summary.total_energy_gain_joules:+.1f} J / Total Ideal: {stats_summary.total_ideal_energy_joules:.1f} J)")
        logger.info(f"Paired t-test on Efficiency Gain: t = {stats_summary.efficiency_stats.t_statistic:.3f}, p = {stats_summary.efficiency_stats.t_test_p_value:.3e} (Significant: {stats_summary.efficiency_stats.is_statistically_significant_05})")
        logger.info(f"Wilcoxon Signed-Rank test: W = {stats_summary.efficiency_stats.wilcoxon_statistic:.1f}, p = {stats_summary.efficiency_stats.wilcoxon_p_value:.3e}")
        logger.info(f"Steady-State Power Ripple: SAC Mean = {stats_summary.ripple_stats_watts.mean_difference + stats_summary.category_summaries['A_STC']['mean_po_ripple_w']:.4f} W vs P&O Mean = {stats_summary.category_summaries['A_STC']['mean_po_ripple_w']:.4f} W")
        logger.info(f"Exported metrics: {metrics_out}")
        logger.info(f"Generated 14 publication figures: {graphs_out}")

    logger.info("Operational status: PHASE 7 MPPT CONTROLLER COMPARATIVE STUDY OPERATIONAL.")
    return 0




if __name__ == "__main__":
    sys.exit(main())
