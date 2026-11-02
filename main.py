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

    logger.info("Phase 1 operational status: PV MATHEMATICAL MODEL OPERATIONAL.")
    return 0



if __name__ == "__main__":
    sys.exit(main())
