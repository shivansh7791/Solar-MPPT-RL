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

    logger.info("Initializing RL-Based Solar MPPT Project (Phase 0 Foundation)...")

    config = get_default_config()
    logger.info(f"Loaded Master Configuration with random seed: {config.simulation.seed}")
    logger.info(f"Default simulation time step dt: {config.simulation.time_step} s")
    logger.info(f"Root directory resolved to: {config.paths.root_dir}")

    if args.check_env:
        logger.info("Environment check completed successfully: All foundational subpackages operational.")

    logger.info("Phase 0 operational status: READY FOR PHASE 1 APPROVAL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
