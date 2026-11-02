"""Phase 0 Smoke Tests: Foundation, Imports, Configuration, and Logging.

Verifies the baseline operational readiness of the project infrastructure
without claiming any electrical or RL validation.
"""

import importlib
import logging
import pytest


def test_third_party_dependencies_import():
    """Verify all project third-party libraries import cleanly."""
    required_libraries = [
        "numpy",
        "scipy",
        "pandas",
        "pvlib",
        "gymnasium",
        "stable_baselines3",
        "torch",
        "matplotlib",
        "plotly",
        "streamlit",
        "pytest",
    ]
    for lib in required_libraries:
        module = importlib.import_module(lib)
        assert module is not None, f"Failed to import core dependency: {lib}"


def test_internal_packages_import():
    """Verify all internal application subpackages can be imported."""
    internal_packages = [
        "app",
        "app.pv",
        "app.converter",
        "app.controllers",
        "app.environment",
        "app.training",
        "app.experiments",
        "app.dashboard",
        "app.utils",
        "config",
    ]
    for pkg in internal_packages:
        module = importlib.import_module(pkg)
        assert module is not None, f"Failed to import internal package: {pkg}"


def test_default_configuration_structure():
    """Verify configuration architecture initializes with expected defaults."""
    from config import MasterConfig, get_default_config

    config = get_default_config()
    assert isinstance(config, MasterConfig)
    assert config.simulation.seed == 42
    assert config.simulation.time_step > 0
    assert config.pv.stc_irradiance == 1000.0
    assert config.pv.stc_temperature == 25.0
    assert config.converter.switching_frequency == 20000.0

    # Test dictionary export
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "simulation" in config_dict
    assert "pv" in config_dict
    assert "converter" in config_dict
    assert "paths" in config_dict


def test_project_paths_resolution():
    """Verify that ProjectPaths correctly maps to existing project directories."""
    from config import ProjectPaths

    paths = ProjectPaths()
    assert paths.root_dir.exists(), "Root directory path does not exist"
    assert paths.data_dir.exists(), "Data directory path does not exist"
    assert paths.profiles_dir.exists(), "Profiles directory path does not exist"
    assert paths.models_dir.exists(), "Models directory path does not exist"
    assert paths.results_dir.exists(), "Results directory path does not exist"
    assert paths.graphs_dir.exists(), "Graphs directory path does not exist"
    assert paths.metrics_dir.exists(), "Metrics directory path does not exist"
    assert paths.logs_dir.exists(), "Logs directory path does not exist"


def test_logger_initialization():
    """Verify logger setup, message handling, and log levels."""
    from app.utils.logger import get_logger, setup_logger

    logger = setup_logger("test_smoke_logger", level=logging.DEBUG)
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) >= 1

    # Verify logging does not raise exceptions
    logger.debug("Smoke test debug log message.")
    logger.info("Smoke test info log message.")

    retrieved_logger = get_logger("test_smoke_logger")
    assert retrieved_logger is logger
