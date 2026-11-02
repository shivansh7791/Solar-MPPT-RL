"""Centralized Configuration Architecture for RL-Based Solar MPPT Project.

Defines modular dataclasses for simulation defaults, PV array parameters,
boost converter parameters, MPPT controllers, environment settings, and
reinforcement learning hyperparameters.

In Phase 0, foundational simulation defaults and architectural schemas are
established without prematurely hardcoding component values reserved for
future engineering design phases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SimulationConfig:
    """Core simulation parameters and numerical defaults."""

    seed: int = 42
    time_step: float = 1e-4  # Default simulation step dt [seconds]
    total_duration: float = 1.0  # Default simulation duration [seconds]
    log_level: str = "INFO"


@dataclass
class PVConfig:
    """Placeholder configuration schema for PV array models.

    Detailed physical ratings (e.g., Voc, Isc, Vmp, Imp, temp coefficients)
    will be established and mathematically justified in Phase 1.
    """

    panel_model: str = "GENERIC_PV"
    num_series_modules: int = 1
    num_parallel_strings: int = 1
    stc_irradiance: float = 1000.0  # W/m^2
    stc_temperature: float = 25.0  # °C
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConverterConfig:
    """Placeholder configuration schema for DC-DC Boost Converter.

    Component ratings (inductance L, input/output capacitance C_in/C_out,
    load resistance R_L, and switching frequency f_sw) will be derived in Phase 2.
    """

    switching_frequency: float = 20_000.0  # Default 20 kHz [Hz]
    duty_cycle_min: float = 0.05
    duty_cycle_max: float = 0.95
    duty_cycle_init: float = 0.50
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MPPTConfig:
    """Placeholder configuration schema for MPPT controllers.

    Perturb & Observe step sizes and sampling frequencies will be defined in Phase 3.
    """

    algorithm: str = "PO"  # 'PO', 'RL_SAC', etc.
    sampling_interval: float = 1e-3  # Control loop interval [seconds]
    delta_duty: float = 0.01  # Perturbation step for P&O
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    """Placeholder configuration schema for the Gymnasium RL Environment.

    State observation features, action scaling, and reward functions will be
    formalized in Phase 4.
    """

    max_steps_per_episode: int = 1000
    action_type: str = "continuous"  # Continuous delta-duty or absolute duty
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLConfig:
    """Placeholder configuration schema for Soft Actor-Critic (SAC) agent.

    Hyperparameters (learning rate, buffer size, gamma, tau, batch size)
    will be tuned and validated in Phase 5.
    """

    algorithm: str = "SAC"
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: str = "auto"
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectPaths:
    """Directory paths for the project repository."""

    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def results_dir(self) -> Path:
        return self.root_dir / "results"

    @property
    def graphs_dir(self) -> Path:
        return self.results_dir / "graphs"

    @property
    def metrics_dir(self) -> Path:
        return self.results_dir / "metrics"

    @property
    def logs_dir(self) -> Path:
        return self.results_dir / "experiment_logs"


@dataclass
class MasterConfig:
    """Aggregated project configuration container."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    pv: PVConfig = field(default_factory=PVConfig)
    converter: ConverterConfig = field(default_factory=ConverterConfig)
    mppt: MPPTConfig = field(default_factory=MPPTConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    paths: ProjectPaths = field(default_factory=ProjectPaths)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration hierarchy into a nested dictionary."""
        data = asdict(self)
        # Convert Path objects to string representations
        data["paths"] = {
            "root_dir": str(self.paths.root_dir),
            "data_dir": str(self.paths.data_dir),
            "profiles_dir": str(self.paths.profiles_dir),
            "models_dir": str(self.paths.models_dir),
            "results_dir": str(self.paths.results_dir),
            "graphs_dir": str(self.paths.graphs_dir),
            "metrics_dir": str(self.paths.metrics_dir),
            "logs_dir": str(self.paths.logs_dir),
        }
        return data


def get_default_config() -> MasterConfig:
    """Instantiate and return the project default configuration."""
    return MasterConfig()
