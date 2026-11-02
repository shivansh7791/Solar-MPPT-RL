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
from typing import Any, Dict, Optional, Tuple


@dataclass
class SimulationConfig:
    """Core simulation parameters and numerical defaults."""

    seed: int = 42
    time_step: float = 1e-4  # Default simulation step dt [seconds]
    total_duration: float = 1.0  # Default simulation duration [seconds]
    log_level: str = "INFO"


@dataclass
class PVConfig:
    """Configuration for PV module and array based on Canadian Solar CS6P-240P.

    Parameters derived from the California Energy Commission (CEC) database via pvlib.
    """

    manufacturer: str = "Canadian Solar Inc."
    model_name: str = "CS6P-240P"
    panel_model: str = "Canadian_Solar_Inc__CS6P_240P"
    technology: str = "Multi-c-Si"
    num_cells_in_series: int = 60
    num_series_modules: int = 1
    num_parallel_strings: int = 1

    # Standard Test Condition (STC) Reference Values
    stc_irradiance: float = 1000.0  # W/m^2
    stc_temperature: float = 25.0  # °C

    # Reference STC Datasheet Ratings
    stc_i_sc: float = 8.59  # Short-circuit current [A]
    stc_v_oc: float = 37.0  # Open-circuit voltage [V]
    stc_v_mp: float = 29.9  # Voltage at maximum power point [V]
    stc_i_mp: float = 8.03  # Current at maximum power point [A]
    stc_p_mp: float = 240.097  # Maximum power [W]

    # Temperature Coefficients
    alpha_sc: float = 0.005472  # Temp coeff of Isc [A/°C]
    beta_oc: float = -0.135198  # Temp coeff of Voc [V/°C]
    gamma_r: float = -0.4538  # Temp coeff of Pmp [%/°C]
    t_noct: float = 44.1  # Nominal Operating Cell Temperature [°C]

    # Single-Diode CEC 5-Parameter Empirical Values (De Soto Model)
    a_ref: float = 1.577654  # Modified ideality factor [V]
    I_L_ref: float = 8.599262  # Light-generated current at STC [A]
    I_o_ref: float = 5.52853e-10  # Diode reverse saturation current at STC [A]
    R_s: float = 0.310448  # Series resistance [Ohm]
    R_sh_ref: float = 287.92276  # Shunt resistance at STC [Ohm]
    Adjust: float = 3.568598  # Temperature adjustment parameter

    custom_params: Dict[str, Any] = field(default_factory=dict)

    def get_cec_parameters(self) -> Dict[str, Any]:
        """Return parameters dictionary formatted for pvlib CEC single-diode functions."""
        return {
            "alpha_sc": self.alpha_sc,
            "a_ref": self.a_ref,
            "I_L_ref": self.I_L_ref,
            "I_o_ref": self.I_o_ref,
            "R_sh_ref": self.R_sh_ref,
            "R_s": self.R_s,
            "Adjust": self.Adjust,
            "N_s": self.num_cells_in_series,
        }

    def get_stc_reference_dict(self) -> Dict[str, float]:
        """Return reference STC electrical values."""
        return {
            "i_sc": self.stc_i_sc,
            "v_oc": self.stc_v_oc,
            "v_mp": self.stc_v_mp,
            "i_mp": self.stc_i_mp,
            "p_mp": self.stc_p_mp,
        }



@dataclass
class ConverterConfig:
    """Configuration for DC-DC Boost Converter matched to Canadian Solar CS6P-240P.

    Component ratings sized for Continuous Conduction Mode (CCM) operation
    with nominal 48V-50V DC bus voltage and 240W nominal throughput.
    """

    inductance: float = 1.0e-3  # Boost inductance L [H] (1.0 mH)
    input_capacitance: float = 470.0e-6  # Input decoupling capacitance C_pv [F] (470 uF)
    output_capacitance: float = 470.0e-6  # Output filter capacitance C_out [F] (470 uF)
    load_resistance: float = 10.0  # Output resistive load R_load [Ohm]
    inductor_resistance: float = 0.05  # Inductor series resistance r_L [Ohm] (50 mOhm)

    switching_frequency: float = 20_000.0  # Switching frequency f_sw [Hz] (20 kHz)
    duty_cycle_min: float = 0.05  # Minimum permitted duty cycle
    duty_cycle_max: float = 0.90  # Maximum permitted duty cycle
    duty_cycle_init: float = 0.35  # Nominal initial duty cycle

    time_step: float = 1.0e-4  # Converter simulation step dt [s] (0.1 ms)

    # Initial physical state conditions [V_pv, i_L, V_out]
    v_pv_init: float = 29.9  # Initial PV voltage [V]
    i_l_init: float = 8.0  # Initial inductor current [A]
    v_out_init: float = 48.0  # Initial output voltage [V]

    custom_params: Dict[str, Any] = field(default_factory=dict)

    def get_initial_state(self) -> Tuple[float, float, float]:
        """Return initial state vector [V_pv, i_L, V_out]."""
        return (self.v_pv_init, self.i_l_init, self.v_out_init)



@dataclass
class POConfig:
    """Configuration for Classical Perturb & Observe (P&O) MPPT Controller.

    Parameters
    ----------
    initial_duty : float
        Initial duty cycle D_0 commanded at start of control [0.05, 0.90].
    duty_min : float
        Minimum permitted duty cycle saturation bound.
    duty_max : float
        Maximum permitted duty cycle saturation bound.
    perturbation_step : float
        Fixed duty cycle perturbation increment delta_D (e.g. 0.002).
    sampling_interval : float
        MPPT control update period T_mppt [s] (e.g. 0.005 s = 5 ms).
    power_deadband : float
        Minimum absolute power change |Delta P| required to trigger perturbation [W].
    voltage_deadband : float
        Minimum absolute voltage change |Delta V| required to trigger perturbation [V].
    simulation_duration : float
        Standard duration for P&O simulation runs [s].
    """

    initial_duty: float = 0.35  # Off-MPP starting point to test convergence towards 0.3939
    duty_min: float = 0.05
    duty_max: float = 0.90
    perturbation_step: float = 0.002
    sampling_interval: float = 0.005  # 5 ms (50 converter integration steps)
    power_deadband: float = 0.0
    voltage_deadband: float = 0.0
    simulation_duration: float = 1.5
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MPPTConfig:
    """Configuration schema for MPPT controllers."""

    algorithm: str = "PO"  # 'PO', 'RL_SAC', etc.
    sampling_interval: float = 0.005  # Default control loop interval [seconds] (5 ms)
    delta_duty: float = 0.002  # Perturbation step for P&O
    po: POConfig = field(default_factory=POConfig)
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLEnvConfig:
    """Configuration for Custom Gymnasium Solar MPPT Environment.

    Defines action limits, control timing, observation scaling, reward weights,
    initial condition randomization, and safety thresholds for the Gymnasium environment.
    """

    # Timing & Steps
    control_interval: float = 0.005  # RL control step T_RL = 5 ms (50 plant integration steps)
    max_steps_per_episode: int = 300  # 300 steps * 5 ms = 1.5 s episode duration

    # Action Space & Slew Limits
    action_min: float = -1.0
    action_max: float = 1.0
    delta_duty_max: float = 0.01  # Maximum duty adjustment per step: Delta D = a * delta_duty_max
    duty_min: float = 0.05  # Converter physical saturation bounds
    duty_max: float = 0.90

    # Initial State Randomization
    randomize_initial_duty: bool = True
    initial_duty_min: float = 0.20
    initial_duty_max: float = 0.60
    initial_duty_default: float = 0.35  # Deterministic STC starting duty

    # Environmental Randomization & Range
    randomize_environment: bool = False
    profile_type: str = "stc"  # 'stc', 'random_constant', 'irradiance_step', 'temperature_step', 'combined'
    irradiance_min: float = 400.0  # W/m^2
    irradiance_max: float = 1000.0
    temperature_min: float = 15.0  # °C
    temperature_max: float = 55.0

    # Observation Space & Normalization Scales
    include_mpp_in_obs: bool = True  # If True: 10-dim privileged obs; If False: 8-dim realistic obs
    v_scale: float = 45.0  # PV voltage scale [V] (V_oc ~ 37V)
    i_scale: float = 10.0  # PV current scale [A] (I_sc ~ 8.6A)
    p_scale: float = 300.0  # PV power scale [W] (P_mp ~ 240W)
    delta_v_scale: float = 5.0  # Delta V scale [V]
    delta_p_scale: float = 50.0  # Delta P scale [W]
    g_scale: float = 1000.0  # Irradiance scale [W/m^2]
    t_ref: float = 25.0  # Reference temperature [°C]
    t_scale: float = 50.0  # Temperature scale [°C]
    obs_clip_min: float = -2.0
    obs_clip_max: float = 2.0

    # Reward Function Settings
    reward_type: str = "hybrid_tracking"  # 'hybrid_tracking', 'normalized_power', 'power_delta'
    weight_power: float = 1.0  # w_p: linear power extraction weight
    weight_tracking_error: float = 2.0  # w_e: quadratic tracking error penalty weight
    weight_action_penalty: float = 0.05  # w_d: control effort penalty weight
    epsilon_power: float = 1e-4  # Epsilon to prevent division by zero [W]
    fault_penalty: float = 10.0  # Penalty applied upon numerical or physical safety violation

    # Safety Thresholds
    max_output_voltage: float = 120.0  # Extreme over-voltage termination threshold [V]

    # Legacy compatibility fields
    action_type: str = "continuous"
    custom_params: Dict[str, Any] = field(default_factory=dict)


# Type alias for EnvironmentConfig to maintain backward compatibility
EnvironmentConfig = RLEnvConfig


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
