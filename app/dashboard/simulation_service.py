"""Simulation Service for Streamlit Dashboard.

Bridges user interface controls with the physical simulation engine (ComparisonRunner),
supporting single-controller runs, paired comparative evaluations, custom scenarios,
and standard atmospheric benchmark profiles.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.evaluation.comparison_runner import ComparisonRunner, PairedScenarioResult
from app.evaluation.controller_adapters import (
    BaseMPPTController,
    POControllerAdapter,
    SACControllerAdapter,
)
from app.evaluation.metrics import ComparisonMetrics
from app.evaluation.scenario_manager import (
    ScenarioDefinition,
    create_scenario_a,
    create_scenario_b,
    create_scenario_c,
    create_scenario_d,
    create_scenario_e_list,
    create_scenario_f_list,
)
from app.pv.pv_model import PVOutput, SolarPVModel
from app.utils.logger import get_logger
from config.default_config import ConverterConfig, MasterConfig, get_default_config

logger = get_logger("simulation_service")


@dataclass
class SimulationRequest:
    """User simulation configuration payload."""

    controller_mode: str  # 'P&O', 'SAC', or 'Comparison (P&O vs SAC)'
    profile_type: str  # 'Static (Constant G & T)', 'Scenario A (STC)', 'Scenario B (Irradiance Steps)', ...
    irradiance: float = 1000.0  # W/m^2 (used for static mode)
    temperature: float = 25.0  # °C (used for static mode)
    load_resistance: float = 10.0  # Ohm
    initial_duty: float = 0.35
    duration: float = 1.5  # seconds
    seed: int = 42
    sac_checkpoint_path: Optional[str] = None
    use_privileged_policy: bool = False


@dataclass
class SingleSimulationResult:
    """Outcome of a single-controller simulation."""

    controller_name: str
    scenario: ScenarioDefinition
    timeseries: pd.DataFrame
    metrics: ComparisonMetrics
    converter_config: ConverterConfig


class SimulationService:
    """Encapsulates simulation execution and parameter validation for the dashboard."""

    def __init__(self, config: Optional[MasterConfig] = None) -> None:
        self.base_config = config if config is not None else get_default_config()
        self.pv_model = SolarPVModel(module_parameters=self.base_config.pv)

    def validate_request(self, req: SimulationRequest) -> None:
        """Validate all parameters in the simulation request."""
        if req.irradiance <= 0.0 or req.irradiance > 1500.0:
            raise ValueError(
                f"Irradiance must be between 1.0 and 1500.0 W/m^2. Received: {req.irradiance}"
            )
        if req.temperature < -40.0 or req.temperature > 85.0:
            raise ValueError(
                f"Cell temperature must be between -40.0 and 85.0 °C. Received: {req.temperature}"
            )
        if req.load_resistance <= 0.5 or req.load_resistance > 100.0:
            raise ValueError(
                f"Load resistance must be between 0.5 and 100.0 Ω. Received: {req.load_resistance}"
            )
        if req.initial_duty < 0.05 or req.initial_duty > 0.90:
            raise ValueError(
                f"Initial duty cycle must be within converter bounds [0.05, 0.90]. Received: {req.initial_duty}"
            )
        if req.duration < 0.05 or req.duration > 5.0:
            raise ValueError(
                f"Simulation duration must be between 0.05 and 5.0 seconds. Received: {req.duration}"
            )

    def get_available_profiles(self) -> Dict[str, str]:
        """Return human-readable names and descriptions of supported environmental profiles."""
        return {
            "Static (Constant G & T)": "Constant solar irradiance and cell temperature over time.",
            "Scenario A: Standard Test Conditions (STC)": "Standard rating baseline (1000 W/m², 25°C, D0=0.35).",
            "Scenario B: Multistep Irradiance Changes": "Rapid solar transitions (1000 -> 600 -> 800 -> 400 W/m²).",
            "Scenario C: Multistep Temperature Steps": "Thermal junction shifts (25 -> 35 -> 45 -> 25°C).",
            "Scenario D: Combined Ramp & Thermal Step": "Simultaneous linear solar decrease and 25°C heat jump.",
            "Scenario E.1: Cloud Shading Dip": "Temporary cloud passage with rapid recovery (Seed 101).",
            "Scenario E.2: Morning Solar Ramp": "Continuous irradiance rise from 400 to 950 W/m² (Seed 102).",
            "Scenario E.4: Low Irradiance Overcast": "Challenging diffuse daylight (420 W/m², Seed 104).",
            "Scenario F.4: Duty Mis-match (D0=0.50)": "Off-MPP startup stressing transient recovery.",
            "Scenario F.5: Extreme Mis-match (D0=0.60)": "Severe duty-cycle offset stressing directional hunting.",
        }

    def build_scenario(self, req: SimulationRequest) -> ScenarioDefinition:
        """Resolve or construct ScenarioDefinition from user request."""
        profile_key = req.profile_type

        if profile_key == "Scenario A: Standard Test Conditions (STC)":
            sc = create_scenario_a()
            sc.duration = req.duration
            sc.initial_duty = req.initial_duty
            return sc
        elif profile_key == "Scenario B: Multistep Irradiance Changes":
            sc = create_scenario_b()
            sc.duration = req.duration
            sc.initial_duty = req.initial_duty
            return sc
        elif profile_key == "Scenario C: Multistep Temperature Steps":
            sc = create_scenario_c()
            sc.duration = req.duration
            sc.initial_duty = req.initial_duty
            return sc
        elif profile_key == "Scenario D: Combined Ramp & Thermal Step":
            sc = create_scenario_d()
            sc.duration = req.duration
            sc.initial_duty = req.initial_duty
            return sc
        elif profile_key == "Scenario E.1: Cloud Shading Dip":
            sc = create_scenario_e_list(num_scenarios=1, base_seed=101)[0]
            sc.duration = req.duration
            sc.initial_duty = req.initial_duty
            return sc
        elif profile_key == "Scenario E.2: Morning Solar Ramp":
            sc = create_scenario_e_list(num_scenarios=2, base_seed=101)[1]
            sc.duration = req.duration
            sc.initial_duty = req.initial_duty
            return sc
        elif profile_key == "Scenario E.4: Low Irradiance Overcast":
            sc = create_scenario_e_list(num_scenarios=4, base_seed=101)[3]
            sc.duration = req.duration
            sc.initial_duty = req.initial_duty
            return sc
        elif profile_key == "Scenario F.4: Duty Mis-match (D0=0.50)":
            sc = create_scenario_f_list()[3]
            sc.duration = req.duration
            return sc
        elif profile_key == "Scenario F.5: Extreme Mis-match (D0=0.60)":
            sc = create_scenario_f_list()[4]
            sc.duration = req.duration
            return sc
        else:
            # Static constant condition
            g_val = float(req.irradiance)
            t_val = float(req.temperature)
            return ScenarioDefinition(
                scenario_id="custom_static_profile",
                name=f"Custom Static Profile ({g_val:.0f} W/m², {t_val:.1f}°C)",
                category="CUSTOM",
                duration=float(req.duration),
                initial_duty=float(req.initial_duty),
                irradiance_fn=lambda t: g_val,
                temperature_fn=lambda t: t_val,
                description=f"User-configured static ambient conditions: G={g_val} W/m², T={t_val}°C, Rload={req.load_resistance}Ω.",
                seed=req.seed,
            )

    def _prepare_runner_and_config(self, req: SimulationRequest) -> Tuple[ComparisonRunner, MasterConfig]:
        """Construct isolated ComparisonRunner with customized converter parameters."""
        run_config = copy.deepcopy(self.base_config)
        run_config.converter.load_resistance = float(req.load_resistance)
        runner = ComparisonRunner(config=run_config)
        return runner, run_config

    def execute_single_simulation(
        self,
        req: SimulationRequest,
        controller: BaseMPPTController,
    ) -> SingleSimulationResult:
        """Run a single controller over the requested environmental scenario."""
        self.validate_request(req)
        scenario = self.build_scenario(req)
        runner, run_cfg = self._prepare_runner_and_config(req)

        timeseries_df, metrics = runner.simulate_controller(
            controller=controller,
            scenario=scenario,
        )

        return SingleSimulationResult(
            controller_name=controller.name,
            scenario=scenario,
            timeseries=timeseries_df,
            metrics=metrics,
            converter_config=run_cfg.converter,
        )

    def execute_comparison_simulation(
        self,
        req: SimulationRequest,
        po_controller: BaseMPPTController,
        sac_controller: BaseMPPTController,
    ) -> PairedScenarioResult:
        """Run synchronized paired comparison of P&O and SAC under identical conditions."""
        self.validate_request(req)
        scenario = self.build_scenario(req)
        runner, _ = self._prepare_runner_and_config(req)

        return runner.run_paired_scenario(
            scenario=scenario,
            po_adapter=po_controller,  # type: ignore
            sac_adapter=sac_controller,  # type: ignore
        )

    def calculate_pv_operating_curve(
        self,
        irradiance: float,
        temperature: float,
        num_points: int = 250,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PVOutput]:
        """Generate I-V and P-V curve arrays and theoretical MPP for ambient conditions."""
        from app.pv.pv_characteristics import PVCharacteristicAnalyzer

        analyzer = PVCharacteristicAnalyzer(self.pv_model)
        curve = analyzer.generate_curve(
            irradiance=irradiance,
            cell_temperature=temperature,
            num_points=num_points,
        )
        return curve.voltage, curve.current, curve.power, curve.pv_output
