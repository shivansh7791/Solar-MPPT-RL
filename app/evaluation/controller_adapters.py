"""Standardized Controller Adapters for MPPT Comparative Evaluation.

Provides a unified interface enabling Classical P&O and Trained SAC agents to interact
with identical plant simulation environments under identical operating conditions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from stable_baselines3 import SAC

from app.controllers.po_controller import PerturbAndObserveController
from app.rl.observations import ObservationProcessor
from app.utils.logger import get_logger
from config.default_config import MasterConfig, POConfig, RLEnvConfig, get_default_config

logger = get_logger("controller_adapters")


class BaseMPPTController(ABC):
    """Abstract base class defining the standardized MPPT controller interface."""

    def __init__(self, name: str, duty_min: float = 0.05, duty_max: float = 0.90) -> None:
        self.name = name
        self.duty_min = duty_min
        self.duty_max = duty_max
        self.current_duty: float = 0.35
        self.step_count: int = 0

    @abstractmethod
    def reset(
        self,
        initial_voltage: float,
        initial_current: float,
        initial_duty: float = 0.35,
    ) -> float:
        """Reset internal controller state and return initial commanded duty cycle.

        Parameters
        ----------
        initial_voltage : float
            Initial PV terminal voltage [V].
        initial_current : float
            Initial PV array current [A].
        initial_duty : float
            Starting duty cycle command.

        Returns
        -------
        float
            Initial commanded duty cycle.
        """
        pass

    @abstractmethod
    def predict(
        self,
        v_pv: float,
        i_pv: float,
        current_duty: float,
        simulation_time: float,
        irradiance: float,
        cell_temperature: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate and return next commanded converter duty cycle and diagnostic metadata.

        Parameters
        ----------
        v_pv : float
            Measured PV terminal voltage [V].
        i_pv : float
            Measured PV terminal current [A].
        current_duty : float
            Current active converter duty cycle.
        simulation_time : float
            Elapsed simulation physical time [s].
        irradiance : float
            Incident solar irradiance [W/m^2].
        cell_temperature : float
            Cell junction temperature [°C].

        Returns
        -------
        Tuple[float, Dict[str, Any]]
            (next_duty_cycle, diagnostic_info_dict)
        """
        pass


class POControllerAdapter(BaseMPPTController):
    """Adapter wrapping Classical Perturb & Observe (P&O) MPPT controller."""

    def __init__(
        self,
        config: Optional[POConfig] = None,
        name: str = "P&O",
    ) -> None:
        po_cfg = config if config is not None else get_default_config().mppt.po
        super().__init__(name=name, duty_min=po_cfg.duty_min, duty_max=po_cfg.duty_max)
        self.po = PerturbAndObserveController(config=po_cfg)
        self._last_diagnostics: Dict[str, Any] = {}

    def reset(
        self,
        initial_voltage: float,
        initial_current: float,
        initial_duty: float = 0.35,
    ) -> float:
        self.current_duty = float(self.po.reset(initial_duty=initial_duty))
        self.step_count = 0
        # Initialize internal P&O state with t=0 measurement
        self.current_duty = float(self.po.update(initial_voltage, initial_current))
        state = self.po.get_state()
        self._last_diagnostics = {
            "controller": self.name,
            "raw_action": float(state.duty_adjustment),
            "delta_duty": float(state.duty_adjustment),
            "perturbation_direction": int(state.perturbation_direction),
            "is_saturated": bool(state.is_saturated),
        }
        return self.current_duty

    def predict(
        self,
        v_pv: float,
        i_pv: float,
        current_duty: float,
        simulation_time: float,
        irradiance: float,
        cell_temperature: float,
    ) -> Tuple[float, Dict[str, Any]]:
        self.step_count += 1
        new_duty = float(self.po.update(v_pv, i_pv))
        state = self.po.get_state()
        self.current_duty = new_duty

        diag = {
            "controller": self.name,
            "raw_action": float(state.duty_adjustment),
            "delta_duty": float(state.duty_adjustment),
            "perturbation_direction": int(state.perturbation_direction),
            "is_saturated": bool(state.is_saturated),
            "step": self.step_count,
        }
        self._last_diagnostics = diag
        return new_duty, diag


class SACControllerAdapter(BaseMPPTController):
    """Adapter wrapping Trained Soft Actor-Critic (SAC) RL Agent."""

    def __init__(
        self,
        model: SAC,
        env_config: Optional[RLEnvConfig] = None,
        name: str = "SAC",
        deterministic: bool = True,
    ) -> None:
        self.model = model
        self.env_config = env_config if env_config is not None else RLEnvConfig(include_mpp_in_obs=False)
        super().__init__(
            name=name,
            duty_min=self.env_config.duty_min,
            duty_max=self.env_config.duty_max,
        )
        self.delta_duty_max = float(self.env_config.delta_duty_max)
        self.deterministic = deterministic

        # Initialize observation processor for realistic 8-dim state space
        self.obs_processor = ObservationProcessor(self.env_config)
        self._last_diagnostics: Dict[str, Any] = {}

    def reset(
        self,
        initial_voltage: float,
        initial_current: float,
        initial_duty: float = 0.35,
    ) -> float:
        self.current_duty = float(np.clip(initial_duty, self.duty_min, self.duty_max))
        self.step_count = 0
        p_init = initial_voltage * initial_current
        self.obs_processor.reset(initial_voltage=initial_voltage, initial_power=p_init)

        self._last_diagnostics = {
            "controller": self.name,
            "raw_action": 0.0,
            "delta_duty": 0.0,
            "is_saturated": False,
        }
        return self.current_duty

    def predict(
        self,
        v_pv: float,
        i_pv: float,
        current_duty: float,
        simulation_time: float,
        irradiance: float,
        cell_temperature: float,
    ) -> Tuple[float, Dict[str, Any]]:
        self.step_count += 1
        p_pv = v_pv * i_pv

        # Construct exact normalized observation vector s_t
        obs = self.obs_processor.process(
            v_pv=v_pv,
            i_pv=i_pv,
            p_pv=p_pv,
            duty_cycle=current_duty,
            irradiance=irradiance,
            cell_temperature=cell_temperature,
        )

        # Query deterministic policy action a_t in [-1, +1]
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        act_val = float(np.clip(action[0], -1.0, 1.0))

        # Scale action to duty adjustment Delta D = a_t * Delta D_max
        delta_d = act_val * self.delta_duty_max
        raw_new_duty = current_duty + delta_d
        clamped_duty = float(np.clip(raw_new_duty, self.duty_min, self.duty_max))
        is_sat = bool(math.isclose(clamped_duty, self.duty_min) or math.isclose(clamped_duty, self.duty_max))
        actual_delta_d = clamped_duty - current_duty

        self.current_duty = clamped_duty

        diag = {
            "controller": self.name,
            "raw_action": act_val,
            "delta_duty": actual_delta_d,
            "is_saturated": is_sat,
            "step": self.step_count,
        }
        self._last_diagnostics = diag
        return clamped_duty, diag
