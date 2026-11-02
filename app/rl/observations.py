"""Observation Space and Processing for Gymnasium Solar MPPT Environment.

Normalizes physical measurements (V_pv, I_pv, P_pv, D, Delta V, Delta P, G, T)
and optional privileged reference values (V_mp, P_mp) into a bounded continuous
observation vector of dtype float32 suitable for reinforcement learning agents.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium.spaces as spaces
import numpy as np

from config.default_config import RLEnvConfig


class ObservationProcessor:
    """Processes, scales, and normalizes physical PV measurements into RL observation vectors.

    Features:
    0: Normalized PV Voltage: V_pv / V_scale
    1: Normalized PV Current: I_pv / I_scale
    2: Normalized PV Power: P_pv / P_scale
    3: Converter Duty Cycle: D in [D_min, D_max]
    4: Normalized Delta V: (V_pv - V_prev) / Delta V_scale
    5: Normalized Delta P: (P_pv - P_prev) / Delta P_scale
    6: Normalized Solar Irradiance: G / G_scale
    7: Normalized Cell Temperature: (T - T_ref) / T_scale
    [Optional Privileged Features when include_mpp_in_obs=True]:
    8: Normalized Reference MPP Voltage: V_mp / V_scale
    9: Normalized Reference MPP Power: P_mp / P_scale
    """

    FEATURE_NAMES_STANDARD = [
        "v_pv_norm",
        "i_pv_norm",
        "p_pv_norm",
        "duty_cycle",
        "delta_v_norm",
        "delta_p_norm",
        "irradiance_norm",
        "temperature_norm",
    ]

    FEATURE_NAMES_PRIVILEGED = [
        "v_pv_norm",
        "i_pv_norm",
        "p_pv_norm",
        "duty_cycle",
        "delta_v_norm",
        "delta_p_norm",
        "irradiance_norm",
        "temperature_norm",
        "v_mp_norm",
        "p_mp_norm",
    ]

    def __init__(self, config: Optional[RLEnvConfig] = None) -> None:
        """Initialize the observation processor with scaling configuration.

        Parameters
        ----------
        config : Optional[RLEnvConfig], optional
            RL environment configuration. If None, default RLEnvConfig is used.
        """
        self.config = config if config is not None else RLEnvConfig()

        self.include_mpp = bool(self.config.include_mpp_in_obs)
        self.v_scale = float(self.config.v_scale)
        self.i_scale = float(self.config.i_scale)
        self.p_scale = float(self.config.p_scale)
        self.delta_v_scale = float(self.config.delta_v_scale)
        self.delta_p_scale = float(self.config.delta_p_scale)
        self.g_scale = float(self.config.g_scale)
        self.t_ref = float(self.config.t_ref)
        self.t_scale = float(self.config.t_scale)
        self.clip_min = float(self.config.obs_clip_min)
        self.clip_max = float(self.config.obs_clip_max)

        self._validate_scales()

        self.observation_dim = 10 if self.include_mpp else 8
        self.feature_names = self.FEATURE_NAMES_PRIVILEGED if self.include_mpp else self.FEATURE_NAMES_STANDARD

        # State memory for computing finite differences
        self.prev_voltage: Optional[float] = None
        self.prev_power: Optional[float] = None

    def _validate_scales(self) -> None:
        """Validate that all normalization scales are strictly positive."""
        for name, val in [
            ("v_scale", self.v_scale),
            ("i_scale", self.i_scale),
            ("p_scale", self.p_scale),
            ("delta_v_scale", self.delta_v_scale),
            ("delta_p_scale", self.delta_p_scale),
            ("g_scale", self.g_scale),
            ("t_scale", self.t_scale),
        ]:
            if val <= 0.0:
                raise ValueError(f"Observation scale '{name}' must be strictly positive. Received: {val}")

        if self.clip_min >= self.clip_max:
            raise ValueError(f"obs_clip_min ({self.clip_min}) must be strictly less than obs_clip_max ({self.clip_max})")

    def get_observation_space(self) -> spaces.Box:
        """Construct and return the Gymnasium Box observation space."""
        lows = np.full(self.observation_dim, self.clip_min, dtype=np.float32)
        highs = np.full(self.observation_dim, self.clip_max, dtype=np.float32)
        return spaces.Box(low=lows, high=highs, shape=(self.observation_dim,), dtype=np.float32)

    def reset(self, initial_voltage: Optional[float] = None, initial_power: Optional[float] = None) -> None:
        """Reset historical state memory for derivative features.

        Parameters
        ----------
        initial_voltage : Optional[float], optional
            Initial PV voltage. If provided, sets prev_voltage.
        initial_power : Optional[float], optional
            Initial PV power. If provided, sets prev_power.
        """
        self.prev_voltage = float(initial_voltage) if initial_voltage is not None else None
        self.prev_power = float(initial_power) if initial_power is not None else None

    def process(
        self,
        v_pv: float,
        i_pv: float,
        p_pv: float,
        duty_cycle: float,
        irradiance: float,
        cell_temperature: float,
        v_mp: Optional[float] = None,
        p_mp: Optional[float] = None,
    ) -> np.ndarray:
        """Convert instantaneous physical variables into a normalized float32 observation vector.

        Parameters
        ----------
        v_pv : float
            PV terminal voltage [V].
        i_pv : float
            PV output current [A].
        p_pv : float
            PV extracted power [W].
        duty_cycle : float
            Converter duty cycle [0, 1].
        irradiance : float
            Solar irradiance [W/m^2].
        cell_temperature : float
            Cell junction temperature [°C].
        v_mp : Optional[float], optional
            Privileged reference MPP voltage [V]. Required if include_mpp_in_obs=True.
        p_mp : Optional[float], optional
            Privileged reference MPP power [W]. Required if include_mpp_in_obs=True.

        Returns
        -------
        np.ndarray
            Normalized float32 1D numpy array of shape (observation_dim,).
        """
        # Validate finite numeric values
        for name, val in [
            ("v_pv", v_pv),
            ("i_pv", i_pv),
            ("p_pv", p_pv),
            ("duty_cycle", duty_cycle),
            ("irradiance", irradiance),
            ("cell_temperature", cell_temperature),
        ]:
            if not isinstance(val, (int, float, np.number)) or not math.isfinite(float(val)):
                raise TypeError(f"Measurement '{name}' must be a finite numeric value. Received: {val}")

        # Compute finite differences
        if self.prev_voltage is None:
            delta_v = 0.0
        else:
            delta_v = float(v_pv) - self.prev_voltage

        if self.prev_power is None:
            delta_p = 0.0
        else:
            delta_p = float(p_pv) - self.prev_power

        # Update historical memory
        self.prev_voltage = float(v_pv)
        self.prev_power = float(p_pv)

        # Compute normalized quantities
        norm_v = float(v_pv) / self.v_scale
        norm_i = float(i_pv) / self.i_scale
        norm_p = float(p_pv) / self.p_scale
        norm_d = float(duty_cycle)
        norm_dv = delta_v / self.delta_v_scale
        norm_dp = delta_p / self.delta_p_scale
        norm_g = float(irradiance) / self.g_scale
        norm_t = (float(cell_temperature) - self.t_ref) / self.t_scale

        features = [
            norm_v,
            norm_i,
            norm_p,
            norm_d,
            norm_dv,
            norm_dp,
            norm_g,
            norm_t,
        ]

        if self.include_mpp:
            if v_mp is None or p_mp is None:
                raise ValueError("v_mp and p_mp must be provided when include_mpp_in_obs=True.")
            if not math.isfinite(float(v_mp)) or not math.isfinite(float(p_mp)):
                raise TypeError(f"Reference MPP values must be finite. Received: v_mp={v_mp}, p_mp={p_mp}")
            norm_v_mp = float(v_mp) / self.v_scale
            norm_p_mp = float(p_mp) / self.p_scale
            features.extend([norm_v_mp, norm_p_mp])

        obs_arr = np.array(features, dtype=np.float32)
        np.clip(obs_arr, self.clip_min, self.clip_max, out=obs_arr)
        return obs_arr

    def to_dict(self, observation: np.ndarray) -> Dict[str, float]:
        """Convert an observation array into a named dictionary for inspection."""
        if len(observation) != self.observation_dim:
            raise ValueError(f"Observation length ({len(observation)}) does not match expected dim ({self.observation_dim}).")
        return {name: float(val) for name, val in zip(self.feature_names, observation)}
