"""Configurable Reward Functions for RL-Based Solar MPPT Control.

Implements mathematically formulated, physically grounded reward functions designed
to encourage maximum power point tracking while penalizing tracking errors, control
effort chattering, and physical/numerical safety violations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config.default_config import RLEnvConfig


class RewardType(str, Enum):
    """Enumeration of supported reward formulations."""

    HYBRID_TRACKING = "hybrid_tracking"
    NORMALIZED_POWER = "normalized_power"
    POWER_DELTA = "power_delta"


@dataclass
class RewardComponents:
    """Detailed breakdown of individual terms comprising the scalar reward."""

    power_term: float
    error_penalty: float
    action_penalty: float
    fault_penalty: float
    total_reward: float

    def to_dict(self) -> Dict[str, float]:
        """Convert components to dictionary for Gymnasium info reporting."""
        return asdict(self)


class RewardCalculator:
    """Computes scalar reinforcement learning rewards and diagnostic component breakdowns.

    Formulations
    ------------
    1. HYBRID_TRACKING (Primary):
       r_t = w_p * (P_pv / (P_mpp + eps)) - w_e * ((P_mpp - P_pv) / (P_mpp + eps))^2 - w_d * (action)^2 - fault_pen

    2. NORMALIZED_POWER:
       r_t = (P_pv / (P_mpp + eps)) - w_d * (action)^2 - fault_pen

    3. POWER_DELTA:
       r_t = (P_pv - P_prev) / P_scale - w_d * (action)^2 - fault_pen
    """

    def __init__(self, config: Optional[RLEnvConfig] = None) -> None:
        """Initialize reward calculator with weights from configuration.

        Parameters
        ----------
        config : Optional[RLEnvConfig], optional
            Configuration dataclass. If None, default RLEnvConfig is used.
        """
        self.config = config if config is not None else RLEnvConfig()

        self.reward_type = RewardType(str(self.config.reward_type).lower())
        self.w_p = float(self.config.weight_power)
        self.w_e = float(self.config.weight_tracking_error)
        self.w_d = float(self.config.weight_action_penalty)
        self.eps = float(self.config.epsilon_power)
        self.fault_penalty = float(self.config.fault_penalty)
        self.p_scale = float(self.config.p_scale)

        self._validate_parameters()

        # State memory for delta power formulation
        self.prev_power: Optional[float] = None

    def _validate_parameters(self) -> None:
        """Verify weights and numeric constants are physically reasonable and non-negative."""
        if self.w_p < 0.0:
            raise ValueError(f"weight_power must be non-negative. Received: {self.w_p}")
        if self.w_e < 0.0:
            raise ValueError(f"weight_tracking_error must be non-negative. Received: {self.w_e}")
        if self.w_d < 0.0:
            raise ValueError(f"weight_action_penalty must be non-negative. Received: {self.w_d}")
        if self.eps <= 0.0:
            raise ValueError(f"epsilon_power must be strictly positive. Received: {self.eps}")
        if self.fault_penalty < 0.0:
            raise ValueError(f"fault_penalty must be non-negative. Received: {self.fault_penalty}")
        if self.p_scale <= 0.0:
            raise ValueError(f"p_scale must be strictly positive. Received: {self.p_scale}")

    def reset(self, initial_power: Optional[float] = None) -> None:
        """Reset historical power for differential reward tracking."""
        self.prev_power = float(initial_power) if initial_power is not None else None

    def compute_reward(
        self,
        p_pv: float,
        p_mpp: float,
        action: float,
        is_fault: bool = False,
    ) -> Tuple[float, RewardComponents]:
        """Compute the scalar reward and return its component breakdown.

        Parameters
        ----------
        p_pv : float
            Extracted PV output power [W].
        p_mpp : float
            Reference theoretical Maximum Power Point power [W].
        action : float
            Normalized continuous RL action in [-1.0, 1.0].
        is_fault : bool, optional
            Flag indicating whether a numerical or physical fault occurred, by default False.

        Returns
        -------
        Tuple[float, RewardComponents]
            Total scalar reward and structured component breakdown.
        """
        # Validate inputs
        for name, val in [("p_pv", p_pv), ("p_mpp", p_mpp), ("action", action)]:
            if not isinstance(val, (int, float, np.number)) or not math.isfinite(float(val)):
                raise TypeError(f"Reward input '{name}' must be finite numeric. Received: {val}")

        p_pv_val = max(0.0, float(p_pv))
        p_mpp_val = max(0.0, float(p_mpp))
        act_val = float(action)

        # Fault condition immediately returns fault penalty
        if is_fault:
            comps = RewardComponents(
                power_term=0.0,
                error_penalty=0.0,
                action_penalty=0.0,
                fault_penalty=self.fault_penalty,
                total_reward=-self.fault_penalty,
            )
            return -self.fault_penalty, comps

        denom = p_mpp_val + self.eps
        action_penalty = self.w_d * (act_val ** 2)

        if self.reward_type == RewardType.HYBRID_TRACKING:
            power_term = self.w_p * (p_pv_val / denom)
            rel_error = (p_mpp_val - p_pv_val) / denom
            error_penalty = self.w_e * (rel_error ** 2)
            total_reward = power_term - error_penalty - action_penalty

        elif self.reward_type == RewardType.NORMALIZED_POWER:
            power_term = p_pv_val / denom
            error_penalty = 0.0
            total_reward = power_term - action_penalty

        elif self.reward_type == RewardType.POWER_DELTA:
            if self.prev_power is None:
                delta_p = 0.0
            else:
                delta_p = p_pv_val - self.prev_power
            self.prev_power = p_pv_val
            power_term = delta_p / self.p_scale
            error_penalty = 0.0
            total_reward = power_term - action_penalty
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}")

        # Ensure finite result
        if not math.isfinite(total_reward):
            total_reward = -self.fault_penalty

        comps = RewardComponents(
            power_term=float(power_term),
            error_penalty=float(error_penalty),
            action_penalty=float(action_penalty),
            fault_penalty=0.0,
            total_reward=float(total_reward),
        )

        return float(total_reward), comps
