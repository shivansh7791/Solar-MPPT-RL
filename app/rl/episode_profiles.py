"""Environmental Profile Generators for Reinforcement Learning MPPT Episodes.

Defines reproducible solar irradiance G(t) and cell temperature T(t) trajectories
to evaluate RL agent tracking robustness across steady-state, randomized, and
dynamic step/ramp atmospheric conditions.
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from config.default_config import RLEnvConfig


class ProfileType(str, Enum):
    """Supported environmental profile types."""

    STC = "stc"
    RANDOM_CONSTANT = "random_constant"
    IRRADIANCE_STEP = "irradiance_step"
    TEMPERATURE_STEP = "temperature_step"
    COMBINED = "combined"


class EpisodeProfile:
    """Base class and factory for episode environmental profiles."""

    def __init__(
        self,
        profile_type: ProfileType = ProfileType.STC,
        config: Optional[RLEnvConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initialize environmental profile.

        Parameters
        ----------
        profile_type : ProfileType, optional
            Type of environmental profile, by default ProfileType.STC.
        config : Optional[RLEnvConfig], optional
            Configuration dataclass. If None, default RLEnvConfig is used.
        rng : Optional[np.random.Generator], optional
            Numpy random generator for deterministic seeding.
        """
        self.config = config if config is not None else RLEnvConfig()
        self.profile_type = profile_type
        self.rng = rng if rng is not None else np.random.default_rng()

        self.g_min = float(self.config.irradiance_min)
        self.g_max = float(self.config.irradiance_max)
        self.t_min = float(self.config.temperature_min)
        self.t_max = float(self.config.temperature_max)

        # Active episode parameters (assigned on reset)
        self.base_irradiance: float = 1000.0
        self.base_temperature: float = 25.0
        self.reset()

    def reset(self, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
        """Reset profile parameters for a new episode.

        Parameters
        ----------
        rng : Optional[np.random.Generator], optional
            Optional new random generator instance.

        Returns
        -------
        Tuple[float, float]
            Initial environmental conditions (G_0, T_0).
        """
        if rng is not None:
            self.rng = rng

        if self.profile_type == ProfileType.STC:
            self.base_irradiance = 1000.0
            self.base_temperature = 25.0

        elif self.profile_type == ProfileType.RANDOM_CONSTANT:
            self.base_irradiance = float(self.rng.uniform(self.g_min, self.g_max))
            self.base_temperature = float(self.rng.uniform(self.t_min, self.t_max))

        elif self.profile_type == ProfileType.IRRADIANCE_STEP:
            self.base_irradiance = 1000.0
            self.base_temperature = 25.0

        elif self.profile_type == ProfileType.TEMPERATURE_STEP:
            self.base_irradiance = 1000.0
            self.base_temperature = 25.0

        elif self.profile_type == ProfileType.COMBINED:
            self.base_irradiance = 1000.0
            self.base_temperature = 25.0

        return self.get_conditions(0.0)

    def get_conditions(self, time: float) -> Tuple[float, float]:
        """Evaluate solar irradiance G(t) and cell temperature T(t) at simulation time t.

        Parameters
        ----------
        time : float
            Current simulation time [seconds].

        Returns
        -------
        Tuple[float, float]
            (irradiance [W/m^2], cell_temperature [°C]).
        """
        t = max(0.0, float(time))

        if self.profile_type == ProfileType.STC:
            return 1000.0, 25.0

        elif self.profile_type == ProfileType.RANDOM_CONSTANT:
            return self.base_irradiance, self.base_temperature

        elif self.profile_type == ProfileType.IRRADIANCE_STEP:
            # Multi-step irradiance profile:
            # [0.0, 0.4s) -> 1000 W/m^2
            # [0.4, 0.9s) -> 600 W/m^2
            # [0.9s, inf) -> 800 W/m^2
            if t < 0.4:
                g = 1000.0
            elif t < 0.9:
                g = 600.0
            else:
                g = 800.0
            return g, 25.0

        elif self.profile_type == ProfileType.TEMPERATURE_STEP:
            # Multi-step temperature profile:
            # [0.0, 0.4s) -> 25°C
            # [0.4, 0.9s) -> 45°C
            # [0.9s, inf) -> 35°C
            if t < 0.4:
                temp = 25.0
            elif t < 0.9:
                temp = 45.0
            else:
                temp = 35.0
            return 1000.0, temp

        elif self.profile_type == ProfileType.COMBINED:
            # Combined dynamic profile:
            # Continuous irradiance ramp from 1000 to 500 W/m^2 between 0.3s and 0.7s
            # Temperature step from 25°C to 50°C at 0.5s
            if t < 0.3:
                g = 1000.0
            elif t < 0.7:
                # Linear ramp
                g = 1000.0 - (500.0 * (t - 0.3) / 0.4)
            else:
                g = 500.0

            temp = 25.0 if t < 0.5 else 50.0
            return g, temp

        else:
            raise ValueError(f"Unknown profile type: {self.profile_type}")


class EpisodeProfileGenerator:
    """Factory and manager for environmental profiles in Gymnasium MPPT environment."""

    @staticmethod
    def create(
        profile_type: str | ProfileType = ProfileType.STC,
        config: Optional[RLEnvConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> EpisodeProfile:
        """Create an environmental profile instance.

        Parameters
        ----------
        profile_type : str | ProfileType, optional
            Profile name or enum, by default ProfileType.STC.
        config : Optional[RLEnvConfig], optional
            Configuration dataclass. If None, default is used.
        rng : Optional[np.random.Generator], optional
            Random generator for seeding.

        Returns
        -------
        EpisodeProfile
            Instantiated profile.
        """
        if isinstance(profile_type, str):
            profile_type = ProfileType(profile_type.lower())
        return EpisodeProfile(profile_type=profile_type, config=config, rng=rng)
