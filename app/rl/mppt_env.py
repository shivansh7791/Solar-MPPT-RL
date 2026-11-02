"""Custom Gymnasium Environment for Solar MPPT Reinforcement Learning Control.

Exposes a coupled Canadian Solar CS6P-240P PV model and dynamic DC-DC boost converter
as a continuous-control Markov Decision Process (MDP) compatible with Gymnasium and
Stable-Baselines3 (SAC).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from scipy.optimize import brentq

from app.converter.boost_converter import BoostConverter, ConverterMeasurements
from app.pv.pv_model import PVOutput, SolarPVModel
from app.rl.episode_profiles import EpisodeProfile, EpisodeProfileGenerator, ProfileType
from app.rl.observations import ObservationProcessor
from app.rl.reward import RewardCalculator, RewardComponents
from app.utils.logger import get_logger
from config.default_config import MasterConfig, RLEnvConfig, get_default_config

logger = get_logger("mppt_env")


class SolarMPPTEnv(gym.Env):
    """Custom Gymnasium Environment for RL-based Solar MPPT.

    The environment simulates the continuous-time physical interaction between a
    photovoltaic array, a DC-DC boost converter, and an RL MPPT controller operating
    at a discrete control interval T_RL (default 5 ms).

    Observation Space
    -----------------
    Continuous vector of normalized physical measurements:
    [V_pv/V_scale, I_pv/I_scale, P_pv/P_scale, D, Delta V_norm, Delta P_norm, G/G_scale, (T-T_ref)/T_scale]
    If include_mpp_in_obs=True, includes privileged reference inputs [V_mp/V_scale, P_mp/P_scale].
    Box(low=-2.0, high=2.0, shape=(8 or 10,), dtype=np.float32)

    Action Space
    ------------
    Continuous duty cycle adjustment increment:
    a in [-1.0, 1.0] -> D_new = clip(D_current + a * Delta D_max, D_min, D_max)
    Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    """

    metadata = {"render_modes": [], "name": "SolarMPPT-v0"}

    def __init__(
        self,
        config: Optional[MasterConfig] = None,
        rl_env_config: Optional[RLEnvConfig] = None,
        pv_model: Optional[SolarPVModel] = None,
        converter: Optional[BoostConverter] = None,
    ) -> None:
        """Initialize the Solar MPPT Gymnasium environment.

        Parameters
        ----------
        config : Optional[MasterConfig], optional
            Master project configuration. If None, loaded from default.
        rl_env_config : Optional[RLEnvConfig], optional
            Dedicated RL environment config. If None, derived from config.environment.
        pv_model : Optional[SolarPVModel], optional
            Pre-instantiated SolarPVModel instance. If None, created from config.
        converter : Optional[BoostConverter], optional
            Pre-instantiated BoostConverter instance. If None, created from config.
        """
        super().__init__()

        self.master_config = config if config is not None else get_default_config()
        self.env_config = rl_env_config if rl_env_config is not None else self.master_config.environment

        # Core physics models
        self.pv_model = pv_model if pv_model is not None else SolarPVModel(self.master_config.pv)
        self.converter = converter if converter is not None else BoostConverter(self.pv_model, self.master_config.converter)

        # Environmental timing & multirate ratios
        self.dt_plant = float(self.converter.default_dt)  # 100 us
        self.t_rl = float(self.env_config.control_interval)  # 5 ms
        if self.t_rl < self.dt_plant:
            raise ValueError(f"RL control interval ({self.t_rl} s) cannot be less than plant dt ({self.dt_plant} s).")
        self.sub_steps = max(1, int(round(self.t_rl / self.dt_plant)))

        self.max_steps = int(self.env_config.max_steps_per_episode)
        self.current_step: int = 0
        self.simulation_time: float = 0.0

        # Action parameters
        self.delta_duty_max = float(self.env_config.delta_duty_max)
        self.duty_min = float(self.env_config.duty_min)
        self.duty_max = float(self.env_config.duty_max)
        self.current_duty: float = float(self.env_config.initial_duty_default)

        # Action and Observation spaces
        self.action_space = spaces.Box(
            low=float(self.env_config.action_min),
            high=float(self.env_config.action_max),
            shape=(1,),
            dtype=np.float32,
        )

        self.obs_processor = ObservationProcessor(self.env_config)
        self.observation_space = self.obs_processor.get_observation_space()

        # Reward calculator
        self.reward_calculator = RewardCalculator(self.env_config)

        # Environmental profile generator
        self.profile = EpisodeProfileGenerator.create(
            profile_type=self.env_config.profile_type,
            config=self.env_config,
            rng=self.np_random,
        )

        # MPP analytical point caching
        self._mpp_cache: Dict[Tuple[float, float], PVOutput] = {}

        logger.debug(
            f"Initialized SolarMPPTEnv: T_rl={self.t_rl*1e3:.1f}ms, dt_plant={self.dt_plant*1e6:.1f}us, "
            f"sub_steps={self.sub_steps}, obs_dim={self.observation_space.shape[0]}"
        )

    def _get_mpp(self, irradiance: float, cell_temperature: float) -> PVOutput:
        """Query or calculate analytical reference MPP for given environmental condition."""
        key = (round(float(irradiance), 2), round(float(cell_temperature), 2))
        if key not in self._mpp_cache:
            self._mpp_cache[key] = self.pv_model.calculate(key[0], key[1])
        return self._mpp_cache[key]

    def compute_equilibrium_state(
        self,
        irradiance: float,
        cell_temperature: float,
        duty_cycle: float,
        ref_mpp: Optional[PVOutput] = None,
    ) -> Tuple[float, float, float]:
        """Compute exact physical equilibrium state vector (V_pv*, i_L*, V_out*) where all derivatives are 0.

        Solves V_pv = R_in(D) * I_pv(V_pv, G, T) where R_in(D) = (1 - D)^2 * R_load + r_L.
        """
        d = float(duty_cycle)
        r_in = ((1.0 - d) ** 2) * self.converter.R_load + self.converter.r_L
        v_oc = ref_mpp.v_oc if ref_mpp is not None else self._get_mpp(irradiance, cell_temperature).v_oc

        def residual(v: float) -> float:
            i_pv = self.pv_model.current_at_voltage(v, irradiance, cell_temperature, clip_non_negative=True)
            return v - r_in * i_pv

        try:
            v_pv_star = float(brentq(residual, 0.0, v_oc * 1.05))
            i_l_star = float(self.pv_model.current_at_voltage(v_pv_star, irradiance, cell_temperature, clip_non_negative=True))
            v_out_star = float((1.0 - d) * self.converter.R_load * i_l_star)
            return v_pv_star, i_l_star, v_out_star
        except Exception as e:
            logger.warning(f"Equilibrium root-finding failed ({e}), falling back to heuristic initialization.")
            mpp = ref_mpp if ref_mpp is not None else self._get_mpp(irradiance, cell_temperature)
            v_pv_h = float(mpp.v_mp)
            i_l_h = float(mpp.i_mp)
            v_out_h = float(v_pv_h / max(0.1, 1.0 - d))
            return v_pv_h, i_l_h, v_out_h

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to an initial state for a new episode.

        Parameters
        ----------
        seed : Optional[int], optional
            Random number generator seed for deterministic reproducibility.
        options : Optional[Dict[str, Any]], optional
            Optional dictionary to override reset behaviors, e.g.
            - "profile_type": str ('stc', 'random_constant', etc.)
            - "initial_duty": float
            - "randomize_duty": bool

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Initial observation vector and diagnostic info dictionary.
        """
        super().reset(seed=seed)

        options = options or {}
        self.current_step = 0
        self.simulation_time = 0.0

        # Configure environmental profile
        profile_type_str = options.get("profile_type", self.env_config.profile_type)
        self.profile = EpisodeProfileGenerator.create(
            profile_type=profile_type_str,
            config=self.env_config,
            rng=self.np_random,
        )
        g_0, t_0 = self.profile.reset(rng=self.np_random)

        # Determine initial duty cycle
        randomize_duty = options.get("randomize_duty", self.env_config.randomize_initial_duty)
        if "initial_duty" in options:
            self.current_duty = float(options["initial_duty"])
        elif randomize_duty and profile_type_str != "stc":
            self.current_duty = float(
                self.np_random.uniform(self.env_config.initial_duty_min, self.env_config.initial_duty_max)
            )
        else:
            self.current_duty = float(self.env_config.initial_duty_default)

        self.current_duty = float(np.clip(self.current_duty, self.duty_min, self.duty_max))

        # Query reference MPP at initial condition
        ref_mpp = self._get_mpp(g_0, t_0)

        # Initialize converter state at exact physical equilibrium for (g_0, t_0, current_duty)
        # where dV_pv/dt = di_L/dt = dV_out/dt = 0
        init_state = self.compute_equilibrium_state(g_0, t_0, self.current_duty, ref_mpp=ref_mpp)

        meas = self.converter.reset(
            initial_state=init_state,
            irradiance=g_0,
            cell_temperature=t_0,
        )

        # Reset processors
        self.obs_processor.reset(initial_voltage=meas.v_pv, initial_power=meas.p_pv)
        self.reward_calculator.reset(initial_power=meas.p_pv)

        obs = self.obs_processor.process(
            v_pv=meas.v_pv,
            i_pv=meas.i_pv,
            p_pv=meas.p_pv,
            duty_cycle=self.current_duty,
            irradiance=g_0,
            cell_temperature=t_0,
            v_mp=ref_mpp.v_mp,
            p_mp=ref_mpp.p_mp,
        )

        info = self._build_info(
            meas=meas,
            ref_mpp=ref_mpp,
            irradiance=g_0,
            cell_temperature=t_0,
            duty_adjustment=0.0,
            duty_saturated=False,
            reward_components=RewardComponents(0.0, 0.0, 0.0, 0.0, 0.0),
        )

        return obs, info

    def step(self, action: Union[float, np.ndarray, Sequence[float]]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one RL control epoch (T_RL = 5 ms).

        Parameters
        ----------
        action : Union[float, np.ndarray, Sequence[float]]
            Continuous action a in [-1.0, 1.0] representing normalized duty-cycle change.

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
            (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # 1. Parse and validate action
        if isinstance(action, (np.ndarray, list, tuple)):
            if len(action) != 1:
                raise ValueError(f"Action must have shape (1,). Received shape {np.shape(action)}")
            act_val = float(action[0])
        else:
            act_val = float(action)

        if not math.isfinite(act_val):
            raise TypeError(f"Action must be a finite numeric value. Received: {act_val}")

        # Clip action to declared action space bounds
        act_val = float(np.clip(act_val, self.action_space.low[0], self.action_space.high[0]))

        # 2. Map action to converter duty cycle
        delta_d_cmd = act_val * self.delta_duty_max
        new_duty = self.current_duty + delta_d_cmd
        clamped_duty = float(np.clip(new_duty, self.duty_min, self.duty_max))
        actual_delta_d = clamped_duty - self.current_duty
        is_saturated = bool(math.isclose(clamped_duty, self.duty_min) or math.isclose(clamped_duty, self.duty_max))
        self.current_duty = clamped_duty

        # 3. Multirate plant integration (sub_steps = 50 steps at dt_plant = 100 us)
        is_fault = False
        last_meas: Optional[ConverterMeasurements] = None

        for _ in range(self.sub_steps):
            g_t, t_t = self.profile.get_conditions(self.simulation_time)
            try:
                last_meas = self.converter.step(
                    duty_cycle=self.current_duty,
                    irradiance=g_t,
                    cell_temperature=t_t,
                    dt=self.dt_plant,
                )
                self.simulation_time += self.dt_plant
            except (ValueError, ArithmeticError) as e:
                logger.warning(f"Numerical integration failure at step {self.current_step}, t={self.simulation_time:.4f}s: {e}")
                is_fault = True
                break

        # Fallback measurement if fault occurred
        if last_meas is None or is_fault:
            is_fault = True
            last_meas = self.converter.get_measurements(g_t, t_t)

        # 4. Check safety violations (e.g. excessive output overvoltage)
        if last_meas.v_out > float(self.env_config.max_output_voltage) or not np.all(np.isfinite(self.converter.state)):
            is_fault = True

        # 5. Evaluate reference MPP
        current_g, current_t = self.profile.get_conditions(self.simulation_time)
        ref_mpp = self._get_mpp(current_g, current_t)

        # 6. Compute reward
        reward, reward_comps = self.reward_calculator.compute_reward(
            p_pv=last_meas.p_pv,
            p_mpp=ref_mpp.p_mp,
            action=act_val,
            is_fault=is_fault,
        )

        # 7. Construct observation
        if is_fault:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self.obs_processor.process(
                v_pv=last_meas.v_pv,
                i_pv=last_meas.i_pv,
                p_pv=last_meas.p_pv,
                duty_cycle=self.current_duty,
                irradiance=current_g,
                cell_temperature=current_t,
                v_mp=ref_mpp.v_mp,
                p_mp=ref_mpp.p_mp,
            )

        # 8. Termination and Truncation
        terminated = bool(is_fault)
        truncated = bool(self.current_step >= self.max_steps)

        # 9. Build info dictionary
        info = self._build_info(
            meas=last_meas,
            ref_mpp=ref_mpp,
            irradiance=current_g,
            cell_temperature=current_t,
            duty_adjustment=actual_delta_d,
            duty_saturated=is_saturated,
            reward_components=reward_comps,
        )

        return obs, reward, terminated, truncated, info

    def _build_info(
        self,
        meas: ConverterMeasurements,
        ref_mpp: PVOutput,
        irradiance: float,
        cell_temperature: float,
        duty_adjustment: float,
        duty_saturated: bool,
        reward_components: RewardComponents,
    ) -> Dict[str, Any]:
        """Construct comprehensive diagnostic dictionary for Gymnasium step/reset returns."""
        p_pv = max(0.0, float(meas.p_pv))
        p_mpp = max(0.0, float(ref_mpp.p_mp))
        tracking_eff = (p_pv / (p_mpp + 1e-4) * 100.0) if p_mpp > 1e-4 else 0.0
        tracking_error = abs(p_mpp - p_pv)

        return {
            "pv_voltage": float(meas.v_pv),
            "pv_current": float(meas.i_pv),
            "pv_power": float(p_pv),
            "mpp_voltage": float(ref_mpp.v_mp),
            "mpp_current": float(ref_mpp.i_mp),
            "mpp_power": float(p_mpp),
            "duty_cycle": float(self.current_duty),
            "duty_adjustment": float(duty_adjustment),
            "duty_saturated": bool(duty_saturated),
            "irradiance": float(irradiance),
            "cell_temperature": float(cell_temperature),
            "tracking_efficiency": float(min(100.0, max(0.0, tracking_eff))),
            "tracking_error": float(tracking_error),
            "converter_efficiency": float(meas.efficiency),
            "episode_step": int(self.current_step),
            "simulation_time": float(self.simulation_time),
            "is_ccm": bool(meas.is_ccm),
            "reward_components": reward_components.to_dict(),
        }
