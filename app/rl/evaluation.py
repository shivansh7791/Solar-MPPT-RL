"""Deterministic Evaluation Module for Trained SAC Solar MPPT Agents.

Executes standardized evaluation episodes across all five atmospheric profiles,
computing energy tracking efficiency, steady-state ripple, settling time, and converter dynamics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from app.rl.episode_profiles import ProfileType
from app.rl.mppt_env import SolarMPPTEnv
from app.utils.logger import get_logger
from config.default_config import MasterConfig, RLEnvConfig, get_default_config

logger = get_logger("sac_evaluation")


def evaluate_sac_policy_on_profile(
    model: SAC,
    profile_type: Union[str, ProfileType] = ProfileType.STC,
    include_mpp_in_obs: bool = False,
    num_steps: int = 300,
    seed: int = 42,
    deterministic: bool = True,
    config: Optional[MasterConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate a trained SAC policy on a specific environmental profile.

    Parameters
    ----------
    model : SAC
        Trained Stable-Baselines3 SAC agent.
    profile_type : Union[str, ProfileType], optional
        Environmental profile to evaluate, by default ProfileType.STC.
    include_mpp_in_obs : bool, optional
        Whether observation includes privileged MPP reference.
    num_steps : int, optional
        Duration of episode in control steps (e.g. 300 steps = 1.5 s).
    seed : int, optional
        Random seed for deterministic initialization, by default 42.
    deterministic : bool, optional
        Use mean of Gaussian policy, by default True.
    config : Optional[MasterConfig], optional
        Project configuration.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (timeseries_dataframe, scalar_metrics_dictionary)
    """
    cfg = config if config is not None else get_default_config()
    prof_type_str = profile_type.value if isinstance(profile_type, ProfileType) else str(profile_type).lower()

    env_cfg = RLEnvConfig(
        include_mpp_in_obs=include_mpp_in_obs,
        profile_type=prof_type_str,
        max_steps_per_episode=num_steps,
        randomize_environment=False,
    )

    env = SolarMPPTEnv(config=cfg, rl_env_config=env_cfg)
    obs, info = env.reset(seed=seed, options={"profile_type": prof_type_str, "randomize_duty": False, "initial_duty": 0.35})

    records: List[Dict[str, Any]] = []

    # Initial row at t=0
    records.append({
        "step": 0,
        "time": 0.0,
        "action": 0.0,
        "duty_cycle": float(info["duty_cycle"]),
        "pv_voltage": float(info["pv_voltage"]),
        "pv_current": float(info["pv_current"]),
        "pv_power": float(info["pv_power"]),
        "mpp_voltage": float(info["mpp_voltage"]),
        "mpp_current": float(info["mpp_current"]),
        "mpp_power": float(info["mpp_power"]),
        "irradiance": float(info["irradiance"]),
        "cell_temperature": float(info["cell_temperature"]),
        "tracking_efficiency": float(info["tracking_efficiency"]),
        "converter_efficiency": float(info["converter_efficiency"]),
        "reward": 0.0,
    })

    total_reward = 0.0
    terminated = False
    truncated = False

    for s in range(1, num_steps + 1):
        action, _states = model.predict(obs, deterministic=deterministic)
        act_val = float(action[0])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        records.append({
            "step": s,
            "time": float(info["simulation_time"]),
            "action": act_val,
            "duty_cycle": float(info["duty_cycle"]),
            "pv_voltage": float(info["pv_voltage"]),
            "pv_current": float(info["pv_current"]),
            "pv_power": float(info["pv_power"]),
            "mpp_voltage": float(info["mpp_voltage"]),
            "mpp_current": float(info["mpp_current"]),
            "mpp_power": float(info["mpp_power"]),
            "irradiance": float(info["irradiance"]),
            "cell_temperature": float(info["cell_temperature"]),
            "tracking_efficiency": float(info["tracking_efficiency"]),
            "converter_efficiency": float(info["converter_efficiency"]),
            "reward": float(reward),
        })

        if terminated or truncated:
            break

    df = pd.DataFrame(records)

    # Compute standardized energy-based MPPT tracking efficiency
    # eta_MPPT = sum(P_pv * dt) / sum(P_mpp * dt) * 100%
    dt_ctrl = env.t_rl
    energy_extracted = float(np.sum(df["pv_power"].iloc[1:]) * dt_ctrl)
    energy_theoretical = float(np.sum(df["mpp_power"].iloc[1:]) * dt_ctrl)
    eta_mppt_energy = (energy_extracted / (energy_theoretical + 1e-6)) * 100.0

    # Steady-state window (last 50% of steps)
    half_idx = len(df) // 2
    df_ss = df.iloc[half_idx:]
    ss_tracking_eff = float(df_ss["tracking_efficiency"].mean())
    ss_power_mean = float(df_ss["pv_power"].mean())
    ss_power_std = float(df_ss["pv_power"].std())
    ss_power_ripple = float(df_ss["pv_power"].max() - df_ss["pv_power"].min())

    # Settling time (2% band of MPP)
    settling_time = float("nan")
    target_p = float(df["mpp_power"].iloc[-1])
    lower_band = 0.98 * target_p
    upper_band = 1.02 * target_p

    for i in range(len(df)):
        subsequent = df.iloc[i:]
        if np.all((subsequent["pv_power"] >= lower_band) & (subsequent["pv_power"] <= upper_band)):
            settling_time = float(df["time"].iloc[i])
            break

    metrics = {
        "profile": prof_type_str,
        "energy_tracking_efficiency_percent": float(min(100.0, max(0.0, eta_mppt_energy))),
        "mean_instantaneous_efficiency_percent": float(df["tracking_efficiency"].mean()),
        "ss_tracking_efficiency_percent": float(ss_tracking_eff),
        "mean_extracted_power_W": float(df["pv_power"].mean()),
        "mean_mpp_power_W": float(df["mpp_power"].mean()),
        "ss_power_mean_W": ss_power_mean,
        "ss_power_std_W": ss_power_std,
        "ss_power_ripple_W": ss_power_ripple,
        "settling_time_s": settling_time,
        "mean_converter_efficiency_percent": float(df["converter_efficiency"].mean()),
        "total_reward": total_reward,
        "final_duty_cycle": float(df["duty_cycle"].iloc[-1]),
        "steps_completed": len(df) - 1,
        "terminated_early": bool(terminated),
    }

    return df, metrics


def evaluate_sac_across_all_profiles(
    model: SAC,
    include_mpp_in_obs: bool = False,
    num_steps: int = 300,
    seed: int = 42,
    config: Optional[MasterConfig] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Execute evaluation across all five standardized atmospheric profiles."""
    profiles = [
        ProfileType.STC,
        ProfileType.RANDOM_CONSTANT,
        ProfileType.IRRADIANCE_STEP,
        ProfileType.TEMPERATURE_STEP,
        ProfileType.COMBINED,
    ]

    all_timeseries: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, Any]] = []

    for prof in profiles:
        logger.info(f"Evaluating SAC model on {prof.value} profile...")
        df, metrics = evaluate_sac_policy_on_profile(
            model=model,
            profile_type=prof,
            include_mpp_in_obs=include_mpp_in_obs,
            num_steps=num_steps,
            seed=seed,
            config=config,
        )
        all_timeseries[prof.value] = df
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    return all_timeseries, summary_df
