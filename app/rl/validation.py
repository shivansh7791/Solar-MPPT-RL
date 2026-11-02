"""Validation and Diagnostic Utilities for Solar MPPT Gymnasium Environment.

Provides API compliance verification (Gymnasium, Stable-Baselines3), baseline
comparisons (P&O, Fixed Duty, Random Actions), environmental profile sweeps,
and publication-quality visualization routines at 300 DPI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium.utils.env_checker import check_env as gym_check_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3.common.monitor import Monitor

from app.controllers.po_controller import PerturbAndObserveController
from app.rl.episode_profiles import EpisodeProfileGenerator, ProfileType
from app.rl.mppt_env import SolarMPPTEnv
from app.rl.reward import RewardCalculator, RewardType
from app.utils.logger import get_logger
from config.default_config import MasterConfig, RLEnvConfig, get_default_config

logger = get_logger("rl_validation")

# Academic visualization formatting
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def validate_gymnasium_api(env: SolarMPPTEnv) -> Dict[str, Any]:
    """Execute Gymnasium's official environment checker and verify interfaces."""
    logger.info("Executing Gymnasium API checker...")
    gym_check_env(env.unwrapped)
    return {
        "gym_check_passed": True,
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "observation_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.shape[0],
    }


def validate_sb3_compatibility(env: SolarMPPTEnv) -> Dict[str, Any]:
    """Execute Stable-Baselines3 environment checker and test Monitor wrapper."""
    logger.info("Executing Stable-Baselines3 compatibility checker...")
    sb3_check_env(env.unwrapped)
    monitored_env = Monitor(env)
    obs, info = monitored_env.reset(seed=42)
    step_obs, reward, terminated, truncated, step_info = monitored_env.step(np.array([0.0], dtype=np.float32))
    return {
        "sb3_check_passed": True,
        "monitor_wrapper_passed": True,
        "initial_obs_shape": list(obs.shape),
        "step_reward_finite": bool(np.isfinite(reward)),
    }


def run_random_action_trajectory(
    env: SolarMPPTEnv,
    seed: int = 42,
    num_steps: int = 300,
) -> pd.DataFrame:
    """Execute a single episode under uniformly random actions and capture timeseries."""
    obs, info = env.reset(seed=seed, options={"profile_type": "stc", "randomize_duty": False, "initial_duty": 0.35})
    records: List[Dict[str, Any]] = []

    init_row = {
        "step": 0,
        "time": 0.0,
        "action": 0.0,
        "duty_cycle": info["duty_cycle"],
        "pv_voltage": info["pv_voltage"],
        "pv_current": info["pv_current"],
        "pv_power": info["pv_power"],
        "mpp_power": info["mpp_power"],
        "tracking_efficiency": info["tracking_efficiency"],
        "reward": 0.0,
        "power_term": 0.0,
        "error_penalty": 0.0,
        "action_penalty": 0.0,
    }
    for idx, name in enumerate(env.obs_processor.feature_names):
        init_row[f"obs_{name}"] = float(obs[idx])
    records.append(init_row)

    rng = np.random.default_rng(seed)
    for step_i in range(1, num_steps + 1):
        action = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        r_comps = info["reward_components"]
        row = {
            "step": step_i,
            "time": info["simulation_time"],
            "action": float(action[0]),
            "duty_cycle": info["duty_cycle"],
            "pv_voltage": info["pv_voltage"],
            "pv_current": info["pv_current"],
            "pv_power": info["pv_power"],
            "mpp_power": info["mpp_power"],
            "tracking_efficiency": info["tracking_efficiency"],
            "reward": float(reward),
            "power_term": r_comps["power_term"],
            "error_penalty": r_comps["error_penalty"],
            "action_penalty": r_comps["action_penalty"],
        }
        for idx, name in enumerate(env.obs_processor.feature_names):
            row[f"obs_{name}"] = float(obs[idx])
        records.append(row)

        if terminated or truncated:
            break

    return pd.DataFrame(records)


def run_baseline_comparison(
    config: Optional[MasterConfig] = None,
    num_steps: int = 300,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute and compare STC baseline trajectories: P&O, Fixed Duty, and Random Actions.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (timeseries_df, summary_metrics_df)
    """
    cfg = config if config is not None else get_default_config()

    # 1. P&O Baseline
    env_po = SolarMPPTEnv(config=cfg)
    po_ctrl = PerturbAndObserveController(config=cfg.mppt.po)
    obs_po, info_po = env_po.reset(seed=seed, options={"profile_type": "stc", "randomize_duty": False, "initial_duty": 0.35})
    po_ctrl.reset(initial_duty=0.35)

    po_records = []
    curr_meas_v = info_po["pv_voltage"]
    curr_meas_i = info_po["pv_current"]

    for s_idx in range(num_steps):
        # PO computes new duty cycle
        target_d = po_ctrl.update(curr_meas_v, curr_meas_i)
        curr_d = env_po.current_duty
        # Convert delta duty to normalized action: a = (target_d - curr_d) / delta_duty_max
        delta_d = target_d - curr_d
        act = np.array([np.clip(delta_d / env_po.delta_duty_max, -1.0, 1.0)], dtype=np.float32)

        obs, r, term, trunc, info = env_po.step(act)
        curr_meas_v = info["pv_voltage"]
        curr_meas_i = info["pv_current"]

        po_records.append({
            "step": s_idx + 1,
            "time": info["simulation_time"],
            "policy": "Classical P&O",
            "duty_cycle": info["duty_cycle"],
            "pv_power": info["pv_power"],
            "mpp_power": info["mpp_power"],
            "tracking_efficiency": info["tracking_efficiency"],
            "tracking_error": info["tracking_error"],
            "reward": r,
        })
        if term or trunc:
            break

    df_po = pd.DataFrame(po_records)

    # 2. Fixed Duty Baseline (D = 0.35)
    env_fixed = SolarMPPTEnv(config=cfg)
    obs_fx, info_fx = env_fixed.reset(seed=seed, options={"profile_type": "stc", "randomize_duty": False, "initial_duty": 0.35})
    fixed_records = []
    zero_act = np.array([0.0], dtype=np.float32)

    for s_idx in range(num_steps):
        obs, r, term, trunc, info = env_fixed.step(zero_act)
        fixed_records.append({
            "step": s_idx + 1,
            "time": info["simulation_time"],
            "policy": "Fixed Duty (D=0.35)",
            "duty_cycle": info["duty_cycle"],
            "pv_power": info["pv_power"],
            "mpp_power": info["mpp_power"],
            "tracking_efficiency": info["tracking_efficiency"],
            "tracking_error": info["tracking_error"],
            "reward": r,
        })
        if term or trunc:
            break

    df_fixed = pd.DataFrame(fixed_records)

    # 3. Random Actions Baseline
    env_rnd = SolarMPPTEnv(config=cfg)
    obs_rd, info_rd = env_rnd.reset(seed=seed, options={"profile_type": "stc", "randomize_duty": False, "initial_duty": 0.35})
    rnd_records = []
    rng = np.random.default_rng(seed)

    for s_idx in range(num_steps):
        act = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        obs, r, term, trunc, info = env_rnd.step(act)
        rnd_records.append({
            "step": s_idx + 1,
            "time": info["simulation_time"],
            "policy": "Random Actions",
            "duty_cycle": info["duty_cycle"],
            "pv_power": info["pv_power"],
            "mpp_power": info["mpp_power"],
            "tracking_efficiency": info["tracking_efficiency"],
            "tracking_error": info["tracking_error"],
            "reward": r,
        })
        if term or trunc:
            break

    df_rnd = pd.DataFrame(rnd_records)

    df_all = pd.concat([df_po, df_fixed, df_rnd], ignore_index=True)

    # Compute summary metrics for each policy
    summaries = []
    for policy_name, grp in df_all.groupby("policy"):
        # Steady state is last 50% of episode
        ss_grp = grp.iloc[len(grp) // 2 :]
        summaries.append({
            "policy": policy_name,
            "mean_power_W": float(grp["pv_power"].mean()),
            "max_power_W": float(grp["pv_power"].max()),
            "mean_tracking_eff_percent": float(grp["tracking_efficiency"].mean()),
            "ss_tracking_eff_percent": float(ss_grp["tracking_efficiency"].mean()),
            "mean_tracking_error_W": float(grp["tracking_error"].mean()),
            "mean_reward": float(grp["reward"].mean()),
            "cumulative_reward": float(grp["reward"].sum()),
            "final_duty": float(grp["duty_cycle"].iloc[-1]),
        })

    df_summary = pd.DataFrame(summaries)
    return df_all, df_summary


def run_profile_diagnostics(
    config: Optional[MasterConfig] = None,
    duration: float = 1.5,
    dt: float = 0.005,
) -> pd.DataFrame:
    """Evaluate all 5 environmental profiles across time."""
    cfg = config if config is not None else get_default_config()
    time_pts = np.arange(0.0, duration + dt, dt)
    records = []

    profiles = [
        ("Profile A: Constant STC", ProfileType.STC),
        ("Profile B: Randomized Constant", ProfileType.RANDOM_CONSTANT),
        ("Profile C: Irradiance Step", ProfileType.IRRADIANCE_STEP),
        ("Profile D: Temperature Step", ProfileType.TEMPERATURE_STEP),
        ("Profile E: Combined Dynamic", ProfileType.COMBINED),
    ]

    for label, p_type in profiles:
        prof = EpisodeProfileGenerator.create(profile_type=p_type, config=cfg.environment, rng=np.random.default_rng(42))
        prof.reset(rng=np.random.default_rng(42))
        for t in time_pts:
            g, temp = prof.get_conditions(t)
            records.append({
                "time": t,
                "profile": label,
                "profile_type": p_type.value,
                "irradiance": g,
                "temperature": temp,
            })

    return pd.DataFrame(records)


def run_reward_diagnostics(
    config: Optional[MasterConfig] = None,
) -> pd.DataFrame:
    """Evaluate and compare reward formulations across varying power extraction levels."""
    cfg = config if config is not None else get_default_config()
    p_mpp = 240.097  # STC MPP
    p_pv_values = np.linspace(0.0, 260.0, 100)
    records = []

    for r_type in [RewardType.HYBRID_TRACKING, RewardType.NORMALIZED_POWER]:
        env_cfg = RLEnvConfig(reward_type=r_type.value)
        calc = RewardCalculator(env_cfg)
        for p_pv in p_pv_values:
            r_val, comps = calc.compute_reward(p_pv=p_pv, p_mpp=p_mpp, action=0.0)
            records.append({
                "p_pv": p_pv,
                "p_mpp": p_mpp,
                "power_ratio": p_pv / p_mpp,
                "reward_type": r_type.value,
                "total_reward": r_val,
                "power_term": comps.power_term,
                "error_penalty": comps.error_penalty,
                "action_penalty": comps.action_penalty,
            })

    return pd.DataFrame(records)


# -------------------------------------------------------------------------
# Visualization Routines (300 DPI)
# -------------------------------------------------------------------------

def plot_observation_space_ranges(df_rnd: pd.DataFrame, save_path: Path) -> None:
    """Generate figure showing observation feature ranges during dynamic exploration."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    feature_cols = [c for c in df_rnd.columns if c.startswith("obs_")]
    display_names = [
        r"$\tilde{V}_{pv}$ (Norm Voltage)",
        r"$\tilde{I}_{pv}$ (Norm Current)",
        r"$\tilde{P}_{pv}$ (Norm Power)",
        r"$D$ (Duty Cycle)",
        r"$\widetilde{\Delta V}$ (Norm $\Delta V$)",
        r"$\widetilde{\Delta P}$ (Norm $\Delta P$)",
        r"$\tilde{G}$ (Norm Irradiance)",
        r"$\tilde{T}$ (Norm Temperature)",
        r"$\tilde{V}_{mp}$ (Ref $V_{mp}$)",
    ]

    for i in range(min(9, len(feature_cols))):
        col = feature_cols[i]
        ax = axes[i]
        ax.plot(df_rnd["time"], df_rnd[col], color="#1f77b4", lw=1.5)
        ax.axhline(0.0, color="gray", ls=":", alpha=0.6)
        ax.set_title(display_names[i])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Value")
        ax.set_ylim(-2.1, 2.1)

    fig.suptitle("Gymnasium Solar MPPT: Observation Vector Feature Trajectories (Random Actions)", y=0.99)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved observation space ranges figure: {save_path}")


def plot_reward_components(df_rew: pd.DataFrame, save_path: Path) -> None:
    """Generate figure comparing reward formulations and component breakdowns."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Formulation comparison
    for r_type, grp in df_rew.groupby("reward_type"):
        label = "Hybrid Tracking (Primary)" if r_type == "hybrid_tracking" else "Normalized Power (Linear)"
        ax1.plot(grp["p_pv"], grp["total_reward"], lw=2.2, label=label)

    ax1.axvline(240.097, color="r", ls="--", label="STC MPP (240.1 W)")
    ax1.set_title("Reward vs. Extracted PV Power ($P_{pv}$)")
    ax1.set_xlabel("PV Power $P_{pv}$ [W]")
    ax1.set_ylabel("Scalar Reward $r_t$")
    ax1.legend(loc="upper left")

    # Hybrid component breakdown
    hybrid_df = df_rew[df_rew["reward_type"] == "hybrid_tracking"]
    ax2.plot(hybrid_df["p_pv"], hybrid_df["power_term"], label="Power Term $w_p (P_{pv}/P_{mpp})$", color="green", lw=1.8)
    ax2.plot(hybrid_df["p_pv"], -hybrid_df["error_penalty"], label="Tracking Error Penalty $-w_e ((P_{mpp}-P_{pv})/P_{mpp})^2$", color="crimson", lw=1.8)
    ax2.plot(hybrid_df["p_pv"], hybrid_df["total_reward"], label="Total Reward $r_t$", color="black", lw=2.2, ls="--")
    ax2.axvline(240.097, color="r", ls=":", alpha=0.7)
    ax2.set_title("Hybrid Tracking Reward Component Decomposition")
    ax2.set_xlabel("PV Power $P_{pv}$ [W]")
    ax2.set_ylabel("Term Value")
    ax2.legend(loc="lower right")

    fig.suptitle("Reward Function Formulation and Gradient Characteristics", y=0.98)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved reward components figure: {save_path}")


def plot_random_action_episode(df_rnd: pd.DataFrame, save_path: Path) -> None:
    """Generate figure illustrating complete physical response under random exploratory actions."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # 1. Commanded action and duty cycle
    axes[0].plot(df_rnd["time"], df_rnd["duty_cycle"], color="#1f77b4", lw=1.5, label="Converter Duty Cycle $D$")
    axes[0].set_ylabel("Duty Cycle $D$")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Converter Duty Cycle Evolution Under Random Slew Actions")

    # 2. PV Voltage and Current
    ax2_twin = axes[1].twinx()
    l1 = axes[1].plot(df_rnd["time"], df_rnd["pv_voltage"], color="#d62728", lw=1.5, label="PV Voltage $V_{pv}$ [V]")
    l2 = ax2_twin.plot(df_rnd["time"], df_rnd["pv_current"], color="#2ca02c", lw=1.5, label="PV Current $I_{pv}$ [A]")
    axes[1].set_ylabel("Voltage [V]")
    ax2_twin.set_ylabel("Current [A]")
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    axes[1].legend(lines, labels, loc="upper right")

    # 3. PV Power vs MPP Reference
    axes[2].plot(df_rnd["time"], df_rnd["pv_power"], color="#ff7f0e", lw=1.5, label="Actual PV Power $P_{pv}$ [W]")
    axes[2].plot(df_rnd["time"], df_rnd["mpp_power"], color="black", ls="--", lw=1.5, label="Reference $P_{mp}$ (240.1 W)")
    axes[2].set_ylabel("Power [W]")
    axes[2].legend(loc="lower right")

    # 4. Reward signal
    axes[3].plot(df_rnd["time"], df_rnd["reward"], color="#9467bd", lw=1.2, label="Scalar Reward $r_t$")
    axes[3].set_ylabel("Reward")
    axes[3].set_xlabel("Simulation Time [s]")
    axes[3].legend(loc="upper right")

    fig.suptitle("Physical Simulation Dynamics: Random Action Exploration Episode", y=0.99)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved random action episode figure: {save_path}")


def plot_environment_profiles(df_prof: pd.DataFrame, save_path: Path) -> None:
    """Generate figure showing irradiance and temperature trajectories across all 5 profiles."""
    profiles = df_prof["profile"].unique()
    fig, axes = plt.subplots(len(profiles), 2, figsize=(14, 2.5 * len(profiles)), sharex=True)

    for idx, prof_name in enumerate(profiles):
        grp = df_prof[df_prof["profile"] == prof_name]
        ax_g = axes[idx, 0]
        ax_t = axes[idx, 1]

        ax_g.plot(grp["time"], grp["irradiance"], color="#ff7f0e", lw=2.0)
        ax_g.set_ylabel(r"$G$ [W/m$^2$]")
        ax_g.set_title(f"{prof_name} — Irradiance", fontsize=11)
        ax_g.set_ylim(200, 1100)

        ax_t.plot(grp["time"], grp["temperature"], color="#d62728", lw=2.0)
        ax_t.set_ylabel(r"$T$ [$^\circ$C]")
        ax_t.set_title(f"{prof_name} — Temperature", fontsize=11)
        ax_t.set_ylim(10, 60)

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")

    fig.suptitle("Standardized Atmospheric Training and Evaluation Profiles", y=0.99)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved environment profiles figure: {save_path}")


def plot_baseline_comparison(df_base: pd.DataFrame, save_path: Path) -> None:
    """Generate figure comparing Classical P&O, Fixed Duty, and Random Actions."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    colors = {"Classical P&O": "#1f77b4", "Fixed Duty (D=0.35)": "#2ca02c", "Random Actions": "#d62728"}

    for policy_name, grp in df_base.groupby("policy"):
        ax1.plot(grp["time"], grp["pv_power"], label=policy_name, color=colors.get(policy_name, "gray"), lw=1.8)
        ax2.plot(grp["time"], grp["duty_cycle"], label=policy_name, color=colors.get(policy_name, "gray"), lw=1.8)

    ax1.axhline(240.097, color="black", ls="--", label="STC MPP (240.1 W)")
    ax1.set_ylabel("PV Extracted Power [W]")
    ax1.set_title("Extracted PV Power Comparison Across Baseline Policies")
    ax1.legend(loc="lower right")

    ax2.set_ylabel("Duty Cycle $D$")
    ax2.set_xlabel("Simulation Time [s]")
    ax2.set_title("Converter Duty Cycle Evolution")
    ax2.legend(loc="upper right")

    fig.suptitle("Environment Verification: Classical P&O vs. Fixed Duty vs. Random Actions", y=0.99)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved baseline comparison figure: {save_path}")
