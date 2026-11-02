"""Comprehensive Environment Validation and Benchmark Script for Phase 5.

Executes Gymnasium API checks, Stable-Baselines3 compatibility verification,
deterministic seeding tests, environmental profile sweeps, reward formulation diagnostics,
and baseline comparisons (P&O vs Fixed Duty vs Random Actions).

Outputs metrics to `results/metrics/phase_5/` and publication plots to `results/graphs/phase_5/`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.rl.mppt_env import SolarMPPTEnv
from app.rl.validation import (
    plot_baseline_comparison,
    plot_environment_profiles,
    plot_observation_space_ranges,
    plot_random_action_episode,
    plot_reward_components,
    run_baseline_comparison,
    run_profile_diagnostics,
    run_random_action_trajectory,
    run_reward_diagnostics,
    validate_gymnasium_api,
    validate_sb3_compatibility,
)
from app.utils.logger import get_logger
from config.default_config import MasterConfig, get_default_config

logger = get_logger("validate_rl_environment")


def main() -> int:
    """Execute complete validation suite and generate artifacts."""
    config: MasterConfig = get_default_config()

    metrics_dir = config.paths.results_dir / "metrics" / "phase_5"
    graphs_dir = config.paths.results_dir / "graphs" / "phase_5"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 5: CUSTOM GYMNASIUM SOLAR MPPT ENVIRONMENT VALIDATION")
    print("=" * 80)

    # 1. Instantiate Environment
    env = SolarMPPTEnv(config=config)
    print(f"\n[1/7] Environment Instantiated: SolarMPPTEnv")
    print(f"      Observation Space: {env.observation_space} (Dim: {env.observation_space.shape[0]})")
    print(f"      Action Space:      {env.action_space} (Dim: {env.action_space.shape[0]})")
    print(f"      Control Interval:  {env.t_rl * 1000:.1f} ms ({env.sub_steps} plant steps @ {env.dt_plant * 1e6:.1f} us)")

    # 2. Gymnasium Official Environment Checker
    print("\n[2/7] Running Gymnasium API Compliance Check...")
    gym_res = validate_gymnasium_api(env)
    print("      ✓ Gymnasium API check passed successfully.")

    # 3. Stable-Baselines3 Official Environment Checker & Monitor Wrapper
    print("\n[3/7] Running Stable-Baselines3 Compatibility Check...")
    sb3_res = validate_sb3_compatibility(env)
    print("      ✓ SB3 check_env passed successfully.")
    print("      ✓ SB3 Monitor wrapper reset and step verified.")

    # 4. Deterministic Seeding Test
    print("\n[4/7] Verifying Deterministic Seeding...")
    env_seed_a = SolarMPPTEnv(config=config)
    env_seed_b = SolarMPPTEnv(config=config)
    obs_a1, info_a1 = env_seed_a.reset(seed=12345, options={"profile_type": "random_constant", "randomize_duty": True})
    obs_b1, info_b1 = env_seed_b.reset(seed=12345, options={"profile_type": "random_constant", "randomize_duty": True})
    np.testing.assert_allclose(obs_a1, obs_b1, err_msg="Identical seeds produced divergent observations on reset.")
    assert info_a1["duty_cycle"] == info_b1["duty_cycle"], "Identical seeds produced divergent initial duties."

    # Different seeds produce different conditions
    obs_c1, info_c1 = env_seed_a.reset(seed=99999, options={"profile_type": "random_constant", "randomize_duty": True})
    assert not np.allclose(obs_a1, obs_c1), "Different seeds should produce different observations in randomized mode."
    print("      ✓ Deterministic seeding verified (identical seeds match, different seeds diverge).")

    # 5. Random Action Trajectory & Visualizations
    print("\n[5/7] Executing Random Action Trajectory & Generating Plots...")
    df_rnd = run_random_action_trajectory(env, seed=42, num_steps=300)
    plot_observation_space_ranges(df_rnd, graphs_dir / "observation_space_ranges.png")
    plot_random_action_episode(df_rnd, graphs_dir / "random_action_episode.png")
    print("      ✓ Saved 'observation_space_ranges.png' (300 DPI)")
    print("      ✓ Saved 'random_action_episode.png' (300 DPI)")

    # 6. Environmental Profile Sweeps
    print("\n[6/7] Sweeping Atmospheric Profiles & Reward Formulations...")
    df_prof = run_profile_diagnostics(config, duration=1.5, dt=0.005)
    df_prof.to_csv(metrics_dir / "environment_profile_summary.csv", index=False)
    plot_environment_profiles(df_prof, graphs_dir / "environment_profiles.png")
    print("      ✓ Saved 'environment_profile_summary.csv'")
    print("      ✓ Saved 'environment_profiles.png' (300 DPI)")

    df_rew = run_reward_diagnostics(config)
    df_rew.to_csv(metrics_dir / "reward_diagnostic_summary.csv", index=False)
    plot_reward_components(df_rew, graphs_dir / "reward_components.png")
    print("      ✓ Saved 'reward_diagnostic_summary.csv'")
    print("      ✓ Saved 'reward_components.png' (300 DPI)")

    # 7. Baseline Policy Comparison (P&O vs Fixed vs Random)
    print("\n[7/7] Benchmarking Baseline Policies (P&O, Fixed Duty, Random)...")
    df_base_timeseries, df_base_summary = run_baseline_comparison(config, num_steps=300, seed=42)
    df_base_summary.to_csv(metrics_dir / "baseline_comparison.csv", index=False)
    plot_baseline_comparison(df_base_timeseries, graphs_dir / "baseline_comparison.png")
    print("      ✓ Saved 'baseline_comparison.csv'")
    print("      ✓ Saved 'baseline_comparison.png' (300 DPI)")

    # Save complete JSON validation summary
    validation_summary = {
        "gymnasium_api": gym_res,
        "stable_baselines3": sb3_res,
        "timing": {
            "control_interval_s": env.t_rl,
            "plant_dt_s": env.dt_plant,
            "sub_steps": env.sub_steps,
            "max_steps": env.max_steps,
            "episode_duration_s": env.t_rl * env.max_steps,
        },
        "observation_space": {
            "dimension": env.observation_space.shape[0],
            "low": float(env.observation_space.low[0]),
            "high": float(env.observation_space.high[0]),
            "features": env.obs_processor.feature_names,
        },
        "action_space": {
            "dimension": env.action_space.shape[0],
            "low": float(env.action_space.low[0]),
            "high": float(env.action_space.high[0]),
            "delta_duty_max": env.delta_duty_max,
        },
        "baseline_summary": df_base_summary.to_dict(orient="records"),
    }

    with open(metrics_dir / "environment_validation_summary.json", "w") as f:
        json.dump(validation_summary, f, indent=4)
    print("      ✓ Saved 'environment_validation_summary.json'")

    print("\n" + "=" * 80)
    print("BASELINE POLICY PERFORMANCE SUMMARY (STC: 1000 W/m², 25°C, Reference Pmp: 240.10 W)")
    print("=" * 80)
    print(df_base_summary.to_string(index=False))
    print("=" * 80)
    print("\nPhase 5 Environment Validation completed successfully with 0 errors.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
