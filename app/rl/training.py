"""Staged Training Pipeline for SAC Reinforcement Learning MPPT Agent.

Coordinates:
- Stage A: Smoke Test (2,000 steps)
- Stage B: Realistic 8-Dimensional Agent Training (up to 100,000 steps)
- Stage C: Privileged 10-Dimensional Agent Training (optional)
- TensorBoard logging, checkpoint management, and automatic evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import torch

from app.rl.callbacks import MPPTTrainingCallback
from app.rl.evaluation import evaluate_sac_across_all_profiles
from app.rl.mppt_env import SolarMPPTEnv
from app.rl.sac_agent import create_sac_agent, get_agent_summary
from app.rl.sac_config import SACConfig
from app.rl.training_visualization import generate_all_phase_6_plots
from app.utils.logger import get_logger
from config.default_config import MasterConfig, RLEnvConfig, get_default_config

logger = get_logger("sac_training")


def set_deterministic_seeds(seed: int = 42) -> None:
    """Set global random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_sac_training(
    sac_config: Optional[SACConfig] = None,
    master_config: Optional[MasterConfig] = None,
    mode: str = "realistic",
    timesteps: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute SAC training pipeline with metrics tracking, checkpointing, and evaluation.

    Parameters
    ----------
    sac_config : Optional[SACConfig], optional
        SAC hyperparameters and network settings. If None, loaded from defaults.
    master_config : Optional[MasterConfig], optional
        Master project configuration. If None, loaded from project defaults.
    mode : str, optional
        Experiment mode: 'realistic' (8-dim) or 'privileged' (10-dim).
    timesteps : Optional[int], optional
        Override total timesteps. If None, uses sac_config.total_timesteps.
    seed : Optional[int], optional
        Override random seed. If None, uses sac_config.seed.

    Returns
    -------
    Dict[str, Any]
        Structured training summary dictionary.
    """
    m_cfg = master_config if master_config is not None else get_default_config()
    s_cfg = sac_config if sac_config is not None else SACConfig()

    effective_seed = seed if seed is not None else s_cfg.seed
    set_deterministic_seeds(effective_seed)

    is_privileged = (mode.lower() == "privileged")
    s_cfg.include_mpp_in_obs = is_privileged
    if timesteps is not None:
        s_cfg.total_timesteps = timesteps

    # Set up directory layout
    root_dir = m_cfg.paths.root_dir
    mode_name = "privileged" if is_privileged else "realistic"
    artifact_dir = root_dir / "artifacts" / "rl" / mode_name
    checkpoint_dir = artifact_dir / "checkpoints"
    best_model_dir = artifact_dir / "best_model"
    final_model_dir = artifact_dir / "final_model"
    log_dir = artifact_dir / "logs"
    graphs_dir = root_dir / "results" / "graphs" / "phase_6"
    metrics_dir = root_dir / "results" / "metrics" / "phase_6"

    for d in [checkpoint_dir, best_model_dir, final_model_dir, log_dir, graphs_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    s_cfg.artifacts_dir = str(artifact_dir)
    s_cfg.tensorboard_log = str(log_dir)

    # Save hyperparameters
    config_json_path = artifact_dir / "config.json"
    s_cfg.save_json(config_json_path)

    # Also save to results/metrics/phase_6
    s_cfg.save_json(metrics_dir / "hyperparameters.json")

    logger.info("=" * 80)
    logger.info(f"STARTING SAC MPPT TRAINING [{mode_name.upper()} MODE - {s_cfg.total_timesteps:,} STEPS]")
    logger.info("=" * 80)
    logger.info(f"Observation Mode: {'Privileged (10-dim)' if is_privileged else 'Realistic (8-dim)'}")
    logger.info(f"Seed: {effective_seed} | Target Steps: {s_cfg.total_timesteps:,}")

    # Build Training Environment
    env_cfg = RLEnvConfig(
        include_mpp_in_obs=is_privileged,
        profile_type=s_cfg.profile_type,
        randomize_initial_duty=True,
        randomize_environment=True,
    )
    raw_env = SolarMPPTEnv(config=m_cfg, rl_env_config=env_cfg)
    train_env = Monitor(raw_env, filename=str(log_dir / "train_monitor.csv"))

    # Instantiate SAC Agent
    agent = create_sac_agent(
        env=train_env,
        config=s_cfg,
        seed=effective_seed,
        tensorboard_log=str(log_dir),
    )
    agent_summary = get_agent_summary(agent)
    logger.info(f"Model Summary: Total Trainable Parameters = {agent_summary['total_trainable_parameters']:,}")

    # Set up callback
    callback = MPPTTrainingCallback(
        checkpoint_dir=checkpoint_dir,
        best_model_dir=best_model_dir,
        checkpoint_interval=s_cfg.checkpoint_interval,
        eval_interval=s_cfg.eval_interval,
        eval_episodes=s_cfg.eval_episodes,
        verbose=s_cfg.verbose,
    )

    # Train Model
    t_start = time.perf_counter()
    logger.info(f"Beginning policy optimization over {s_cfg.total_timesteps:,} timesteps...")
    agent.learn(total_timesteps=s_cfg.total_timesteps, callback=callback)
    t_end = time.perf_counter()
    duration_s = t_end - t_start
    fps = s_cfg.total_timesteps / max(0.001, duration_s)

    logger.info(f"Training completed in {duration_s:.2f} s ({fps:.1f} steps/s).")

    # Save Final Model
    final_model_path = final_model_dir / "sac_mppt_final"
    agent.save(str(final_model_path))
    logger.info(f"Saved final model to: {final_model_path}.zip")

    # Save Training History CSV
    df_history = callback.get_history_dataframe()
    history_csv_path = metrics_dir / "training_history.csv"
    callback.save_history_csv(history_csv_path)
    callback.save_history_csv(artifact_dir / "training_history.csv")

    # Evaluate Final Model Across All Five Profiles
    logger.info("Executing comprehensive post-training evaluation across all 5 profiles...")
    eval_timeseries, df_eval = evaluate_sac_across_all_profiles(
        model=agent,
        include_mpp_in_obs=is_privileged,
        num_steps=300,
        seed=effective_seed,
        config=m_cfg,
    )

    # Save Evaluation History CSV
    eval_csv_path = metrics_dir / "evaluation_history.csv"
    df_eval.to_csv(eval_csv_path, index=False)
    df_eval.to_csv(artifact_dir / "evaluation_history.csv", index=False)
    logger.info(f"Saved evaluation metrics to: {eval_csv_path}")

    # Generate 8 Training Diagnostic Figures
    plot_paths = generate_all_phase_6_plots(
        df_history=df_history,
        df_eval=df_eval,
        eval_timeseries=eval_timeseries,
        output_dir=graphs_dir,
    )

    # Build Training Summary
    summary = {
        "mode": mode_name,
        "is_privileged": is_privileged,
        "total_timesteps_target": s_cfg.total_timesteps,
        "total_timesteps_completed": int(agent.num_timesteps),
        "total_episodes_completed": len(df_history),
        "duration_seconds": round(duration_s, 2),
        "throughput_steps_per_second": round(fps, 1),
        "device_used": agent_summary["device"],
        "random_seed": effective_seed,
        "model_summary": agent_summary,
        "final_model_path": str(final_model_path) + ".zip",
        "best_model_path": str(best_model_dir / "best_mppt_model.zip"),
        "best_tracking_efficiency_achieved": round(float(callback.best_tracking_efficiency), 2),
        "mean_evaluation_tracking_efficiency": round(float(df_eval["energy_tracking_efficiency_percent"].mean()), 2),
        "evaluation_summary": df_eval.to_dict(orient="records"),
    }

    summary_json_path = metrics_dir / "training_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=4)
    with open(artifact_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Saved training summary to: {summary_json_path}")
    logger.info("=" * 80)
    logger.info("PHASE 6 SAC TRAINING AND EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    return summary
