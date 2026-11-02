"""Training Diagnostics and Visualization Suite for SAC Solar MPPT Control.

Generates publication-quality 300 DPI figures for:
1. Episode reward vs training episodes
2. Moving average episode reward
3. Actor loss vs training steps
4. Critic loss vs training steps
5. Entropy coefficient vs training steps
6. Mean MPP tracking efficiency during evaluation
7. Duty-cycle distribution
8. Episode length and termination diagnostics
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger("training_visualization")

# Publication plot styling
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


def generate_all_phase_6_plots(
    df_history: pd.DataFrame,
    df_eval: pd.DataFrame,
    eval_timeseries: Optional[Dict[str, pd.DataFrame]] = None,
    output_dir: Path | str = "results/graphs/phase_6",
) -> Dict[str, Path]:
    """Generate and save all 8 required training diagnostic figures at 300 DPI."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    has_history = df_history is not None and len(df_history) > 0

    # 1. Episode reward vs training episodes
    p1 = out_dir / "episode_reward_vs_episodes.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    if has_history:
        ax.plot(df_history["episode"], df_history["episode_reward"], color="#1f77b4", alpha=0.6, label="Raw Episode Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Episode Reward")
        ax.set_title("SAC Training Progress: Episode Reward vs. Training Episodes")
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close()
    plot_paths["episode_reward"] = p1

    # 2. Moving average episode reward
    p2 = out_dir / "moving_average_reward.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    if has_history:
        window = min(20, max(2, len(df_history) // 10))
        roll_mean = df_history["episode_reward"].rolling(window=window, min_periods=1).mean()
        roll_std = df_history["episode_reward"].rolling(window=window, min_periods=1).std().fillna(0)

        ax.plot(df_history["episode"], roll_mean, color="#2ca02c", lw=2.0, label=f"Moving Average (window={window})")
        ax.fill_between(df_history["episode"], roll_mean - roll_std, roll_mean + roll_std, color="#2ca02c", alpha=0.2, label=r"$\pm 1\sigma$ Band")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("SAC Convergence: Moving Average Episode Reward")
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close()
    plot_paths["moving_average_reward"] = p2

    # 3. Actor loss vs training steps
    p3 = out_dir / "actor_loss_vs_steps.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    if has_history and "actor_loss" in df_history.columns and df_history["actor_loss"].notna().any():
        valid = df_history.dropna(subset=["actor_loss"])
        ax.plot(valid["step"], valid["actor_loss"], color="#d62728", lw=1.5, label="Policy Loss")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Loss")
        ax.set_title("SAC Optimization: Actor (Policy) Loss vs. Training Steps")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "Actor loss recorded at gradient updates", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(p3, dpi=300)
    plt.close()
    plot_paths["actor_loss"] = p3

    # 4. Critic loss vs training steps
    p4 = out_dir / "critic_loss_vs_steps.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    if has_history and "critic_loss" in df_history.columns and df_history["critic_loss"].notna().any():
        valid = df_history.dropna(subset=["critic_loss"])
        ax.plot(valid["step"], valid["critic_loss"], color="#9467bd", lw=1.5, label="Q-Value Loss")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Loss")
        ax.set_title("SAC Optimization: Twin Critic (QF) Loss vs. Training Steps")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "Critic loss recorded at gradient updates", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(p4, dpi=300)
    plt.close()
    plot_paths["critic_loss"] = p4

    # 5. Entropy coefficient vs training steps
    p5 = out_dir / "entropy_coef_vs_steps.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    if has_history and "ent_coef" in df_history.columns and df_history["ent_coef"].notna().any():
        valid = df_history.dropna(subset=["ent_coef"])
        ax.plot(valid["step"], valid["ent_coef"], color="#ff7f0e", lw=1.8, label=r"Temperature $\alpha$")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(r"Entropy Coefficient $\alpha$")
        ax.set_title(r"SAC Exploration: Automatic Entropy Coefficient $\alpha$ vs. Steps")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "Automatic entropy tuning: target = -dim(A)", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(p5, dpi=300)
    plt.close()
    plot_paths["entropy_coef"] = p5

    # 6. Mean MPP tracking efficiency during evaluation
    p6 = out_dir / "tracking_efficiency_evaluation.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    if df_eval is not None and len(df_eval) > 0:
        bars = ax.bar(
            df_eval["profile"],
            df_eval["energy_tracking_efficiency_percent"],
            color="#17becf",
            edgecolor="black",
            width=0.5,
        )
        ax.set_ylabel("Energy Tracking Efficiency [%]")
        ax.set_ylim(80, 105)
        ax.set_title("Deterministic Policy Evaluation: MPPT Tracking Efficiency by Profile")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(p6, dpi=300)
    plt.close()
    plot_paths["tracking_efficiency"] = p6

    # 7. Duty-cycle distribution
    p7 = out_dir / "duty_cycle_distribution.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    if has_history and "final_duty_cycle" in df_history.columns:
        ax.hist(df_history["final_duty_cycle"], bins=20, color="#8c564b", edgecolor="black", alpha=0.7)
        ax.axvline(0.3939, color="r", ls="--", lw=2, label="STC MPP Duty D* = 0.3939")
        ax.set_xlabel("Converter Duty Cycle $D$")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Operating Duty Cycles Across Training Episodes")
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(p7, dpi=300)
    plt.close()
    plot_paths["duty_distribution"] = p7

    # 8. Episode length and termination diagnostics
    p8 = out_dir / "episode_length_diagnostics.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    if has_history:
        ax.plot(df_history["episode"], df_history["episode_length"], color="#bcbd22", lw=1.5, label="Episode Steps Completed")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.set_title("Episode Stability & Length Diagnostics")
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(p8, dpi=300)
    plt.close()
    plot_paths["episode_length"] = p8

    logger.info(f"Generated {len(plot_paths)} Phase 6 diagnostic plots at 300 DPI in: {out_dir}")
    return plot_paths
