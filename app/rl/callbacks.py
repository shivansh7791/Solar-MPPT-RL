"""Custom Callbacks and Metrics Tracking for SAC Solar MPPT Training.

Collects power electronics telemetry, logs to TensorBoard, saves periodic checkpoints,
tracks best models based on tracking efficiency, and exports structured CSV training histories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from app.utils.logger import get_logger

logger = get_logger("sac_callbacks")


class MPPTTrainingCallback(BaseCallback):
    """Custom callback tracking solar MPPT physical metrics during SAC policy optimization."""

    def __init__(
        self,
        checkpoint_dir: Path | str,
        best_model_dir: Path | str,
        checkpoint_interval: int = 10_000,
        eval_interval: int = 5_000,
        eval_episodes: int = 5,
        verbose: int = 1,
    ) -> None:
        """Initialize callback.

        Parameters
        ----------
        checkpoint_dir : Path | str
            Directory to store periodic model checkpoints.
        best_model_dir : Path | str
            Directory to store the best model checkpoint.
        checkpoint_interval : int, optional
            Step cadence for periodic checkpoints, by default 10_000.
        eval_interval : int, optional
            Step cadence for logging summary statistics, by default 5_000.
        eval_episodes : int, optional
            Number of evaluation episodes, by default 5.
        verbose : int, optional
            Verbosity level, by default 1.
        """
        super().__init__(verbose)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_model_dir = Path(best_model_dir)
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

        self.best_mean_reward: float = -float("inf")
        self.best_tracking_efficiency: float = 0.0

        # Detailed step-by-step history tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_efficiencies: List[float] = []
        self.episode_powers: List[float] = []
        self.history_records: List[Dict[str, Any]] = []

        # Running window tracking
        self._current_ep_rewards: List[float] = []
        self._current_ep_efficiencies: List[float] = []
        self._current_ep_powers: List[float] = []
        self._fault_count: int = 0
        self._total_episodes: int = 0

    def _on_step(self) -> bool:
        """Execute on every environment step."""
        # Extract info from environment
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        if infos and len(infos) > 0:
            info = infos[0]
            step_reward = float(rewards[0]) if len(rewards) > 0 else 0.0

            p_pv = float(info.get("pv_power", 0.0))
            eff = float(info.get("tracking_efficiency", 0.0))
            duty = float(info.get("duty_cycle", 0.0))

            self._current_ep_rewards.append(step_reward)
            self._current_ep_efficiencies.append(eff)
            self._current_ep_powers.append(p_pv)

            # Check if episode ended
            dones = self.locals.get("dones", [False])
            if dones[0]:
                self._total_episodes += 1
                ep_reward = sum(self._current_ep_rewards)
                ep_length = len(self._current_ep_rewards)
                mean_eff = float(np.mean(self._current_ep_efficiencies)) if self._current_ep_efficiencies else 0.0
                mean_power = float(np.mean(self._current_ep_powers)) if self._current_ep_powers else 0.0

                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.episode_efficiencies.append(mean_eff)
                self.episode_powers.append(mean_power)

                # Record history row
                record = {
                    "step": self.num_timesteps,
                    "episode": self._total_episodes,
                    "episode_reward": ep_reward,
                    "episode_length": ep_length,
                    "mean_tracking_efficiency": mean_eff,
                    "mean_pv_power": mean_power,
                    "final_duty_cycle": duty,
                    "fault_count": self._fault_count,
                }

                # Try to extract SAC loss values from logger if available
                if hasattr(self.model, "logger") and self.model.logger:
                    name_to_value = getattr(self.model.logger, "name_to_value", {})
                    record["actor_loss"] = float(name_to_value.get("train/actor_loss", np.nan))
                    record["critic_loss"] = float(name_to_value.get("train/critic_loss", np.nan))
                    record["ent_coef"] = float(name_to_value.get("train/ent_coef", np.nan))

                self.history_records.append(record)

                # Log to TensorBoard
                self.logger.record("mppt/episode_reward", ep_reward)
                self.logger.record("mppt/episode_length", ep_length)
                self.logger.record("mppt/tracking_efficiency", mean_eff)
                self.logger.record("mppt/mean_pv_power", mean_power)
                self.logger.record("mppt/duty_cycle", duty)

                # Check if this is the best model so far (based on tracking efficiency and reward)
                if mean_eff > self.best_tracking_efficiency and self.num_timesteps >= 2_000:
                    self.best_tracking_efficiency = mean_eff
                    self.best_mean_reward = ep_reward
                    best_model_path = self.best_model_dir / "best_mppt_model"
                    self.model.save(str(best_model_path))
                    if self.verbose > 0:
                        logger.info(
                            f"New best model saved at step {self.num_timesteps} "
                            f"(Tracking Eff: {mean_eff:.2f}%, Reward: {ep_reward:.2f})"
                        )

                # Reset episode accumulators
                self._current_ep_rewards = []
                self._current_ep_efficiencies = []
                self._current_ep_powers = []

        # Periodic model checkpointing
        if self.checkpoint_interval > 0 and self.num_timesteps % self.checkpoint_interval == 0:
            ckpt_path = self.checkpoint_dir / f"sac_mppt_step_{self.num_timesteps}"
            self.model.save(str(ckpt_path))
            if self.verbose > 0:
                logger.info(f"Saved periodic checkpoint: {ckpt_path}.zip")

        return True

    def get_history_dataframe(self) -> pd.DataFrame:
        """Return collected training history as a DataFrame."""
        return pd.DataFrame(self.history_records)

    def save_history_csv(self, file_path: Path | str) -> None:
        """Export training history records to CSV."""
        df = self.get_history_dataframe()
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        logger.info(f"Saved training history ({len(df)} episodes) to: {p}")
