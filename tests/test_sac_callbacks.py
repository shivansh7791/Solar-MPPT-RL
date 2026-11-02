"""Unit Tests for MPPT Training Callbacks."""

from pathlib import Path
import numpy as np
import pytest
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from app.rl.callbacks import MPPTTrainingCallback
from app.rl.mppt_env import SolarMPPTEnv
from app.rl.sac_agent import create_sac_agent
from app.rl.sac_config import SACConfig
from config.default_config import RLEnvConfig


class TestMPPTTrainingCallback:
    """Test telemetry logging, checkpoint creation, and history recording."""

    @pytest.fixture
    def env(self) -> Monitor:
        raw_env = SolarMPPTEnv(rl_env_config=RLEnvConfig(include_mpp_in_obs=False, max_steps_per_episode=20))
        return Monitor(raw_env)

    def test_callback_records_episodes_and_saves_history(self, env: Monitor, tmp_path: Path) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        best_dir = tmp_path / "best_model"

        callback = MPPTTrainingCallback(
            checkpoint_dir=ckpt_dir,
            best_model_dir=best_dir,
            checkpoint_interval=50,
            eval_interval=20,
            verbose=0,
        )

        cfg = SACConfig(learning_starts=20, buffer_size=500, batch_size=32)
        agent = create_sac_agent(env=env, config=cfg, seed=42)

        # Train for short 60 steps (3 episodes of 20 steps)
        agent.learn(total_timesteps=60, callback=callback)

        history_df = callback.get_history_dataframe()
        assert len(history_df) >= 2
        assert "mean_tracking_efficiency" in history_df.columns
        assert "mean_pv_power" in history_df.columns
        assert "episode_reward" in history_df.columns

        # Verify periodic checkpoint created at step 50
        ckpts = list(ckpt_dir.glob("*.zip"))
        assert len(ckpts) >= 1

        # Test CSV export
        csv_file = tmp_path / "history.csv"
        callback.save_history_csv(csv_file)
        assert csv_file.is_file()
