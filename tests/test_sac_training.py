"""Unit Tests for SAC Training Pipeline."""

from pathlib import Path
import pytest

from app.rl.sac_config import SACConfig
from app.rl.training import run_sac_training, set_deterministic_seeds
from config.default_config import get_default_config


class TestSACTrainingPipeline:
    """Test full training pipeline execution and output artifacts."""

    def test_seed_setting(self) -> None:
        set_deterministic_seeds(42)

    def test_training_pipeline_smoke_run(self, tmp_path: Path) -> None:
        cfg = get_default_config()
        # Override paths to temporary directory
        cfg.paths.root_dir = tmp_path

        s_cfg = SACConfig(
            total_timesteps=60,
            learning_starts=20,
            buffer_size=500,
            batch_size=32,
            checkpoint_interval=30,
            seed=42,
        )

        summary = run_sac_training(
            sac_config=s_cfg,
            master_config=cfg,
            mode="realistic",
            timesteps=60,
            seed=42,
        )

        assert summary["mode"] == "realistic"
        assert summary["is_privileged"] is False
        assert summary["total_timesteps_completed"] == 60
        assert summary["duration_seconds"] > 0
        assert summary["throughput_steps_per_second"] > 0
        assert Path(summary["final_model_path"]).is_file()
        assert len(summary["evaluation_summary"]) == 5
