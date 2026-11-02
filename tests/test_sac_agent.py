"""Unit Tests for SAC Agent Factory, Architecture, and Checkpoint Management."""

from pathlib import Path
import numpy as np
import pytest
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from app.rl.mppt_env import SolarMPPTEnv
from app.rl.sac_agent import create_sac_agent, get_agent_summary, load_sac_agent, predict_mppt_action
from app.rl.sac_config import SACConfig
from config.default_config import RLEnvConfig


class TestSACAgentCreation:
    """Test SAC model instantiation and network properties."""

    @pytest.fixture
    def env(self) -> Monitor:
        raw_env = SolarMPPTEnv(rl_env_config=RLEnvConfig(include_mpp_in_obs=False))
        return Monitor(raw_env)

    def test_agent_initialization(self, env: Monitor) -> None:
        cfg = SACConfig(learning_starts=100, buffer_size=1000, batch_size=64)
        agent = create_sac_agent(env=env, config=cfg, seed=42)

        assert isinstance(agent, SAC)
        assert agent.observation_space.shape == (8,)
        assert agent.action_space.shape == (1,)
        assert agent.batch_size == 64
        assert agent.buffer_size == 1000

    def test_agent_summary_diagnostics(self, env: Monitor) -> None:
        cfg = SACConfig(net_arch_pi=[256, 256], net_arch_qf=[256, 256])
        agent = create_sac_agent(env=env, config=cfg, seed=42)
        summary = get_agent_summary(agent)

        assert summary["algorithm"] == "SAC"
        assert summary["total_trainable_parameters"] > 200_000
        assert summary["actor_parameters"] > 60_000
        assert summary["critic_parameters"] > 130_000

    def test_action_prediction_bounds(self, env: Monitor) -> None:
        cfg = SACConfig()
        agent = create_sac_agent(env=env, config=cfg, seed=42)

        obs, info = env.reset()
        action = predict_mppt_action(agent, obs, deterministic=True)

        assert isinstance(action, float)
        assert -1.0 <= action <= 1.0


class TestSACCheckpointing:
    """Test model serialization, saving, and reloading."""

    @pytest.fixture
    def env(self) -> Monitor:
        raw_env = SolarMPPTEnv(rl_env_config=RLEnvConfig(include_mpp_in_obs=False))
        return Monitor(raw_env)

    def test_save_and_reload_deterministic_inference(self, env: Monitor, tmp_path: Path) -> None:
        cfg = SACConfig(learning_starts=50, buffer_size=1000)
        agent = create_sac_agent(env=env, config=cfg, seed=42)

        model_path = tmp_path / "test_sac_model.zip"
        agent.save(str(model_path))
        assert model_path.is_file()

        # Reload model
        loaded_agent = load_sac_agent(model_path, env=env)
        assert isinstance(loaded_agent, SAC)

        obs, info = env.reset()
        act_orig = predict_mppt_action(agent, obs, deterministic=True)
        act_reloaded = predict_mppt_action(loaded_agent, obs, deterministic=True)

        assert np.isclose(act_orig, act_reloaded, atol=1e-5)

    def test_nonexistent_model_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_sac_agent("nonexistent_model_path_12345.zip")
