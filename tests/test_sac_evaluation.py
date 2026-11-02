"""Unit Tests for SAC Policy Evaluation Module."""

import numpy as np
import pytest
from stable_baselines3.common.monitor import Monitor

from app.rl.episode_profiles import ProfileType
from app.rl.evaluation import evaluate_sac_across_all_profiles, evaluate_sac_policy_on_profile
from app.rl.mppt_env import SolarMPPTEnv
from app.rl.sac_agent import create_sac_agent
from app.rl.sac_config import SACConfig
from config.default_config import RLEnvConfig


class TestSACEvaluation:
    """Test policy evaluation across atmospheric profiles."""

    @pytest.fixture
    def agent(self) -> any:
        raw_env = SolarMPPTEnv(rl_env_config=RLEnvConfig(include_mpp_in_obs=False))
        train_env = Monitor(raw_env)
        cfg = SACConfig(learning_starts=20, buffer_size=500, batch_size=32)
        agent = create_sac_agent(env=train_env, config=cfg, seed=42)
        return agent

    def test_single_profile_evaluation(self, agent) -> None:
        df, metrics = evaluate_sac_policy_on_profile(
            model=agent,
            profile_type=ProfileType.STC,
            include_mpp_in_obs=False,
            num_steps=30,
            seed=42,
        )

        assert len(df) == 31  # 30 steps + initial row
        assert "energy_tracking_efficiency_percent" in metrics
        assert "mean_extracted_power_W" in metrics
        assert "ss_power_ripple_W" in metrics
        assert metrics["steps_completed"] == 30
        assert 0.0 <= metrics["energy_tracking_efficiency_percent"] <= 100.0

    def test_evaluation_determinism(self, agent) -> None:
        df1, m1 = evaluate_sac_policy_on_profile(
            model=agent,
            profile_type=ProfileType.STC,
            include_mpp_in_obs=False,
            num_steps=20,
            seed=123,
        )
        df2, m2 = evaluate_sac_policy_on_profile(
            model=agent,
            profile_type=ProfileType.STC,
            include_mpp_in_obs=False,
            num_steps=20,
            seed=123,
        )

        np.testing.assert_allclose(df1["pv_power"].values, df2["pv_power"].values)
        np.testing.assert_allclose(df1["duty_cycle"].values, df2["duty_cycle"].values)
        assert m1["energy_tracking_efficiency_percent"] == m2["energy_tracking_efficiency_percent"]

    def test_all_profiles_evaluation_sweep(self, agent) -> None:
        timeseries_dict, summary_df = evaluate_sac_across_all_profiles(
            model=agent,
            include_mpp_in_obs=False,
            num_steps=20,
            seed=42,
        )

        assert len(summary_df) == 5
        assert set(summary_df["profile"]) == {
            "stc",
            "random_constant",
            "irradiance_step",
            "temperature_step",
            "combined",
        }
        for p in summary_df["profile"]:
            assert p in timeseries_dict
