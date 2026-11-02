"""Comprehensive Unit and Integration Tests for SolarMPPTEnv."""

import math
import gymnasium as gym
from gymnasium.utils.env_checker import check_env as gym_check_env
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3.common.monitor import Monitor

from app.rl.mppt_env import SolarMPPTEnv
from config.default_config import MasterConfig, RLEnvConfig, get_default_config


class TestSolarMPPTEnvGymnasiumAPI:
    """Test Gymnasium API conformance, space definitions, and return types."""

    def test_environment_initialization(self) -> None:
        env = SolarMPPTEnv()
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.observation_space.shape == (10,)
        assert env.observation_space.dtype == np.float32
        assert env.action_space.shape == (1,)
        assert env.action_space.dtype == np.float32
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0

    def test_non_privileged_mode(self) -> None:
        cfg = get_default_config()
        cfg.environment.include_mpp_in_obs = False
        env = SolarMPPTEnv(config=cfg)
        assert env.observation_space.shape == (8,)
        obs, info = env.reset(seed=42)
        assert obs.shape == (8,)

    def test_official_gymnasium_checker_passes(self) -> None:
        env = SolarMPPTEnv()
        gym_check_env(env.unwrapped)

    def test_official_sb3_checker_passes(self) -> None:
        env = SolarMPPTEnv()
        sb3_check_env(env.unwrapped)

    def test_sb3_monitor_wrapper_compatibility(self) -> None:
        env = SolarMPPTEnv()
        monitored_env = Monitor(env)
        obs, info = monitored_env.reset(seed=42)
        assert isinstance(obs, np.ndarray)

        step_obs, reward, term, trunc, step_info = monitored_env.step(np.array([0.5], dtype=np.float32))
        assert math.isfinite(reward)
        assert not term
        assert not trunc


class TestEnvironmentResetAndSeeding:
    """Test reset behavior, options overrides, and deterministic reproducibility."""

    def test_deterministic_seeding_identical(self) -> None:
        env1 = SolarMPPTEnv()
        env2 = SolarMPPTEnv()

        obs1, info1 = env1.reset(seed=1234, options={"profile_type": "random_constant", "randomize_duty": True})
        obs2, info2 = env2.reset(seed=1234, options={"profile_type": "random_constant", "randomize_duty": True})

        np.testing.assert_allclose(obs1, obs2)
        assert info1["duty_cycle"] == info2["duty_cycle"]
        assert info1["irradiance"] == info2["irradiance"]
        assert info1["cell_temperature"] == info2["cell_temperature"]

    def test_deterministic_seeding_divergent(self) -> None:
        env = SolarMPPTEnv()
        obs1, info1 = env.reset(seed=1234, options={"profile_type": "random_constant", "randomize_duty": True})
        obs2, info2 = env.reset(seed=5678, options={"profile_type": "random_constant", "randomize_duty": True})

        assert not np.allclose(obs1, obs2)

    def test_custom_options_override(self) -> None:
        env = SolarMPPTEnv()
        obs, info = env.reset(options={"initial_duty": 0.45, "profile_type": "stc"})
        assert math.isclose(info["duty_cycle"], 0.45, abs_tol=1e-4)
        assert info["irradiance"] == 1000.0
        assert info["cell_temperature"] == 25.0


class TestEnvironmentSteppingAndPhysics:
    """Test plant multirate integration, duty cycle updates, and safety bounds."""

    def test_duty_cycle_incremental_update(self) -> None:
        env = SolarMPPTEnv()
        obs, info = env.reset(seed=42, options={"initial_duty": 0.35, "randomize_duty": False})
        assert math.isclose(env.current_duty, 0.35, abs_tol=1e-4)

        # Action = +1.0 -> Delta D = +0.01 -> New D = 0.36
        obs, r, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
        assert math.isclose(env.current_duty, 0.36, abs_tol=1e-4)
        assert math.isclose(info["duty_adjustment"], 0.01, abs_tol=1e-4)
        assert not info["duty_saturated"]

        # Action = -0.5 -> Delta D = -0.005 -> New D = 0.355
        obs, r, term, trunc, info = env.step(np.array([-0.5], dtype=np.float32))
        assert math.isclose(env.current_duty, 0.355, abs_tol=1e-4)
        assert math.isclose(info["duty_adjustment"], -0.005, abs_tol=1e-4)

    def test_duty_cycle_saturation_limits(self) -> None:
        env = SolarMPPTEnv()
        # Start near upper limit 0.90
        env.reset(options={"initial_duty": 0.895})
        obs, r, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))

        # Commanded 0.895 + 0.01 = 0.905 -> saturated to 0.90
        assert math.isclose(env.current_duty, 0.90, abs_tol=1e-4)
        assert math.isclose(info["duty_adjustment"], 0.005, abs_tol=1e-4)
        assert info["duty_saturated"]

    def test_action_clipping(self) -> None:
        env = SolarMPPTEnv()
        env.reset(options={"initial_duty": 0.35})

        # Action = +5.0 (exceeds high=1.0) -> clipped to 1.0 -> Delta D = 0.01
        obs, r, term, trunc, info = env.step(np.array([5.0], dtype=np.float32))
        assert math.isclose(env.current_duty, 0.36, abs_tol=1e-4)

    @pytest.mark.parametrize("bad_act", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_action_rejection(self, bad_act) -> None:
        env = SolarMPPTEnv()
        env.reset()
        with pytest.raises(TypeError):
            env.step(np.array([bad_act], dtype=np.float32))

    def test_multirate_timing_advancement(self) -> None:
        env = SolarMPPTEnv()
        env.reset()
        assert env.current_step == 0
        assert env.simulation_time == 0.0

        for s in range(1, 11):
            obs, r, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
            assert env.current_step == s
            assert math.isclose(env.simulation_time, s * 0.005, abs_tol=1e-6)
            assert math.isclose(info["simulation_time"], s * 0.005, abs_tol=1e-6)

    def test_episode_truncation_at_max_steps(self) -> None:
        cfg = get_default_config()
        cfg.environment.max_steps_per_episode = 10
        env = SolarMPPTEnv(config=cfg)
        env.reset()

        for s in range(9):
            obs, r, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
            assert not term
            assert not trunc

        # 10th step should trigger truncation
        obs, r, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
        assert not term
        assert trunc
        assert info["episode_step"] == 10

    def test_info_dictionary_structure(self) -> None:
        env = SolarMPPTEnv()
        obs, info = env.reset()
        obs, r, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))

        required_keys = [
            "pv_voltage",
            "pv_current",
            "pv_power",
            "mpp_voltage",
            "mpp_current",
            "mpp_power",
            "duty_cycle",
            "duty_adjustment",
            "duty_saturated",
            "irradiance",
            "cell_temperature",
            "tracking_efficiency",
            "tracking_error",
            "converter_efficiency",
            "episode_step",
            "simulation_time",
            "reward_components",
            "is_ccm",
        ]
        for key in required_keys:
            assert key in info, f"Missing key in info dictionary: {key}"
