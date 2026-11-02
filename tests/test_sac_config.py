"""Unit Tests for SACConfig Dataclass and Hyperparameters."""

from pathlib import Path
import pytest
import torch.nn as nn

from app.rl.sac_config import SACConfig


class TestSACConfigInitialization:
    """Test hyperparameter defaults and physical sanity validation."""

    def test_default_hyperparameters(self) -> None:
        cfg = SACConfig()
        assert cfg.algorithm == "SAC"
        assert cfg.policy == "MlpPolicy"
        assert cfg.learning_rate == 3.0e-4
        assert cfg.buffer_size == 100_000
        assert cfg.learning_starts == 2_000
        assert cfg.batch_size == 256
        assert cfg.tau == 0.005
        assert cfg.gamma == 0.99
        assert cfg.ent_coef == "auto"
        assert cfg.net_arch_pi == [256, 256]
        assert cfg.net_arch_qf == [256, 256]
        assert cfg.activation_fn_name == "relu"
        assert cfg.seed == 42
        assert cfg.include_mpp_in_obs is False

    @pytest.mark.parametrize(
        ("param", "bad_val"),
        [
            ("learning_rate", 0.0),
            ("learning_rate", -1e-4),
            ("buffer_size", 0),
            ("batch_size", -10),
            ("tau", 0.0),
            ("tau", 1.5),
            ("gamma", -0.1),
            ("gamma", 1.1),
            ("learning_starts", -1),
            ("total_timesteps", 0),
        ],
    )
    def test_invalid_hyperparameters_rejected(self, param: str, bad_val: float) -> None:
        kwargs = {param: bad_val}
        with pytest.raises(ValueError):
            SACConfig(**kwargs)

    def test_activation_function_resolution(self) -> None:
        cfg_relu = SACConfig(activation_fn_name="relu")
        assert cfg_relu.get_activation_fn() == nn.ReLU

        cfg_tanh = SACConfig(activation_fn_name="tanh")
        assert cfg_tanh.get_activation_fn() == nn.Tanh

        cfg_elu = SACConfig(activation_fn_name="elu")
        assert cfg_elu.get_activation_fn() == nn.ELU

        cfg_bad = SACConfig(activation_fn_name="invalid_act")
        with pytest.raises(ValueError, match="Unsupported activation function"):
            cfg_bad.get_activation_fn()

    def test_policy_kwargs_generation(self) -> None:
        cfg = SACConfig(net_arch_pi=[128, 128], net_arch_qf=[128, 128], activation_fn_name="relu")
        kwargs = cfg.get_policy_kwargs()
        assert "net_arch" in kwargs
        assert kwargs["net_arch"]["pi"] == [128, 128]
        assert kwargs["net_arch"]["qf"] == [128, 128]
        assert kwargs["activation_fn"] == nn.ReLU

    def test_serialization_and_json_save(self, tmp_path: Path) -> None:
        cfg = SACConfig(total_timesteps=5_000, seed=123)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["total_timesteps"] == 5_000

        loaded = SACConfig.from_dict(d)
        assert loaded.total_timesteps == 5_000
        assert loaded.seed == 123

        json_file = tmp_path / "test_sac_config.json"
        cfg.save_json(json_file)
        assert json_file.is_file()
