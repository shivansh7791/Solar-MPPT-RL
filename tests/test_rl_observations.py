"""Unit Tests for ObservationProcessor and Observation Spaces."""

import math
import numpy as np
import pytest

from app.rl.observations import ObservationProcessor
from config.default_config import RLEnvConfig


class TestObservationProcessorInitialization:
    """Test configuration validation and space generation."""

    def test_default_initialization(self) -> None:
        processor = ObservationProcessor()
        assert processor.observation_dim == 10
        assert len(processor.feature_names) == 10
        assert "v_mp_norm" in processor.feature_names

        space = processor.get_observation_space()
        assert space.shape == (10,)
        assert space.dtype == np.float32
        assert np.all(space.low == -2.0)
        assert np.all(space.high == 2.0)

    def test_standard_non_privileged_mode(self) -> None:
        cfg = RLEnvConfig(include_mpp_in_obs=False)
        processor = ObservationProcessor(cfg)
        assert processor.observation_dim == 8
        assert len(processor.feature_names) == 8
        assert "v_mp_norm" not in processor.feature_names
        assert processor.get_observation_space().shape == (8,)

    def test_invalid_scales_rejected(self) -> None:
        with pytest.raises(ValueError, match="v_scale"):
            ObservationProcessor(RLEnvConfig(v_scale=0.0))

        with pytest.raises(ValueError, match="i_scale"):
            ObservationProcessor(RLEnvConfig(i_scale=-5.0))

        with pytest.raises(ValueError, match="obs_clip_min"):
            ObservationProcessor(RLEnvConfig(obs_clip_min=2.0, obs_clip_max=1.0))


class TestObservationProcessingLogic:
    """Test mathematical normalization and finite differencing."""

    @pytest.fixture
    def processor(self) -> ObservationProcessor:
        cfg = RLEnvConfig(
            v_scale=50.0,
            i_scale=10.0,
            p_scale=250.0,
            delta_v_scale=5.0,
            delta_p_scale=50.0,
            g_scale=1000.0,
            t_ref=25.0,
            t_scale=50.0,
            include_mpp_in_obs=True,
        )
        return ObservationProcessor(cfg)

    def test_first_step_finite_differences_are_zero(self, processor: ObservationProcessor) -> None:
        processor.reset()
        obs = processor.process(
            v_pv=30.0,
            i_pv=8.0,
            p_pv=240.0,
            duty_cycle=0.40,
            irradiance=1000.0,
            cell_temperature=25.0,
            v_mp=30.0,
            p_mp=240.0,
        )

        assert obs.dtype == np.float32
        assert obs.shape == (10,)
        assert math.isclose(obs[0], 30.0 / 50.0, abs_tol=1e-5)  # v_norm
        assert math.isclose(obs[1], 8.0 / 10.0, abs_tol=1e-5)   # i_norm
        assert math.isclose(obs[2], 240.0 / 250.0, abs_tol=1e-5) # p_norm
        assert math.isclose(obs[3], 0.40, abs_tol=1e-5)          # duty
        assert math.isclose(obs[4], 0.0, abs_tol=1e-5)           # delta_v (step 0)
        assert math.isclose(obs[5], 0.0, abs_tol=1e-5)           # delta_p (step 0)
        assert math.isclose(obs[6], 1.0, abs_tol=1e-5)           # g_norm
        assert math.isclose(obs[7], 0.0, abs_tol=1e-5)           # t_norm
        assert math.isclose(obs[8], 30.0 / 50.0, abs_tol=1e-5)  # v_mp_norm
        assert math.isclose(obs[9], 240.0 / 250.0, abs_tol=1e-5) # p_mp_norm

    def test_second_step_finite_differences(self, processor: ObservationProcessor) -> None:
        processor.reset(initial_voltage=30.0, initial_power=240.0)

        # Move to V=32.0 (+2V), P=245.0 (+5W)
        obs = processor.process(
            v_pv=32.0,
            i_pv=7.656,
            p_pv=245.0,
            duty_cycle=0.38,
            irradiance=1000.0,
            cell_temperature=25.0,
            v_mp=30.0,
            p_mp=240.0,
        )

        # delta_v = (32 - 30) / 5.0 = 0.4
        # delta_p = (245 - 240) / 50.0 = 0.1
        assert math.isclose(obs[4], 0.4, abs_tol=1e-5)
        assert math.isclose(obs[5], 0.1, abs_tol=1e-5)

    def test_clipping_bounds_enforced(self, processor: ObservationProcessor) -> None:
        processor.reset()
        # Extreme values that exceed normalization bounds
        obs = processor.process(
            v_pv=250.0,  # 250/50 = 5.0 -> should clip to 2.0
            i_pv=50.0,   # 50/10 = 5.0 -> should clip to 2.0
            p_pv=1000.0, # 1000/250 = 4.0 -> should clip to 2.0
            duty_cycle=0.5,
            irradiance=3000.0, # 3000/1000 = 3.0 -> should clip to 2.0
            cell_temperature=200.0, # (200-25)/50 = 3.5 -> should clip to 2.0
            v_mp=200.0,
            p_mp=1000.0,
        )

        assert np.all(obs <= 2.0)
        assert np.all(obs >= -2.0)

    def test_missing_privileged_reference_raises_value_error(self, processor: ObservationProcessor) -> None:
        with pytest.raises(ValueError, match="v_mp and p_mp must be provided"):
            processor.process(
                v_pv=30.0,
                i_pv=8.0,
                p_pv=240.0,
                duty_cycle=0.4,
                irradiance=1000.0,
                cell_temperature=25.0,
                v_mp=None,
                p_mp=None,
            )

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf"), "invalid"])
    def test_invalid_measurements_raise_type_error(self, processor: ObservationProcessor, bad_val) -> None:
        with pytest.raises(TypeError):
            processor.process(
                v_pv=bad_val,
                i_pv=8.0,
                p_pv=240.0,
                duty_cycle=0.4,
                irradiance=1000.0,
                cell_temperature=25.0,
                v_mp=30.0,
                p_mp=240.0,
            )

    def test_to_dict_conversion(self, processor: ObservationProcessor) -> None:
        processor.reset()
        obs = processor.process(
            v_pv=30.0,
            i_pv=8.0,
            p_pv=240.0,
            duty_cycle=0.4,
            irradiance=1000.0,
            cell_temperature=25.0,
            v_mp=30.0,
            p_mp=240.0,
        )
        d = processor.to_dict(obs)
        assert isinstance(d, dict)
        assert len(d) == 10
        assert "v_pv_norm" in d
        assert "p_mp_norm" in d
