"""Unit Tests for RL Reward Formulation and Component Decompositions."""

import math
import pytest

from app.rl.reward import RewardCalculator, RewardComponents, RewardType
from config.default_config import RLEnvConfig


class TestRewardCalculatorInitialization:
    """Test parameter validation and initialization."""

    def test_default_initialization(self) -> None:
        calc = RewardCalculator()
        assert calc.reward_type == RewardType.HYBRID_TRACKING
        assert calc.w_p == 1.0
        assert calc.w_e == 2.0
        assert calc.w_d == 0.05
        assert calc.fault_penalty == 10.0

    def test_invalid_parameters_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="weight_power"):
            RewardCalculator(RLEnvConfig(weight_power=-1.0))

        with pytest.raises(ValueError, match="weight_tracking_error"):
            RewardCalculator(RLEnvConfig(weight_tracking_error=-0.5))

        with pytest.raises(ValueError, match="weight_action_penalty"):
            RewardCalculator(RLEnvConfig(weight_action_penalty=-0.1))

        with pytest.raises(ValueError, match="epsilon_power"):
            RewardCalculator(RLEnvConfig(epsilon_power=0.0))

        with pytest.raises(ValueError, match="fault_penalty"):
            RewardCalculator(RLEnvConfig(fault_penalty=-5.0))


class TestHybridTrackingReward:
    """Test mathematical exactness and gradient behavior of the primary hybrid reward."""

    @pytest.fixture
    def hybrid_calc(self) -> RewardCalculator:
        cfg = RLEnvConfig(
            reward_type="hybrid_tracking",
            weight_power=1.0,
            weight_tracking_error=2.0,
            weight_action_penalty=0.1,
            epsilon_power=1e-4,
            fault_penalty=10.0,
        )
        return RewardCalculator(cfg)

    def test_exact_mpp_tracking_produces_maximum_reward(self, hybrid_calc: RewardCalculator) -> None:
        p_mpp = 240.0
        p_pv = 240.0
        action = 0.0

        r, comps = hybrid_calc.compute_reward(p_pv=p_pv, p_mpp=p_mpp, action=action)

        # At MPP: power_term = 1.0 * (240 / (240 + 1e-4)) ~ 1.0
        # error_term = 2.0 * (0)^2 = 0
        # action_penalty = 0.1 * 0^2 = 0
        assert math.isclose(r, 1.0, abs_tol=1e-4)
        assert math.isclose(comps.power_term, 1.0, abs_tol=1e-4)
        assert math.isclose(comps.error_penalty, 0.0, abs_tol=1e-5)
        assert math.isclose(comps.action_penalty, 0.0, abs_tol=1e-5)

    def test_monotonic_increase_towards_mpp(self, hybrid_calc: RewardCalculator) -> None:
        p_mpp = 240.0
        powers = [50.0, 100.0, 180.0, 220.0, 239.0, 240.0]
        rewards = [hybrid_calc.compute_reward(p, p_mpp, action=0.0)[0] for p in powers]

        # Rewards must be strictly monotonically increasing towards MPP
        for i in range(len(rewards) - 1):
            assert rewards[i + 1] > rewards[i], f"Reward did not increase: {rewards[i+1]} <= {rewards[i]}"

    def test_action_penalty_quadratic_scaling(self, hybrid_calc: RewardCalculator) -> None:
        p_mpp = 240.0
        p_pv = 240.0

        r_zero, comps_zero = hybrid_calc.compute_reward(p_pv, p_mpp, action=0.0)
        r_half, comps_half = hybrid_calc.compute_reward(p_pv, p_mpp, action=0.5)
        r_full, comps_full = hybrid_calc.compute_reward(p_pv, p_mpp, action=1.0)

        # w_d = 0.1: penalty at a=0.5 is 0.1*(0.25)=0.025; at a=1.0 is 0.1*(1)=0.1
        assert math.isclose(comps_half.action_penalty, 0.025, abs_tol=1e-5)
        assert math.isclose(comps_full.action_penalty, 0.100, abs_tol=1e-5)
        assert r_zero > r_half > r_full

    def test_fault_penalty_application(self, hybrid_calc: RewardCalculator) -> None:
        r_fault, comps_fault = hybrid_calc.compute_reward(p_pv=200.0, p_mpp=240.0, action=0.0, is_fault=True)
        assert r_fault == -10.0
        assert comps_fault.fault_penalty == 10.0
        assert comps_fault.power_term == 0.0
        assert comps_fault.error_penalty == 0.0

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), "invalid"])
    def test_non_finite_input_rejection(self, hybrid_calc: RewardCalculator, bad_val) -> None:
        with pytest.raises(TypeError):
            hybrid_calc.compute_reward(p_pv=bad_val, p_mpp=240.0, action=0.0)

    def test_near_zero_mpp_power_handled_safely(self, hybrid_calc: RewardCalculator) -> None:
        # P_mpp = 0.0 (night / shaded)
        r, comps = hybrid_calc.compute_reward(p_pv=0.0, p_mpp=0.0, action=0.0)
        assert math.isfinite(r)
        assert not math.isnan(r)


class TestAlternativeRewardFormulations:
    """Test secondary reward formulations."""

    def test_normalized_power_formulation(self) -> None:
        cfg = RLEnvConfig(reward_type="normalized_power", weight_action_penalty=0.05)
        calc = RewardCalculator(cfg)

        r, comps = calc.compute_reward(p_pv=120.0, p_mpp=240.0, action=0.0)
        assert math.isclose(r, 0.5, abs_tol=1e-3)
        assert comps.error_penalty == 0.0

    def test_power_delta_formulation(self) -> None:
        cfg = RLEnvConfig(reward_type="power_delta", p_scale=200.0, weight_action_penalty=0.0)
        calc = RewardCalculator(cfg)
        calc.reset(initial_power=100.0)

        # Power increased from 100 to 150 (+50W): r = 50 / 200 = 0.25
        r, comps = calc.compute_reward(p_pv=150.0, p_mpp=240.0, action=0.0)
        assert math.isclose(r, 0.25, abs_tol=1e-4)
