"""Unit Tests for Evaluation Metrics Module."""

import numpy as np
import pandas as pd
import pytest

from app.evaluation.metrics import ComparisonMetrics, calculate_comparison_metrics


class TestEvaluationMetrics:
    """Test suite for calculate_comparison_metrics."""

    def test_constant_ideal_tracking(self):
        """When actual power equals ideal power, efficiency should be 100% and ripple 0."""
        t = np.linspace(0, 1.0, 101)
        df = pd.DataFrame({
            "time": t,
            "v_pv": np.full_like(t, 30.0),
            "i_pv": np.full_like(t, 8.0),
            "p_pv": np.full_like(t, 240.0),
            "duty_cycle": np.full_like(t, 0.39),
            "p_mpp": np.full_like(t, 240.0),
            "irradiance": np.full_like(t, 1000.0),
            "cell_temperature": np.full_like(t, 25.0),
        })

        m = calculate_comparison_metrics(
            df=df,
            controller_name="TestController",
            scenario_id="test_stc",
            scenario_name="Test STC",
            category="TEST",
        )

        assert pytest.approx(m.tracking_efficiency, rel=1e-3) == 100.0
        assert pytest.approx(m.energy_harvested_joules, rel=1e-3) == 240.0
        assert pytest.approx(m.energy_loss_joules, abs=1e-3) == 0.0
        assert m.is_converged is True
        assert m.settling_time_s == 0.0
        assert pytest.approx(m.steady_state_power_ripple_w, abs=1e-4) == 0.0
        assert pytest.approx(m.steady_state_power_std_w, abs=1e-4) == 0.0

    def test_constant_under_tracking(self):
        """When actual power is exactly 50% of ideal power, efficiency should be 50%."""
        t = np.linspace(0, 2.0, 201)
        df = pd.DataFrame({
            "time": t,
            "v_pv": np.full_like(t, 25.0),
            "i_pv": np.full_like(t, 4.0),
            "p_pv": np.full_like(t, 100.0),
            "duty_cycle": np.full_like(t, 0.30),
            "p_mpp": np.full_like(t, 200.0),
            "irradiance": np.full_like(t, 1000.0),
            "cell_temperature": np.full_like(t, 25.0),
        })

        m = calculate_comparison_metrics(
            df=df,
            controller_name="UnderTracker",
            scenario_id="test_under",
            scenario_name="Under Tracking",
            category="TEST",
        )

        assert pytest.approx(m.tracking_efficiency, rel=1e-3) == 50.0
        assert pytest.approx(m.energy_harvested_joules, rel=1e-3) == 200.0  # 100W * 2s
        assert pytest.approx(m.energy_loss_joules, rel=1e-3) == 200.0  # (200-100)*2s
        assert m.is_converged is False
        assert m.settling_time_s is None

    def test_settling_time_identification(self):
        """System starts outside 2% band and enters band at t=0.2s."""
        t = np.linspace(0, 1.0, 101)  # step = 0.01s
        p_ideal = np.full_like(t, 200.0)
        p_actual = np.full_like(t, 200.0)
        # First 20 steps (t=0 to 0.19) are at 180W (outside 2% = 196W)
        p_actual[:20] = 180.0

        df = pd.DataFrame({
            "time": t,
            "v_pv": np.full_like(t, 28.0),
            "i_pv": p_actual / 28.0,
            "p_pv": p_actual,
            "duty_cycle": np.full_like(t, 0.38),
            "p_mpp": p_ideal,
            "irradiance": np.full_like(t, 1000.0),
            "cell_temperature": np.full_like(t, 25.0),
        })

        m = calculate_comparison_metrics(
            df=df,
            controller_name="Settler",
            scenario_id="test_settle",
            scenario_name="Settling Test",
            category="TEST",
        )

        assert m.is_converged is True
        assert pytest.approx(m.settling_time_s, abs=0.015) == 0.20

    def test_steady_state_ripple_calculation(self):
        """Square wave oscillation of +/- 2W in steady-state window."""
        t = np.linspace(0, 1.0, 101)
        p_actual = np.full_like(t, 200.0)
        # Add square wave oscillation to last 30%
        for i in range(70, 101):
            p_actual[i] = 202.0 if (i % 2 == 0) else 198.0

        df = pd.DataFrame({
            "time": t,
            "v_pv": np.full_like(t, 28.0),
            "i_pv": p_actual / 28.0,
            "p_pv": p_actual,
            "duty_cycle": np.full_like(t, 0.38),
            "p_mpp": np.full_like(t, 200.0),
            "irradiance": np.full_like(t, 1000.0),
            "cell_temperature": np.full_like(t, 25.0),
        })

        m = calculate_comparison_metrics(
            df=df,
            controller_name="Oscillator",
            scenario_id="test_osc",
            scenario_name="Oscillation Test",
            category="TEST",
            steady_state_window_ratio=0.3,
        )

        assert pytest.approx(m.steady_state_power_ripple_w, abs=1e-3) == 4.0
        assert m.steady_state_power_std_w > 1.5

    def test_empty_dataframe_raises_value_error(self):
        with pytest.raises(ValueError, match="empty DataFrame"):
            calculate_comparison_metrics(
                df=pd.DataFrame(),
                controller_name="Fail",
                scenario_id="fail",
                scenario_name="Fail",
                category="FAIL",
            )
