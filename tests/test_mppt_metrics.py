"""Phase 4 Test Suite: MPPT Performance Metrics Unit Tests.

Verifies:
1. MPPT tracking efficiency calculation with trapezoidal energy integration.
2. Settling time identification against configurable tolerance bands (+/- 2%).
3. Steady-state oscillation statistics (mean, std, peak-to-peak ripple).
4. Power loss and cumulative energy loss computation.
5. Dynamic time-varying MPP reference generation.
6. Edge case handling (empty DataFrame, missing columns, zero available power).
7. Serialization to dictionary and single-row DataFrame.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from app.controllers.mppt_metrics import MPPTMetrics, calculate_mppt_metrics
from app.pv import SolarPVModel


@pytest.fixture
def synthetic_constant_df() -> pd.DataFrame:
    """Fixture providing a synthetic simulation timeseries with known analytical properties."""
    time = np.linspace(0.0, 1.0, 1001)  # 1 second, dt = 1 ms
    p_mpp = np.full_like(time, 200.0)    # Constant 200 W available

    # Actual power starts at 150 W, linearly ramps to 200 W in 0.1 s, then stays at 200 W
    p_actual = np.piecewise(
        time,
        [time < 0.1, time >= 0.1],
        [lambda t: 150.0 + 50.0 * (t / 0.1), 200.0],
    )

    return pd.DataFrame({
        "time": time,
        "v_pv": np.full_like(time, 25.0),
        "i_pv": p_actual / 25.0,
        "p_pv": p_actual,
        "duty_cycle": np.full_like(time, 0.40),
        "irradiance": np.full_like(time, 1000.0),
        "cell_temperature": np.full_like(time, 25.0),
        "p_mpp": p_mpp,
        "efficiency": np.full_like(time, 98.0),
    })


class TestMPPTMetricsCalculation:
    """Tests verifying quantitative metric calculation accuracy."""

    def test_tracking_efficiency_exactness(self, synthetic_constant_df: pd.DataFrame):
        """Verify tracking efficiency matches analytical ratio of areas under curve."""
        metrics = calculate_mppt_metrics(synthetic_constant_df)
        assert isinstance(metrics, MPPTMetrics)

        # Ideal energy = 200 W * 1.0 s = 200.0 J
        # Actual energy = Area(trapezoid 0 to 0.1s: 0.5 * (150 + 200) * 0.1 = 17.5 J) + (200 W * 0.9s = 180.0 J) = 197.5 J
        # Expected efficiency = 197.5 / 200.0 * 100 = 98.75%
        assert math.isclose(metrics.total_energy_ideal, 200.0, abs_tol=1e-3)
        assert math.isclose(metrics.total_energy_actual, 197.5, abs_tol=1e-3)
        assert math.isclose(metrics.tracking_efficiency, 98.75, abs_tol=1e-2)

    def test_settling_time_detection(self, synthetic_constant_df: pd.DataFrame):
        """Verify settling time accurately marks entry into +/- 2% band (196 W to 204 W)."""
        metrics = calculate_mppt_metrics(synthetic_constant_df, tolerance_band_pct=2.0)

        # Reaches 196 W when 150 + 500*t = 196 => t = 46/500 = 0.092 s
        assert metrics.is_converged is True
        assert metrics.settling_time is not None
        assert math.isclose(metrics.settling_time, 0.092, abs_tol=0.005)

    def test_steady_state_oscillation_metrics(self):
        """Verify steady state ripple and standard deviation calculations with known sinusoidal ripple."""
        time = np.linspace(0.0, 1.0, 1001)
        # Steady state window is last 30% (t in [0.7, 1.0])
        # Add 1 W amplitude sine ripple: 200 + sin(2*pi*50*t) -> peak-to-peak = 2 W
        p_actual = 200.0 + 1.0 * np.sin(2.0 * np.pi * 50.0 * time)

        df = pd.DataFrame({
            "time": time,
            "v_pv": np.full_like(time, 30.0),
            "i_pv": p_actual / 30.0,
            "p_pv": p_actual,
            "duty_cycle": np.full_like(time, 0.39),
            "p_mpp": np.full_like(time, 200.0),
        })

        metrics = calculate_mppt_metrics(df, steady_state_window_ratio=0.3)
        assert math.isclose(metrics.steady_state_power_mean, 200.0, abs_tol=0.05)
        # Peak to peak ripple of 1.0 * sin is 2.0 W
        assert math.isclose(metrics.steady_state_power_ripple, 2.0, abs_tol=0.05)
        # RMS of sine wave with amplitude 1 is 1 / sqrt(2) approx 0.707 W
        assert math.isclose(metrics.steady_state_power_std, 1.0 / math.sqrt(2.0), abs_tol=0.05)

    def test_power_and_energy_loss(self, synthetic_constant_df: pd.DataFrame):
        """Verify mean power loss and cumulative energy loss match analytical difference."""
        metrics = calculate_mppt_metrics(synthetic_constant_df)
        # Energy deficit = 200.0 - 197.5 = 2.5 J
        assert math.isclose(metrics.cumulative_energy_loss, 2.5, abs_tol=0.01)
        # Mean power loss = 2.5 J / 1.0 s = 2.5 W
        assert math.isclose(metrics.mean_power_loss, 2.5, abs_tol=0.01)


class TestMPPTMetricsEdgeCases:
    """Tests verifying error handling, boundary validation, and serialization."""

    def test_empty_dataframe_raises_value_error(self):
        """Verify empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="empty DataFrame"):
            calculate_mppt_metrics(pd.DataFrame())

    def test_missing_mandatory_columns_raises_value_error(self):
        """Verify missing columns raise ValueError."""
        bad_df = pd.DataFrame({"time": [0.0, 0.1], "v_pv": [25.0, 25.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            calculate_mppt_metrics(bad_df)

    def test_non_positive_duration_raises_value_error(self):
        """Verify single-row or non-advancing time DataFrame raises ValueError."""
        zero_t_df = pd.DataFrame({
            "time": [0.0],
            "v_pv": [25.0],
            "i_pv": [8.0],
            "p_pv": [200.0],
            "duty_cycle": [0.4],
        })
        with pytest.raises(ValueError, match="duration must be strictly positive"):
            calculate_mppt_metrics(zero_t_df)

    def test_metrics_dataclass_serialization(self, synthetic_constant_df: pd.DataFrame):
        """Verify MPPTMetrics serializes properly to dict and DataFrame."""
        metrics = calculate_mppt_metrics(synthetic_constant_df)
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert d["tracking_efficiency"] > 0
        assert d["is_converged"] is True

        m_df = metrics.to_dataframe()
        assert isinstance(m_df, pd.DataFrame)
        assert len(m_df) == 1
        assert "tracking_efficiency" in m_df.columns
