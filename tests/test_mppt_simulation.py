"""Phase 4 Test Suite: Closed-Loop MPPT Simulation Integration Tests.

Verifies:
1. Multirate simulation execution (plant steps at dt=100 us, controller updates at T_mppt=5 ms).
2. Closed-loop feedback coupling between PV model, boost converter plant, and P&O controller.
3. Convergence towards theoretical MPP under Standard Test Conditions (STC).
4. Tracking recovery and dynamic adaptation under irradiance step disturbances.
5. Tracking adaptation under temperature disturbances.
6. Robust convergence across diverse initial duty cycles (D_0 from 0.20 to 0.60).
7. Preservation of physical finiteness (no NaNs, Infs, or negative states).
8. Parameter boundary rejection on simulator configuration.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from app.controllers import PerturbAndObserveController
from app.controllers.mppt_metrics import calculate_mppt_metrics
from app.controllers.mppt_simulation import MPPTSimulator
from app.converter import BoostConverter
from app.pv import SolarPVModel
from config.default_config import get_default_config


@pytest.fixture
def default_simulator() -> MPPTSimulator:
    """Fixture providing a default closed-loop MPPTSimulator instance."""
    return MPPTSimulator()


class TestMPPTSimulatorArchitecture:
    """Tests verifying simulator initialization, input checks, and multirate step handling."""

    def test_default_initialization(self, default_simulator: MPPTSimulator):
        """Verify simulator constructs with valid default subcomponents."""
        assert isinstance(default_simulator.pv_model, SolarPVModel)
        assert isinstance(default_simulator.converter, BoostConverter)
        assert isinstance(default_simulator.controller, PerturbAndObserveController)
        assert default_simulator.sampling_interval == 0.005

    def test_invalid_sampling_interval_rejection(self):
        """Verify non-positive or smaller-than-dt sampling intervals are rejected."""
        with pytest.raises(ValueError, match="sampling interval must be positive"):
            MPPTSimulator(sampling_interval=0.0)
        with pytest.raises(ValueError, match="sampling interval must be positive"):
            MPPTSimulator(sampling_interval=-0.005)

    def test_sampling_interval_smaller_than_dt_rejection(self, default_simulator: MPPTSimulator):
        """Verify simulation rejects sampling interval smaller than plant integration step."""
        default_simulator.sampling_interval = 5e-5  # 50 us < 100 us plant dt
        with pytest.raises(ValueError, match="cannot be smaller than converter integration step"):
            default_simulator.run(duration=0.05, dt=1e-4)

    def test_multirate_timing_ratio(self, default_simulator: MPPTSimulator):
        """Verify controller is updated only once every N plant integration steps."""
        duration = 0.05  # 50 ms
        dt = 1e-4        # 100 us -> 500 plant steps
        t_mppt = 0.005   # 5 ms -> should have 1 initial + 10 updates = 11 updates

        df = default_simulator.run(duration=duration, dt=dt)
        assert len(df) == 501  # t=0 row + 500 steps
        mppt_updates = df["is_mppt_step"].sum()
        # Should be exactly 11 controller updates (t=0, 5ms, 10ms, ..., 50ms)
        assert mppt_updates == 11


class TestSTCTrackingPerformance:
    """Tests verifying convergence, tracking accuracy, and steady state under STC."""

    def test_convergence_to_mpp(self, default_simulator: MPPTSimulator):
        """Verify P&O controller converges toward STC MPP (Pmp ~ 240.1 W, D* ~ 0.394)."""
        df = default_simulator.run(duration=1.0, irradiance=1000.0, cell_temperature=25.0, initial_duty=0.35)
        metrics = calculate_mppt_metrics(df, pv_model=default_simulator.pv_model)

        # 1. High tracking efficiency (> 99.0%)
        assert metrics.tracking_efficiency > 99.0, f"Tracking efficiency {metrics.tracking_efficiency:.2f}% < 99%"

        # 2. Convergence within tolerance band
        assert metrics.is_converged is True
        assert metrics.settling_time is not None
        assert metrics.settling_time < 0.10, f"Settling time {metrics.settling_time} s took longer than 100 ms"

        # 3. Final electrical values match analytical MPP closely
        assert math.isclose(metrics.final_p_pv, metrics.final_p_mpp, abs_tol=1.0)
        assert math.isclose(metrics.final_duty_cycle, 0.3939, abs_tol=0.01)

        # 4. Steady-state power ripple is very small (< 0.2 W with delta_D=0.002)
        assert metrics.steady_state_power_ripple < 0.20

    def test_finiteness_and_positivity_throughout(self, default_simulator: MPPTSimulator):
        """Verify all recorded state variables remain finite, bounded, and physically positive."""
        df = default_simulator.run(duration=0.5, irradiance=1000.0, cell_temperature=25.0)

        for col in ["v_pv", "i_pv", "p_pv", "i_l", "v_out", "i_out", "p_out", "duty_cycle"]:
            vals = df[col].to_numpy()
            assert np.all(np.isfinite(vals)), f"Non-finite values found in {col}"
            assert np.all(vals >= 0.0), f"Negative values found in {col}"


class TestEnvironmentalDisturbances:
    """Tests verifying dynamic tracking adaptation under environmental changes."""

    def test_irradiance_step_adaptation(self, default_simulator: MPPTSimulator):
        """Verify controller adapts to irradiance step change from 1000 down to 600 W/m^2."""
        def g_profile(t: float) -> float:
            return 1000.0 if t < 0.4 else 600.0

        df = default_simulator.run(duration=1.2, irradiance=g_profile, cell_temperature=25.0, initial_duty=0.35)
        metrics = calculate_mppt_metrics(df, pv_model=default_simulator.pv_model)

        # Before step (t=0.35s): power should be near 240 W
        p_pre = df[np.isclose(df["time"], 0.35, atol=1e-4)]["p_pv"].iloc[0]
        assert math.isclose(p_pre, 240.1, abs_tol=2.0)

        # After step settles (t=1.1s): power should be near 600 W/m^2 MPP (~145.0 W)
        p_post = df[np.isclose(df["time"], 1.10, atol=1e-4)]["p_pv"].iloc[0]
        assert math.isclose(p_post, 145.0, abs_tol=2.0)

        assert metrics.tracking_efficiency > 90.0

    def test_temperature_step_adaptation(self, default_simulator: MPPTSimulator):
        """Verify controller adapts to temperature step from 25 to 45 °C."""
        def t_profile(t: float) -> float:
            return 25.0 if t < 0.4 else 45.0

        df = default_simulator.run(duration=0.8, irradiance=1000.0, cell_temperature=t_profile, initial_duty=0.35)
        metrics = calculate_mppt_metrics(df, pv_model=default_simulator.pv_model)

        # Voltage should drop due to negative temperature coefficient
        v_pre = df[np.isclose(df["time"], 0.35, atol=1e-4)]["v_pv"].iloc[0]
        v_post = df[np.isclose(df["time"], 0.75, atol=1e-4)]["v_pv"].iloc[0]
        assert v_post < v_pre
        assert metrics.tracking_efficiency > 95.0


class TestInitialConditionSensitivity:
    """Tests verifying convergence across widely different initial operating duty cycles."""

    @pytest.mark.parametrize("d_init", [0.20, 0.30, 0.50, 0.60])
    def test_convergence_from_arbitrary_initial_duty(self, default_simulator: MPPTSimulator, d_init: float):
        """Verify controller converges to near MPP from both left and right sides of MPP."""
        df = default_simulator.run(duration=1.2, irradiance=1000.0, cell_temperature=25.0, initial_duty=d_init)
        metrics = calculate_mppt_metrics(df, pv_model=default_simulator.pv_model)

        assert metrics.is_converged is True
        assert math.isclose(metrics.final_duty_cycle, 0.3939, abs_tol=0.01)
        assert math.isclose(metrics.final_p_pv, 240.097, abs_tol=1.0)
