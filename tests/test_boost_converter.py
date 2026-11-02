"""Phase 3 Test Suite: DC-DC Boost Converter Plant Modelling and Simulation.

Comprehensive unit and integration tests verifying:
1. Converter initialization, default and custom configuration.
2. Component parameter validation (L, C_pv, C_out, R_load, r_L, duty bounds).
3. State reset and initial condition enforcement.
4. Duty cycle validation and operational bounds.
5. 4th-order Runge-Kutta numerical integration stability.
6. Integration with SolarPVModel equivalent circuit generator.
7. Instantaneous electrical measurement calculations.
8. Steady-state equilibrium at multiple operating points.
9. Duty cycle step response and PV voltage impedance-matching dynamics.
10. Environmental sensitivity (irradiance and temperature steps).
11. First-law energy conservation and power balance verification.
12. Continuous Conduction Mode (CCM) verification.
13. Timeseries simulation helper interface.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from app.converter import BoostConverter, ConverterMeasurements
from app.pv import SolarPVModel
from config.default_config import ConverterConfig, get_default_config


@pytest.fixture
def default_converter() -> BoostConverter:
    """Fixture providing a fresh default BoostConverter instance."""
    return BoostConverter()


@pytest.fixture
def custom_config() -> ConverterConfig:
    """Fixture providing custom valid converter configuration."""
    return ConverterConfig(
        inductance=0.002,
        input_capacitance=0.00022,
        output_capacitance=0.00033,
        load_resistance=15.0,
        inductor_resistance=0.08,
        switching_frequency=25000.0,
        duty_cycle_min=0.10,
        duty_cycle_max=0.85,
        duty_cycle_init=0.40,
        time_step=2e-5,
        v_pv_init=28.0,
        i_l_init=4.0,
        v_out_init=45.0,
    )


class TestConverterInitialization:
    """Tests verifying converter construction, configuration loading, and state setup."""

    def test_default_initialization(self, default_converter: BoostConverter):
        """Verify default initialization loads project default parameters."""
        cfg = get_default_config().converter
        assert default_converter.L == cfg.inductance
        assert default_converter.C_pv == cfg.input_capacitance
        assert default_converter.C_out == cfg.output_capacitance
        assert default_converter.R_load == cfg.load_resistance
        assert default_converter.r_L == cfg.inductor_resistance
        assert default_converter.f_sw == cfg.switching_frequency
        assert default_converter.d_min == cfg.duty_cycle_min
        assert default_converter.d_max == cfg.duty_cycle_max
        assert default_converter.time == 0.0

        state = default_converter.get_state()
        assert math.isclose(state[0], cfg.v_pv_init)
        assert math.isclose(state[1], cfg.i_l_init)
        assert math.isclose(state[2], cfg.v_out_init)

    def test_custom_configuration(self, custom_config: ConverterConfig):
        """Verify initialization with custom parameters overrides defaults correctly."""
        conv = BoostConverter(config=custom_config)
        assert conv.L == 0.002
        assert conv.C_pv == 0.00022
        assert conv.C_out == 0.00033
        assert conv.R_load == 15.0
        assert conv.r_L == 0.08
        assert conv.f_sw == 25000.0
        assert conv.d_min == 0.10
        assert conv.d_max == 0.85

        state = conv.get_state()
        assert math.isclose(state[0], 28.0)
        assert math.isclose(state[1], 4.0)
        assert math.isclose(state[2], 45.0)

    @pytest.mark.parametrize(
        "param_name,invalid_value",
        [
            ("inductance", 0.0),
            ("inductance", -1e-3),
            ("input_capacitance", 0.0),
            ("input_capacitance", -100e-6),
            ("output_capacitance", 0.0),
            ("output_capacitance", -470e-6),
            ("load_resistance", 0.0),
            ("load_resistance", -10.0),
            ("inductor_resistance", -0.05),
        ],
    )
    def test_invalid_component_rejection(self, param_name: str, invalid_value: float):
        """Verify non-positive components and negative resistance raise ValueError."""
        cfg_dict = get_default_config().converter.__dict__.copy()
        cfg_dict[param_name] = invalid_value
        bad_cfg = ConverterConfig(**cfg_dict)

        with pytest.raises(ValueError, match=r"(?i)inductance|resistance|capacitance"):
            BoostConverter(config=bad_cfg)

    @pytest.mark.parametrize(
        "d_min,d_max",
        [
            (0.0, 0.9),
            (0.1, 1.0),
            (0.7, 0.3),
            (0.5, 0.5),
            (-0.1, 0.8),
        ],
    )
    def test_invalid_duty_cycle_bounds(self, d_min: float, d_max: float):
        """Verify duty cycle bounds outside (0, 1) or inverted bounds raise ValueError."""
        cfg_dict = get_default_config().converter.__dict__.copy()
        cfg_dict["duty_cycle_min"] = d_min
        cfg_dict["duty_cycle_max"] = d_max
        bad_cfg = ConverterConfig(**cfg_dict)

        with pytest.raises(ValueError, match="Duty cycle bounds"):
            BoostConverter(config=bad_cfg)


class TestStateManagement:
    """Tests verifying state retrieval, reset, and boundary clamping."""

    def test_reset_to_default(self, default_converter: BoostConverter):
        """Verify reset restores the initial configuration state vector and resets time."""
        # Advance simulation
        for _ in range(10):
            default_converter.step(0.40)
        assert default_converter.time > 0.0

        # Reset
        meas = default_converter.reset()
        assert default_converter.time == 0.0
        assert isinstance(meas, ConverterMeasurements)
        state = default_converter.get_state()
        cfg = get_default_config().converter
        assert math.isclose(state[0], cfg.v_pv_init)
        assert math.isclose(state[1], cfg.i_l_init)
        assert math.isclose(state[2], cfg.v_out_init)

    def test_reset_custom_initial_state(self, default_converter: BoostConverter):
        """Verify reset accepts custom (V_pv, i_L, V_out) state vector."""
        custom_state = (31.5, 6.2, 52.0)
        meas = default_converter.reset(initial_state=custom_state)
        assert default_converter.time == 0.0
        assert math.isclose(meas.v_pv, 31.5)
        assert math.isclose(meas.i_l, 6.2)
        assert math.isclose(meas.v_out, 52.0)

    def test_reset_invalid_length(self, default_converter: BoostConverter):
        """Verify reset raises ValueError if initial_state does not contain 3 elements."""
        with pytest.raises(ValueError, match="must have 3 elements"):
            default_converter.reset(initial_state=(25.0, 5.0))  # type: ignore


class TestDutyCycleValidation:
    """Tests verifying duty cycle argument checking and bounds enforcement."""

    def test_valid_duty_cycle_range(self, default_converter: BoostConverter):
        """Verify stepping across full permissible duty cycle range executes successfully."""
        for d in [0.05, 0.20, 0.39, 0.50, 0.75, 0.90]:
            meas = default_converter.step(duty_cycle=d)
            assert np.isfinite(meas.v_pv)
            assert np.isfinite(meas.i_l)
            assert np.isfinite(meas.v_out)

    @pytest.mark.parametrize("bad_d", [-0.1, 0.01, 0.049, 0.901, 0.95, 1.2])
    def test_out_of_bounds_duty_cycle(self, default_converter: BoostConverter, bad_d: float):
        """Verify duty cycle outside [d_min, d_max] raises ValueError."""
        with pytest.raises(ValueError, match="operational limits"):
            default_converter.step(duty_cycle=bad_d)

    @pytest.mark.parametrize("bad_type", [float("nan"), float("inf"), -float("inf"), "0.4", None])
    def test_non_finite_duty_cycle(self, default_converter: BoostConverter, bad_type):
        """Verify NaN, Inf, string, or None raises TypeError."""
        with pytest.raises(TypeError, match="finite numeric"):
            default_converter.step(duty_cycle=bad_type)


class TestRK4NumericalIntegration:
    """Tests verifying 4th-order Runge-Kutta numerical integration accuracy and non-divergence."""

    def test_single_step_execution(self, default_converter: BoostConverter):
        """Verify single integration step updates state, advances time, and preserves finiteness."""
        dt = 1e-4
        t0 = default_converter.time
        meas = default_converter.step(duty_cycle=0.35, dt=dt)

        assert math.isclose(default_converter.time, t0 + dt, abs_tol=1e-12)
        assert np.all(np.isfinite(default_converter.state))
        assert meas.v_pv >= 0.0
        assert meas.i_l >= 0.0
        assert meas.v_out >= 0.0

    def test_invalid_dt_rejection(self, default_converter: BoostConverter):
        """Verify non-positive integration time steps are rejected."""
        with pytest.raises(ValueError, match="time step dt must be positive"):
            default_converter.step(duty_cycle=0.35, dt=0.0)
        with pytest.raises(ValueError, match="time step dt must be positive"):
            default_converter.step(duty_cycle=0.35, dt=-1e-4)

    def test_derivatives_structure(self, default_converter: BoostConverter):
        """Verify compute_derivatives returns a 3-element finite array."""
        state = np.array([25.0, 5.0, 35.0])
        derivs = default_converter.compute_derivatives(state, duty_cycle=0.40, irradiance=1000.0, cell_temperature=25.0)

        assert isinstance(derivs, np.ndarray)
        assert derivs.shape == (3,)
        assert np.all(np.isfinite(derivs))


class TestSolarPVIntegration:
    """Tests verifying dynamic coupling with SolarPVModel equivalent circuit generator."""

    def test_pv_current_matches_voltage(self, default_converter: BoostConverter):
        """Verify PV current reported by converter exactly matches SolarPVModel at that voltage."""
        pv_model = default_converter.pv_model
        meas = default_converter.step(duty_cycle=0.38, irradiance=950.0, cell_temperature=30.0)

        expected_i_pv = pv_model.current_at_voltage(meas.v_pv, irradiance=950.0, cell_temperature=30.0)
        assert math.isclose(meas.i_pv, expected_i_pv, rel_tol=1e-5)

    def test_pv_operating_point_shift_with_duty(self, default_converter: BoostConverter):
        """Verify higher duty cycle decreases effective input impedance, drawing higher current."""
        # Settle at D = 0.20 (high reflected impedance -> higher Vpv, lower Ipv)
        default_converter.reset((30.0, 4.0, 40.0))
        for _ in range(500):
            m_low_d = default_converter.step(duty_cycle=0.20, dt=1e-4)

        # Settle at D = 0.55 (low reflected impedance -> lower Vpv, higher Ipv)
        default_converter.reset((30.0, 4.0, 40.0))
        for _ in range(500):
            m_high_d = default_converter.step(duty_cycle=0.55, dt=1e-4)

        assert m_high_d.v_pv < m_low_d.v_pv
        assert m_high_d.i_pv > m_low_d.i_pv


class TestElectricalMeasurements:
    """Tests verifying power calculations, boost ratio, efficiency, and Ohm's law consistency."""

    def test_power_and_efficiency_consistency(self, default_converter: BoostConverter):
        """Verify Pin = Vpv*Ipv, Pout = Vout*Iout, Ploss = rL*iL^2, and efficiency is bounded."""
        meas = default_converter.step(duty_cycle=0.39, irradiance=1000.0, cell_temperature=25.0)

        assert math.isclose(meas.p_pv, meas.v_pv * meas.i_pv, rel_tol=1e-6)
        assert math.isclose(meas.i_out, meas.v_out / default_converter.R_load, rel_tol=1e-6)
        assert math.isclose(meas.p_out, meas.v_out * meas.i_out, rel_tol=1e-6)
        assert math.isclose(meas.power_loss, default_converter.r_L * (meas.i_l ** 2), rel_tol=1e-6)

        expected_eff = (meas.p_out / meas.p_pv) * 100.0
        assert math.isclose(meas.efficiency, expected_eff, rel_tol=1e-5)
        assert 0.0 <= meas.efficiency <= 100.0

    def test_steady_state_boost_voltage_ratio(self, default_converter: BoostConverter):
        """Verify in steady state, Vout > Vpv (boost functionality) for duty cycles D > 0."""
        # Simulate to steady state at D = 0.40
        default_converter.reset((25.0, 5.0, 35.0))
        for _ in range(600):
            meas = default_converter.step(duty_cycle=0.40, dt=1e-4)

        # Boost converter must step voltage up: Vout > Vpv
        assert meas.v_out > meas.v_pv
        # Theoretical lossless ideal ratio: Vout / Vpv = 1 / (1 - D) = 1 / 0.6 = 1.667
        ideal_ratio = 1.0 / (1.0 - 0.40)
        actual_ratio = meas.v_out / meas.v_pv
        # Due to inductor resistance r_L, actual ratio is slightly below ideal ratio
        assert actual_ratio < ideal_ratio
        assert actual_ratio > 0.90 * ideal_ratio


class TestSteadyStateEquilibrium:
    """Tests verifying convergence to steady state where state derivatives approach zero."""

    @pytest.mark.parametrize("duty_cycle", [0.20, 0.39, 0.60])
    def test_steady_state_derivatives_approach_zero(self, default_converter: BoostConverter, duty_cycle: float):
        """Verify state derivatives vanish at equilibrium for various duty cycles."""
        default_converter.reset((25.0, 5.0, 35.0))
        # Step for 350 ms (3500 steps of 100 us) to allow settling well past 5 time constants
        for _ in range(3500):
            default_converter.step(duty_cycle=duty_cycle, dt=1e-4)

        state = default_converter.state
        derivs = default_converter.compute_derivatives(state, duty_cycle, irradiance=1000.0, cell_temperature=25.0)

        # Tolerances on settled derivative magnitudes:
        # |dVpv/dt| < 0.1 V/s
        # |diL/dt| < 0.1 A/s
        # |dVout/dt| < 0.1 V/s
        assert abs(derivs[0]) < 0.1, f"Vpv derivative not settled: {derivs[0]:.4f} V/s"
        assert abs(derivs[1]) < 0.1, f"iL derivative not settled: {derivs[1]:.4f} A/s"
        assert abs(derivs[2]) < 0.1, f"Vout derivative not settled: {derivs[2]:.4f} V/s"


class TestTransientResponses:
    """Tests verifying dynamic step responses for duty cycle, irradiance, and temperature."""

    def test_duty_cycle_step_transient(self, default_converter: BoostConverter):
        """Verify step increase in D lowers steady-state Vpv and step decrease raises Vpv."""
        default_converter.reset((28.0, 5.0, 40.0))
        # Settle at D = 0.35
        for _ in range(800):
            m_pre = default_converter.step(duty_cycle=0.35, dt=1e-4)

        # Step D = 0.35 -> 0.48
        for _ in range(800):
            m_post = default_converter.step(duty_cycle=0.48, dt=1e-4)

        # Stepping D higher reduces reflected input impedance, pulling Vpv down
        assert m_post.v_pv < m_pre.v_pv
        # Boost ratio Vout/Vpv increases with duty cycle
        assert (m_post.v_out / m_post.v_pv) > (m_pre.v_out / m_pre.v_pv)

        # Step D = 0.48 -> 0.25 (lower D increases reflected impedance, raising Vpv)
        for _ in range(800):
            m_down = default_converter.step(duty_cycle=0.25, dt=1e-4)

        assert m_down.v_pv > m_post.v_pv
        assert (m_down.v_out / m_down.v_pv) < (m_post.v_out / m_post.v_pv)

    def test_irradiance_step_transient(self, default_converter: BoostConverter):
        """Verify irradiance drop (1000 -> 400 W/m^2) reduces inductor and PV current."""
        default_converter.reset((28.0, 7.0, 42.0))
        # Settle at 1000 W/m^2
        for _ in range(800):
            m_1000 = default_converter.step(duty_cycle=0.39, irradiance=1000.0, dt=1e-4)

        # Step irradiance to 400 W/m^2
        for _ in range(800):
            m_400 = default_converter.step(duty_cycle=0.39, irradiance=400.0, dt=1e-4)

        # Lower irradiance produces proportionally less photocurrent
        assert m_400.i_pv < 0.5 * m_1000.i_pv
        assert m_400.i_l < 0.5 * m_1000.i_l
        assert m_400.p_out < 0.5 * m_1000.p_out

    def test_temperature_step_transient(self, default_converter: BoostConverter):
        """Verify temperature increase (25 -> 45 °C) reduces PV terminal voltage."""
        default_converter.reset((28.0, 7.0, 42.0))
        # Settle at 25 °C
        for _ in range(800):
            m_25c = default_converter.step(duty_cycle=0.39, cell_temperature=25.0, dt=1e-4)

        # Step temperature to 45 °C
        for _ in range(800):
            m_45c = default_converter.step(duty_cycle=0.39, cell_temperature=45.0, dt=1e-4)

        # Negative temperature coefficient of silicon PV cells causes voltage drop
        assert m_45c.v_pv < m_25c.v_pv


class TestEnergyConservation:
    """Tests verifying the First Law of Thermodynamics and power balance."""

    def test_instantaneous_power_balance(self, default_converter: BoostConverter):
        """Verify Pin - Pout - Ploss - dE/dt residual vanishes in steady state and is bounded in transients."""
        default_converter.reset((25.0, 5.0, 35.0))
        # Transient phase: relative residual is bounded by finite difference truncation (< 5%)
        for _ in range(50):
            meas = default_converter.step(duty_cycle=0.42, dt=1e-4)
            rel_error = meas.power_balance_residual / max(meas.p_pv, 1.0)
            assert rel_error < 0.05, f"Transient power balance relative error {rel_error*100:.3f}% exceeds 5%"

        # Steady-state phase: residual vanishes (< 0.001 W)
        for _ in range(600):
            meas = default_converter.step(duty_cycle=0.42, dt=1e-4)
        assert meas.power_balance_residual < 0.001, f"Steady-state residual {meas.power_balance_residual:.6e} W > 1 mW"

    def test_integrated_energy_conservation(self, default_converter: BoostConverter):
        """Verify integral of power flow over a transient matches change in stored energy within 1%."""
        default_converter.reset((20.0, 2.0, 25.0))
        dt = 1e-4
        steps = 500

        e_initial = default_converter._calculate_stored_energy(default_converter.state)
        total_p_net_integrated = 0.0

        for _ in range(steps):
            meas = default_converter.step(duty_cycle=0.40, dt=dt)
            p_net = meas.p_pv - meas.p_out - meas.power_loss
            total_p_net_integrated += p_net * dt

        e_final = default_converter._calculate_stored_energy(default_converter.state)
        delta_e = e_final - e_initial

        # Relative error between trapezoidal/Euler work and delta E
        rel_error = abs(total_p_net_integrated - delta_e) / max(abs(delta_e), 1.0)
        assert rel_error < 0.01, f"Energy conservation error {rel_error*100:.3f}% exceeds 1% tolerance"


class TestContinuousConductionMode:
    """Tests verifying the converter maintains continuous inductor conduction."""

    @pytest.mark.parametrize("d_val", [0.15, 0.35, 0.50, 0.70])
    @pytest.mark.parametrize("g_val", [300.0, 600.0, 1000.0])
    def test_inductor_current_positivity(self, default_converter: BoostConverter, d_val: float, g_val: float):
        """Verify inductor current remains strictly positive (CCM) across operating points."""
        default_converter.reset((25.0, 3.0, 35.0))
        for _ in range(300):
            meas = default_converter.step(duty_cycle=d_val, irradiance=g_val, dt=1e-4)

        assert meas.i_l > 0.0
        assert meas.is_ccm is True


class TestSimulationHelper:
    """Tests verifying the simulate() timeseries DataFrame generation helper."""

    def test_simulate_scalar_duty(self, default_converter: BoostConverter):
        """Verify simulate returns expected DataFrame columns and rows for constant duty cycle."""
        duration = 0.01  # 10 ms
        dt = 1e-4
        df = default_converter.simulate(duration=duration, duty_cycle=0.38, dt=dt)

        assert isinstance(df, pd.DataFrame)
        # Duration / dt = 100 steps + 1 initial row = 101 rows
        assert len(df) == 101
        expected_cols = {
            "time", "v_pv", "i_pv", "p_pv", "i_l", "v_out", "i_out", "p_out",
            "duty_cycle", "efficiency", "is_ccm", "stored_energy", "power_loss",
            "dE_dt", "power_balance_residual",
        }
        assert expected_cols.issubset(set(df.columns))
        assert df["time"].iloc[0] == 0.0
        assert math.isclose(df["time"].iloc[-1], duration, abs_tol=1e-9)

    def test_simulate_callable_profile(self, default_converter: BoostConverter):
        """Verify simulate works with time-varying function callables for D(t) and G(t)."""
        duration = 0.005  # 5 ms
        dt = 1e-4

        d_func = lambda t: 0.30 if t < 0.002 else 0.45
        g_func = lambda t: 1000.0 if t < 0.002 else 800.0

        df = default_converter.simulate(duration=duration, duty_cycle=d_func, irradiance=g_func, dt=dt)
        assert len(df) == 51
        assert df["duty_cycle"].iloc[10] == 0.30
        assert df["duty_cycle"].iloc[35] == 0.45
