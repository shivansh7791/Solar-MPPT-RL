"""Phase 2 Test Suite: Photovoltaic Characteristics & Maximum Power Point Analysis.

Verifies I-V and P-V curve synthesis, numerical MPP identification against analytical solutions,
fill factor physical boundaries, environmental sensitivity sweeps, and DataFrame export.
"""

import math
import numpy as np
import pandas as pd
import pytest

from app.pv import PVCharacteristicAnalyzer, PVCharacteristicCurve, SolarPVModel


@pytest.fixture
def analyzer() -> PVCharacteristicAnalyzer:
    """Fixture providing a default PVCharacteristicAnalyzer instance."""
    return PVCharacteristicAnalyzer()


class TestCurveGeneration:
    """Tests verifying I-V and P-V curve array synthesis, dimensional consistency, and endpoint values."""

    def test_curve_generation_dimensions_and_types(self, analyzer: PVCharacteristicAnalyzer):
        """Verify generated arrays have exact specified dimensions, correct types, and finite values."""
        num_points = 350
        curve = analyzer.generate_curve(irradiance=1000.0, cell_temperature=25.0, num_points=num_points)

        assert isinstance(curve, PVCharacteristicCurve)
        assert len(curve.voltage) == num_points
        assert len(curve.current) == num_points
        assert len(curve.power) == num_points

        # Verify all values are finite real numbers
        assert np.all(np.isfinite(curve.voltage))
        assert np.all(np.isfinite(curve.current))
        assert np.all(np.isfinite(curve.power))

    def test_short_circuit_endpoint_current(self, analyzer: PVCharacteristicAnalyzer):
        """Verify current at zero voltage equals short-circuit current Isc."""
        curve = analyzer.generate_curve(irradiance=1000.0, cell_temperature=25.0, num_points=500)

        # First point is at V = 0.0
        assert math.isclose(curve.voltage[0], 0.0, abs_tol=1e-9)
        assert math.isclose(curve.current[0], curve.i_sc_analytical, rel_tol=1e-4)
        assert math.isclose(curve.power[0], 0.0, abs_tol=1e-9)

    def test_open_circuit_endpoint_current(self, analyzer: PVCharacteristicAnalyzer):
        """Verify current at Voc approaches zero and power equals zero."""
        curve = analyzer.generate_curve(irradiance=1000.0, cell_temperature=25.0, num_points=500)

        # Last point is at V = Voc
        assert math.isclose(curve.voltage[-1], curve.v_oc_analytical, rel_tol=1e-4)
        assert math.isclose(curve.current[-1], 0.0, abs_tol=1e-3)
        assert math.isclose(curve.power[-1], 0.0, abs_tol=1e-3)

    def test_power_calculation_exactness(self, analyzer: PVCharacteristicAnalyzer):
        """Verify power is strictly equal to voltage multiplied by current across all points."""
        curve = analyzer.generate_curve(irradiance=800.0, cell_temperature=30.0, num_points=200)
        expected_power = curve.voltage * curve.current
        np.testing.assert_allclose(curve.power, expected_power, rtol=1e-6)

    def test_non_negative_current_enforcement(self, analyzer: PVCharacteristicAnalyzer):
        """Verify currents are non-negative across the entire generated curve."""
        curve = analyzer.generate_curve(irradiance=1000.0, cell_temperature=25.0, v_max_ratio=1.05)
        assert np.all(curve.current >= 0.0), "Negative currents detected in quadrant I"


class TestMPPIdentificationAndVerification:
    """Tests verifying discrete numerical MPP search vs analytical pvlib solution."""

    def test_numerical_mpp_agreement_with_analytical_solution(self, analyzer: PVCharacteristicAnalyzer):
        """Verify numerical MPP from a 500-point sweep agrees with analytical solution (< 0.05% power error)."""
        curve = analyzer.generate_curve(irradiance=1000.0, cell_temperature=25.0, num_points=500)

        # Analytical benchmarks
        p_ana = curve.p_mp_analytical
        v_ana = curve.v_mp_analytical
        i_ana = curve.i_mp_analytical

        # Numerical sweep results
        p_num = curve.p_mp_num
        v_num = curve.v_mp_num
        i_num = curve.i_mp_num

        # Check percentage errors
        assert curve.p_mp_error_pct < 0.01, f"Pmp numerical error {curve.p_mp_error_pct}% exceeded 0.01%"
        assert curve.v_mp_error_pct < 0.15, f"Vmp numerical error {curve.v_mp_error_pct}% exceeded 0.15%"
        assert curve.i_mp_error_pct < 0.15, f"Imp numerical error {curve.i_mp_error_pct}% exceeded 0.15%"

        # Consistency: Pmp = Vmp * Imp
        assert math.isclose(p_num, v_num * i_num, rel_tol=1e-4)

    def test_fill_factor_physical_boundaries(self, analyzer: PVCharacteristicAnalyzer):
        """Verify Fill Factor satisfies 0 < FF < 1 and falls in typical crystalline silicon range (0.70-0.85)."""
        for g in [200.0, 600.0, 1000.0]:
            curve = analyzer.generate_curve(irradiance=g, cell_temperature=25.0)
            ff = curve.fill_factor
            assert 0.0 < ff < 1.0, f"Fill factor {ff} violated physical boundary (0, 1)"
            assert 0.70 <= ff <= 0.85, f"Fill factor {ff} outside commercial silicon range [0.70, 0.85]"

    def test_mpp_index_bounds(self, analyzer: PVCharacteristicAnalyzer):
        """Verify mpp_index is strictly strictly interior (not at V=0 or V=Voc)."""
        curve = analyzer.generate_curve(irradiance=1000.0, cell_temperature=25.0, num_points=500)
        assert 0 < curve.mpp_index < len(curve.voltage) - 1, "MPP should be interior to [0, Voc]"


class TestParametricSweeps:
    """Tests verifying physical trends across irradiance and temperature variations."""

    def test_irradiance_sweep_trends(self, analyzer: PVCharacteristicAnalyzer):
        """Verify that increasing irradiance increases Isc, Imp, and Pmp monotonically."""
        irradiances = [200.0, 400.0, 600.0, 800.0, 1000.0]
        curves = analyzer.analyze_irradiance_variation(irradiances, temperature=25.0)

        assert len(curves) == len(irradiances)

        for i in range(len(curves) - 1):
            c_low = curves[i]
            c_high = curves[i + 1]

            # Current and power must strictly increase
            assert c_high.i_sc_analytical > c_low.i_sc_analytical
            assert c_high.i_mp_analytical > c_low.i_mp_analytical
            assert c_high.p_mp_analytical > c_low.p_mp_analytical
            assert c_high.p_mp_num > c_low.p_mp_num

            # Voc increases logarithmically
            assert c_high.v_oc_analytical > c_low.v_oc_analytical

    def test_temperature_sweep_trends(self, analyzer: PVCharacteristicAnalyzer):
        """Verify that increasing temperature decreases Voc, Vmp, and Pmp monotonically."""
        temperatures = [15.0, 25.0, 35.0, 45.0, 55.0]
        curves = analyzer.analyze_temperature_variation(temperatures, irradiance=1000.0)

        assert len(curves) == len(temperatures)

        for i in range(len(curves) - 1):
            c_cold = curves[i]
            c_hot = curves[i + 1]

            # Voc and Pmp must strictly decrease
            assert c_hot.v_oc_analytical < c_cold.v_oc_analytical
            assert c_hot.v_mp_analytical < c_cold.v_mp_analytical
            assert c_hot.p_mp_analytical < c_cold.p_mp_analytical
            assert c_hot.p_mp_num < c_cold.p_mp_num

            # Isc slightly increases due to bandgap contraction
            assert c_hot.i_sc_analytical > c_cold.i_sc_analytical

    def test_combined_environmental_scenarios(self, analyzer: PVCharacteristicAnalyzer):
        """Verify multi-scenario execution and DataFrame conversion."""
        scenarios = [(1000.0, 25.0), (800.0, 30.0), (600.0, 35.0), (400.0, 45.0)]
        curves = analyzer.analyze_environmental_scenarios(scenarios)
        df = analyzer.curves_to_mpp_dataframe(curves)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "irradiance" in df.columns
        assert "p_mp" in df.columns
        assert "p_mp_numerical" in df.columns
        assert "p_mp_error_pct" in df.columns
        assert np.all(df["p_mp_error_pct"] < 0.05)


class TestEdgeCasesAndValidation:
    """Tests verifying input boundary handling, zero irradiance, and DataFrame export."""

    def test_zero_irradiance_produces_zero_arrays(self, analyzer: PVCharacteristicAnalyzer):
        """Verify G=0 cleanly produces zero-filled arrays without exceptions."""
        curve = analyzer.generate_curve(irradiance=0.0, cell_temperature=25.0, num_points=100)
        assert np.all(curve.voltage == 0.0)
        assert np.all(curve.current == 0.0)
        assert np.all(curve.power == 0.0)
        assert curve.p_mp_num == 0.0
        assert curve.fill_factor == 0.0

    def test_invalid_num_points_raises_value_error(self, analyzer: PVCharacteristicAnalyzer):
        with pytest.raises(ValueError, match="num_points must be >= 2"):
            analyzer.generate_curve(irradiance=1000.0, cell_temperature=25.0, num_points=1)

    def test_invalid_v_max_ratio_raises_value_error(self, analyzer: PVCharacteristicAnalyzer):
        with pytest.raises(ValueError, match="v_max_ratio must be positive"):
            analyzer.generate_curve(irradiance=1000.0, cell_temperature=25.0, v_max_ratio=-0.1)

    def test_curve_to_dataframe(self, analyzer: PVCharacteristicAnalyzer):
        """Verify to_dataframe method on curve produces correct structure."""
        curve = analyzer.generate_curve(irradiance=1000.0, cell_temperature=25.0, num_points=50)
        df = curve.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert list(df.columns) == ["voltage", "current", "power"]
