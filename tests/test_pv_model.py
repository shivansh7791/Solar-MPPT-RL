"""Phase 1 Test Suite: Photovoltaic Mathematical Model Verification.

Validates the physical single-diode equivalent circuit model against STC benchmarks,
environmental sensitivity (irradiance, temperature), input boundary constraints,
and physical semiconductor sanity rules.
"""

import math
import pytest

from app.pv import PVOutput, SolarPVModel
from config import get_default_config


class TestPVModelInitialization:
    """Tests for model construction, configuration loading, and parameter validation."""

    def test_default_initialization(self):
        """Verify model initializes with default Canadian Solar CS6P-240P parameters."""
        model = SolarPVModel()
        info = model.get_module_information()
        assert info["manufacturer"] == "Canadian Solar Inc."
        assert info["model_name"] == "CS6P-240P"
        assert info["technology"] == "Multi-c-Si"
        assert info["num_cells_in_series"] == 60
        assert "a_ref" in info["cec_parameters"]
        assert "I_L_ref" in info["cec_parameters"]

    def test_initialization_from_pvconfig(self):
        """Verify initialization directly from a PVConfig instance."""
        config = get_default_config()
        model = SolarPVModel(config.pv)
        assert model.model_name == "CS6P-240P"
        assert model.reference_stc["p_mp"] == 240.097

    def test_initialization_from_custom_dict(self):
        """Verify initialization with a custom valid parameter dictionary."""
        custom_params = {
            "manufacturer": "CustomPV",
            "model_name": "TestMod-100",
            "technology": "Mono-c-Si",
            "N_s": 36,
            "alpha_sc": 0.002,
            "a_ref": 1.2,
            "I_L_ref": 5.0,
            "I_o_ref": 1e-10,
            "R_sh_ref": 500.0,
            "R_s": 0.25,
            "Adjust": 2.0,
            "stc_p_mp": 100.0,
        }
        model = SolarPVModel(custom_params)
        assert model.manufacturer == "CustomPV"
        assert model.num_cells_in_series == 36

    def test_missing_mandatory_parameter_raises_value_error(self):
        """Verify that omitting a mandatory CEC parameter raises ValueError."""
        incomplete_params = {
            "alpha_sc": 0.002,
            "a_ref": 1.2,
            # Missing I_L_ref, I_o_ref, etc.
        }
        with pytest.raises(ValueError, match="Missing mandatory CEC parameter"):
            SolarPVModel(incomplete_params)

    def test_non_positive_parameters_raise_value_error(self):
        """Verify non-physical negative/zero parameters are rejected."""
        valid_config = get_default_config().pv.get_cec_parameters()

        # Test negative I_L_ref
        bad_il = dict(valid_config, I_L_ref=-1.0)
        with pytest.raises(ValueError, match="strictly positive"):
            SolarPVModel(bad_il)

        # Test negative R_s
        bad_rs = dict(valid_config, R_s=-0.5)
        with pytest.raises(ValueError, match="cannot be negative"):
            SolarPVModel(bad_rs)

    def test_unsupported_parameter_type_raises_type_error(self):
        """Verify non-dict / non-PVConfig inputs raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported type for module_parameters"):
            SolarPVModel("invalid_string_parameter")


class TestSTCNumericalValidation:
    """Verifies that model calculations at STC match documented reference values."""

    @pytest.fixture
    def pv_model(self):
        return SolarPVModel()

    def test_stc_electrical_parameters_accuracy(self, pv_model):
        """Verify calculated electrical parameters match reference STC within < 0.1% tolerance."""
        stc_output = pv_model.get_stc_parameters()
        ref = pv_model.reference_stc

        assert isinstance(stc_output, PVOutput)

        # Numerical comparison against certified reference values
        # Acceptance tolerance: 0.1% maximum deviation
        tolerance_pct = 0.1

        for key in ["i_sc", "v_oc", "v_mp", "i_mp", "p_mp"]:
            calc_val = getattr(stc_output, key)
            ref_val = ref[key]
            pct_error = (abs(calc_val - ref_val) / abs(ref_val)) * 100.0
            assert pct_error < tolerance_pct, (
                f"STC parameter '{key}' error {pct_error:.4f}% exceeded tolerance {tolerance_pct}% "
                f"(calc: {calc_val}, ref: {ref_val})"
            )

    def test_pmp_power_consistency(self, pv_model):
        """Verify Pmp is mathematically consistent with Vmp * Imp."""
        stc = pv_model.get_stc_parameters()
        product_power = stc.v_mp * stc.i_mp
        assert math.isclose(stc.p_mp, product_power, rel_tol=1e-5), (
            f"Pmp ({stc.p_mp}) does not match Vmp * Imp ({product_power})"
        )

    def test_fill_factor_physical_range(self, pv_model):
        """Verify Fill Factor (FF = Pmp / (Voc * Isc)) is physically plausible for silicon."""
        stc = pv_model.get_stc_parameters()
        ff = stc.p_mp / (stc.v_oc * stc.i_sc)
        # Commercial crystalline silicon typically exhibits 0.70 <= FF <= 0.85
        assert 0.70 <= ff <= 0.85, f"Fill factor {ff:.4f} is outside normal physical silicon range [0.70, 0.85]"


class TestPhysicalSensitivities:
    """Verifies physical behavior of semiconductor laws under environmental variations."""

    @pytest.fixture
    def pv_model(self):
        return SolarPVModel()

    def test_irradiance_monotonicity(self, pv_model):
        """Rule 1: Increasing irradiance at constant temperature must strictly increase photocurrent, Isc, and Pmp."""
        irradiances = [200.0, 400.0, 600.0, 800.0, 1000.0]
        t_const = 25.0

        outputs = [pv_model.calculate(g, t_const) for g in irradiances]

        for i in range(len(outputs) - 1):
            assert outputs[i + 1].i_ph > outputs[i].i_ph, "Photocurrent did not strictly increase with irradiance"
            assert outputs[i + 1].i_sc > outputs[i].i_sc, "Isc did not strictly increase with irradiance"
            assert outputs[i + 1].p_mp > outputs[i].p_mp, "Pmp did not strictly increase with irradiance"
            assert outputs[i + 1].v_oc >= outputs[i].v_oc, "Voc did not increase (logarithmic) with irradiance"

    def test_temperature_negative_coefficient_on_voltage_and_power(self, pv_model):
        """Rule 2: Increasing temperature at constant irradiance must reduce open-circuit voltage and maximum power."""
        temperatures = [15.0, 25.0, 45.0, 65.0]
        g_const = 1000.0

        outputs = [pv_model.calculate(g_const, t) for t in temperatures]

        for i in range(len(outputs) - 1):
            assert outputs[i + 1].v_oc < outputs[i].v_oc, (
                f"Voc at {temperatures[i+1]}°C ({outputs[i+1].v_oc}V) should be lower than at {temperatures[i]}°C ({outputs[i].v_oc}V)"
            )
            assert outputs[i + 1].p_mp < outputs[i].p_mp, (
                f"Pmp at {temperatures[i+1]}°C ({outputs[i+1].p_mp}W) should be lower than at {temperatures[i]}°C ({outputs[i].p_mp}W)"
            )

    def test_short_circuit_current_positive_temperature_coefficient(self, pv_model):
        """Rule 3: In crystalline silicon, Isc slightly increases with temperature (alpha_sc > 0)."""
        out_cold = pv_model.calculate(irradiance=1000.0, cell_temperature=15.0)
        out_hot = pv_model.calculate(irradiance=1000.0, cell_temperature=65.0)
        assert out_hot.i_sc > out_cold.i_sc, "Isc should exhibit small positive temperature slope"

    def test_zero_irradiance_produces_zero_power(self, pv_model):
        """Rule 4: Zero irradiance must cleanly produce zero power without crashing."""
        out = pv_model.calculate(irradiance=0.0, cell_temperature=25.0)
        assert out.p_mp == 0.0
        assert out.i_sc == 0.0
        assert out.v_oc == 0.0
        assert out.v_mp == 0.0
        assert out.i_mp == 0.0
        assert out.i_ph == 0.0

    def test_high_irradiance_supported(self, pv_model):
        """Verify elevated irradiance (e.g. 1200 W/m² during cloud enhancement) calculates correctly."""
        out = pv_model.calculate(irradiance=1200.0, cell_temperature=30.0)
        assert out.p_mp > pv_model.reference_stc["p_mp"]
        assert math.isfinite(out.p_mp)


class TestInputValidationAndEdgeCases:
    """Verifies rigorous error handling for unphysical or invalid inputs."""

    @pytest.fixture
    def pv_model(self):
        return SolarPVModel()

    def test_negative_irradiance_raises_value_error(self, pv_model):
        with pytest.raises(ValueError, match="cannot be negative"):
            pv_model.calculate(irradiance=-50.0, cell_temperature=25.0)

    def test_extreme_irradiance_raises_value_error(self, pv_model):
        with pytest.raises(ValueError, match="exceeds upper terrestrial limit"):
            pv_model.calculate(irradiance=3000.0, cell_temperature=25.0)

    def test_temperature_below_absolute_zero_raises_value_error(self, pv_model):
        with pytest.raises(ValueError, match="below absolute zero"):
            pv_model.calculate(irradiance=1000.0, cell_temperature=-300.0)

    def test_extreme_high_temperature_raises_value_error(self, pv_model):
        with pytest.raises(ValueError, match="outside supported operational bounds"):
            pv_model.calculate(irradiance=1000.0, cell_temperature=160.0)

    def test_non_numeric_irradiance_raises_type_error(self, pv_model):
        with pytest.raises(TypeError, match="finite numerical value"):
            pv_model.calculate(irradiance="one_thousand", cell_temperature=25.0)

    def test_nan_irradiance_raises_type_error(self, pv_model):
        with pytest.raises(TypeError, match="finite numerical value"):
            pv_model.calculate(irradiance=float("nan"), cell_temperature=25.0)

    def test_nan_temperature_raises_type_error(self, pv_model):
        with pytest.raises(TypeError, match="finite numerical value"):
            pv_model.calculate(irradiance=1000.0, cell_temperature=float("nan"))

    def test_output_dataclass_serialization(self, pv_model):
        """Verify PVOutput produces valid dictionary."""
        out = pv_model.calculate(1000.0, 25.0)
        d = out.to_dict()
        assert isinstance(d, dict)
        assert "p_mp" in d
        assert "v_mp" in d
        assert "i_mp" in d
        assert "v_oc" in d
        assert "i_sc" in d
