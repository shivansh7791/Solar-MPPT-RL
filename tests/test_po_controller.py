"""Phase 4 Test Suite: Perturb & Observe (P&O) MPPT Controller Unit Tests.

Verifies:
1. Controller initialization and default/custom parameter validation.
2. Parameter boundary enforcement (duty bounds, positive step, non-negative deadbands).
3. Reset functionality and state memory clearing.
4. First measurement handling (no arbitrary perturbation jump at step 0).
5. All 4 quadrants of the P&O decision matrix (Delta P x Delta V):
   - (+DP, +DV) -> Decrease D (delta_d = -step)
   - (+DP, -DV) -> Increase D (delta_d = +step)
   - (-DP, +DV) -> Increase D (delta_d = +step)
   - (-DP, -DV) -> Decrease D (delta_d = -step)
6. Duty cycle clamping and saturation detection.
7. Input validation (negative voltage/current, NaN, Inf, non-numeric).
8. Power and voltage deadband thresholds.
9. Diagnostic state reporting.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from app.controllers.po_controller import PerturbAndObserveController, PODiagnostics
from config.default_config import POConfig, get_default_config


@pytest.fixture
def default_po() -> PerturbAndObserveController:
    """Fixture providing a default PerturbAndObserveController instance."""
    return PerturbAndObserveController()


@pytest.fixture
def custom_po() -> PerturbAndObserveController:
    """Fixture providing a PerturbAndObserveController with specific test parameters."""
    return PerturbAndObserveController(
        initial_duty=0.40,
        perturbation_step=0.005,
        min_duty=0.10,
        max_duty=0.85,
        power_deadband=0.1,
        voltage_deadband=0.05,
    )


class TestPOInitialization:
    """Tests verifying controller parameter assignment and validation."""

    def test_default_initialization(self, default_po: PerturbAndObserveController):
        """Verify default initialization matches project default configuration."""
        cfg = get_default_config().mppt.po
        assert default_po.initial_duty == cfg.initial_duty
        assert default_po.perturbation_step == cfg.perturbation_step
        assert default_po.min_duty == cfg.duty_min
        assert default_po.max_duty == cfg.duty_max
        assert default_po.power_deadband == cfg.power_deadband
        assert default_po.voltage_deadband == cfg.voltage_deadband
        assert default_po.duty_cycle == cfg.initial_duty
        assert default_po.prev_voltage is None
        assert default_po.prev_power is None
        assert default_po.update_count == 0

    def test_custom_initialization(self, custom_po: PerturbAndObserveController):
        """Verify custom initialization overrides parameters correctly."""
        assert custom_po.initial_duty == 0.40
        assert custom_po.perturbation_step == 0.005
        assert custom_po.min_duty == 0.10
        assert custom_po.max_duty == 0.85
        assert custom_po.power_deadband == 0.1
        assert custom_po.voltage_deadband == 0.05

    @pytest.mark.parametrize(
        "min_d,max_d",
        [
            (0.0, 0.9),
            (0.1, 1.0),
            (0.7, 0.3),
            (0.5, 0.5),
            (-0.1, 0.8),
        ],
    )
    def test_invalid_duty_limits_rejected(self, min_d: float, max_d: float):
        """Verify invalid duty boundaries raise ValueError."""
        with pytest.raises(ValueError, match="Duty limits must satisfy"):
            PerturbAndObserveController(min_duty=min_d, max_duty=max_d)

    @pytest.mark.parametrize("d0", [0.04, 0.91, -0.1, 1.2])
    def test_initial_duty_out_of_bounds_rejected(self, d0: float):
        """Verify initial duty outside [min_duty, max_duty] raises ValueError."""
        with pytest.raises(ValueError, match="must be within bounds"):
            PerturbAndObserveController(initial_duty=d0, min_duty=0.05, max_duty=0.90)

    @pytest.mark.parametrize("bad_step", [0.0, -0.002, -1.0])
    def test_non_positive_step_rejected(self, bad_step: float):
        """Verify non-positive perturbation step raises ValueError."""
        with pytest.raises(ValueError, match="Perturbation step must be strictly positive"):
            PerturbAndObserveController(perturbation_step=bad_step)

    @pytest.mark.parametrize("bad_db", [-0.01, -1.0])
    def test_negative_deadbands_rejected(self, bad_db: float):
        """Verify negative deadbands raise ValueError."""
        with pytest.raises(ValueError, match="deadband cannot be negative"):
            PerturbAndObserveController(power_deadband=bad_db)
        with pytest.raises(ValueError, match="deadband cannot be negative"):
            PerturbAndObserveController(voltage_deadband=bad_db)


class TestPOResetAndFirstMeasurement:
    """Tests verifying reset operations and initial sample handling."""

    def test_reset_to_default(self, default_po: PerturbAndObserveController):
        """Verify reset restores configured initial duty and clears memory."""
        # Step controller
        default_po.update(25.0, 8.0)
        default_po.update(26.0, 8.1)
        assert default_po.update_count > 0

        d = default_po.reset()
        assert d == default_po.initial_duty
        assert default_po.duty_cycle == default_po.initial_duty
        assert default_po.prev_voltage is None
        assert default_po.prev_power is None
        assert default_po.update_count == 0

    def test_reset_with_custom_initial_duty(self, default_po: PerturbAndObserveController):
        """Verify reset with custom starting duty assigns it correctly."""
        d = default_po.reset(initial_duty=0.55)
        assert d == 0.55
        assert default_po.duty_cycle == 0.55

    def test_first_measurement_no_perturbation(self, default_po: PerturbAndObserveController):
        """Verify first update establishes baseline without altering duty cycle."""
        d0 = default_po.duty_cycle
        d_out = default_po.update(voltage=30.0, current=8.0)

        # Duty cycle must not jump on step 1
        assert d_out == d0
        assert default_po.duty_adjustment == 0.0
        assert default_po.perturbation_direction == 0
        assert default_po.update_count == 1
        assert default_po.prev_voltage == 30.0
        assert default_po.prev_power == 240.0


class TestPODecisionMatrix:
    """Tests verifying all 4 quadrants of the boost converter P&O decision matrix."""

    def test_quadrant_1_positive_dp_positive_dv(self, default_po: PerturbAndObserveController):
        """Quadrant 1: Delta P > 0 and Delta V > 0 (Left of MPP, moving right).

        Desired action: Increase V -> Commanded action: DECREASE D (delta_D = -step).
        """
        step = default_po.perturbation_step
        # Initial step
        d0 = default_po.update(voltage=26.0, current=8.0)  # P0 = 208.0 W

        # Step 2: V increases (26 -> 28), P increases (208 -> 224)
        d1 = default_po.update(voltage=28.0, current=8.0)  # P1 = 224.0 W, DP = +16, DV = +2

        assert math.isclose(d1, d0 - step, rel_tol=1e-9)
        assert default_po.perturbation_direction == -1
        assert default_po.duty_adjustment == -step

    def test_quadrant_2_positive_dp_negative_dv(self, default_po: PerturbAndObserveController):
        """Quadrant 2: Delta P > 0 and Delta V < 0 (Right of MPP, moving left).

        Desired action: Decrease V -> Commanded action: INCREASE D (delta_D = +step).
        """
        step = default_po.perturbation_step
        # Initial step
        d0 = default_po.update(voltage=34.0, current=6.0)  # P0 = 204.0 W

        # Step 2: V decreases (34 -> 32), P increases (204 -> 224)
        d1 = default_po.update(voltage=32.0, current=7.0)  # P1 = 224.0 W, DP = +20, DV = -2

        assert math.isclose(d1, d0 + step, rel_tol=1e-9)
        assert default_po.perturbation_direction == +1
        assert default_po.duty_adjustment == +step

    def test_quadrant_3_negative_dp_positive_dv(self, default_po: PerturbAndObserveController):
        """Quadrant 3: Delta P < 0 and Delta V > 0 (Overshot to Right of MPP).

        Desired action: Decrease V -> Commanded action: INCREASE D (delta_D = +step).
        """
        step = default_po.perturbation_step
        # Initial step
        d0 = default_po.update(voltage=31.0, current=7.5)  # P0 = 232.5 W

        # Step 2: V increases (31 -> 33), P decreases (232.5 -> 214.5)
        d1 = default_po.update(voltage=33.0, current=6.5)  # P1 = 214.5 W, DP = -18, DV = +2

        assert math.isclose(d1, d0 + step, rel_tol=1e-9)
        assert default_po.perturbation_direction == +1
        assert default_po.duty_adjustment == +step

    def test_quadrant_4_negative_dp_negative_dv(self, default_po: PerturbAndObserveController):
        """Quadrant 4: Delta P < 0 and Delta V < 0 (Overshot to Left of MPP).

        Desired action: Increase V -> Commanded action: DECREASE D (delta_D = -step).
        """
        step = default_po.perturbation_step
        # Initial step
        d0 = default_po.update(voltage=28.0, current=8.0)  # P0 = 224.0 W

        # Step 2: V decreases (28 -> 26), P decreases (224 -> 208)
        d1 = default_po.update(voltage=26.0, current=8.0)  # P1 = 208.0 W, DP = -16, DV = -2

        assert math.isclose(d1, d0 - step, rel_tol=1e-9)
        assert default_po.perturbation_direction == -1
        assert default_po.duty_adjustment == -step


class TestPOSaturationAndDeadbands:
    """Tests verifying saturation limits and deadband insensitivity."""

    def test_upper_saturation_clamping(self):
        """Verify duty cycle does not exceed max_duty and is_saturated is flagged."""
        ctrl = PerturbAndObserveController(initial_duty=0.899, perturbation_step=0.005, max_duty=0.90)
        # Quadrant 2: Increase D
        ctrl.update(34.0, 6.0)
        d_clamped = ctrl.update(32.0, 7.0)

        assert d_clamped == 0.90
        assert ctrl.is_saturated is True

    def test_lower_saturation_clamping(self):
        """Verify duty cycle does not drop below min_duty and is_saturated is flagged."""
        ctrl = PerturbAndObserveController(initial_duty=0.052, perturbation_step=0.005, min_duty=0.05)
        # Quadrant 1: Decrease D
        ctrl.update(26.0, 8.0)
        d_clamped = ctrl.update(28.0, 8.0)

        assert d_clamped == 0.05
        assert ctrl.is_saturated is True

    def test_power_deadband_hold(self):
        """Verify power variations within power_deadband result in delta_D = 0."""
        ctrl = PerturbAndObserveController(initial_duty=0.40, perturbation_step=0.005, power_deadband=1.0)
        ctrl.update(28.0, 8.0)  # P0 = 224.0 W

        # Change power by 0.5 W (less than 1.0 W deadband)
        d_next = ctrl.update(28.05, 8.0)  # P1 = 224.4 W, DP = +0.4 W

        assert d_next == 0.40
        assert ctrl.duty_adjustment == 0.0
        assert ctrl.perturbation_direction == 0

    def test_voltage_deadband_hold(self):
        """Verify voltage variations within voltage_deadband result in delta_D = 0."""
        ctrl = PerturbAndObserveController(initial_duty=0.40, perturbation_step=0.005, voltage_deadband=0.5)
        ctrl.update(28.0, 8.0)

        # Change voltage by 0.2 V (less than 0.5 V deadband)
        d_next = ctrl.update(28.2, 8.2)

        assert d_next == 0.40
        assert ctrl.duty_adjustment == 0.0
        assert ctrl.perturbation_direction == 0


class TestPOInputValidationAndDiagnostics:
    """Tests verifying robust rejection of non-physical inputs and diagnostics structure."""

    @pytest.mark.parametrize("bad_v", [-1.0, -1e-5])
    def test_negative_voltage_raises_value_error(self, default_po: PerturbAndObserveController, bad_v: float):
        """Verify negative voltage raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            default_po.update(voltage=bad_v, current=8.0)

    @pytest.mark.parametrize("bad_i", [-1.0, -1e-5])
    def test_negative_current_raises_value_error(self, default_po: PerturbAndObserveController, bad_i: float):
        """Verify negative current raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            default_po.update(voltage=28.0, current=bad_i)

    @pytest.mark.parametrize("bad_input", [float("nan"), float("inf"), -float("inf"), "28.0", None])
    def test_non_finite_inputs_raise_type_error(self, default_po: PerturbAndObserveController, bad_input):
        """Verify NaN, Inf, strings, or None raise TypeError."""
        with pytest.raises(TypeError, match="finite numeric"):
            default_po.update(voltage=bad_input, current=8.0)
        with pytest.raises(TypeError, match="finite numeric"):
            default_po.update(voltage=28.0, current=bad_input)

    def test_diagnostics_state_retrieval(self, default_po: PerturbAndObserveController):
        """Verify get_state returns a complete, accurate PODiagnostics dataclass."""
        default_po.update(28.0, 8.0)
        default_po.update(29.0, 8.1)

        diag = default_po.get_state()
        assert isinstance(diag, PODiagnostics)
        assert diag.curr_voltage == 29.0
        assert diag.curr_current == 8.1
        assert math.isclose(diag.curr_power, 29.0 * 8.1, rel_tol=1e-6)
        assert diag.prev_voltage == 28.0
        assert math.isclose(diag.prev_power, 28.0 * 8.0, rel_tol=1e-6)
        assert math.isclose(diag.delta_voltage, 1.0, rel_tol=1e-6)
        assert diag.update_count == 2
        assert isinstance(diag.to_dict(), dict)
