"""Unit Tests for Scenario Manager Module."""

from pathlib import Path
import pytest

from app.evaluation.scenario_manager import (
    ScenarioDefinition,
    ScenarioManager,
    create_scenario_a,
    create_scenario_b,
    create_scenario_c,
    create_scenario_d,
    create_scenario_e_list,
    create_scenario_f_list,
    create_standard_scenarios,
)


class TestScenarioManager:
    """Test suite for ScenarioManager and scenario factory functions."""

    def test_standard_scenarios_count_and_uniqueness(self):
        """Standard suite must contain 19 distinct scenarios."""
        scenarios = create_standard_scenarios()
        assert len(scenarios) == 19
        ids = [s.scenario_id for s in scenarios]
        assert len(set(ids)) == 19  # All IDs must be unique

    def test_scenario_a_stc_conditions(self):
        """Scenario A must be constant 1000 W/m^2, 25°C, duration 1.5s."""
        sc = create_scenario_a()
        assert sc.scenario_id == "scenario_a_stc"
        assert sc.duration == 1.5
        assert sc.initial_duty == 0.35
        g, t = sc.get_conditions(0.5)
        assert g == 1000.0
        assert t == 25.0

    def test_scenario_b_irradiance_step_transitions(self):
        """Scenario B must step through 1000 -> 600 -> 800 -> 400 W/m^2."""
        sc = create_scenario_b()
        assert sc.category == "B_IRRADIANCE_STEP"
        assert sc.get_conditions(0.1)[0] == 1000.0
        assert sc.get_conditions(0.4)[0] == 600.0
        assert sc.get_conditions(0.8)[0] == 800.0
        assert sc.get_conditions(1.2)[0] == 400.0
        # Temperature must remain 25°C throughout
        assert sc.get_conditions(0.4)[1] == 25.0

    def test_scenario_c_temperature_step_transitions(self):
        """Scenario C must step through 25 -> 35 -> 45 -> 25 °C."""
        sc = create_scenario_c()
        assert sc.category == "C_TEMPERATURE_STEP"
        assert sc.get_conditions(0.1)[1] == 25.0
        assert sc.get_conditions(0.4)[1] == 35.0
        assert sc.get_conditions(0.8)[1] == 45.0
        assert sc.get_conditions(1.2)[1] == 25.0
        # Irradiance must remain 1000 W/m^2 throughout
        assert sc.get_conditions(0.4)[0] == 1000.0

    def test_scenario_d_combined_dynamics(self):
        """Scenario D combines linear G ramp and sudden thermal step."""
        sc = create_scenario_d()
        assert sc.category == "D_COMBINED"
        # At t=0: G=1000, T=25
        g0, t0 = sc.get_conditions(0.0)
        assert pytest.approx(g0, abs=1e-3) == 1000.0
        assert t0 == 25.0
        # At t=0.75: G=750, T steps to 50
        g_mid, t_mid = sc.get_conditions(0.75)
        assert pytest.approx(g_mid, abs=1e-3) == 750.0
        assert t_mid == 50.0
        # At t=1.5: G=500, T=50
        g_end, t_end = sc.get_conditions(1.5)
        assert pytest.approx(g_end, abs=1e-3) == 500.0
        assert t_end == 50.0

    def test_scenario_e_randomized_reproducibility(self):
        """Scenario E list must be reproducible with fixed seeds."""
        list1 = create_scenario_e_list(num_scenarios=5, base_seed=101)
        list2 = create_scenario_e_list(num_scenarios=5, base_seed=101)
        assert len(list1) == 5
        for s1, s2 in zip(list1, list2):
            assert s1.scenario_id == s2.scenario_id
            assert s1.seed == s2.seed
            assert s1.get_conditions(0.7) == s2.get_conditions(0.7)

    def test_scenario_f_initial_duty_range(self):
        """Scenario F must test duty cycles [0.20, 0.30, 0.40, 0.50, 0.60]."""
        f_list = create_scenario_f_list()
        assert len(f_list) == 5
        d_vals = [s.initial_duty for s in f_list]
        assert d_vals == [0.20, 0.30, 0.40, 0.50, 0.60]

    def test_scenario_manager_query_and_export(self, tmp_path):
        """ScenarioManager should retrieve by ID and export valid JSON."""
        sm = ScenarioManager()
        assert len(sm) == 19
        sc = sm.get("scenario_a_stc")
        assert sc.name.startswith("Scenario A")

        with pytest.raises(KeyError):
            sm.get("non_existent_id")

        out_json = tmp_path / "scenarios.json"
        exported = sm.export_definitions_json(out_json)
        assert exported.is_file()
        assert exported.stat().st_size > 500
