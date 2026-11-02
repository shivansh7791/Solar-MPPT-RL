"""Unit and Integration Tests for Phase 8 Streamlit Dashboard and Integration Services."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

import numpy as np
import pandas as pd

from app.dashboard.controller_service import ControllerService
from app.dashboard.data_export import (
    export_configuration_to_json,
    export_metrics_to_csv,
    export_paired_comparison_to_csv,
    export_timeseries_to_csv,
    load_phase_7_controller_summary_csv,
    load_phase_7_statistical_summary_json,
)
from app.dashboard.simulation_service import SimulationRequest, SimulationService
from app.dashboard.visualizations import (
    create_comparison_response_plots,
    create_dynamic_response_dashboard,
    create_pv_characteristics_plot,
)
from config.default_config import get_default_config


class TestDashboardServices:
    """Test suite for controller and simulation backend services."""

    @pytest.fixture
    def setup_services(self):
        cfg = get_default_config()
        ctrl_service = ControllerService(config=cfg)
        sim_service = SimulationService(config=cfg)
        return cfg, ctrl_service, sim_service

    def test_dashboard_imports(self):
        """Verify dashboard package and core modules import without errors."""
        import app.dashboard
        from app.dashboard import (
            ControllerService,
            SimulationService,
            create_pv_characteristics_plot,
        )
        assert ControllerService is not None
        assert SimulationService is not None
        assert create_pv_characteristics_plot is not None

    def test_checkpoint_discovery_and_metadata(self, setup_services):
        """Verify discovery of realistic and privileged checkpoints with accurate metadata."""
        _, ctrl_service, _ = setup_services
        cps = ctrl_service.get_available_checkpoints()
        assert len(cps) >= 1
        assert any("Realistic SAC" in k for k in cps.keys())

        default_path = ctrl_service.get_default_checkpoint_path()
        assert default_path.is_file()

    def test_sac_checkpoint_loading_and_dimension_validation(self, setup_services):
        """Verify model loads onto CPU and validates observation space dimension."""
        _, ctrl_service, _ = setup_services
        model = ctrl_service.load_sac_model()
        assert model is not None
        assert model.observation_space.shape == (8,)

        # Validating 8 dimensions should pass
        ctrl_service.validate_sac_observation_dim(model, expected_dim=8)

        # Validating 10 dimensions should fail with ValueError
        with pytest.raises(ValueError, match="Observation dimension mismatch"):
            ctrl_service.validate_sac_observation_dim(model, expected_dim=10)

    def test_pv_characteristics_generation(self, setup_services):
        """Verify non-linear I-V/P-V curves and theoretical MPP calculation."""
        _, _, sim_service = setup_services
        v_curve, i_curve, p_curve, mpp = sim_service.calculate_pv_operating_curve(
            irradiance=1000.0,
            temperature=25.0,
            num_points=100,
        )

        assert len(v_curve) == 100
        assert len(i_curve) == 100
        assert len(p_curve) == 100
        assert pytest.approx(mpp.p_mp, rel=1e-2) == 240.1
        assert pytest.approx(mpp.v_mp, rel=1e-2) == 29.9
        assert pytest.approx(mpp.i_mp, rel=1e-2) == 8.03
        assert np.max(p_curve) <= mpp.p_mp + 0.1

    def test_pv_characteristics_plot_creation(self, setup_services):
        """Verify Plotly figure builder creates dual-axis figure with operating points."""
        _, _, sim_service = setup_services
        v_c, i_c, p_c, mpp = sim_service.calculate_pv_operating_curve(1000.0, 25.0, 50)
        fig = create_pv_characteristics_plot(v_c, i_c, p_c, mpp, operating_v=28.0, operating_p=220.0)
        assert fig is not None
        assert len(fig.data) >= 3

    def test_simulation_service_single_po_execution(self, setup_services):
        """Verify single Classical P&O simulation execution on static profile."""
        _, ctrl_service, sim_service = setup_services
        req = SimulationRequest(
            controller_mode="P&O",
            profile_type="Static (Constant G & T)",
            irradiance=1000.0,
            temperature=25.0,
            load_resistance=10.0,
            initial_duty=0.35,
            duration=0.1,  # short 20 control steps for fast test
        )
        po_adapter = ctrl_service.create_po_adapter()
        res = sim_service.execute_single_simulation(req, po_adapter)

        assert res.controller_name in ("P&O", "Classical P&O")
        assert len(res.timeseries) == 21  # step 0 to 20
        assert res.metrics.tracking_efficiency > 90.0
        assert res.metrics.energy_harvested_joules > 0.0
        assert abs(res.metrics.energy_ideal_joules - (res.metrics.energy_harvested_joules + res.metrics.energy_loss_joules)) < 1e-6

    def test_simulation_service_single_sac_execution(self, setup_services):
        """Verify single SAC simulation execution on static profile."""
        _, ctrl_service, sim_service = setup_services
        req = SimulationRequest(
            controller_mode="SAC",
            profile_type="Static (Constant G & T)",
            irradiance=1000.0,
            temperature=25.0,
            load_resistance=10.0,
            initial_duty=0.35,
            duration=0.1,  # 20 steps
        )
        sac_adapter = ctrl_service.create_sac_adapter()
        res = sim_service.execute_single_simulation(req, sac_adapter)

        assert "SAC" in res.controller_name
        assert len(res.timeseries) == 21
        assert res.metrics.tracking_efficiency > 90.0
        assert abs(res.metrics.energy_ideal_joules - (res.metrics.energy_harvested_joules + res.metrics.energy_loss_joules)) < 1e-6

    def test_simulation_service_comparison_execution(self, setup_services):
        """Verify side-by-side comparison mode executes paired run on identical conditions."""
        _, ctrl_service, sim_service = setup_services
        req = SimulationRequest(
            controller_mode="Comparison (P&O vs SAC)",
            profile_type="Static (Constant G & T)",
            irradiance=1000.0,
            temperature=25.0,
            load_resistance=10.0,
            initial_duty=0.35,
            duration=0.1,
        )
        po_adapter = ctrl_service.create_po_adapter()
        sac_adapter = ctrl_service.create_sac_adapter()

        paired_res = sim_service.execute_comparison_simulation(req, po_adapter, sac_adapter)

        assert paired_res.po_metrics.tracking_efficiency > 0.0
        assert paired_res.sac_metrics.tracking_efficiency > 0.0
        assert len(paired_res.po_timeseries) == 21
        assert len(paired_res.sac_timeseries) == 21
        assert "delta_efficiency" in paired_res.to_paired_record()

    def test_predefined_profiles_building(self, setup_services):
        """Verify standard scenarios (STC, steps, ramps) resolve correctly."""
        _, _, sim_service = setup_services
        profiles = sim_service.get_available_profiles()

        for prof in ["Scenario A: Standard Test Conditions (STC)", "Scenario B: Multistep Irradiance Changes"]:
            req = SimulationRequest(
                controller_mode="Comparison",
                profile_type=prof,
                duration=0.5,
            )
            sc = sim_service.build_scenario(req)
            assert sc.duration == 0.5
            g0, t0 = sc.get_conditions(0.0)
            assert g0 > 0.0
            assert t0 > 0.0

    def test_data_export_serialization(self, setup_services):
        """Verify CSV and JSON serialization utilities produce valid outputs."""
        _, ctrl_service, sim_service = setup_services
        req = SimulationRequest(
            controller_mode="P&O",
            profile_type="Static (Constant G & T)",
            duration=0.05,
        )
        po_adapter = ctrl_service.create_po_adapter()
        res = sim_service.execute_single_simulation(req, po_adapter)

        csv_ts = export_timeseries_to_csv(res.timeseries)
        assert "v_pv" in csv_ts
        assert "p_pv" in csv_ts

        csv_m = export_metrics_to_csv(res.metrics)
        assert "tracking_efficiency" in csv_m

        json_cfg = export_configuration_to_json(res.scenario, res.converter_config)
        parsed = json.loads(json_cfg)
        assert "scenario" in parsed
        assert "converter" in parsed

    def test_invalid_input_validation(self, setup_services):
        """Verify out-of-bounds parameters are rejected with descriptive errors."""
        _, _, sim_service = setup_services

        # Negative irradiance
        with pytest.raises(ValueError, match="Irradiance must be"):
            sim_service.validate_request(SimulationRequest(controller_mode="P&O", profile_type="Static", irradiance=-50.0))

        # Extreme temperature
        with pytest.raises(ValueError, match="Cell temperature must be"):
            sim_service.validate_request(SimulationRequest(controller_mode="P&O", profile_type="Static", temperature=150.0))

        # Invalid load resistance
        with pytest.raises(ValueError, match="Load resistance must be"):
            sim_service.validate_request(SimulationRequest(controller_mode="P&O", profile_type="Static", load_resistance=0.0))

        # Out-of-bound initial duty
        with pytest.raises(ValueError, match="Initial duty cycle must be"):
            sim_service.validate_request(SimulationRequest(controller_mode="P&O", profile_type="Static", initial_duty=0.95))

    def test_reproducibility_with_fixed_seed(self, setup_services):
        """Verify identical runs produce bitwise-identical metrics under fixed seed."""
        _, ctrl_service, sim_service = setup_services
        req = SimulationRequest(
            controller_mode="P&O",
            profile_type="Static (Constant G & T)",
            irradiance=1000.0,
            temperature=25.0,
            initial_duty=0.35,
            duration=0.1,
            seed=42,
        )
        po1 = ctrl_service.create_po_adapter()
        po2 = ctrl_service.create_po_adapter()

        res1 = sim_service.execute_single_simulation(req, po1)
        res2 = sim_service.execute_single_simulation(req, po2)

        assert res1.metrics.tracking_efficiency == res2.metrics.tracking_efficiency
        assert res1.metrics.energy_harvested_joules == res2.metrics.energy_harvested_joules
