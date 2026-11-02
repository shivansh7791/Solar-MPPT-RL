"""Unit Tests for Controller Adapters."""

import numpy as np
import pytest

from app.evaluation.controller_adapters import POControllerAdapter, SACControllerAdapter
from app.rl.sac_agent import load_sac_agent
from config.default_config import get_default_config


class TestControllerAdapters:
    """Test suite for POControllerAdapter and SACControllerAdapter."""

    def test_po_controller_adapter_lifecycle(self):
        adapter = POControllerAdapter()
        assert adapter.name == "P&O"

        # Reset with initial measurement
        init_d = adapter.reset(initial_voltage=31.5, initial_current=7.5, initial_duty=0.35)
        assert 0.05 <= init_d <= 0.90

        # Step forward
        next_d, diag = adapter.predict(
            v_pv=30.0,
            i_pv=7.8,
            current_duty=init_d,
            simulation_time=0.005,
            irradiance=1000.0,
            cell_temperature=25.0,
        )
        assert 0.05 <= next_d <= 0.90
        assert "delta_duty" in diag
        assert "is_saturated" in diag

    def test_sac_controller_adapter_lifecycle(self):
        cfg = get_default_config()
        model_path = cfg.paths.root_dir / "artifacts/rl/realistic/best_model/best_mppt_model.zip"
        if not model_path.is_file():
            model_path = cfg.paths.root_dir / "artifacts/rl/realistic/final_model/sac_mppt_final.zip"

        model = load_sac_agent(model_path, device="cpu")
        adapter = SACControllerAdapter(model=model, deterministic=True)
        assert adapter.name == "SAC"

        init_d = adapter.reset(initial_voltage=31.5, initial_current=7.5, initial_duty=0.35)
        assert pytest.approx(init_d, abs=1e-4) == 0.35

        next_d, diag = adapter.predict(
            v_pv=30.0,
            i_pv=7.8,
            current_duty=init_d,
            simulation_time=0.005,
            irradiance=1000.0,
            cell_temperature=25.0,
        )
        assert 0.05 <= next_d <= 0.90
        assert "raw_action" in diag
        assert -1.0 <= diag["raw_action"] <= 1.0
        # Duty step should not exceed delta_duty_max (0.01)
        assert abs(next_d - init_d) <= 0.01 + 1e-6
