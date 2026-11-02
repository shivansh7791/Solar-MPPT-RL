"""Unit Tests for Comparison Runner Module."""

import pytest

from app.evaluation.comparison_runner import ComparisonRunner
from app.evaluation.controller_adapters import POControllerAdapter, SACControllerAdapter
from app.evaluation.scenario_manager import create_scenario_a
from app.rl.sac_agent import load_sac_agent
from config.default_config import get_default_config


class TestComparisonRunner:
    """Test suite for ComparisonRunner."""

    def test_paired_scenario_execution(self):
        cfg = get_default_config()
        model_path = cfg.paths.root_dir / "artifacts/rl/realistic/best_model/best_mppt_model.zip"
        if not model_path.is_file():
            model_path = cfg.paths.root_dir / "artifacts/rl/realistic/final_model/sac_mppt_final.zip"

        model = load_sac_agent(model_path, device="cpu")
        po_adapter = POControllerAdapter()
        sac_adapter = SACControllerAdapter(model=model, deterministic=True)

        runner = ComparisonRunner(config=cfg)
        sc = create_scenario_a()
        # Shorten scenario duration to 0.05s (10 control steps) for quick test
        sc.duration = 0.05

        paired_res = runner.run_paired_scenario(sc, po_adapter, sac_adapter)

        assert paired_res.po_metrics.tracking_efficiency > 0.0
        assert paired_res.sac_metrics.tracking_efficiency > 0.0
        assert len(paired_res.po_timeseries) == 11  # step 0 to 10
        assert len(paired_res.sac_timeseries) == 11
        assert "delta_efficiency" in paired_res.to_paired_record()
