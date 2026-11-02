"""Unit Tests for Statistical Analysis Module."""

import numpy as np
import pytest

from app.evaluation.comparison_runner import PairedScenarioResult
from app.evaluation.metrics import ComparisonMetrics
from app.evaluation.scenario_manager import create_scenario_a
from app.evaluation.statistical_analysis import (
    _compute_paired_stats,
    perform_statistical_analysis,
)


def _make_dummy_metrics(ctrl: str, eff: float, energy: float, ripple: float) -> ComparisonMetrics:
    return ComparisonMetrics(
        controller=ctrl,
        scenario_id="dummy",
        scenario_name="Dummy",
        category="TEST",
        duration=1.5,
        initial_duty=0.35,
        tracking_efficiency=eff,
        energy_harvested_joules=energy,
        energy_harvested_wh=energy / 3600.0,
        energy_ideal_joules=energy / (eff / 100.0),
        energy_ideal_wh=(energy / (eff / 100.0)) / 3600.0,
        energy_loss_joules=10.0,
        energy_loss_wh=10.0 / 3600.0,
        settling_time_s=0.01,
        is_converged=True,
        tolerance_band_pct=2.0,
        steady_state_power_mean_w=240.0,
        steady_state_power_std_w=0.05,
        steady_state_power_ripple_w=ripple,
        steady_state_rms_error_w=0.02,
        steady_state_duty_mean=0.39,
        steady_state_duty_std=0.001,
        steady_state_duty_ripple=0.002,
        mean_pv_voltage_v=30.0,
        mean_pv_current_a=8.0,
        mean_pv_power_w=240.0,
        mean_duty_cycle=0.39,
        duty_cycle_std=0.005,
        max_tracking_error_w=5.0,
        mean_converter_efficiency=98.5,
        energy_converter_efficiency=98.5,
        total_saturation_events=0,
        fault_count=0,
    )


class TestStatisticalAnalysis:
    """Test suite for perform_statistical_analysis and paired stats."""

    def test_compute_paired_stats_positive_gain(self):
        # SAC outperforms P&O by ~1.0% on average
        diffs = np.array([0.8, 1.2, 0.9, 1.1, 1.0, 1.3, 0.7, 1.0])
        stats = _compute_paired_stats(diffs, "Efficiency Gain")

        assert pytest.approx(stats.mean_difference, abs=0.01) == 1.0
        assert stats.ci_95_lower > 0.0  # CI strictly positive
        assert stats.t_statistic > 0.0
        assert stats.t_test_p_value < 0.001
        assert stats.wilcoxon_p_value < 0.05
        assert stats.is_statistically_significant_05 is True
        assert stats.cohens_d > 4.0

    def test_perform_statistical_analysis_pipeline(self):
        sc = create_scenario_a()
        results = []
        for i in range(5):
            po_m = _make_dummy_metrics("P&O", eff=98.0 + 0.1 * i, energy=350.0, ripple=0.05)
            sac_m = _make_dummy_metrics("SAC", eff=99.5 + 0.1 * i, energy=355.0, ripple=0.001)
            results.append(
                PairedScenarioResult(
                    scenario=sc,
                    po_metrics=po_m,
                    sac_metrics=sac_m,
                    delta_efficiency=sac_m.tracking_efficiency - po_m.tracking_efficiency,
                    delta_energy_joules=sac_m.energy_harvested_joules - po_m.energy_harvested_joules,
                    delta_energy_wh=sac_m.energy_harvested_wh - po_m.energy_harvested_wh,
                    delta_ripple_w=sac_m.steady_state_power_ripple_w - po_m.steady_state_power_ripple_w,
                    delta_settling_time_s=0.0,
                    po_timeseries=None,
                    sac_timeseries=None,
                )
            )

        summary = perform_statistical_analysis(results)
        assert summary.total_scenarios == 5
        assert pytest.approx(summary.overall_mean_efficiency_gain, abs=0.01) == 1.5
        assert summary.total_energy_gain_joules == 25.0
        assert summary.efficiency_stats.is_statistically_significant_05 is True
        assert summary.overall_energy_weighted_sac_efficiency > 99.0
        assert summary.overall_energy_weighted_po_efficiency > 98.0
        assert summary.total_ideal_energy_joules > 0.0
