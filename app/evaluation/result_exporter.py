"""Structured Results Exporter for MPPT Comparative Study.

Persists experiment results into structured CSV and JSON artifacts under results/metrics/phase_7/:
- scenario_definitions.json
- raw_simulation_data.csv
- scenario_metrics.csv
- paired_comparison.csv
- statistical_summary.json
- controller_summary.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from app.evaluation.comparison_runner import PairedScenarioResult
from app.evaluation.metrics import ComparisonMetrics
from app.evaluation.scenario_manager import ScenarioManager
from app.evaluation.statistical_analysis import StatisticalSummary
from app.utils.logger import get_logger

logger = get_logger("result_exporter")


class ResultExporter:
    """Manages serialization and structured data export for Phase 7."""

    def __init__(self, output_dir: Union[str, Path]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        scenario_manager: ScenarioManager,
        paired_results: List[PairedScenarioResult],
        statistical_summary: StatisticalSummary,
    ) -> Dict[str, Path]:
        """Export all Phase 7 structured data files and return dictionary of paths."""
        logger.info(f"Exporting Phase 7 results to: {self.output_dir}")

        exported_paths: Dict[str, Path] = {}

        # 1. Scenario definitions
        sc_def_path = self.output_dir / "scenario_definitions.json"
        scenario_manager.export_definitions_json(sc_def_path)
        exported_paths["scenario_definitions"] = sc_def_path

        # 2. Raw simulation data (control-step resolution for both controllers)
        raw_dfs = []
        for r in paired_results:
            raw_dfs.append(r.po_timeseries)
            raw_dfs.append(r.sac_timeseries)
        combined_raw_df = pd.concat(raw_dfs, ignore_index=True)
        raw_path = self.output_dir / "raw_simulation_data.csv"
        combined_raw_df.to_csv(raw_path, index=False)
        exported_paths["raw_simulation_data"] = raw_path

        # 3. Individual scenario metrics (38 rows)
        metric_rows = []
        for r in paired_results:
            metric_rows.append(r.po_metrics.to_dict())
            metric_rows.append(r.sac_metrics.to_dict())
        scenario_metrics_df = pd.DataFrame(metric_rows)
        metrics_path = self.output_dir / "scenario_metrics.csv"
        scenario_metrics_df.to_csv(metrics_path, index=False)
        exported_paths["scenario_metrics"] = metrics_path

        # 4. Paired comparison rows (19 rows)
        paired_rows = [r.to_paired_record() for r in paired_results]
        paired_df = pd.DataFrame(paired_rows)
        paired_path = self.output_dir / "paired_comparison.csv"
        paired_df.to_csv(paired_path, index=False)
        exported_paths["paired_comparison"] = paired_path

        # 5. Statistical summary JSON
        stats_path = self.output_dir / "statistical_summary.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(statistical_summary.to_dict(), f, indent=4)
        exported_paths["statistical_summary"] = stats_path

        # 6. Controller summary table (P&O vs SAC aggregate comparison)
        summary_path = self._export_controller_summary(scenario_metrics_df)
        exported_paths["controller_summary"] = summary_path

        logger.info(f"Successfully exported {len(exported_paths)} Phase 7 artifacts.")
        return exported_paths

    def _export_controller_summary(self, metrics_df: pd.DataFrame) -> Path:
        """Create and save high-level controller comparison table."""
        po_mask = metrics_df["controller"] == "P&O"
        sac_mask = metrics_df["controller"] == "SAC"

        po_df = metrics_df[po_mask]
        sac_df = metrics_df[sac_mask]

        summary_data = [
            {
                "metric": "Mean Tracking Efficiency (%)",
                "PO": round(float(po_df["tracking_efficiency"].mean()), 2),
                "SAC": round(float(sac_df["tracking_efficiency"].mean()), 2),
                "Difference (SAC - PO)": round(float(sac_df["tracking_efficiency"].mean() - po_df["tracking_efficiency"].mean()), 2),
            },
            {
                "metric": "Total Energy Harvested (Joules)",
                "PO": round(float(po_df["energy_harvested_joules"].sum()), 2),
                "SAC": round(float(sac_df["energy_harvested_joules"].sum()), 2),
                "Difference (SAC - PO)": round(float(sac_df["energy_harvested_joules"].sum() - po_df["energy_harvested_joules"].sum()), 2),
            },
            {
                "metric": "Total Energy Harvested (Wh)",
                "PO": round(float(po_df["energy_harvested_wh"].sum()), 4),
                "SAC": round(float(sac_df["energy_harvested_wh"].sum()), 4),
                "Difference (SAC - PO)": round(float(sac_df["energy_harvested_wh"].sum() - po_df["energy_harvested_wh"].sum()), 4),
            },
            {
                "metric": "Total Energy Loss (Joules)",
                "PO": round(float(po_df["energy_loss_joules"].sum()), 2),
                "SAC": round(float(sac_df["energy_loss_joules"].sum()), 2),
                "Difference (SAC - PO)": round(float(sac_df["energy_loss_joules"].sum() - po_df["energy_loss_joules"].sum()), 2),
            },
            {
                "metric": "Mean Steady-State Power Ripple (W)",
                "PO": round(float(po_df["steady_state_power_ripple_w"].mean()), 4),
                "SAC": round(float(sac_df["steady_state_power_ripple_w"].mean()), 4),
                "Difference (SAC - PO)": round(float(sac_df["steady_state_power_ripple_w"].mean() - po_df["steady_state_power_ripple_w"].mean()), 4),
            },
            {
                "metric": "Mean Steady-State Power Std (W)",
                "PO": round(float(po_df["steady_state_power_std_w"].mean()), 4),
                "SAC": round(float(sac_df["steady_state_power_std_w"].mean()), 4),
                "Difference (SAC - PO)": round(float(sac_df["steady_state_power_std_w"].mean() - po_df["steady_state_power_std_w"].mean()), 4),
            },
            {
                "metric": "Mean Steady-State RMS Error (W)",
                "PO": round(float(po_df["steady_state_rms_error_w"].mean()), 4),
                "SAC": round(float(sac_df["steady_state_rms_error_w"].mean()), 4),
                "Difference (SAC - PO)": round(float(sac_df["steady_state_rms_error_w"].mean() - po_df["steady_state_rms_error_w"].mean()), 4),
            },
            {
                "metric": "Mean Converter Efficiency (%)",
                "PO": round(float(po_df["mean_converter_efficiency"].mean()), 2),
                "SAC": round(float(sac_df["mean_converter_efficiency"].mean()), 2),
                "Difference (SAC - PO)": round(float(sac_df["mean_converter_efficiency"].mean() - po_df["mean_converter_efficiency"].mean()), 2),
            },
            {
                "metric": "Mean Duty Cycle Std",
                "PO": round(float(po_df["duty_cycle_std"].mean()), 4),
                "SAC": round(float(sac_df["duty_cycle_std"].mean()), 4),
                "Difference (SAC - PO)": round(float(sac_df["duty_cycle_std"].mean() - po_df["duty_cycle_std"].mean()), 4),
            },
            {
                "metric": "Convergence Success Rate (%)",
                "PO": round(float(po_df["is_converged"].mean() * 100.0), 1),
                "SAC": round(float(sac_df["is_converged"].mean() * 100.0), 1),
                "Difference (SAC - PO)": round(float((sac_df["is_converged"].mean() - po_df["is_converged"].mean()) * 100.0), 1),
            },
        ]

        summary_df = pd.DataFrame(summary_data)
        out_path = self.output_dir / "controller_summary.csv"
        summary_df.to_csv(out_path, index=False)
        return out_path
