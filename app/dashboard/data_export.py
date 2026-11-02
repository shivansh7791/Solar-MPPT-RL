"""Data Export Serialization Utilities for Streamlit Dashboard.

Formats simulation timeseries, metrics, configurations, and comparison summaries
into downloadable CSV and JSON representations for user export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from app.evaluation.comparison_runner import PairedScenarioResult
from app.evaluation.metrics import ComparisonMetrics
from app.evaluation.scenario_manager import ScenarioDefinition
from config.default_config import ConverterConfig, MasterConfig


def export_timeseries_to_csv(df: pd.DataFrame) -> str:
    """Serialize timeseries dataframe to UTF-8 CSV string."""
    return df.to_csv(index=False)


def export_metrics_to_csv(metrics: Union[ComparisonMetrics, Dict[str, Any]]) -> str:
    """Serialize scalar metrics to CSV string."""
    if isinstance(metrics, ComparisonMetrics):
        df = metrics.to_dataframe()
    else:
        df = pd.DataFrame([metrics])
    return df.to_csv(index=False)


def export_paired_comparison_to_csv(paired_result: PairedScenarioResult) -> str:
    """Serialize paired comparison summary row to CSV string."""
    df = pd.DataFrame([paired_result.to_paired_record()])
    return df.to_csv(index=False)


def export_configuration_to_json(
    scenario: ScenarioDefinition,
    converter_config: ConverterConfig,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Format simulation parameters and configuration into structured JSON string."""
    data: Dict[str, Any] = {
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "name": scenario.name,
            "category": scenario.category,
            "duration_s": scenario.duration,
            "initial_duty": scenario.initial_duty,
            "seed": scenario.seed,
            "description": scenario.description,
        },
        "converter": {
            "inductance_h": converter_config.inductance,
            "input_capacitance_f": converter_config.input_capacitance,
            "output_capacitance_f": converter_config.output_capacitance,
            "load_resistance_ohm": converter_config.load_resistance,
            "inductor_series_resistance_ohm": converter_config.inductor_resistance,
            "switching_frequency_hz": converter_config.switching_frequency,
            "duty_min": converter_config.duty_cycle_min,
            "duty_max": converter_config.duty_cycle_max,
            "time_step_s": converter_config.time_step,
        },
    }

    if additional_metadata:
        data["metadata"] = additional_metadata

    return json.dumps(data, indent=4)


def load_phase_7_statistical_summary_json() -> Optional[str]:
    """Read audited Phase 7 statistical summary if available on disk."""
    p = Path("results/metrics/phase_7/statistical_summary.json")
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return None


def load_phase_7_controller_summary_csv() -> Optional[str]:
    """Read audited Phase 7 high-level controller comparison table if available on disk."""
    p = Path("results/metrics/phase_7/controller_summary.csv")
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return None
