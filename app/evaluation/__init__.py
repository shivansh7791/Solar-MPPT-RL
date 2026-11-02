"""Phase 7: Comparative Performance Evaluation of SAC and P&O MPPT Controllers.

Provides:
- Common MPPT controller interface and adapters (POControllerAdapter, SACControllerAdapter).
- Standardized scenario generation and management (STC, steps, ramp, randomized, duty sensitivity).
- Rigorous metric computation (energy tracking efficiency, settling time, steady-state ripple).
- Paired simulation runner ensuring identical plant and environmental conditions.
- Formal statistical hypothesis testing (paired t-test, Wilcoxon signed-rank, 95% CI, Cohen's d).
- 14 publication-quality visualization figures at 300 DPI.
- Structured CSV and JSON data exporters.
"""

from app.evaluation.controller_adapters import (
    BaseMPPTController,
    POControllerAdapter,
    SACControllerAdapter,
)
from app.evaluation.metrics import ComparisonMetrics, calculate_comparison_metrics
from app.evaluation.scenario_manager import (
    ScenarioDefinition,
    ScenarioManager,
    create_standard_scenarios,
)
from app.evaluation.comparison_runner import ComparisonRunner, PairedScenarioResult
from app.evaluation.statistical_analysis import (
    StatisticalSummary,
    perform_statistical_analysis,
)
from app.evaluation.result_exporter import ResultExporter
from app.evaluation.visualization import generate_all_phase_7_plots

__all__ = [
    "BaseMPPTController",
    "POControllerAdapter",
    "SACControllerAdapter",
    "ComparisonMetrics",
    "calculate_comparison_metrics",
    "ScenarioDefinition",
    "ScenarioManager",
    "create_standard_scenarios",
    "ComparisonRunner",
    "PairedScenarioResult",
    "StatisticalSummary",
    "perform_statistical_analysis",
    "ResultExporter",
    "generate_all_phase_7_plots",
]
