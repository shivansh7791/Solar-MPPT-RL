"""Formal Statistical Hypothesis Testing and Distribution Analysis for MPPT Comparison.

Performs paired scenario-level statistical inference:
- Paired differences: Delta eta = eta_SAC - eta_PO, Delta E = E_SAC - E_PO, Delta Ripple.
- Parametric Paired Student's t-test (scipy.stats.ttest_rel).
- Non-parametric Wilcoxon Signed-Rank test (scipy.stats.wilcoxon).
- 95% Confidence Interval using Student's t-distribution.
- Standardized effect size (Cohen's d).
- Category-level aggregations and descriptive summaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from app.evaluation.comparison_runner import PairedScenarioResult
from app.utils.logger import get_logger

logger = get_logger("statistical_analysis")


@dataclass
class MetricDifferenceStats:
    """Statistical summary of paired difference Delta X = X_SAC - X_PO."""

    metric_name: str
    n_samples: int
    mean_difference: float
    median_difference: float
    std_difference: float
    min_difference: float
    max_difference: float
    ci_95_lower: float
    ci_95_upper: float
    t_statistic: float
    t_test_p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    cohens_d: float
    is_statistically_significant_05: bool
    is_statistically_significant_01: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StatisticalSummary:
    """Complete multi-metric and category-wise statistical summary."""

    total_scenarios: int
    efficiency_stats: MetricDifferenceStats
    energy_stats_joules: MetricDifferenceStats
    ripple_stats_watts: MetricDifferenceStats
    overall_mean_sac_efficiency: float
    overall_mean_po_efficiency: float
    overall_mean_efficiency_gain: float
    overall_energy_weighted_sac_efficiency: float
    overall_energy_weighted_po_efficiency: float
    overall_energy_weighted_efficiency_gain: float
    total_ideal_energy_joules: float
    total_sac_energy_harvested_joules: float
    total_po_energy_harvested_joules: float
    total_energy_gain_joules: float
    total_sac_energy_loss_joules: float
    total_po_energy_loss_joules: float
    category_summaries: Dict[str, Dict[str, Any]]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_scenarios": self.total_scenarios,
            "overall_mean_sac_efficiency": self.overall_mean_sac_efficiency,
            "overall_mean_po_efficiency": self.overall_mean_po_efficiency,
            "overall_mean_efficiency_gain": self.overall_mean_efficiency_gain,
            "overall_energy_weighted_sac_efficiency": self.overall_energy_weighted_sac_efficiency,
            "overall_energy_weighted_po_efficiency": self.overall_energy_weighted_po_efficiency,
            "overall_energy_weighted_efficiency_gain": self.overall_energy_weighted_efficiency_gain,
            "total_ideal_energy_joules": self.total_ideal_energy_joules,
            "total_sac_energy_harvested_joules": self.total_sac_energy_harvested_joules,
            "total_po_energy_harvested_joules": self.total_po_energy_harvested_joules,
            "total_energy_gain_joules": self.total_energy_gain_joules,
            "total_sac_energy_loss_joules": self.total_sac_energy_loss_joules,
            "total_po_energy_loss_joules": self.total_po_energy_loss_joules,
            "efficiency_difference_stats": self.efficiency_stats.to_dict(),
            "energy_difference_stats": self.energy_stats_joules.to_dict(),
            "ripple_difference_stats": self.ripple_stats_watts.to_dict(),
            "category_summaries": self.category_summaries,
            "notes": self.notes,
        }



def _compute_paired_stats(
    diff_arr: np.ndarray,
    metric_name: str,
) -> MetricDifferenceStats:
    """Compute parametric and non-parametric paired statistics on difference array."""
    n = len(diff_arr)
    if n < 2:
        raise ValueError(f"Need at least 2 samples for statistical analysis. Received {n}")

    mean_diff = float(np.mean(diff_arr))
    median_diff = float(np.median(diff_arr))
    std_diff = float(np.std(diff_arr, ddof=1)) if n > 1 else 0.0
    min_diff = float(np.min(diff_arr))
    max_diff = float(np.max(diff_arr))

    # Standard error of the mean
    sem = std_diff / math.sqrt(n) if n > 0 else 0.0

    # 95% Confidence Interval via Student's t-distribution
    t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 1.96
    ci_lower = float(mean_diff - t_crit * sem)
    ci_upper = float(mean_diff + t_crit * sem)

    # Paired t-test against H0: mu_diff = 0
    if std_diff > 1e-12:
        t_stat, t_pval = stats.ttest_1samp(diff_arr, popmean=0.0)
        t_stat = float(t_stat)
        t_pval = float(t_pval)
        cohens_d = float(mean_diff / std_diff)
    else:
        t_stat = 0.0
        t_pval = 1.0 if math.isclose(mean_diff, 0.0) else 0.0
        cohens_d = float("inf") if mean_diff > 0 else 0.0

    # Wilcoxon signed-rank test against H0: median_diff = 0
    # Note: if all differences are identical or zero, wilcoxon returns edge cases
    try:
        # zero_method='wilcox' discards zero differences
        if np.all(np.isclose(diff_arr, 0.0)):
            w_stat = 0.0
            w_pval = 1.0
        else:
            w_res = stats.wilcoxon(diff_arr, zero_method="wilcox", alternative="two-sided")
            w_stat = float(w_res.statistic)
            w_pval = float(w_res.pvalue)
    except Exception:
        w_stat = float("nan")
        w_pval = 1.0

    is_sig_05 = bool(t_pval < 0.05 or w_pval < 0.05)
    is_sig_01 = bool(t_pval < 0.01 or w_pval < 0.01)

    return MetricDifferenceStats(
        metric_name=metric_name,
        n_samples=n,
        mean_difference=mean_diff,
        median_difference=median_diff,
        std_difference=std_diff,
        min_difference=min_diff,
        max_difference=max_diff,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        t_statistic=t_stat,
        t_test_p_value=t_pval,
        wilcoxon_statistic=w_stat,
        wilcoxon_p_value=w_pval,
        cohens_d=cohens_d,
        is_statistically_significant_05=is_sig_05,
        is_statistically_significant_01=is_sig_01,
    )


def perform_statistical_analysis(
    results: List[PairedScenarioResult],
) -> StatisticalSummary:
    """Perform formal statistical analysis across all paired evaluation scenarios.

    Parameters
    ----------
    results : List[PairedScenarioResult]
        List of paired simulation outcomes from ComparisonRunner.

    Returns
    -------
    StatisticalSummary
        Comprehensive statistical summary object.
    """
    if not results:
        raise ValueError("Cannot perform statistical analysis on empty results list.")

    n_scenarios = len(results)

    delta_effs = np.array([r.delta_efficiency for r in results], dtype=float)
    delta_energies = np.array([r.delta_energy_joules for r in results], dtype=float)
    delta_ripples = np.array([r.delta_ripple_w for r in results], dtype=float)

    sac_effs = np.array([r.sac_metrics.tracking_efficiency for r in results], dtype=float)
    po_effs = np.array([r.po_metrics.tracking_efficiency for r in results], dtype=float)
    sac_energies = np.array([r.sac_metrics.energy_harvested_joules for r in results], dtype=float)
    po_energies = np.array([r.po_metrics.energy_harvested_joules for r in results], dtype=float)
    ideal_energies = np.array([r.po_metrics.energy_ideal_joules for r in results], dtype=float)
    sac_losses = np.array([r.sac_metrics.energy_loss_joules for r in results], dtype=float)
    po_losses = np.array([r.po_metrics.energy_loss_joules for r in results], dtype=float)

    total_ideal_energy = float(np.sum(ideal_energies))
    total_sac_energy = float(np.sum(sac_energies))
    total_po_energy = float(np.sum(po_energies))
    total_sac_loss = float(np.sum(sac_losses))
    total_po_loss = float(np.sum(po_losses))

    energy_weighted_sac_eff = float((total_sac_energy / max(1e-6, total_ideal_energy)) * 100.0)
    energy_weighted_po_eff = float((total_po_energy / max(1e-6, total_ideal_energy)) * 100.0)
    energy_weighted_gain = float(energy_weighted_sac_eff - energy_weighted_po_eff)

    eff_stats = _compute_paired_stats(delta_effs, "Tracking Efficiency [%]")
    energy_stats = _compute_paired_stats(delta_energies, "Energy Harvested [Joules]")
    ripple_stats = _compute_paired_stats(delta_ripples, "Steady-State Ripple [Watts]")

    # Category breakdowns
    categories = sorted(list(set(r.scenario.category for r in results)))
    category_summaries: Dict[str, Dict[str, Any]] = {}

    for cat in categories:
        cat_results = [r for r in results if r.scenario.category == cat]
        cat_n = len(cat_results)
        cat_delta_eff = np.array([r.delta_efficiency for r in cat_results])
        cat_sac_eff = np.array([r.sac_metrics.tracking_efficiency for r in cat_results])
        cat_po_eff = np.array([r.po_metrics.tracking_efficiency for r in cat_results])
        cat_sac_e = np.array([r.sac_metrics.energy_harvested_joules for r in cat_results])
        cat_po_e = np.array([r.po_metrics.energy_harvested_joules for r in cat_results])
        cat_ideal_e = np.array([r.po_metrics.energy_ideal_joules for r in cat_results])
        cat_delta_e = np.array([r.delta_energy_joules for r in cat_results])
        cat_sac_ripple = np.array([r.sac_metrics.steady_state_power_ripple_w for r in cat_results])
        cat_po_ripple = np.array([r.po_metrics.steady_state_power_ripple_w for r in cat_results])

        cat_tot_ideal = float(np.sum(cat_ideal_e))
        cat_sac_w_eff = float((np.sum(cat_sac_e) / max(1e-6, cat_tot_ideal)) * 100.0)
        cat_po_w_eff = float((np.sum(cat_po_e) / max(1e-6, cat_tot_ideal)) * 100.0)

        category_summaries[cat] = {
            "n_scenarios": cat_n,
            "mean_sac_efficiency": float(np.mean(cat_sac_eff)),
            "mean_po_efficiency": float(np.mean(cat_po_eff)),
            "mean_delta_efficiency": float(np.mean(cat_delta_eff)),
            "arithmetic_mean_sac_efficiency": float(np.mean(cat_sac_eff)),
            "arithmetic_mean_po_efficiency": float(np.mean(cat_po_eff)),
            "arithmetic_mean_delta_efficiency": float(np.mean(cat_delta_eff)),
            "energy_weighted_sac_efficiency": cat_sac_w_eff,
            "energy_weighted_po_efficiency": cat_po_w_eff,
            "energy_weighted_delta_efficiency": float(cat_sac_w_eff - cat_po_w_eff),
            "min_delta_efficiency": float(np.min(cat_delta_eff)),
            "max_delta_efficiency": float(np.max(cat_delta_eff)),
            "total_ideal_energy_j": cat_tot_ideal,
            "total_sac_energy_j": float(np.sum(cat_sac_e)),
            "total_po_energy_j": float(np.sum(cat_po_e)),
            "mean_delta_energy_j": float(np.mean(cat_delta_e)),
            "mean_sac_ripple_w": float(np.mean(cat_sac_ripple)),
            "mean_po_ripple_w": float(np.mean(cat_po_ripple)),
        }

    notes = [
        "Statistical study evaluates performance across environmental scenarios using fixed single checkpoint.",
        "Both controllers executed on identical PV and converter plant models with exact equilibrium initial conditions.",
        "Paired difference computed as Delta = SAC - P&O.",
        "Both parametric (Student's t) and non-parametric (Wilcoxon signed-rank) tests conducted at alpha=0.05 and 0.01.",
        "Arithmetic mean efficiency reflects unweighted scenario average.",
        "Energy-weighted aggregate efficiency reflects total energy harvested divided by total ideal reference energy.",
    ]

    return StatisticalSummary(
        total_scenarios=n_scenarios,
        efficiency_stats=eff_stats,
        energy_stats_joules=energy_stats,
        ripple_stats_watts=ripple_stats,
        overall_mean_sac_efficiency=float(np.mean(sac_effs)),
        overall_mean_po_efficiency=float(np.mean(po_effs)),
        overall_mean_efficiency_gain=float(np.mean(delta_effs)),
        overall_energy_weighted_sac_efficiency=energy_weighted_sac_eff,
        overall_energy_weighted_po_efficiency=energy_weighted_po_eff,
        overall_energy_weighted_efficiency_gain=energy_weighted_gain,
        total_ideal_energy_joules=total_ideal_energy,
        total_sac_energy_harvested_joules=total_sac_energy,
        total_po_energy_harvested_joules=total_po_energy,
        total_energy_gain_joules=float(np.sum(delta_energies)),
        total_sac_energy_loss_joules=total_sac_loss,
        total_po_energy_loss_joules=total_po_loss,
        category_summaries=category_summaries,
        notes=notes,
    )

