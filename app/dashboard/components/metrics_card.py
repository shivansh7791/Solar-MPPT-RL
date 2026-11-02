"""Metrics Display Component for Streamlit Dashboard.

Renders KPI performance metric cards for single controller runs and side-by-side
differential comparison summaries.
"""

from __future__ import annotations

import streamlit as st

from app.evaluation.comparison_runner import PairedScenarioResult
from app.evaluation.metrics import ComparisonMetrics


def render_single_metrics_card(metrics: ComparisonMetrics) -> None:
    """Render summary KPI cards for a single controller simulation run."""
    st.subheader(f"📊 Performance Metrics: {metrics.controller}")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="Tracking Efficiency (η)",
            value=f"{metrics.tracking_efficiency:.2f}%",
            help="Integrated PV energy divided by theoretical ideal MPP energy: ∫P_pv dt / ∫P_mpp dt.",
        )
        st.metric(
            label="Energy Deficit (Loss)",
            value=f"{metrics.energy_loss_joules:.2f} J",
            help="Theoretical energy lost due to tracking delay and oscillations: E_ideal - E_harvested.",
        )

    with c2:
        st.metric(
            label="Energy Harvested",
            value=f"{metrics.energy_harvested_joules:.2f} J",
            delta=f"{metrics.energy_harvested_wh:.4f} Wh",
            delta_color="off",
            help="Total electrical energy extracted from the PV array over the simulation duration.",
        )
        st.metric(
            label="Ideal MPP Energy",
            value=f"{metrics.energy_ideal_joules:.2f} J",
            help="Maximum theoretical solar energy available if operating continuously at exact MPP.",
        )

    with c3:
        st.metric(
            label="Steady-State Ripple",
            value=f"{metrics.steady_state_power_ripple_w:.3f} W",
            help="Peak-to-peak power oscillation (max - min) evaluated over the steady-state window.",
        )
        st.metric(
            label="RMS Tracking Error",
            value=f"{metrics.steady_state_rms_error_w:.3f} W",
            help="Root mean square error relative to instantaneous theoretical MPP power.",
        )

    with c4:
        conv_str = "Yes (Converged)" if metrics.is_converged else "No (Hunting/Lag)"
        st.metric(
            label="Convergence Success",
            value=conv_str,
            help="True if power permanently stays within ±2.0% tolerance band of dynamic MPP.",
        )
        settling_str = f"{metrics.settling_time_s:.3f} s" if metrics.settling_time_s is not None else "N/A"
        st.metric(
            label="Dynamic Settling Time",
            value=settling_str,
            help="Time taken to enter and permanently remain within ±2.0% of dynamic MPP.",
        )


def render_comparison_metrics_card(paired_res: PairedScenarioResult) -> None:
    """Render side-by-side comparative KPI cards contrasting P&O and SAC."""
    st.subheader("⚖️ Side-by-Side MPPT Comparative Performance")

    po = paired_res.po_metrics
    sac = paired_res.sac_metrics

    delta_eff = sac.tracking_efficiency - po.tracking_efficiency
    delta_energy = sac.energy_harvested_joules - po.energy_harvested_joules
    deficit_reduction = ((po.energy_loss_joules - sac.energy_loss_joules) / max(1e-6, po.energy_loss_joules)) * 100.0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="SAC Tracking Efficiency",
            value=f"{sac.tracking_efficiency:.2f}%",
            delta=f"{delta_eff:+.2f}% vs P&O",
            delta_color="normal",
            help="P&O Efficiency: {:.2f}% | SAC Efficiency: {:.2f}%".format(po.tracking_efficiency, sac.tracking_efficiency),
        )
        st.caption(f"Classical P&O: **{po.tracking_efficiency:.2f}%**")

    with c2:
        st.metric(
            label="Total Energy Harvested",
            value=f"{sac.energy_harvested_joules:.2f} J",
            delta=f"{delta_energy:+.2f} J ({delta_energy/max(1e-6, po.energy_harvested_joules)*100:+.1f}%)",
            delta_color="normal",
            help="Cumulative energy extracted over the scenario duration.",
        )
        st.caption(f"P&O: **{po.energy_harvested_joules:.2f} J** ({po.energy_harvested_wh:.4f} Wh)")

    with c3:
        st.metric(
            label="Energy Deficit Reduction",
            value=f"{deficit_reduction:.1f}%",
            delta=f"-{po.energy_loss_joules - sac.energy_loss_joules:.2f} J Lost",
            delta_color="normal",
            help="Percentage reduction in wasted solar energy compared to Classical P&O.",
        )
        st.caption(f"P&O Loss: **{po.energy_loss_joules:.2f} J** | SAC: **{sac.energy_loss_joules:.2f} J**")

    with c4:
        ripple_diff = sac.steady_state_power_ripple_w - po.steady_state_power_ripple_w
        st.metric(
            label="Steady-State Ripple",
            value=f"{sac.steady_state_power_ripple_w:.3f} W",
            delta=f"{ripple_diff:+.3f} W vs P&O",
            delta_color="inverse",
            help="Peak-to-peak perturbation ripple in the steady-state window.",
        )
        st.caption(f"P&O Ripple: **{po.steady_state_power_ripple_w:.3f} W**")
