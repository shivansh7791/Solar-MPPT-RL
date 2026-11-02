"""Interactive Plotly Visualizations for MPPT Streamlit Dashboard.

Generates publication-grade, responsive scientific charts:
- Non-linear I-V and P-V characteristic curves with operating point projection.
- High-resolution dynamic response timeseries (Voltage, Current, Power, Duty, Energy, Ambient).
- Paired comparison overlays contrasting Classical P&O against Trained SAC.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.evaluation.metrics import ComparisonMetrics
from app.pv.pv_model import PVOutput

# Standardized visual color palette matching project publication figures
COLOR_PO = "#1F77B4"  # Classical P&O Blue
COLOR_SAC = "#FF7F0E"  # Trained SAC Amber/Orange
COLOR_MPP = "#2CA02C"  # Theoretical MPP Green
COLOR_GRID = "#E5E7EB"
COLOR_BG = "#FFFFFF"
COLOR_TEXT = "#1F2937"


def create_pv_characteristics_plot(
    v_curve: np.ndarray,
    i_curve: np.ndarray,
    p_curve: np.ndarray,
    mpp: PVOutput,
    operating_v: Optional[float] = None,
    operating_i: Optional[float] = None,
    operating_p: Optional[float] = None,
    controller_label: Optional[str] = None,
) -> go.Figure:
    """Create interactive dual-axis I-V and P-V curve with theoretical MPP and operating point."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. I-V Curve
    fig.add_trace(
        go.Scatter(
            x=v_curve,
            y=i_curve,
            name="Current (I-V)",
            mode="lines",
            line=dict(color="#2563EB", width=2.8),
            hovertemplate="Voltage: %{x:.2f} V<br>Current: %{y:.2f} A<extra></extra>",
        ),
        secondary_y=False,
    )

    # 2. P-V Curve
    fig.add_trace(
        go.Scatter(
            x=v_curve,
            y=p_curve,
            name="Power (P-V)",
            mode="lines",
            line=dict(color="#DC2626", width=2.8, dash="solid"),
            hovertemplate="Voltage: %{x:.2f} V<br>Power: %{y:.2f} W<extra></extra>",
        ),
        secondary_y=True,
    )

    # 3. Maximum Power Point (MPP) Marker
    fig.add_trace(
        go.Scatter(
            x=[mpp.v_mp],
            y=[mpp.p_mp],
            name=f"Theoretical MPP ({mpp.p_mp:.1f} W)",
            mode="markers",
            marker=dict(symbol="star", size=14, color="#16A34A", line=dict(width=1.5, color="#FFFFFF")),
            hovertemplate=f"<b>Theoretical MPP</b><br>Vmp: {mpp.v_mp:.2f} V<br>Pmp: {mpp.p_mp:.2f} W<extra></extra>",
        ),
        secondary_y=True,
    )

    # 4. Optional Controller Operating Point Marker
    if operating_v is not None and operating_p is not None:
        label = controller_label if controller_label else "Current Operating Point"
        fig.add_trace(
            go.Scatter(
                x=[operating_v],
                y=[operating_p],
                name=label,
                mode="markers",
                marker=dict(symbol="diamond", size=12, color="#9333EA", line=dict(width=1.5, color="#FFFFFF")),
                hovertemplate=f"<b>{label}</b><br>Vpv: {operating_v:.2f} V<br>Ppv: {operating_p:.2f} W<extra></extra>",
            ),
            secondary_y=True,
        )

    # Layout styling
    fig.update_layout(
        title="<b>PV Module Non-Linear Electrical Characteristics (CS6P-240P)</b>",
        xaxis_title="<b>PV Terminal Voltage V_pv (V)</b>",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor=COLOR_BG,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLOR_GRID, zeroline=True)
    fig.update_yaxes(title_text="<b>Output Current I_pv (A)</b>", secondary_y=False, showgrid=True, gridwidth=1, gridcolor=COLOR_GRID)
    fig.update_yaxes(title_text="<b>Generated Power P_pv (W)</b>", secondary_y=True, showgrid=False)

    return fig


def create_dynamic_response_dashboard(
    timeseries_df: pd.DataFrame,
    metrics: ComparisonMetrics,
    controller_name: str,
) -> go.Figure:
    """Generate 6-panel interactive timeseries dashboard for a single controller run."""
    t = timeseries_df["time"]

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "<b>1. Harvested PV Power vs. Ideal MPP</b>",
            "<b>2. Converter Duty Cycle Command</b>",
            "<b>3. PV Terminal Voltage</b>",
            "<b>4. PV Output Current</b>",
            "<b>5. Cumulative Energy Harvested</b>",
            "<b>6. Environmental Conditions (G & T)</b>",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    color_ctrl = COLOR_SAC if "SAC" in controller_name else COLOR_PO

    # Panel 1: Power vs MPP
    fig.add_trace(
        go.Scatter(x=t, y=timeseries_df["p_mpp"], name="Ideal MPP P_mpp", line=dict(color=COLOR_MPP, width=2, dash="dash")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=timeseries_df["p_pv"], name=f"{controller_name} P_pv", line=dict(color=color_ctrl, width=2)),
        row=1, col=1,
    )

    # Panel 2: Duty Cycle
    fig.add_trace(
        go.Scatter(x=t, y=timeseries_df["duty_cycle"], name="Duty Cycle D", line=dict(color=color_ctrl, width=2)),
        row=1, col=2,
    )
    fig.add_hline(y=0.90, line_dash="dot", line_color="#EF4444", row=1, col=2, annotation_text="D_max")
    fig.add_hline(y=0.05, line_dash="dot", line_color="#EF4444", row=1, col=2, annotation_text="D_min")

    # Panel 3: PV Voltage
    fig.add_trace(
        go.Scatter(x=t, y=timeseries_df["v_pv"], name="Voltage V_pv", line=dict(color="#3B82F6", width=2)),
        row=2, col=1,
    )
    if "v_mpp" in timeseries_df.columns:
        fig.add_trace(
            go.Scatter(x=t, y=timeseries_df["v_mpp"], name="Ideal V_mpp", line=dict(color=COLOR_MPP, width=1.5, dash="dot")),
            row=2, col=1,
        )

    # Panel 4: PV Current
    fig.add_trace(
        go.Scatter(x=t, y=timeseries_df["i_pv"], name="Current I_pv", line=dict(color="#10B981", width=2)),
        row=2, col=2,
    )

    # Panel 5: Cumulative Energy (Integral P dt)
    cum_actual = np.cumulative_trapezoid(timeseries_df["p_pv"], t, initial=0.0)
    cum_ideal = np.cumulative_trapezoid(timeseries_df["p_mpp"], t, initial=0.0)
    fig.add_trace(
        go.Scatter(x=t, y=cum_ideal, name="Ideal Energy Available (J)", line=dict(color=COLOR_MPP, width=1.8, dash="dash")),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=cum_actual, name="Harvested Energy (J)", line=dict(color=color_ctrl, width=2)),
        row=3, col=1,
    )

    # Panel 6: Ambient Environment
    fig.add_trace(
        go.Scatter(x=t, y=timeseries_df["irradiance"], name="Irradiance (W/m²)", line=dict(color="#F59E0B", width=2)),
        row=3, col=2,
    )
    if "cell_temperature" in timeseries_df.columns:
        fig.add_trace(
            go.Scatter(x=t, y=timeseries_df["cell_temperature"], name="Cell Temp (°C)", line=dict(color="#EC4899", width=1.5, dash="dot")),
            row=3, col=2,
        )

    # Axis Labels
    fig.update_xaxes(title_text="Time (s)", showgrid=True, gridcolor=COLOR_GRID)
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID)

    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Duty D", row=1, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=2, col=1)
    fig.update_yaxes(title_text="Current (A)", row=2, col=2)
    fig.update_yaxes(title_text="Energy (Joules)", row=3, col=1)
    fig.update_yaxes(title_text="Irradiance (W/m²)", row=3, col=2)

    fig.update_layout(
        height=850,
        title=f"<b>Dynamic Simulation Response: {controller_name} (Tracking Efficiency: {metrics.tracking_efficiency:.2f}%)</b>",
        showlegend=False,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=40, r=40, t=70, b=40),
    )

    return fig


def create_comparison_response_plots(
    po_df: pd.DataFrame,
    sac_df: pd.DataFrame,
    po_metrics: ComparisonMetrics,
    sac_metrics: ComparisonMetrics,
) -> go.Figure:
    """Generate side-by-side overlay comparison plots contrasting P&O and SAC waveforms."""
    t = po_df["time"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "<b>1. MPPT Power Tracking Overlay (W)</b>",
            "<b>2. Converter Duty Cycle Trajectory D(t)</b>",
            "<b>3. PV Terminal Voltage Response (V)</b>",
            "<b>4. Instantaneous Tracking Error (P_mpp - P_pv)</b>",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    # 1. Power Tracking
    fig.add_trace(
        go.Scatter(x=t, y=po_df["p_mpp"], name="Ideal MPP", line=dict(color=COLOR_MPP, width=2, dash="dash")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=po_df["p_pv"], name=f"P&O (η={po_metrics.tracking_efficiency:.2f}%)", line=dict(color=COLOR_PO, width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=sac_df["p_pv"], name=f"SAC (η={sac_metrics.tracking_efficiency:.2f}%)", line=dict(color=COLOR_SAC, width=2)),
        row=1, col=1,
    )

    # 2. Duty Cycle
    fig.add_trace(
        go.Scatter(x=t, y=po_df["duty_cycle"], name="P&O Duty Cycle", line=dict(color=COLOR_PO, width=2)),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=sac_df["duty_cycle"], name="SAC Duty Cycle", line=dict(color=COLOR_SAC, width=2)),
        row=1, col=2,
    )
    fig.add_hline(y=0.90, line_dash="dot", line_color="#EF4444", row=1, col=2)
    fig.add_hline(y=0.05, line_dash="dot", line_color="#EF4444", row=1, col=2)

    # 3. Voltage Response
    fig.add_trace(
        go.Scatter(x=t, y=po_df["v_pv"], name="P&O Voltage", line=dict(color=COLOR_PO, width=2)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=sac_df["v_pv"], name="SAC Voltage", line=dict(color=COLOR_SAC, width=2)),
        row=2, col=1,
    )
    if "v_mpp" in po_df.columns:
        fig.add_trace(
            go.Scatter(x=t, y=po_df["v_mpp"], name="V_mpp Ref", line=dict(color=COLOR_MPP, width=1.5, dash="dot")),
            row=2, col=1,
        )

    # 4. Tracking Error
    err_po = po_df["p_mpp"] - po_df["p_pv"]
    err_sac = sac_df["p_mpp"] - sac_df["p_pv"]
    fig.add_trace(
        go.Scatter(x=t, y=err_po, name="P&O Error (W)", line=dict(color=COLOR_PO, width=1.8)),
        row=2, col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=err_sac, name="SAC Error (W)", line=dict(color=COLOR_SAC, width=1.8)),
        row=2, col=2,
    )
    fig.add_hline(y=0.0, line_dash="solid", line_color="#9CA3AF", row=2, col=2)

    # Axes
    fig.update_xaxes(title_text="Time (s)", showgrid=True, gridcolor=COLOR_GRID)
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID)

    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Duty Cycle D", row=1, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=2, col=1)
    fig.update_yaxes(title_text="Error (W)", row=2, col=2)

    gain = sac_metrics.tracking_efficiency - po_metrics.tracking_efficiency
    fig.update_layout(
        height=680,
        title=f"<b>Comparative Tracking Dynamics: SAC vs. Classical P&O (Efficiency Gain: {gain:+.2f}%)</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor=COLOR_BG,
        margin=dict(l=40, r=40, t=70, b=40),
    )

    return fig
