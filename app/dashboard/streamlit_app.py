"""Main Streamlit Application for Solar MPPT Reinforcement Learning Study.

Interactive engineering dashboard comparing Classical Perturb & Observe (P&O)
with Deep Reinforcement Learning (Soft Actor-Critic) on a physics-validated
solar PV boost converter plant.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure repository root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import streamlit as st

from app.dashboard.components.metrics_card import (
    render_comparison_metrics_card,
    render_single_metrics_card,
)
from app.dashboard.components.pv_display import render_pv_specifications_table
from app.dashboard.components.sidebar import render_sidebar
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

# Page configuration
st.set_page_config(
    page_title="Solar MPPT: SAC vs Classical P&O Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main dashboard rendering routine."""
    # Instantiate core services
    config = get_default_config()
    simulation_service = SimulationService(config=config)
    controller_service = ControllerService(config=config)

    # Header section
    st.title("☀️ Reinforcement Learning-Based Maximum Power Point Tracking (MPPT)")
    st.markdown(
        "**Design and Performance Analysis of a Soft Actor-Critic (SAC) Controller "
        "vs. Classical Perturb & Observe (P&O) for Solar Photovoltaic Systems**  \n"
        "*EEE Minor Project | Canadian Solar CS6P-240P | CCM Boost Converter ($20\\text{ kHz}$) | $dt = 100\\ \\mu\\text{s}$, $T_{ctrl} = 5.0\\text{ ms}$*"
    )
    st.caption("🔬 Physics-grounded simulation environment using state-space averaged differential equations and single-diode De Soto formulation.")

    # Render sidebar controls
    req, run_clicked = render_sidebar(simulation_service, controller_service)

    # Session state initialization for persistent simulation outcomes
    if "last_single_result" not in st.session_state:
        st.session_state.last_single_result = None
    if "last_paired_result" not in st.session_state:
        st.session_state.last_paired_result = None
    if "last_simulation_request" not in st.session_state:
        st.session_state.last_simulation_request = None

    # Handle Simulation Execution
    if run_clicked:
        try:
            with st.spinner("⏳ Simulating non-linear PV and converter plant dynamics (RK4 integration)..."):
                is_comparison = "Comparison" in req.controller_mode

                if is_comparison:
                    po_adapter = controller_service.create_po_adapter()
                    sac_cp_path = Path(req.sac_checkpoint_path) if req.sac_checkpoint_path else None
                    expected_dim = 10 if req.use_privileged_policy else 8
                    sac_adapter = controller_service.create_sac_adapter(
                        checkpoint_path=sac_cp_path,
                        expected_dim=expected_dim,
                    )
                    paired_res = simulation_service.execute_comparison_simulation(
                        req=req,
                        po_controller=po_adapter,
                        sac_controller=sac_adapter,
                    )
                    st.session_state.last_paired_result = paired_res
                    st.session_state.last_single_result = None
                    st.session_state.last_simulation_request = req
                    st.success("✅ Paired comparative simulation completed successfully!")

                else:
                    if "SAC" in req.controller_mode:
                        sac_cp_path = Path(req.sac_checkpoint_path) if req.sac_checkpoint_path else None
                        expected_dim = 10 if req.use_privileged_policy else 8
                        ctrl = controller_service.create_sac_adapter(
                            checkpoint_path=sac_cp_path,
                            expected_dim=expected_dim,
                        )
                    else:
                        ctrl = controller_service.create_po_adapter()

                    single_res = simulation_service.execute_single_simulation(
                        req=req,
                        controller=ctrl,
                    )
                    st.session_state.last_single_result = single_res
                    st.session_state.last_paired_result = None
                    st.session_state.last_simulation_request = req
                    st.success(f"✅ Single simulation completed for {ctrl.name}!")

        except Exception as e:
            st.error(f"❌ Simulation execution error: {str(e)}")

    # Main Content Tabs
    tab_overview, tab_pv, tab_sim, tab_compare, tab_exports = st.tabs([
        "📖 1. System Overview",
        "☀️ 2. PV Characteristics",
        "⚡ 3. MPPT Simulation",
        "⚖️ 4. P&O vs SAC Comparison",
        "📥 5. Data Export & Audit",
    ])

    # =========================================================================
    # TAB 1: SYSTEM OVERVIEW
    # =========================================================================
    with tab_overview:
        st.header("Project Overview & System Architecture")

        col_a, col_b = st.columns([1.1, 0.9])
        with col_a:
            st.markdown(r"""
            ### Scientific Context
            Solar photovoltaic (PV) power generation exhibits non-linear current-voltage ($I-V$) and power-voltage ($P-V$) 
            characteristics governed by semiconductor physics. Under fluctuating ambient irradiance and cell temperature, 
            the Maximum Power Point (MPP) continuously drifts.
            
            This project investigates whether deep reinforcement learning using **Soft Actor-Critic (SAC)** can outperform the 
            industry-standard **Classical Perturb & Observe (P&O)** algorithm by eliminating steady-state limit-cycle oscillations 
            and accelerating tracking agility under abrupt environmental transients.
            
            ### Physical Plant Specifications
            - **Photovoltaic Module**: Canadian Solar CS6P-240P ($240.1\text{ W}$ STC rating, 60 polycrystalline cells).
            - **PV Model Formulation**: Single-Diode 5-Parameter Empirical Model (CEC De Soto formulation).
            - **DC-DC Converter**: Non-isolated boost converter operating strictly in Continuous Conduction Mode (CCM).
            - **Converter Dynamics**: Modeled using state-space averaged differential equations integrated via 4th-order Runge-Kutta (RK4) at $dt = 100\ \mu\text{s}$.
            """)

        with col_b:
            st.markdown(r"""
            ### Mathematical Model Equations
            **Single-Diode PV Current Equation:**
            $$I_{pv} = I_L - I_0 \left[\exp\left(\frac{V_{pv} + I_{pv} R_s}{a}\right) - 1\right] - \frac{V_{pv} + I_{pv} R_s}{R_{sh}}$$
            
            **Boost Converter State Derivatives:**
            $$\frac{dV_{pv}}{dt} = \frac{I_{pv} - i_L}{C_{pv}}$$
            $$\frac{di_L}{dt} = \frac{V_{pv} - r_L i_L - (1 - D) V_{out}}{L}$$
            $$\frac{dV_{out}}{dt} = \frac{(1 - D) i_L - \frac{V_{out}}{R_{load}}}{C_{out}}$$
            """)

        st.markdown("---")
        st.subheader("Model Assumptions & Engineering Boundaries")
        col_warn1, col_warn2 = st.columns(2)
        with col_warn1:
            st.info(r"""
            **Valid Operational Envelope:**
            - Continuous Conduction Mode (CCM) strictly maintained across evaluated load and duty bounds.
            - Converter passive components sized for nominal $48\text{ V} - 50\text{ V}$ bus and $240\text{ W}$ throughput.
            - Exact equilibrium initial state root-finding incorporating inductor series resistance $r_L = 50\text{ m}\Omega$.
            """)
        with col_warn2:
            st.warning("""
            **Simulation Limitations:**
            - **State-Space Averaged Plant**: Deliberately averages out $20\text{ kHz}$ high-frequency PWM switching ripple.
            - **Uniform Solar Irradiance**: Does not model multi-peak partial shading conditions.
            - **Simulation Environment**: Runs in Python simulation; physical embedded deployment on DSP is an analytical projection.
            """)

    # =========================================================================
    # TAB 2: PV CHARACTERISTICS
    # =========================================================================
    with tab_pv:
        st.header("PV Module Electrical Characteristics")
        st.markdown("Evaluate non-linear electrical curves under the currently selected sidebar ambient conditions.")

        # Calculate curves
        g_now = req.irradiance
        t_now = req.temperature
        v_curve, i_curve, p_curve, mpp_now = simulation_service.calculate_pv_operating_curve(
            irradiance=g_now,
            temperature=t_now,
        )

        render_pv_specifications_table(mpp_now, g_now, t_now)

        # Plot curves
        fig_pv = create_pv_characteristics_plot(
            v_curve=v_curve,
            i_curve=i_curve,
            p_curve=p_curve,
            mpp=mpp_now,
        )
        st.plotly_chart(fig_pv, use_container_width=True)

    # =========================================================================
    # TAB 3: MPPT SIMULATION (SINGLE CONTROLLER)
    # =========================================================================
    with tab_sim:
        st.header("Interactive Closed-Loop MPPT Simulation")

        res_single = st.session_state.last_single_result

        if res_single is not None:
            render_single_metrics_card(res_single.metrics)

            fig_dash = create_dynamic_response_dashboard(
                timeseries_df=res_single.timeseries,
                metrics=res_single.metrics,
                controller_name=res_single.controller_name,
            )
            st.plotly_chart(fig_dash, use_container_width=True)

            st.markdown("### 📥 Export Run Data")
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                csv_ts = export_timeseries_to_csv(res_single.timeseries)
                st.download_button(
                    "Download Timeseries Data (CSV)",
                    data=csv_ts,
                    file_name=f"{res_single.controller_name.lower().replace(' ', '_')}_timeseries.csv",
                    mime="text/csv",
                )
            with c_d2:
                json_cfg = export_configuration_to_json(res_single.scenario, res_single.converter_config)
                st.download_button(
                    "Download Run Configuration (JSON)",
                    data=json_cfg,
                    file_name="simulation_config.json",
                    mime="application/json",
                )
        else:
            st.info("👈 Select a single controller ('Soft Actor-Critic' or 'Classical P&O') in the sidebar and click **Run MPPT Simulation**.")

    # =========================================================================
    # TAB 4: P&O VS SAC COMPARISON
    # =========================================================================
    with tab_compare:
        st.header("Side-by-Side MPPT Comparative Evaluation")

        res_paired = st.session_state.last_paired_result

        if res_paired is not None:
            render_comparison_metrics_card(res_paired)

            fig_comp = create_comparison_response_plots(
                po_df=res_paired.po_timeseries,
                sac_df=res_paired.sac_timeseries,
                po_metrics=res_paired.po_metrics,
                sac_metrics=res_paired.sac_metrics,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # Summary comparison dataframe
            st.subheader("📋 Numerical Comparison Table")
            summary_dict = res_paired.to_paired_record()
            display_cols = {
                "po_efficiency": "P&O Tracking Efficiency (%)",
                "sac_efficiency": "SAC Tracking Efficiency (%)",
                "delta_efficiency": "Efficiency Gain Δη (%)",
                "po_energy_harvested_j": "P&O Energy (J)",
                "sac_energy_harvested_j": "SAC Energy (J)",
                "delta_energy_joules": "Net Energy Gain ΔE (J)",
                "po_steady_state_ripple_w": "P&O Steady Ripple (W)",
                "sac_steady_state_ripple_w": "SAC Steady Ripple (W)",
            }
            sub_df = pd.DataFrame([{v: summary_dict.get(k, 0.0) for k, v in display_cols.items()}])
            st.dataframe(sub_df, use_container_width=True)

        else:
            st.info("👈 In the sidebar, select **Comparison (P&O vs SAC)** and click **Run MPPT Simulation** to view paired results.")

    # =========================================================================
    # TAB 5: DATA EXPORT & AUDIT REPORT
    # =========================================================================
    with tab_exports:
        st.header("Data Downloads & Phase 7 Engineering Audit")
        st.markdown(
            "Download validated simulation records and review the reconciled 19-scenario "
            "comparative performance audit."
        )

        st.subheader("📦 Audited Phase 7 Comparative Study Artifacts")
        col_exp1, col_exp2 = st.columns(2)

        summary_csv = load_phase_7_controller_summary_csv()
        if summary_csv:
            with col_exp1:
                st.markdown("**Phase 7 Controller Summary (19 Scenarios)**")
                st.download_button(
                    "Download controller_summary.csv",
                    data=summary_csv,
                    file_name="phase_7_controller_summary.csv",
                    mime="text/csv",
                )

        stats_json = load_phase_7_statistical_summary_json()
        if stats_json:
            with col_exp2:
                st.markdown("**Phase 7 Statistical Hypothesis Testing**")
                st.download_button(
                    "Download statistical_summary.json",
                    data=stats_json,
                    file_name="phase_7_statistical_summary.json",
                    mime="application/json",
                )

        st.markdown("---")
        st.subheader("📜 Documentation References")
        st.markdown("""
        - [`docs/PHASE_7_ANALYSIS.md`](file:///Users/shivanshupadhyay/Documents/Minor%20Project/docs/PHASE_7_ANALYSIS.md): Comprehensive 10-section comparative engineering analysis.
        - [`docs/PHASE_7_DOD.md`](file:///Users/shivanshupadhyay/Documents/Minor%20Project/docs/PHASE_7_DOD.md): Formal Definition of Done with exact energy conservation and statistical proof.
        - [`docs/PHASE_8_IMPLEMENTATION_PLAN.md`](file:///Users/shivanshupadhyay/Documents/Minor%20Project/docs/PHASE_8_IMPLEMENTATION_PLAN.md): Technical integration blueprint for Streamlit dashboard.
        """)


if __name__ == "__main__":
    main()
