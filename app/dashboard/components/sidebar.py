"""Sidebar Configuration Component for Streamlit Dashboard.

Renders user controls for environmental profiles, electrical parameters,
controller selection, checkpoint loading, and simulation execution triggers.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import streamlit as st

from app.dashboard.controller_service import ControllerService
from app.dashboard.simulation_service import SimulationRequest, SimulationService


def render_sidebar(
    simulation_service: SimulationService,
    controller_service: ControllerService,
) -> Tuple[SimulationRequest, bool]:
    """Render the sidebar controls and return the configured SimulationRequest and trigger flag.

    Parameters
    ----------
    simulation_service : SimulationService
        Simulation manager instance.
    controller_service : ControllerService
        Controller manager instance.

    Returns
    -------
    Tuple[SimulationRequest, bool]
        (configured_request, run_button_clicked)
    """
    st.sidebar.header("⚡ Simulation Controls")

    # 1. Controller Mode Selection
    controller_mode = st.sidebar.selectbox(
        "Controller Mode",
        options=[
            "Comparison (P&O vs SAC)",
            "Soft Actor-Critic (SAC)",
            "Classical Perturb & Observe (P&O)",
        ],
        index=0,
        help="Select whether to simulate a single controller or execute a paired side-by-side comparative benchmark.",
    )

    # 2. Environmental Profile Selection
    profiles = simulation_service.get_available_profiles()
    profile_options = list(profiles.keys())
    selected_profile = st.sidebar.selectbox(
        "Environmental Profile",
        options=profile_options,
        index=0,
        help="Choose a standard atmospheric transition scenario or configure static ambient conditions.",
    )

    # Context-dependent controls: If static profile, enable Irradiance and Temperature sliders
    is_static = selected_profile == "Static (Constant G & T)"
    if is_static:
        irradiance = st.sidebar.slider(
            "Solar Irradiance (G)",
            min_value=100.0,
            max_value=1200.0,
            value=1000.0,
            step=25.0,
            format="%.0f W/m²",
            help="Incident solar irradiance on the PV array.",
        )
        temperature = st.sidebar.slider(
            "Cell Temperature (T)",
            min_value=-10.0,
            max_value=75.0,
            value=25.0,
            step=2.5,
            format="%.1f °C",
            help="Operating cell junction temperature.",
        )
    else:
        st.sidebar.info(f"**Profile Note**: {profiles[selected_profile]}")
        irradiance = 1000.0
        temperature = 25.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔌 Plant & Initial State")

    # 3. Converter & Initial Condition Sliders
    load_resistance = st.sidebar.slider(
        "Load Resistance (R_load)",
        min_value=2.0,
        max_value=40.0,
        value=10.0,
        step=1.0,
        format="%.1f Ω",
        help="Output DC resistive load (nominal design baseline is 10.0 Ω for 240W bus matching).",
    )

    initial_duty = st.sidebar.slider(
        "Initial Duty Cycle (D_0)",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.05,
        format="%.2f",
        help="Initial converter duty cycle command commanded at t=0 before MPPT feedback begins.",
    )

    duration = st.sidebar.slider(
        "Simulation Duration",
        min_value=0.2,
        max_value=2.5,
        value=1.5,
        step=0.1,
        format="%.1f s",
        help="Physical simulation time. 1.5 s corresponds to 300 control intervals (15,000 RK4 integration steps).",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 SAC Policy Model")

    # 4. Checkpoint Selection
    checkpoints = controller_service.get_available_checkpoints()
    checkpoint_names = list(checkpoints.keys())

    if checkpoint_names:
        selected_cp_name = st.sidebar.selectbox(
            "SAC Checkpoint",
            options=checkpoint_names,
            index=0,
            help="Select the trained Soft Actor-Critic neural network checkpoint.",
        )
        selected_cp = checkpoints[selected_cp_name]
        sac_checkpoint_path = str(selected_cp.path)
        is_privileged = selected_cp.mode == "privileged"

        if is_privileged:
            st.sidebar.warning(
                "⚠️ **Experimental Checkpoint**: This policy expects 10 observation dimensions "
                "(including internal inductor current and output voltage states)."
            )
        else:
            st.sidebar.caption(
                f"✅ Verified {selected_cp.obs_dim}-Dim Input | {selected_cp.active_eval_params:,} Active Weights"
            )
    else:
        st.sidebar.error("No SAC model checkpoints found in `artifacts/rl/`.")
        sac_checkpoint_path = None
        is_privileged = False

    st.sidebar.markdown("---")

    # 5. Execution Trigger Button
    run_button = st.sidebar.button(
        "🚀 Run MPPT Simulation",
        type="primary",
        use_container_width=True,
    )

    request = SimulationRequest(
        controller_mode=controller_mode,
        profile_type=selected_profile,
        irradiance=irradiance,
        temperature=temperature,
        load_resistance=load_resistance,
        initial_duty=initial_duty,
        duration=duration,
        sac_checkpoint_path=sac_checkpoint_path,
        use_privileged_policy=is_privileged,
    )

    return request, run_button
