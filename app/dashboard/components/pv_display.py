"""PV Specifications Display Component for Streamlit Dashboard."""

from __future__ import annotations

import streamlit as st

from app.pv.pv_model import PVOutput


def render_pv_specifications_table(mpp: PVOutput, irradiance: float, temperature: float) -> None:
    """Render physical electrical parameter cards for the PV module at current conditions."""
    st.subheader(f"☀️ Canadian Solar CS6P-240P Electrical Ratings (at {irradiance:.0f} W/m², {temperature:.1f}°C)")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            label="Maximum Power (P_mp)",
            value=f"{mpp.p_mp:.2f} W",
            help="Theoretical maximum power available at the MPP.",
        )
    with c2:
        st.metric(
            label="Voltage at MPP (V_mp)",
            value=f"{mpp.v_mp:.2f} V",
            help="Optimal terminal voltage at maximum power.",
        )
    with c3:
        st.metric(
            label="Current at MPP (I_mp)",
            value=f"{mpp.i_mp:.2f} A",
            help="Optimal terminal current at maximum power.",
        )
    with c4:
        st.metric(
            label="Open-Circuit Voltage (V_oc)",
            value=f"{mpp.v_oc:.2f} V",
            help="Array terminal voltage at zero current output.",
        )
    with c5:
        st.metric(
            label="Short-Circuit Current (I_sc)",
            value=f"{mpp.i_sc:.2f} A",
            help="Array current output at zero terminal voltage.",
        )
