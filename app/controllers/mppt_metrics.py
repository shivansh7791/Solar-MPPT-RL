"""Performance Evaluation Metrics for Maximum Power Point Tracking (MPPT) Controllers.

Calculates standardized engineering performance indicators:
1. MPPT Tracking Efficiency (eta_MPPT [%]).
2. Settling Time (t_s [s] to within +/- 2% of MPP).
3. Steady-State Power Oscillation (standard deviation and peak-to-peak ripple).
4. Power and Energy Loss relative to theoretical maximum power yield.
5. Separation of MPPT tracking efficiency and DC-DC converter conversion efficiency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.pv.pv_model import SolarPVModel
from app.utils.logger import get_logger

logger = get_logger("mppt_metrics")


@dataclass(frozen=True)
class MPPTMetrics:
    """Quantitative performance evaluation metrics of an MPPT controller execution.

    Attributes
    ----------
    tracking_efficiency : float
        Overall energy tracking efficiency (Integral(P_actual) / Integral(P_MPP) * 100) [%].
    settling_time : Optional[float]
        Time required for PV power to enter and permanently stay within tolerance band of P_mpp [s].
    is_converged : bool
        True if the system entered and remained within the steady-state tolerance band.
    tolerance_band_pct : float
        Tolerance band threshold used for settling time identification (e.g. 2.0%).
    steady_state_power_mean : float
        Mean electrical power in steady-state evaluation window [W].
    steady_state_power_std : float
        Standard deviation of power in steady-state evaluation window [W].
    steady_state_power_ripple : float
        Peak-to-peak power variation (max - min) in steady-state evaluation window [W].
    steady_state_duty_mean : float
        Mean duty cycle in steady-state evaluation window.
    steady_state_duty_std : float
        Standard deviation of duty cycle in steady-state evaluation window.
    steady_state_duty_ripple : float
        Peak-to-peak duty cycle ripple (max - min) in steady-state evaluation window.
    mean_power_loss : float
        Mean instantaneous power difference (P_MPP - P_actual) [W].
    cumulative_energy_loss : float
        Total energy deficit Integral(P_MPP - P_actual) dt [Joules].
    total_energy_actual : float
        Total electrical energy extracted Integral(P_actual) dt [Joules].
    total_energy_ideal : float
        Theoretical maximum electrical energy Integral(P_MPP) dt [Joules].
    final_v_pv : float
        Terminal PV voltage at conclusion of simulation [V].
    final_i_pv : float
        Terminal PV current at conclusion of simulation [A].
    final_p_pv : float
        Extracted PV power at conclusion of simulation [W].
    final_v_mpp : float
        Theoretical reference MPP voltage at conclusion of simulation [V].
    final_i_mpp : float
        Theoretical reference MPP current at conclusion of simulation [A].
    final_p_mpp : float
        Theoretical reference MPP power at conclusion of simulation [W].
    final_duty_cycle : float
        Duty cycle command at conclusion of simulation.
    mean_converter_efficiency : float
        Average DC-DC power conversion efficiency (P_out / P_pv * 100) [%].
    total_controller_updates : int
        Total number of MPPT sampling updates executed.
    total_saturation_events : int
        Total number of steps where duty cycle hit min_duty or max_duty.
    duration : float
        Total simulation duration [s].
    """

    tracking_efficiency: float
    settling_time: Optional[float]
    is_converged: bool
    tolerance_band_pct: float
    steady_state_power_mean: float
    steady_state_power_std: float
    steady_state_power_ripple: float
    steady_state_duty_mean: float
    steady_state_duty_std: float
    steady_state_duty_ripple: float
    mean_power_loss: float
    cumulative_energy_loss: float
    total_energy_actual: float
    total_energy_ideal: float
    final_v_pv: float
    final_i_pv: float
    final_p_pv: float
    final_v_mpp: float
    final_i_mpp: float
    final_p_mpp: float
    final_duty_cycle: float
    mean_converter_efficiency: float
    total_controller_updates: int
    total_saturation_events: int
    duration: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a standard Python dictionary."""
        return asdict(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a single-row Pandas DataFrame."""
        return pd.DataFrame([self.to_dict()])


def calculate_mppt_metrics(
    df: pd.DataFrame,
    pv_model: Optional[SolarPVModel] = None,
    tolerance_band_pct: float = 2.0,
    steady_state_window_ratio: float = 0.3,
) -> MPPTMetrics:
    """Calculate comprehensive performance metrics from an MPPT simulation timeseries DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation timeseries DataFrame containing at minimum:
        ['time', 'v_pv', 'i_pv', 'p_pv', 'duty_cycle', 'irradiance', 'cell_temperature'].
        Optional: ['p_mpp', 'efficiency', 'is_saturated'].
    pv_model : Optional[SolarPVModel], optional
        SolarPVModel instance used to calculate reference MPP if 'p_mpp' is not in df.
    tolerance_band_pct : float, optional
        Percentage tolerance (+/- %) around P_mpp for settling time determination, by default 2.0.
    steady_state_window_ratio : float, optional
        Fraction of timeseries at the end reserved for steady-state evaluation, by default 0.3 (last 30%).

    Returns
    -------
    MPPTMetrics
        Dataclass containing all evaluated performance indicators.

    Raises
    ------
    ValueError
        If DataFrame is empty, missing mandatory columns, or has non-positive duration.
    """
    if df.empty:
        raise ValueError("Cannot calculate MPPT metrics on an empty DataFrame.")

    required_cols = {"time", "v_pv", "i_pv", "p_pv", "duty_cycle"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    time_arr = df["time"].to_numpy(dtype=float)
    duration = float(time_arr[-1] - time_arr[0])
    if duration <= 0.0:
        raise ValueError(f"Simulation duration must be strictly positive. Received: {duration} s")

    p_actual = df["p_pv"].to_numpy(dtype=float)
    duty_arr = df["duty_cycle"].to_numpy(dtype=float)

    # 1. Resolve or compute theoretical reference P_MPP timeseries
    if "p_mpp" in df.columns:
        p_ideal = df["p_mpp"].to_numpy(dtype=float)
        v_ideal = df["v_mpp"].to_numpy(dtype=float) if "v_mpp" in df.columns else np.full_like(p_ideal, np.nan)
        i_ideal = df["i_mpp"].to_numpy(dtype=float) if "i_mpp" in df.columns else np.full_like(p_ideal, np.nan)
    else:
        model = pv_model if pv_model is not None else SolarPVModel()
        irrad_col = df["irradiance"].to_numpy(dtype=float) if "irradiance" in df.columns else np.full_like(time_arr, 1000.0)
        temp_col = df["cell_temperature"].to_numpy(dtype=float) if "cell_temperature" in df.columns else np.full_like(time_arr, 25.0)

        # Cache distinct environmental points for efficiency
        unique_conds = {}
        p_ideal = np.empty_like(time_arr)
        v_ideal = np.empty_like(time_arr)
        i_ideal = np.empty_like(time_arr)

        for idx, (g, t) in enumerate(zip(irrad_col, temp_col)):
            key = (round(g, 2), round(t, 2))
            if key not in unique_conds:
                stc_point = model.calculate(key[0], key[1])
                unique_conds[key] = (stc_point.p_mp, stc_point.v_mp, stc_point.i_mp)
            p_ideal[idx], v_ideal[idx], i_ideal[idx] = unique_conds[key]

    # 2. Energy integration using Trapezoidal rule
    total_energy_actual = float(np.trapezoid(p_actual, time_arr))
    total_energy_ideal = float(np.trapezoid(p_ideal, time_arr))

    if total_energy_ideal > 1e-6:
        tracking_efficiency = float((total_energy_actual / total_energy_ideal) * 100.0)
    else:
        tracking_efficiency = 0.0

    # 3. Power and Energy Loss
    power_loss_arr = p_ideal - p_actual
    cumulative_energy_loss = float(np.trapezoid(np.maximum(0.0, power_loss_arr), time_arr))
    mean_power_loss = float(cumulative_energy_loss / duration)

    # 4. Settling Time Identification
    # Tolerance band: |P_actual(t) - P_ideal(t)| <= (tolerance_band_pct / 100) * P_ideal(t)
    tol_fraction = tolerance_band_pct / 100.0
    tol_band = tol_fraction * p_ideal
    within_band = np.abs(p_actual - p_ideal) <= tol_band

    settling_time: Optional[float] = None
    is_converged = False

    # Find the earliest index after which the system NEVER leaves the tolerance band
    # (Checking from the end backwards to find the last exit point)
    outside_indices = np.where(~within_band)[0]
    if len(outside_indices) == 0:
        # Was within band from the very first step
        settling_time = float(time_arr[0])
        is_converged = True
    else:
        last_exit_idx = outside_indices[-1]
        if last_exit_idx < len(time_arr) - 1:
            settling_time = float(time_arr[last_exit_idx + 1])
            is_converged = True
        else:
            # Exited at the very end of simulation -> did not settle
            settling_time = None
            is_converged = False

    # 5. Steady-State Evaluation Window (last steady_state_window_ratio of timeseries)
    num_samples = len(time_arr)
    ss_start_idx = max(0, int(num_samples * (1.0 - steady_state_window_ratio)))
    ss_p = p_actual[ss_start_idx:]
    ss_d = duty_arr[ss_start_idx:]

    ss_p_mean = float(np.mean(ss_p))
    ss_p_std = float(np.std(ss_p, ddof=1)) if len(ss_p) > 1 else 0.0
    ss_p_ripple = float(np.max(ss_p) - np.min(ss_p))

    ss_d_mean = float(np.mean(ss_d))
    ss_d_std = float(np.std(ss_d, ddof=1)) if len(ss_d) > 1 else 0.0
    ss_d_ripple = float(np.max(ss_d) - np.min(ss_d))

    # 6. Final Electrical Points
    final_v_pv = float(df["v_pv"].iloc[-1])
    final_i_pv = float(df["i_pv"].iloc[-1])
    final_p_pv = float(p_actual[-1])
    final_duty = float(duty_arr[-1])
    final_p_mpp = float(p_ideal[-1])
    final_v_mpp = float(v_ideal[-1])
    final_i_mpp = float(i_ideal[-1])

    # 7. Converter Efficiency & Saturation Stats
    mean_conv_eff = float(df["efficiency"].mean()) if "efficiency" in df.columns else 0.0
    total_updates = int(df["is_mppt_step"].sum()) if "is_mppt_step" in df.columns else len(df)
    total_saturations = int(df["is_saturated"].sum()) if "is_saturated" in df.columns else 0

    return MPPTMetrics(
        tracking_efficiency=tracking_efficiency,
        settling_time=settling_time,
        is_converged=is_converged,
        tolerance_band_pct=tolerance_band_pct,
        steady_state_power_mean=ss_p_mean,
        steady_state_power_std=ss_p_std,
        steady_state_power_ripple=ss_p_ripple,
        steady_state_duty_mean=ss_d_mean,
        steady_state_duty_std=ss_d_std,
        steady_state_duty_ripple=ss_d_ripple,
        mean_power_loss=mean_power_loss,
        cumulative_energy_loss=cumulative_energy_loss,
        total_energy_actual=total_energy_actual,
        total_energy_ideal=total_energy_ideal,
        final_v_pv=final_v_pv,
        final_i_pv=final_i_pv,
        final_p_pv=final_p_pv,
        final_v_mpp=final_v_mpp,
        final_i_mpp=final_i_mpp,
        final_p_mpp=final_p_mpp,
        final_duty_cycle=final_duty,
        mean_converter_efficiency=mean_conv_eff,
        total_controller_updates=total_updates,
        total_saturation_events=total_saturations,
        duration=duration,
    )
