"""Integrated MPPT Simulation Runner for PV-Converter-Controller System.

Coordinates multirate execution between:
1. Plant Model: DC-DC Boost Converter stepping at fine integration interval (dt = 100 us).
2. Controller: Perturb & Observe MPPT executing at sampled control period (T_mppt = 5 ms).
3. Environment: Time-varying solar irradiance and cell temperature profiles.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from app.controllers.po_controller import PerturbAndObserveController
from app.converter.boost_converter import BoostConverter, ConverterMeasurements
from app.pv.pv_model import SolarPVModel
from app.utils.logger import get_logger
from config.default_config import MasterConfig, get_default_config

logger = get_logger("mppt_simulation")


class MPPTSimulator:
    """Integrated Multirate Simulation Runner for Solar PV MPPT Systems.

    Simulates the closed-loop feedback system where a digital MPPT controller samples
    PV voltage and current at interval T_mppt and commands a duty cycle held constant
    over multiple continuous-time plant integration steps dt.
    """

    def __init__(
        self,
        converter: Optional[BoostConverter] = None,
        controller: Optional[PerturbAndObserveController] = None,
        pv_model: Optional[SolarPVModel] = None,
        sampling_interval: Optional[float] = None,
        config: Optional[MasterConfig] = None,
    ) -> None:
        """Initialize the MPPT simulation environment.

        Parameters
        ----------
        converter : Optional[BoostConverter], optional
            BoostConverter plant instance. If None, created from configuration.
        controller : Optional[PerturbAndObserveController], optional
            P&O controller instance. If None, created from configuration.
        pv_model : Optional[SolarPVModel], optional
            SolarPVModel instance. If None, uses converter's model or default.
        sampling_interval : Optional[float], optional
            MPPT controller sampling interval T_mppt [s]. If None, from config.
        config : Optional[MasterConfig], optional
            Master configuration. If None, loaded from project defaults.
        """
        self.config = config if config is not None else get_default_config()

        self.pv_model = pv_model if pv_model is not None else SolarPVModel(self.config.pv)
        self.converter = converter if converter is not None else BoostConverter(self.pv_model, self.config.converter)
        self.controller = controller if controller is not None else PerturbAndObserveController(config=self.config.mppt.po)

        self.sampling_interval = float(
            sampling_interval if sampling_interval is not None else self.config.mppt.po.sampling_interval
        )
        if self.sampling_interval <= 0.0:
            raise ValueError(f"MPPT sampling interval must be positive. Received: {self.sampling_interval} s")

        # Cache reference MPP points to optimize simulation speed
        self._mpp_cache: Dict[Tuple[float, float], Tuple[float, float, float]] = {}

    def _get_reference_mpp(self, irradiance: float, cell_temperature: float) -> Tuple[float, float, float]:
        """Query or calculate analytical reference MPP (P_mpp, V_mpp, I_mpp) for given environmental condition."""
        key = (round(float(irradiance), 2), round(float(cell_temperature), 2))
        if key not in self._mpp_cache:
            stc_point = self.pv_model.calculate(key[0], key[1])
            self._mpp_cache[key] = (stc_point.p_mp, stc_point.v_mp, stc_point.i_mp)
        return self._mpp_cache[key]

    def run(
        self,
        duration: Optional[float] = None,
        irradiance: Union[float, Callable[[float], float]] = 1000.0,
        cell_temperature: Union[float, Callable[[float], float]] = 25.0,
        dt: Optional[float] = None,
        initial_duty: Optional[float] = None,
        initial_state: Optional[Tuple[float, float, float]] = None,
    ) -> pd.DataFrame:
        """Execute a closed-loop MPPT simulation run and return complete timeseries DataFrame.

        Parameters
        ----------
        duration : Optional[float], optional
            Total simulation time [s]. If None, uses config simulation_duration.
        irradiance : Union[float, Callable[[float], float]], optional
            Incident solar irradiance [W/m^2] or function G(t), by default 1000.0.
        cell_temperature : Union[float, Callable[[float], float]], optional
            Cell junction temperature [°C] or function T(t), by default 25.0.
        dt : Optional[float], optional
            Converter RK4 integration step [s]. If None, uses converter.default_dt.
        initial_duty : Optional[float], optional
            Starting duty cycle command. If None, uses controller default.
        initial_state : Optional[Tuple[float, float, float]], optional
            Initial converter state vector [V_pv, i_L, V_out]. If None, resets to converter default.

        Returns
        -------
        pd.DataFrame
            DataFrame with complete electrical, control, and environmental timeseries records.
        """
        sim_duration = float(duration if duration is not None else self.config.mppt.po.simulation_duration)
        step_dt = float(dt if dt is not None else self.converter.default_dt)

        if sim_duration <= 0.0:
            raise ValueError(f"Simulation duration must be strictly positive. Received: {sim_duration}")
        if step_dt <= 0.0:
            raise ValueError(f"Simulation step dt must be strictly positive. Received: {step_dt}")
        if self.sampling_interval < step_dt:
            raise ValueError(
                f"MPPT sampling interval ({self.sampling_interval} s) cannot be smaller than "
                f"converter integration step dt ({step_dt} s)."
            )

        steps_per_mppt = max(1, int(round(self.sampling_interval / step_dt)))
        total_steps = int(math.ceil(sim_duration / step_dt))

        # 1. Reset plant and controller
        self.converter.reset(initial_state=initial_state)
        curr_duty = self.controller.reset(initial_duty=initial_duty)

        # 2. Evaluate initial environmental condition and measurement at t=0
        g_0 = float(irradiance(0.0)) if callable(irradiance) else float(irradiance)
        t_0 = float(cell_temperature(0.0)) if callable(cell_temperature) else float(cell_temperature)
        meas_0 = self.converter.get_measurements(irradiance=g_0, cell_temperature=t_0)
        p_mpp_0, v_mpp_0, i_mpp_0 = self._get_reference_mpp(g_0, t_0)

        # Initialize controller with first measurement (step 0, no perturbation)
        curr_duty = self.controller.update(meas_0.v_pv, meas_0.i_pv)
        diag_0 = self.controller.get_state()

        records: List[Dict[str, Any]] = []

        # Record initial row at t = 0
        records.append({
            "time": 0.0,
            "v_pv": meas_0.v_pv,
            "i_pv": meas_0.i_pv,
            "p_pv": meas_0.p_pv,
            "i_l": meas_0.i_l,
            "v_out": meas_0.v_out,
            "i_out": meas_0.i_out,
            "p_out": meas_0.p_out,
            "duty_cycle": curr_duty,
            "efficiency": meas_0.efficiency,
            "is_ccm": meas_0.is_ccm,
            "irradiance": g_0,
            "cell_temperature": t_0,
            "p_mpp": p_mpp_0,
            "v_mpp": v_mpp_0,
            "i_mpp": i_mpp_0,
            "tracking_error": p_mpp_0 - meas_0.p_pv,
            "is_mppt_step": True,
            "delta_v": diag_0.delta_voltage,
            "delta_p": diag_0.delta_power,
            "duty_adjustment": diag_0.duty_adjustment,
            "perturbation_direction": diag_0.perturbation_direction,
            "is_saturated": diag_0.is_saturated,
        })

        time_since_mppt = 0.0

        # 3. Time-stepping loop
        for step_idx in range(1, total_steps + 1):
            sim_time = self.converter.time + step_dt

            # Current environmental condition
            g_t = float(irradiance(sim_time)) if callable(irradiance) else float(irradiance)
            t_t = float(cell_temperature(sim_time)) if callable(cell_temperature) else float(cell_temperature)

            # Advance converter plant using currently held duty cycle
            meas = self.converter.step(duty_cycle=curr_duty, irradiance=g_t, cell_temperature=t_t, dt=step_dt)
            time_since_mppt += step_dt

            is_mppt_step = False
            # Check if MPPT update is due
            if time_since_mppt >= (self.sampling_interval - 1e-9):
                curr_duty = self.controller.update(meas.v_pv, meas.i_pv)
                time_since_mppt = 0.0
                is_mppt_step = True

            diag = self.controller.get_state()
            p_mpp, v_mpp, i_mpp = self._get_reference_mpp(g_t, t_t)

            records.append({
                "time": float(meas.time),
                "v_pv": meas.v_pv,
                "i_pv": meas.i_pv,
                "p_pv": meas.p_pv,
                "i_l": meas.i_l,
                "v_out": meas.v_out,
                "i_out": meas.i_out,
                "p_out": meas.p_out,
                "duty_cycle": curr_duty,
                "efficiency": meas.efficiency,
                "is_ccm": meas.is_ccm,
                "irradiance": g_t,
                "cell_temperature": t_t,
                "p_mpp": p_mpp,
                "v_mpp": v_mpp,
                "i_mpp": i_mpp,
                "tracking_error": p_mpp - meas.p_pv,
                "is_mppt_step": is_mppt_step,
                "delta_v": diag.delta_voltage,
                "delta_p": diag.delta_power,
                "duty_adjustment": diag.duty_adjustment,
                "perturbation_direction": diag.perturbation_direction,
                "is_saturated": diag.is_saturated,
            })

        df = pd.DataFrame(records)
        logger.info(
            f"Completed MPPT simulation: duration={sim_duration:.2f}s, steps={len(df)}, "
            f"mppt_updates={df['is_mppt_step'].sum()}"
        )
        return df
