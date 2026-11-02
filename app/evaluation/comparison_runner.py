"""Paired Comparison Simulation Runner for MPPT Controllers.

Executes identical multirate physical plant simulations for Classical P&O and Trained SAC,
ensuring rigorous experimental control:
- Identical PV single-diode equations and physical parameters.
- Identical state-space averaged DC-DC boost converter parameters.
- Identical exact physical equilibrium initial conditions.
- Identical time-varying irradiance G(t) and temperature T(t) trajectories.
- Identical control period T_ctrl = 5 ms (200 Hz) and plant integration dt = 100 us.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from app.converter.boost_converter import BoostConverter, ConverterMeasurements
from app.evaluation.controller_adapters import (
    BaseMPPTController,
    POControllerAdapter,
    SACControllerAdapter,
)
from app.evaluation.metrics import ComparisonMetrics, calculate_comparison_metrics
from app.evaluation.scenario_manager import ScenarioDefinition, ScenarioManager
from app.pv.pv_model import PVOutput, SolarPVModel
from app.utils.logger import get_logger
from config.default_config import MasterConfig, get_default_config

logger = get_logger("comparison_runner")


@dataclass
class PairedScenarioResult:
    """Encapsulates paired simulation outcomes for a single evaluation scenario."""

    scenario: ScenarioDefinition
    po_metrics: ComparisonMetrics
    sac_metrics: ComparisonMetrics
    delta_efficiency: float  # eta_SAC - eta_PO [%]
    delta_energy_joules: float  # E_SAC - E_PO [J]
    delta_energy_wh: float  # E_SAC - E_PO [Wh]
    delta_ripple_w: float  # Ripple_SAC - Ripple_PO [W]
    delta_settling_time_s: Optional[float]  # ts_SAC - ts_PO [s] (None if either did not settle)
    po_timeseries: pd.DataFrame
    sac_timeseries: pd.DataFrame

    def to_paired_record(self) -> Dict[str, Any]:
        """Convert paired metrics to a flat comparison row."""
        return {
            "scenario_id": self.scenario.scenario_id,
            "scenario_name": self.scenario.name,
            "category": self.scenario.category,
            "duration_s": self.scenario.duration,
            "initial_duty": self.scenario.initial_duty,
            # Tracking Efficiency
            "po_tracking_efficiency": self.po_metrics.tracking_efficiency,
            "sac_tracking_efficiency": self.sac_metrics.tracking_efficiency,
            "delta_efficiency": self.delta_efficiency,
            # Energy Harvested
            "po_energy_harvested_j": self.po_metrics.energy_harvested_joules,
            "sac_energy_harvested_j": self.sac_metrics.energy_harvested_joules,
            "delta_energy_j": self.delta_energy_joules,
            "po_energy_harvested_wh": self.po_metrics.energy_harvested_wh,
            "sac_energy_harvested_wh": self.sac_metrics.energy_harvested_wh,
            "delta_energy_wh": self.delta_energy_wh,
            # Energy Deficit / Loss
            "po_energy_loss_j": self.po_metrics.energy_loss_joules,
            "sac_energy_loss_j": self.sac_metrics.energy_loss_joules,
            # Settling Time
            "po_settling_time_s": self.po_metrics.settling_time_s,
            "sac_settling_time_s": self.sac_metrics.settling_time_s,
            "delta_settling_time_s": self.delta_settling_time_s,
            # Steady-State Power Ripple & Std
            "po_power_ripple_w": self.po_metrics.steady_state_power_ripple_w,
            "sac_power_ripple_w": self.sac_metrics.steady_state_power_ripple_w,
            "delta_ripple_w": self.delta_ripple_w,
            "po_power_std_w": self.po_metrics.steady_state_power_std_w,
            "sac_power_std_w": self.sac_metrics.steady_state_power_std_w,
            "po_rms_error_w": self.po_metrics.steady_state_rms_error_w,
            "sac_rms_error_w": self.sac_metrics.steady_state_rms_error_w,
            # Converter Efficiency
            "po_converter_efficiency": self.po_metrics.mean_converter_efficiency,
            "sac_converter_efficiency": self.sac_metrics.mean_converter_efficiency,
            # Duty Cycle Stability
            "po_duty_std": self.po_metrics.duty_cycle_std,
            "sac_duty_std": self.sac_metrics.duty_cycle_std,
            "po_duty_ripple": self.po_metrics.steady_state_duty_ripple,
            "sac_duty_ripple": self.sac_metrics.steady_state_duty_ripple,
        }


class ComparisonRunner:
    """Executes paired MPPT simulations under strictly controlled physical conditions."""

    def __init__(
        self,
        config: Optional[MasterConfig] = None,
        control_interval: float = 0.005,  # 5 ms (200 Hz)
        plant_dt: float = 0.0001,  # 100 us (10 kHz)
    ) -> None:
        self.config = config if config is not None else get_default_config()
        self.pv_model = SolarPVModel(self.config.pv)
        self.control_interval = float(control_interval)
        self.plant_dt = float(plant_dt)
        self.sub_steps = max(1, int(round(self.control_interval / self.plant_dt)))

        self._mpp_cache: Dict[Tuple[float, float], PVOutput] = {}

    def _get_mpp(self, g: float, t: float) -> PVOutput:
        """Cached lookup of analytical PV maximum power point."""
        key = (round(float(g), 2), round(float(t), 2))
        if key not in self._mpp_cache:
            self._mpp_cache[key] = self.pv_model.calculate(key[0], key[1])
        return self._mpp_cache[key]

    def compute_equilibrium_state(
        self,
        converter: BoostConverter,
        irradiance: float,
        cell_temperature: float,
        duty_cycle: float,
    ) -> Tuple[float, float, float]:
        """Compute exact physical equilibrium state (V_pv*, i_L*, V_out*) where all time derivatives are zero."""
        d = float(duty_cycle)
        r_in = ((1.0 - d) ** 2) * converter.R_load + converter.r_L
        mpp = self._get_mpp(irradiance, cell_temperature)

        def residual(v: float) -> float:
            i_pv = self.pv_model.current_at_voltage(v, irradiance, cell_temperature, clip_non_negative=True)
            return v - r_in * i_pv

        try:
            v_pv_star = float(brentq(residual, 0.0, mpp.v_oc * 1.05))
            i_l_star = float(self.pv_model.current_at_voltage(v_pv_star, irradiance, cell_temperature, clip_non_negative=True))
            v_out_star = float((1.0 - d) * converter.R_load * i_l_star)
            return v_pv_star, i_l_star, v_out_star
        except Exception:
            # Heuristic fallback if numerical root-finding encounters edge-case boundary
            v_pv_h = float(mpp.v_mp)
            i_l_h = float(mpp.i_mp)
            v_out_h = float(v_pv_h / max(0.1, 1.0 - d))
            return v_pv_h, i_l_h, v_out_h

    def simulate_controller(
        self,
        controller: BaseMPPTController,
        scenario: ScenarioDefinition,
    ) -> Tuple[pd.DataFrame, ComparisonMetrics]:
        """Simulate a single controller over the specified scenario.

        Parameters
        ----------
        controller : BaseMPPTController
            MPPT controller adapter instance.
        scenario : ScenarioDefinition
            Evaluation scenario specification.

        Returns
        -------
        Tuple[pd.DataFrame, ComparisonMetrics]
            (timeseries_dataframe, scalar_metrics)
        """
        # Create fresh converter plant instance
        converter = BoostConverter(pv_model=self.pv_model, config=self.config.converter)

        duration = float(scenario.duration)
        initial_duty = float(scenario.initial_duty)
        num_control_steps = int(math.ceil(duration / self.control_interval))

        # Initial environmental conditions at t=0
        g_0, t_0 = scenario.get_conditions(0.0)
        ref_mpp_0 = self._get_mpp(g_0, t_0)

        # Set physical plant to exact equilibrium
        init_state = self.compute_equilibrium_state(converter, g_0, t_0, initial_duty)
        meas_0 = converter.reset(
            initial_state=init_state,
            irradiance=g_0,
            cell_temperature=t_0,
        )

        # Reset controller
        curr_duty = float(controller.reset(
            initial_voltage=meas_0.v_pv,
            initial_current=meas_0.i_pv,
            initial_duty=initial_duty,
        ))

        records: List[Dict[str, Any]] = []

        # Record t=0 state
        records.append({
            "step": 0,
            "time": 0.0,
            "controller": controller.name,
            "scenario_id": scenario.scenario_id,
            "duty_cycle": curr_duty,
            "v_pv": meas_0.v_pv,
            "i_pv": meas_0.i_pv,
            "p_pv": meas_0.p_pv,
            "v_out": meas_0.v_out,
            "i_out": meas_0.i_out,
            "p_out": meas_0.p_out,
            "efficiency": meas_0.efficiency,
            "irradiance": g_0,
            "cell_temperature": t_0,
            "p_mpp": ref_mpp_0.p_mp,
            "v_mpp": ref_mpp_0.v_mp,
            "i_mpp": ref_mpp_0.i_mp,
            "tracking_error": ref_mpp_0.p_mp - meas_0.p_pv,
            "is_saturated": bool(math.isclose(curr_duty, controller.duty_min) or math.isclose(curr_duty, controller.duty_max)),
            "is_fault": False,
        })

        sim_time = 0.0
        is_fault = False

        for k in range(1, num_control_steps + 1):
            # 1. Integrate plant across sub_steps with currently active duty cycle
            last_meas: Optional[ConverterMeasurements] = None
            for _ in range(self.sub_steps):
                g_sub, t_sub = scenario.get_conditions(sim_time)
                try:
                    last_meas = converter.step(
                        duty_cycle=curr_duty,
                        irradiance=g_sub,
                        cell_temperature=t_sub,
                        dt=self.plant_dt,
                    )
                    sim_time += self.plant_dt
                except Exception as e:
                    logger.warning(f"Integration fault in {controller.name} at step {k}, t={sim_time:.4f}s: {e}")
                    is_fault = True
                    break

            if last_meas is None or is_fault:
                is_fault = True
                g_now, t_now = scenario.get_conditions(sim_time)
                last_meas = converter.get_measurements(g_now, t_now)

            # 2. Sample environmental condition and reference MPP at control decision boundary
            g_k, t_k = scenario.get_conditions(sim_time)
            ref_mpp_k = self._get_mpp(g_k, t_k)

            # 3. Controller predicts next duty cycle for following control interval
            next_duty, diag = controller.predict(
                v_pv=last_meas.v_pv,
                i_pv=last_meas.i_pv,
                current_duty=curr_duty,
                simulation_time=sim_time,
                irradiance=g_k,
                cell_temperature=t_k,
            )
            curr_duty = float(next_duty)

            records.append({
                "step": k,
                "time": sim_time,
                "controller": controller.name,
                "scenario_id": scenario.scenario_id,
                "duty_cycle": curr_duty,
                "v_pv": last_meas.v_pv,
                "i_pv": last_meas.i_pv,
                "p_pv": last_meas.p_pv,
                "v_out": last_meas.v_out,
                "i_out": last_meas.i_out,
                "p_out": last_meas.p_out,
                "efficiency": last_meas.efficiency,
                "irradiance": g_k,
                "cell_temperature": t_k,
                "p_mpp": ref_mpp_k.p_mp,
                "v_mpp": ref_mpp_k.v_mp,
                "i_mpp": ref_mpp_k.i_mp,
                "tracking_error": ref_mpp_k.p_mp - last_meas.p_pv,
                "is_saturated": bool(diag.get("is_saturated", False)),
                "is_fault": is_fault,
            })

            if is_fault:
                break

        df = pd.DataFrame(records)
        metrics = calculate_comparison_metrics(
            df=df,
            controller_name=controller.name,
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            category=scenario.category,
            pv_model=self.pv_model,
        )

        return df, metrics

    def run_paired_scenario(
        self,
        scenario: ScenarioDefinition,
        po_adapter: POControllerAdapter,
        sac_adapter: SACControllerAdapter,
    ) -> PairedScenarioResult:
        """Run identical paired simulation for P&O and SAC controllers on one scenario."""
        logger.info(f"Executing paired scenario: {scenario.scenario_id} ({scenario.name})...")

        # Simulate P&O
        po_df, po_metrics = self.simulate_controller(po_adapter, scenario)

        # Simulate SAC
        sac_df, sac_metrics = self.simulate_controller(sac_adapter, scenario)

        # Calculate differential performance indicators
        delta_eff = float(sac_metrics.tracking_efficiency - po_metrics.tracking_efficiency)
        delta_e_j = float(sac_metrics.energy_harvested_joules - po_metrics.energy_harvested_joules)
        delta_e_wh = float(sac_metrics.energy_harvested_wh - po_metrics.energy_harvested_wh)
        delta_ripple = float(sac_metrics.steady_state_power_ripple_w - po_metrics.steady_state_power_ripple_w)

        delta_ts: Optional[float] = None
        if sac_metrics.settling_time_s is not None and po_metrics.settling_time_s is not None:
            delta_ts = float(sac_metrics.settling_time_s - po_metrics.settling_time_s)

        logger.info(
            f"  Result {scenario.scenario_id}: SAC η={sac_metrics.tracking_efficiency:.2f}%, "
            f"P&O η={po_metrics.tracking_efficiency:.2f}%, Δη={delta_eff:+.2f}%, ΔE={delta_e_j:+.2f}J"
        )

        return PairedScenarioResult(
            scenario=scenario,
            po_metrics=po_metrics,
            sac_metrics=sac_metrics,
            delta_efficiency=delta_eff,
            delta_energy_joules=delta_e_j,
            delta_energy_wh=delta_e_wh,
            delta_ripple_w=delta_ripple,
            delta_settling_time_s=delta_ts,
            po_timeseries=po_df,
            sac_timeseries=sac_df,
        )

    def run_all_scenarios(
        self,
        scenario_manager: ScenarioManager,
        po_adapter: POControllerAdapter,
        sac_adapter: SACControllerAdapter,
    ) -> List[PairedScenarioResult]:
        """Execute paired evaluation across all scenarios in manager."""
        results: List[PairedScenarioResult] = []
        all_scenarios = scenario_manager.get_all()
        logger.info(f"Starting complete comparative evaluation over {len(all_scenarios)} paired scenarios...")

        for idx, sc in enumerate(all_scenarios, 1):
            logger.info(f"[{idx}/{len(all_scenarios)}] Scenario: {sc.scenario_id}")
            res = self.run_paired_scenario(sc, po_adapter, sac_adapter)
            results.append(res)

        logger.info(f"Completed all {len(results)} paired MPPT comparative scenarios.")
        return results
