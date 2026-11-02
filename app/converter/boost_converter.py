"""Dynamic Averaged Model of DC-DC Boost Converter for Solar PV Systems.

Implements the continuous-conduction-mode (CCM) state-space averaged differential
equations coupled with the physics-based SolarPVModel generator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from app.pv.pv_model import SolarPVModel
from app.utils.logger import get_logger
from config.default_config import ConverterConfig, get_default_config

logger = get_logger("boost_converter")


@dataclass(frozen=True)
class ConverterMeasurements:
    """Instantaneous electrical measurements and energy metrics of the boost converter.

    Attributes
    ----------
    time : float
        Simulation timestamp [s].
    v_pv : float
        Instantaneous PV terminal voltage / input capacitor voltage [V].
    i_pv : float
        Current delivered by PV module at voltage v_pv [A].
    p_pv : float
        Electrical power supplied by PV module (v_pv * i_pv) [W].
    i_l : float
        Instantaneous inductor current [A].
    v_out : float
        Instantaneous load terminal / output capacitor voltage [V].
    i_out : float
        Current delivered to the load (v_out / R_load) [A].
    p_out : float
        Electrical power absorbed by the load (v_out * i_out) [W].
    duty_cycle : float
        Converter PWM duty cycle D in [D_min, D_max].
    efficiency : float
        Instantaneous power transfer ratio (P_out / P_pv * 100) [%].
    is_ccm : bool
        True if inductor current is positive (Continuous Conduction Mode).
    stored_energy : float
        Total stored electromagnetic energy (0.5*Cpv*Vpv^2 + 0.5*L*iL^2 + 0.5*Cout*Vout^2) [J].
    power_loss : float
        Ohmic power dissipation in inductor series resistance (r_L * i_L^2) [W].
    dE_dt : float
        Rate of change of total reactive stored energy [W].
    power_balance_residual : float
        Instantaneous power conservation residual |P_in - P_out - P_loss - dE/dt| [W].
    """

    time: float
    v_pv: float
    i_pv: float
    p_pv: float
    i_l: float
    v_out: float
    i_out: float
    p_out: float
    duty_cycle: float
    efficiency: float
    is_ccm: bool
    stored_energy: float
    power_loss: float
    dE_dt: float
    power_balance_residual: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert measurements dataclass to a dictionary."""
        return asdict(self)


class BoostConverter:
    """State-Space Averaged DC-DC Boost Converter Plant Model.

    Governing Differential Equations (Continuous Conduction Mode):
        C_pv  * dV_pv/dt  = I_pv(V_pv, G, T) - i_L
        L     * di_L/dt   = V_pv - (1 - D) * V_out - r_L * i_L
        C_out * dV_out/dt = (1 - D) * i_L - V_out / R_load
    """

    def __init__(
        self,
        pv_model: Optional[SolarPVModel] = None,
        config: Optional[ConverterConfig] = None,
    ) -> None:
        """Initialize the Boost Converter with PV model and converter parameters.

        Parameters
        ----------
        pv_model : Optional[SolarPVModel], optional
            Instance of SolarPVModel. If None, default instance is created.
        config : Optional[ConverterConfig], optional
            Converter configuration dataclass. If None, loaded from project defaults.

        Raises
        ------
        ValueError
            If physical component parameters are non-positive or duty cycle bounds are invalid.
        """
        self.pv_model = pv_model if pv_model is not None else SolarPVModel()
        self.config = config if config is not None else get_default_config().converter

        # Extract and validate component ratings
        self.L = float(self.config.inductance)
        self.C_pv = float(self.config.input_capacitance)
        self.C_out = float(self.config.output_capacitance)
        self.R_load = float(self.config.load_resistance)
        self.r_L = float(self.config.inductor_resistance)
        self.f_sw = float(self.config.switching_frequency)
        self.d_min = float(self.config.duty_cycle_min)
        self.d_max = float(self.config.duty_cycle_max)
        self.default_dt = float(self.config.time_step)

        self._validate_component_parameters()

        # State vector: [V_pv, i_L, V_out]
        self.time: float = 0.0
        self.state: np.ndarray = np.array(self.config.get_initial_state(), dtype=float)
        self._last_energy: float = self._calculate_stored_energy(self.state)
        self._last_duty: float = float(self.config.duty_cycle_init)

        logger.debug(
            f"Initialized BoostConverter: L={self.L*1e3:.2f}mH, Cpv={self.C_pv*1e6:.1f}uF, "
            f"Cout={self.C_out*1e6:.1f}uF, Rload={self.R_load:.1f}Ω, rL={self.r_L*1e3:.1f}mΩ"
        )

    def _validate_component_parameters(self) -> None:
        """Verify all physical component parameters are positive and bounds are valid."""
        if self.L <= 0.0:
            raise ValueError(f"Inductance L must be strictly positive. Received: {self.L} H")
        if self.C_pv <= 0.0:
            raise ValueError(f"Input capacitance C_pv must be strictly positive. Received: {self.C_pv} F")
        if self.C_out <= 0.0:
            raise ValueError(f"Output capacitance C_out must be strictly positive. Received: {self.C_out} F")
        if self.R_load <= 0.0:
            raise ValueError(f"Load resistance R_load must be strictly positive. Received: {self.R_load} Ohm")
        if self.r_L < 0.0:
            raise ValueError(f"Inductor series resistance r_L cannot be negative. Received: {self.r_L} Ohm")
        if not (0.0 < self.d_min < self.d_max < 1.0):
            raise ValueError(
                f"Duty cycle bounds must satisfy 0 < d_min < d_max < 1. Received: [{self.d_min}, {self.d_max}]"
            )

    def reset(
        self,
        initial_state: Optional[Tuple[float, float, float]] = None,
    ) -> ConverterMeasurements:
        """Reset the converter state vector and simulation time.

        Parameters
        ----------
        initial_state : Optional[Tuple[float, float, float]], optional
            Initial state vector (V_pv, i_L, V_out). If None, defaults to config initial state.

        Returns
        -------
        ConverterMeasurements
            Measurements evaluated at the reset state.
        """
        self.time = 0.0
        if initial_state is not None:
            if len(initial_state) != 3:
                raise ValueError(f"initial_state must have 3 elements (V_pv, i_L, V_out). Received {initial_state}")
            self.state = np.array(initial_state, dtype=float)
        else:
            self.state = np.array(self.config.get_initial_state(), dtype=float)

        # Enforce non-negative physical bounds on reset
        self.state[0] = max(0.0, self.state[0])
        self.state[1] = max(0.0, self.state[1])
        self.state[2] = max(0.0, self.state[2])

        self._last_energy = self._calculate_stored_energy(self.state)
        self._last_duty = float(self.config.duty_cycle_init)

        logger.debug(f"Reset BoostConverter state to: Vpv={self.state[0]:.2f}V, iL={self.state[1]:.2f}A, Vout={self.state[2]:.2f}V")
        return self.get_measurements(irradiance=1000.0, cell_temperature=25.0)

    def _calculate_stored_energy(self, state: np.ndarray) -> float:
        """Calculate total reactive stored energy E = 0.5*Cpv*Vpv^2 + 0.5*L*iL^2 + 0.5*Cout*Vout^2 [J]."""
        v_pv, i_l, v_out = state
        return 0.5 * self.C_pv * (v_pv ** 2) + 0.5 * self.L * (i_l ** 2) + 0.5 * self.C_out * (v_out ** 2)

    def compute_derivatives(
        self,
        state: np.ndarray,
        duty_cycle: float,
        irradiance: float,
        cell_temperature: float,
    ) -> np.ndarray:
        """Compute state derivative vector d[V_pv, i_L, V_out]/dt.

        Parameters
        ----------
        state : np.ndarray
            Current state vector [V_pv, i_L, V_out].
        duty_cycle : float
            Effective duty cycle D in [d_min, d_max].
        irradiance : float
            Solar irradiance [W/m^2].
        cell_temperature : float
            Cell junction temperature [°C].

        Returns
        -------
        np.ndarray
            Derivative vector [dV_pv/dt, di_L/dt, dV_out/dt].
        """
        v_pv, i_l, v_out = state

        # Query instantaneous PV current from actual physical model
        i_pv = self.pv_model.current_at_voltage(v_pv, irradiance, cell_temperature, clip_non_negative=True)

        # Governing Averaged Differential Equations
        dv_pv = (i_pv - i_l) / self.C_pv
        di_l = (v_pv - (1.0 - duty_cycle) * v_out - self.r_L * i_l) / self.L
        dv_out = ((1.0 - duty_cycle) * i_l - v_out / self.R_load) / self.C_out

        # Discontinuous Conduction Mode (DCM) boundary protection:
        # If inductor current is zero and derivative is negative, inductor current cannot go negative.
        if i_l <= 0.0 and di_l < 0.0:
            di_l = 0.0

        return np.array([dv_pv, di_l, dv_out], dtype=float)

    def step(
        self,
        duty_cycle: float,
        irradiance: float = 1000.0,
        cell_temperature: float = 25.0,
        dt: Optional[float] = None,
    ) -> ConverterMeasurements:
        """Execute one simulation integration step using 4th-order Runge-Kutta (RK4).

        Parameters
        ----------
        duty_cycle : float
            Actuation duty cycle D. Must satisfy d_min <= D <= d_max.
        irradiance : float, optional
            Incident irradiance [W/m^2], by default 1000.0.
        cell_temperature : float, optional
            Cell junction temperature [°C], by default 25.0.
        dt : Optional[float], optional
            Integration step size [s]. If None, uses default_dt.

        Returns
        -------
        ConverterMeasurements
            Calculated electrical measurements after the step.

        Raises
        ------
        ValueError
            If duty cycle is outside [d_min, d_max] or if state becomes non-finite.
        TypeError
            If inputs are non-numeric.
        """
        if not isinstance(duty_cycle, (int, float, np.number)) or not math.isfinite(float(duty_cycle)):
            raise TypeError(f"Duty cycle must be a finite numeric value. Received: {duty_cycle}")

        d = float(duty_cycle)
        if d < self.d_min or d > self.d_max:
            raise ValueError(
                f"Duty cycle {d:.4f} is outside operational limits [{self.d_min:.4f}, {self.d_max:.4f}]."
            )

        step_dt = float(dt) if dt is not None else self.default_dt
        if step_dt <= 0.0:
            raise ValueError(f"Simulation time step dt must be positive. Received: {step_dt}")

        g = float(irradiance)
        t_cell = float(cell_temperature)

        # 4th-Order Runge-Kutta (RK4) Integration
        s = self.state
        k1 = self.compute_derivatives(s, d, g, t_cell)
        k2 = self.compute_derivatives(s + 0.5 * step_dt * k1, d, g, t_cell)
        k3 = self.compute_derivatives(s + 0.5 * step_dt * k2, d, g, t_cell)
        k4 = self.compute_derivatives(s + step_dt * k3, d, g, t_cell)

        next_state = s + (step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Enforce physical positivity
        next_state[0] = max(0.0, float(next_state[0]))
        next_state[1] = max(0.0, float(next_state[1]))
        next_state[2] = max(0.0, float(next_state[2]))

        if not np.all(np.isfinite(next_state)):
            raise ValueError(f"Numerical divergence detected: state vector={next_state}")

        # Compute energy dynamics
        current_energy = self._calculate_stored_energy(next_state)
        dE_dt = (current_energy - self._last_energy) / step_dt

        # Commit state update
        self.state = next_state
        self.time += step_dt
        self._last_energy = current_energy
        self._last_duty = d

        return self.get_measurements(g, t_cell, dE_dt=dE_dt)

    def get_state(self) -> Tuple[float, float, float]:
        """Return the current physical state vector (V_pv, i_L, V_out)."""
        return (float(self.state[0]), float(self.state[1]), float(self.state[2]))

    def get_measurements(
        self,
        irradiance: float = 1000.0,
        cell_temperature: float = 25.0,
        dE_dt: float = 0.0,
    ) -> ConverterMeasurements:
        """Calculate and return full instantaneous electrical measurements.

        Parameters
        ----------
        irradiance : float, optional
            Current solar irradiance [W/m^2], by default 1000.0.
        cell_temperature : float, optional
            Current cell temperature [°C], by default 25.0.
        dE_dt : float, optional
            Rate of energy change in reactive storage elements [W], by default 0.0.

        Returns
        -------
        ConverterMeasurements
            Measurements dataclass.
        """
        v_pv, i_l, v_out = self.state

        # Instantaneous PV electrical points
        i_pv = self.pv_model.current_at_voltage(v_pv, irradiance, cell_temperature, clip_non_negative=True)
        p_pv = v_pv * i_pv

        # Output side measurements
        i_out = v_out / self.R_load
        p_out = v_out * i_out

        # Inductor series loss
        p_loss = self.r_L * (i_l ** 2)

        # Power conversion efficiency
        efficiency = (p_out / p_pv * 100.0) if p_pv > 1e-4 else 0.0
        efficiency = min(100.0, max(0.0, efficiency))

        # CCM status
        is_ccm = bool(i_l > 1e-3)
        stored_energy = self._calculate_stored_energy(self.state)

        # Power conservation residual: Pin - Pout - Ploss - dE/dt
        residual = abs(p_pv - p_out - p_loss - dE_dt)

        return ConverterMeasurements(
            time=self.time,
            v_pv=float(v_pv),
            i_pv=float(i_pv),
            p_pv=float(p_pv),
            i_l=float(i_l),
            v_out=float(v_out),
            i_out=float(i_out),
            p_out=float(p_out),
            duty_cycle=self._last_duty,
            efficiency=float(efficiency),
            is_ccm=is_ccm,
            stored_energy=float(stored_energy),
            power_loss=float(p_loss),
            dE_dt=float(dE_dt),
            power_balance_residual=float(residual),
        )

    def simulate(
        self,
        duration: float,
        duty_cycle: Union[float, Callable[[float], float], Sequence[float]],
        irradiance: Union[float, Callable[[float], float]] = 1000.0,
        cell_temperature: Union[float, Callable[[float], float]] = 25.0,
        dt: Optional[float] = None,
    ) -> pd.DataFrame:
        """Simulate the converter dynamics over a continuous time window.

        Parameters
        ----------
        duration : float
            Total simulation time [seconds].
        duty_cycle : Union[float, Callable[[float], float], Sequence[float]]
            Duty cycle profile. Can be a constant scalar, a time-dependent callable D(t),
            or a sequence of duty values matching total steps.
        irradiance : Union[float, Callable[[float], float]], optional
            Irradiance condition [W/m^2] or function G(t), by default 1000.0.
        cell_temperature : Union[float, Callable[[float], float]], optional
            Cell temperature condition [°C] or function T(t), by default 25.0.
        dt : Optional[float], optional
            Simulation time step [seconds]. If None, uses default_dt.

        Returns
        -------
        pd.DataFrame
            DataFrame containing complete timeseries measurements.
        """
        step_dt = float(dt) if dt is not None else self.default_dt
        num_steps = int(math.ceil(duration / step_dt))

        records: List[Dict[str, Any]] = []

        # Record initial measurement at t=0
        initial_g = irradiance(self.time) if callable(irradiance) else float(irradiance)
        initial_t = cell_temperature(self.time) if callable(cell_temperature) else float(cell_temperature)
        records.append(self.get_measurements(initial_g, initial_t).to_dict())

        for step_idx in range(num_steps):
            current_time = self.time

            # Resolve instantaneous duty cycle
            if callable(duty_cycle):
                d_val = float(duty_cycle(current_time))
            elif isinstance(duty_cycle, (list, tuple, np.ndarray)):
                d_val = float(duty_cycle[min(step_idx, len(duty_cycle) - 1)])
            else:
                d_val = float(duty_cycle)

            # Resolve instantaneous environmental inputs
            g_val = float(irradiance(current_time)) if callable(irradiance) else float(irradiance)
            t_val = float(cell_temperature(current_time)) if callable(cell_temperature) else float(cell_temperature)

            meas = self.step(duty_cycle=d_val, irradiance=g_val, cell_temperature=t_val, dt=step_dt)
            records.append(meas.to_dict())

        return pd.DataFrame(records)
