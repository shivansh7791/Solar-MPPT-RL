"""Classical Perturb & Observe (P&O) MPPT Controller for Solar Photovoltaic Systems.

Implements the standard hill-climbing Perturb and Observe maximum power point tracking
algorithm tailored for a DC-DC boost converter with inverse voltage-duty coupling:
    Increasing Duty Cycle (D)  --> Decreases PV Terminal Voltage (V_pv)
    Decreasing Duty Cycle (D)  --> Increases PV Terminal Voltage (V_pv)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional, Union

import numpy as np

from app.utils.logger import get_logger
from config.default_config import POConfig, get_default_config

logger = get_logger("po_controller")


@dataclass(frozen=True)
class PODiagnostics:
    """Diagnostic state of the Perturb & Observe controller at a sampling instant.

    Attributes
    ----------
    duty_cycle : float
        Current commanded converter duty cycle D in [min_duty, max_duty].
    curr_voltage : float
        Measured PV terminal voltage at current sample instant [V].
    curr_current : float
        Measured PV array current at current sample instant [A].
    curr_power : float
        Calculated instantaneous PV electrical power (V * I) [W].
    prev_voltage : Optional[float]
        PV terminal voltage recorded at previous sampling instant [V].
    prev_power : Optional[float]
        PV electrical power recorded at previous sampling instant [W].
    delta_voltage : float
        Change in PV terminal voltage (curr_voltage - prev_voltage) [V].
    delta_power : float
        Change in PV electrical power (curr_power - prev_power) [W].
    duty_adjustment : float
        Duty cycle perturbation delta_D applied at this update step.
    perturbation_direction : int
        Direction indicator of duty change: +1 (increase D), -1 (decrease D), 0 (none/deadband).
    is_saturated : bool
        True if the raw calculated duty cycle hit min_duty or max_duty boundary.
    update_count : int
        Total number of control update steps executed since initialization or reset.
    """

    duty_cycle: float
    curr_voltage: float
    curr_current: float
    curr_power: float
    prev_voltage: Optional[float]
    prev_power: Optional[float]
    delta_voltage: float
    delta_power: float
    duty_adjustment: float
    perturbation_direction: int
    is_saturated: bool
    update_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic dataclass to dictionary representation."""
        return asdict(self)


class PerturbAndObserveController:
    """Classical Perturb & Observe (P&O) Maximum Power Point Tracking (MPPT) Controller.

    Governing Control Law for Boost Converter:
    -----------------------------------------
        P(k) = V(k) * I(k)
        Delta V = V(k) - V(k-1)
        Delta P = P(k) - P(k-1)

        If |Delta P| <= power_deadband or |Delta V| <= voltage_deadband:
            Delta D = 0  (hold operating point)
        Else:
            If (Delta P * Delta V) > 0:
                Delta D = -perturbation_step  (decrease D to increase V toward MPP)
            Else:
                Delta D = +perturbation_step  (increase D to decrease V toward MPP)

        D(k) = clip(D(k-1) + Delta D, min_duty, max_duty)
    """

    def __init__(
        self,
        initial_duty: Optional[float] = None,
        perturbation_step: Optional[float] = None,
        min_duty: Optional[float] = None,
        max_duty: Optional[float] = None,
        power_deadband: Optional[float] = None,
        voltage_deadband: Optional[float] = None,
        config: Optional[POConfig] = None,
    ) -> None:
        """Initialize the Perturb & Observe MPPT Controller.

        Parameters
        ----------
        initial_duty : Optional[float], optional
            Starting duty cycle command. If None, defaults to config.
        perturbation_step : Optional[float], optional
            Fixed duty cycle perturbation increment delta_D. If None, defaults to config.
        min_duty : Optional[float], optional
            Minimum permitted duty cycle saturation bound. If None, defaults to config.
        max_duty : Optional[float], optional
            Maximum permitted duty cycle saturation bound. If None, defaults to config.
        power_deadband : Optional[float], optional
            Minimum absolute power change |Delta P| to trigger perturbation [W].
        voltage_deadband : Optional[float], optional
            Minimum absolute voltage change |Delta V| to trigger perturbation [V].
        config : Optional[POConfig], optional
            Configuration dataclass instance. If None, loaded from project defaults.

        Raises
        ------
        ValueError
            If configuration parameters violate physical or operational constraints.
        """
        cfg = config if config is not None else get_default_config().mppt.po

        self.initial_duty = float(initial_duty if initial_duty is not None else cfg.initial_duty)
        self.perturbation_step = float(perturbation_step if perturbation_step is not None else cfg.perturbation_step)
        self.min_duty = float(min_duty if min_duty is not None else cfg.duty_min)
        self.max_duty = float(max_duty if max_duty is not None else cfg.duty_max)
        self.power_deadband = float(power_deadband if power_deadband is not None else cfg.power_deadband)
        self.voltage_deadband = float(voltage_deadband if voltage_deadband is not None else cfg.voltage_deadband)

        self._validate_parameters()

        # Controller state variables
        self.duty_cycle: float = self.initial_duty
        self.prev_voltage: Optional[float] = None
        self.prev_power: Optional[float] = None
        self.last_prev_voltage: Optional[float] = None
        self.last_prev_power: Optional[float] = None
        self.curr_voltage: float = 0.0
        self.curr_current: float = 0.0
        self.curr_power: float = 0.0
        self.delta_voltage: float = 0.0
        self.delta_power: float = 0.0
        self.duty_adjustment: float = 0.0
        self.perturbation_direction: int = 0
        self.is_saturated: bool = False
        self.update_count: int = 0

        logger.debug(
            f"Initialized PerturbAndObserveController: D0={self.initial_duty:.4f}, "
            f"delta_D={self.perturbation_step:.4f}, limits=[{self.min_duty:.2f}, {self.max_duty:.2f}]"
        )

    def _validate_parameters(self) -> None:
        """Validate controller parameters against physical bounds."""
        if not (0.0 < self.min_duty < self.max_duty < 1.0):
            raise ValueError(
                f"Duty limits must satisfy 0 < min_duty < max_duty < 1. Received: [{self.min_duty}, {self.max_duty}]"
            )
        if not (self.min_duty <= self.initial_duty <= self.max_duty):
            raise ValueError(
                f"Initial duty {self.initial_duty} must be within bounds [{self.min_duty}, {self.max_duty}]"
            )
        if self.perturbation_step <= 0.0:
            raise ValueError(
                f"Perturbation step must be strictly positive. Received: {self.perturbation_step}"
            )
        if self.power_deadband < 0.0:
            raise ValueError(
                f"Power deadband cannot be negative. Received: {self.power_deadband}"
            )
        if self.voltage_deadband < 0.0:
            raise ValueError(
                f"Voltage deadband cannot be negative. Received: {self.voltage_deadband}"
            )

    def reset(self, initial_duty: Optional[float] = None) -> float:
        """Reset the controller state and memory.

        Parameters
        ----------
        initial_duty : Optional[float], optional
            Starting duty cycle. If None, resets to the configured initial_duty.

        Returns
        -------
        float
            Reset duty cycle command.
        """
        if initial_duty is not None:
            if not isinstance(initial_duty, (int, float, np.number)) or not math.isfinite(float(initial_duty)):
                raise TypeError(f"initial_duty must be a finite numeric value. Received: {initial_duty}")
            d0 = float(initial_duty)
            if not (self.min_duty <= d0 <= self.max_duty):
                raise ValueError(f"initial_duty {d0} outside limits [{self.min_duty}, {self.max_duty}]")
            self.duty_cycle = d0
        else:
            self.duty_cycle = self.initial_duty

        self.prev_voltage = None
        self.prev_power = None
        self.last_prev_voltage = None
        self.last_prev_power = None
        self.curr_voltage = 0.0
        self.curr_current = 0.0
        self.curr_power = 0.0
        self.delta_voltage = 0.0
        self.delta_power = 0.0
        self.duty_adjustment = 0.0
        self.perturbation_direction = 0
        self.is_saturated = False
        self.update_count = 0

        logger.debug(f"Reset PerturbAndObserveController: duty_cycle={self.duty_cycle:.4f}")
        return self.duty_cycle

    def update(self, voltage: float, current: float) -> float:
        """Execute one P&O MPPT control step and return the new commanded duty cycle.

        Parameters
        ----------
        voltage : float
            Measured PV terminal voltage V [V]. Must be non-negative and finite.
        current : float
            Measured PV terminal current I [A]. Must be non-negative and finite.

        Returns
        -------
        float
            Updated commanded duty cycle D in [min_duty, max_duty].

        Raises
        ------
        TypeError
            If voltage or current is not a finite numeric value.
        ValueError
            If voltage or current is negative.
        """
        # 1. Input Validation
        if not isinstance(voltage, (int, float, np.number)) or not math.isfinite(float(voltage)):
            raise TypeError(f"Voltage must be a finite numeric value. Received: {voltage}")
        if not isinstance(current, (int, float, np.number)) or not math.isfinite(float(current)):
            raise TypeError(f"Current must be a finite numeric value. Received: {current}")

        v = float(voltage)
        i = float(current)

        if v < 0.0:
            raise ValueError(f"Measured PV voltage cannot be negative. Received: {v} V")
        if i < 0.0:
            raise ValueError(f"Measured PV current cannot be negative. Received: {i} A")

        p = v * i
        self.curr_voltage = v
        self.curr_current = i
        self.curr_power = p

        # 2. First measurement handling
        # At step 0, no prior (V, P) exists. Initialize state and return initial duty cycle without perturbing.
        if self.prev_voltage is None or self.prev_power is None:
            self.last_prev_voltage = None
            self.last_prev_power = None
            self.prev_voltage = v
            self.prev_power = p
            self.delta_voltage = 0.0
            self.delta_power = 0.0
            self.duty_adjustment = 0.0
            self.perturbation_direction = 0
            self.is_saturated = False
            self.update_count = 1
            logger.debug(f"P&O Step 1 Initialized: V={v:.2f}V, I={i:.2f}A, P={p:.2f}W, D={self.duty_cycle:.4f}")
            return self.duty_cycle

        # 3. Calculate finite differences
        self.last_prev_voltage = self.prev_voltage
        self.last_prev_power = self.prev_power
        delta_v = v - self.prev_voltage
        delta_p = p - self.prev_power
        self.delta_voltage = delta_v
        self.delta_power = delta_p

        # 4. Check deadbands
        if abs(delta_p) <= self.power_deadband or abs(delta_v) <= self.voltage_deadband:
            delta_d = 0.0
            direction = 0
        else:
            # 5. Boost Converter P&O Control Decision
            # D increase --> V decrease; D decrease --> V increase
            # If Delta P * Delta V > 0 (Left of MPP): Need V increase --> Decrease D (delta_d = -step)
            # If Delta P * Delta V < 0 (Right of MPP): Need V decrease --> Increase D (delta_d = +step)
            if (delta_p * delta_v) > 0.0:
                delta_d = -self.perturbation_step
                direction = -1
            else:
                delta_d = +self.perturbation_step
                direction = +1

        self.duty_adjustment = delta_d
        self.perturbation_direction = direction

        # 6. Apply duty cycle update with saturation
        raw_duty = self.duty_cycle + delta_d
        clamped_duty = max(self.min_duty, min(self.max_duty, raw_duty))

        self.is_saturated = bool(raw_duty < self.min_duty or raw_duty > self.max_duty)
        self.duty_cycle = clamped_duty

        # 7. Update history
        self.prev_voltage = v
        self.prev_power = p
        self.update_count += 1

        return self.duty_cycle

    def get_state(self) -> PODiagnostics:
        """Return the current diagnostic state of the controller."""
        return PODiagnostics(
            duty_cycle=float(self.duty_cycle),
            curr_voltage=float(self.curr_voltage),
            curr_current=float(self.curr_current),
            curr_power=float(self.curr_power),
            prev_voltage=float(self.last_prev_voltage) if self.last_prev_voltage is not None else None,
            prev_power=float(self.last_prev_power) if self.last_prev_power is not None else None,
            delta_voltage=float(self.delta_voltage),
            delta_power=float(self.delta_power),
            duty_adjustment=float(self.duty_adjustment),
            perturbation_direction=int(self.perturbation_direction),
            is_saturated=bool(self.is_saturated),
            update_count=int(self.update_count),
        )
