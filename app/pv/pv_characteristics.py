"""Photovoltaic Characteristics Analysis and Maximum Power Point Identification.

Provides functionality to generate I-V and P-V characteristic curves,
identify discrete and analytical Maximum Power Points (MPP), and analyze
environmental variations (irradiance and temperature).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from app.pv.pv_model import PVOutput, SolarPVModel
from app.utils.logger import get_logger

logger = get_logger("pv_characteristics")


@dataclass(frozen=True)
class PVCharacteristicCurve:
    """Container holding generated I-V and P-V curve data and MPP points.

    Attributes
    ----------
    voltage : np.ndarray
        Voltage array from short-circuit to open-circuit [V].
    current : np.ndarray
        Terminal output current array [A].
    power : np.ndarray
        Delivered electrical power array (P = V * I) [W].
    irradiance : float
        Operating irradiance condition [W/m^2].
    temperature : float
        Operating cell junction temperature [°C].
    pv_output : PVOutput
        Analytical operating points and diode parameters from single-diode model.
    v_mp_num : float
        Discrete Maximum Power Point voltage from numerical sweep [V].
    i_mp_num : float
        Discrete Maximum Power Point current from numerical sweep [A].
    p_mp_num : float
        Discrete Maximum Power Point power from numerical sweep [W].
    mpp_index : int
        Array index where power is maximized.
    """

    voltage: np.ndarray
    current: np.ndarray
    power: np.ndarray
    irradiance: float
    temperature: float
    pv_output: PVOutput
    v_mp_num: float
    i_mp_num: float
    p_mp_num: float
    mpp_index: int

    @property
    def v_oc_analytical(self) -> float:
        """Analytical open-circuit voltage [V]."""
        return self.pv_output.v_oc

    @property
    def i_sc_analytical(self) -> float:
        """Analytical short-circuit current [A]."""
        return self.pv_output.i_sc

    @property
    def v_mp_analytical(self) -> float:
        """Analytical MPP voltage [V]."""
        return self.pv_output.v_mp

    @property
    def i_mp_analytical(self) -> float:
        """Analytical MPP current [A]."""
        return self.pv_output.i_mp

    @property
    def p_mp_analytical(self) -> float:
        """Analytical maximum power [W]."""
        return self.pv_output.p_mp

    @property
    def fill_factor(self) -> float:
        """Analytical Fill Factor (FF = Pmp / (Voc * Isc))."""
        denominator = self.pv_output.v_oc * self.pv_output.i_sc
        return float(self.pv_output.p_mp / denominator) if denominator > 0.0 else 0.0

    @property
    def p_mp_error_pct(self) -> float:
        """Percentage error between discrete numerical MPP and analytical MPP [%]."""
        ref = self.pv_output.p_mp
        return float(abs(self.p_mp_num - ref) / ref * 100.0) if ref > 0.0 else 0.0

    @property
    def v_mp_error_pct(self) -> float:
        """Percentage error between discrete numerical Vmp and analytical Vmp [%]."""
        ref = self.pv_output.v_mp
        return float(abs(self.v_mp_num - ref) / ref * 100.0) if ref > 0.0 else 0.0

    @property
    def i_mp_error_pct(self) -> float:
        """Percentage error between discrete numerical Imp and analytical Imp [%]."""
        ref = self.pv_output.i_mp
        return float(abs(self.i_mp_num - ref) / ref * 100.0) if ref > 0.0 else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Export curve voltage, current, and power trajectories to a DataFrame."""
        return pd.DataFrame(
            {
                "voltage": self.voltage,
                "current": self.current,
                "power": self.power,
            }
        )


class PVCharacteristicAnalyzer:
    """Analyzer for generating and comparing photovoltaic I-V and P-V characteristics.

    Wraps the physics-based `SolarPVModel` to provide sweep-based curve synthesis,
    MPP identification, and comparative parametric sweeps across irradiance and temperature.
    """

    def __init__(self, model: Optional[SolarPVModel] = None) -> None:
        """Initialize the analyzer with a SolarPVModel.

        Parameters
        ----------
        model : Optional[SolarPVModel], optional
            Instance of SolarPVModel. If None, default model instance is created.
        """
        self.model = model if model is not None else SolarPVModel()
        logger.debug(
            f"Initialized PVCharacteristicAnalyzer with module: "
            f"{self.model.manufacturer} {self.model.model_name}"
        )

    def generate_curve(
        self,
        irradiance: float,
        cell_temperature: float,
        num_points: int = 500,
        v_max_ratio: float = 1.0,
    ) -> PVCharacteristicCurve:
        """Generate I-V and P-V characteristic curve over the operational voltage range.

        Parameters
        ----------
        irradiance : float
            Effective solar irradiance [W/m^2].
        cell_temperature : float
            Operating cell temperature [°C].
        num_points : int, optional
            Number of discrete voltage evaluation points, by default 500.
        v_max_ratio : float, optional
            Ratio of maximum sweep voltage to Voc (e.g. 1.0 sweeps from 0 to Voc), by default 1.0.

        Returns
        -------
        PVCharacteristicCurve
            Structured curve data containing voltage, current, power arrays,
            and both analytical and discrete numerical MPP points.

        Raises
        ------
        ValueError
            If num_points < 2 or v_max_ratio <= 0.
        """
        if num_points < 2:
            raise ValueError(f"num_points must be >= 2, received {num_points}.")
        if v_max_ratio <= 0.0:
            raise ValueError(f"v_max_ratio must be positive, received {v_max_ratio}.")

        # Compute analytical baseline points and single-diode parameters
        analytical_output = self.model.calculate(irradiance, cell_temperature)

        # Handle zero irradiance boundary (complete darkness)
        if float(irradiance) == 0.0:
            v_arr = np.zeros(num_points, dtype=float)
            i_arr = np.zeros(num_points, dtype=float)
            p_arr = np.zeros(num_points, dtype=float)
            return PVCharacteristicCurve(
                voltage=v_arr,
                current=i_arr,
                power=p_arr,
                irradiance=0.0,
                temperature=float(cell_temperature),
                pv_output=analytical_output,
                v_mp_num=0.0,
                i_mp_num=0.0,
                p_mp_num=0.0,
                mpp_index=0,
            )

        v_upper = analytical_output.v_oc * v_max_ratio
        voltages = np.linspace(0.0, v_upper, num_points, dtype=float)

        # Evaluate terminal currents across the voltage sweep
        currents = self.model.current_at_voltage(
            voltage=voltages,
            irradiance=irradiance,
            cell_temperature=cell_temperature,
            clip_non_negative=True,
        )
        currents = np.asarray(currents, dtype=float)

        # Calculate electrical power: P = V * I
        powers = voltages * currents

        # Identify discrete Maximum Power Point
        mpp_idx = int(np.argmax(powers))
        v_mp_num = float(voltages[mpp_idx])
        i_mp_num = float(currents[mpp_idx])
        p_mp_num = float(powers[mpp_idx])

        return PVCharacteristicCurve(
            voltage=voltages,
            current=currents,
            power=powers,
            irradiance=float(irradiance),
            temperature=float(cell_temperature),
            pv_output=analytical_output,
            v_mp_num=v_mp_num,
            i_mp_num=i_mp_num,
            p_mp_num=p_mp_num,
            mpp_index=mpp_idx,
        )

    def analyze_irradiance_variation(
        self,
        irradiances: Sequence[float],
        temperature: float = 25.0,
        num_points: int = 500,
    ) -> List[PVCharacteristicCurve]:
        """Generate curves across multiple irradiance levels at constant cell temperature.

        Parameters
        ----------
        irradiances : Sequence[float]
            List or array of irradiance levels [W/m^2].
        temperature : float, optional
            Constant cell junction temperature [°C], by default 25.0.
        num_points : int, optional
            Number of points per curve, by default 500.

        Returns
        -------
        List[PVCharacteristicCurve]
            List of generated curves for each irradiance level.
        """
        logger.info(
            f"Analyzing irradiance variation for {len(irradiances)} levels at T = {temperature}°C..."
        )
        return [
            self.generate_curve(g, temperature, num_points=num_points)
            for g in irradiances
        ]

    def analyze_temperature_variation(
        self,
        temperatures: Sequence[float],
        irradiance: float = 1000.0,
        num_points: int = 500,
    ) -> List[PVCharacteristicCurve]:
        """Generate curves across multiple cell temperatures at constant irradiance.

        Parameters
        ----------
        temperatures : Sequence[float]
            List or array of cell temperatures [°C].
        irradiance : float, optional
            Constant solar irradiance [W/m^2], by default 1000.0.
        num_points : int, optional
            Number of points per curve, by default 500.

        Returns
        -------
        List[PVCharacteristicCurve]
            List of generated curves for each temperature level.
        """
        logger.info(
            f"Analyzing temperature variation for {len(temperatures)} levels at G = {irradiance} W/m^2..."
        )
        return [
            self.generate_curve(irradiance, t, num_points=num_points)
            for t in temperatures
        ]

    def analyze_environmental_scenarios(
        self,
        scenarios: Sequence[Tuple[float, float]],
        num_points: int = 500,
    ) -> List[PVCharacteristicCurve]:
        """Generate curves for combined (irradiance, temperature) scenarios.

        Parameters
        ----------
        scenarios : Sequence[Tuple[float, float]]
            List of tuples: (irradiance [W/m^2], cell_temperature [°C]).
        num_points : int, optional
            Number of points per curve, by default 500.

        Returns
        -------
        List[PVCharacteristicCurve]
            List of generated curves for each scenario.
        """
        logger.info(f"Analyzing {len(scenarios)} combined environmental scenarios...")
        return [
            self.generate_curve(g, t, num_points=num_points)
            for g, t in scenarios
        ]

    def curves_to_mpp_dataframe(
        self,
        curves: Sequence[PVCharacteristicCurve],
    ) -> pd.DataFrame:
        """Convert a collection of characteristic curves to a comprehensive MPP comparison DataFrame.

        Parameters
        ----------
        curves : Sequence[PVCharacteristicCurve]
            Collection of evaluated curves.

        Returns
        -------
        pd.DataFrame
            Formatted DataFrame containing environmental parameters, analytical benchmark values,
            numerical sweep values, fill factor, and discrete-sweep percentage errors.
        """
        records = []
        for c in curves:
            records.append(
                {
                    "irradiance": c.irradiance,
                    "cell_temperature": c.temperature,
                    "i_sc": c.i_sc_analytical,
                    "v_oc": c.v_oc_analytical,
                    "v_mp": c.v_mp_analytical,
                    "i_mp": c.i_mp_analytical,
                    "p_mp": c.p_mp_analytical,
                    "fill_factor": c.fill_factor,
                    "v_mp_numerical": c.v_mp_num,
                    "i_mp_numerical": c.i_mp_num,
                    "p_mp_numerical": c.p_mp_num,
                    "p_mp_error_pct": c.p_mp_error_pct,
                    "v_mp_error_pct": c.v_mp_error_pct,
                    "i_mp_error_pct": c.i_mp_error_pct,
                }
            )
        return pd.DataFrame(records)
