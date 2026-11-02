"""Physics-Based Solar Photovoltaic Single-Diode Model.

Implements the single-diode 5-parameter equivalent circuit model for a photovoltaic
module using pvlib's De Soto / CEC formulation and analytical Lambert W evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import pvlib

from app.utils.logger import get_logger
from config.default_config import PVConfig, get_default_config

logger = get_logger("pv_model")

# Physical terrestrial operational limits
MIN_SUPPORTED_TEMP_C = -50.0  # IEC 61215 lower limit is -40°C
MAX_SUPPORTED_TEMP_C = 125.0  # Upper thermal stress threshold
MAX_SUPPORTED_IRRADIANCE = 2500.0  # Extreme solar concentrator threshold [W/m^2]

REQUIRED_CEC_PARAMS = (
    "alpha_sc",
    "a_ref",
    "I_L_ref",
    "I_o_ref",
    "R_sh_ref",
    "R_s",
    "Adjust",
)


@dataclass(frozen=True)
class PVOutput:
    """Structured container holding electrical operating points of the PV module.

    Attributes
    ----------
    i_sc : float
        Short-circuit current [A]
    v_oc : float
        Open-circuit voltage [V]
    v_mp : float
        Voltage at Maximum Power Point [V]
    i_mp : float
        Current at Maximum Power Point [A]
    p_mp : float
        Power at Maximum Power Point [W]
    i_ph : float
        Light-generated photocurrent [A]
    i_o : float
        Diode reverse dark saturation current [A]
    r_s : float
        Lumped equivalent series resistance [Ohm]
    r_sh : float
        Equivalent shunt resistance [Ohm]
    a : float
        Modified diode ideality factor in voltage units (n * Ns * Vt) [V]
    irradiance : float
        Operating effective irradiance [W/m^2]
    temperature : float
        Operating cell junction temperature [°C]
    """

    i_sc: float
    v_oc: float
    v_mp: float
    i_mp: float
    p_mp: float
    i_ph: float
    i_o: float
    r_s: float
    r_sh: float
    a: float
    irradiance: float
    temperature: float

    def to_dict(self) -> Dict[str, float]:
        """Convert output dataclass to a dictionary."""
        return asdict(self)


class SolarPVModel:
    """Single-Diode Equivalent Circuit Photovoltaic Module Model.

    Utilizes pvlib's De Soto / CEC parameter adjustment (`calcparams_cec`)
    and Lambert W closed-form solver (`singlediode`) to simulate the electrical
    characteristics of a photovoltaic module under specified environmental conditions.
    """

    def __init__(
        self,
        module_parameters: Optional[Union[PVConfig, Dict[str, Any], pd.Series]] = None,
    ) -> None:
        """Initialize the SolarPVModel with module parameters.

        Parameters
        ----------
        module_parameters : Optional[Union[PVConfig, Dict[str, Any], pd.Series]], optional
            Module configuration object, dictionary, or pandas Series containing
            CEC single-diode parameters. If None, default Canadian Solar CS6P-240P
            parameters are loaded from project configuration.

        Raises
        ------
        ValueError
            If any required CEC parameter is missing or non-positive.
        TypeError
            If parameter types are invalid.
        """
        self._raw_config: Dict[str, Any] = {}
        self._cec_params: Dict[str, float] = {}
        self.manufacturer: str = "Unknown"
        self.model_name: str = "Unknown"
        self.technology: str = "Unknown"
        self.num_cells_in_series: int = 60
        self.reference_stc: Dict[str, float] = {}

        self._parse_and_validate_parameters(module_parameters)
        logger.debug(
            f"Initialized SolarPVModel for module: {self.manufacturer} {self.model_name}"
        )

    def _parse_and_validate_parameters(
        self,
        module_parameters: Optional[Union[PVConfig, Dict[str, Any], pd.Series]],
    ) -> None:
        """Parse input parameters and validate completeness and physical positivity."""
        if module_parameters is None:
            default_pv_config = get_default_config().pv
            self.manufacturer = default_pv_config.manufacturer
            self.model_name = default_pv_config.model_name
            self.technology = default_pv_config.technology
            self.num_cells_in_series = default_pv_config.num_cells_in_series
            self.reference_stc = default_pv_config.get_stc_reference_dict()
            self._cec_params = default_pv_config.get_cec_parameters()
            self._raw_config = asdict(default_pv_config)
            return

        if isinstance(module_parameters, PVConfig):
            self.manufacturer = module_parameters.manufacturer
            self.model_name = module_parameters.model_name
            self.technology = module_parameters.technology
            self.num_cells_in_series = module_parameters.num_cells_in_series
            self.reference_stc = module_parameters.get_stc_reference_dict()
            self._cec_params = module_parameters.get_cec_parameters()
            self._raw_config = asdict(module_parameters)
            return

        # Handle Dict or pd.Series
        if isinstance(module_parameters, (dict, pd.Series)):
            params_dict = dict(module_parameters)
            self._raw_config = params_dict
            self.manufacturer = str(params_dict.get("manufacturer", params_dict.get("Manufacturer", "Generic")))
            self.model_name = str(params_dict.get("model_name", params_dict.get("Name", "Generic Module")))
            self.technology = str(params_dict.get("technology", params_dict.get("Technology", "c-Si")))
            self.num_cells_in_series = int(params_dict.get("num_cells_in_series", params_dict.get("N_s", 60)))

            # Populate reference STC if available
            self.reference_stc = {
                "i_sc": float(params_dict.get("stc_i_sc", params_dict.get("I_sc_ref", 0.0))),
                "v_oc": float(params_dict.get("stc_v_oc", params_dict.get("V_oc_ref", 0.0))),
                "v_mp": float(params_dict.get("stc_v_mp", params_dict.get("V_mp_ref", 0.0))),
                "i_mp": float(params_dict.get("stc_i_mp", params_dict.get("I_mp_ref", 0.0))),
                "p_mp": float(params_dict.get("stc_p_mp", params_dict.get("STC", 0.0))),
            }

            extracted_cec: Dict[str, float] = {}
            for key in REQUIRED_CEC_PARAMS:
                if key not in params_dict:
                    raise ValueError(
                        f"Missing mandatory CEC parameter '{key}' in module parameters."
                    )
                val = params_dict[key]
                if not isinstance(val, (int, float, np.number)) or math.isnan(float(val)):
                    raise TypeError(
                        f"CEC parameter '{key}' must be a finite numerical value. Received: {val}"
                    )
                extracted_cec[key] = float(val)

            # Check physical positivity
            if extracted_cec["I_L_ref"] <= 0:
                raise ValueError("I_L_ref (reference light current) must be strictly positive.")
            if extracted_cec["I_o_ref"] <= 0:
                raise ValueError("I_o_ref (diode reverse saturation current) must be strictly positive.")
            if extracted_cec["a_ref"] <= 0:
                raise ValueError("a_ref (modified ideality factor) must be strictly positive.")
            if extracted_cec["R_sh_ref"] <= 0:
                raise ValueError("R_sh_ref (shunt resistance) must be strictly positive.")
            if extracted_cec["R_s"] < 0:
                raise ValueError("R_s (series resistance) cannot be negative.")

            extracted_cec["N_s"] = float(self.num_cells_in_series)
            self._cec_params = extracted_cec
            return

        raise TypeError(
            f"Unsupported type for module_parameters: {type(module_parameters)}. "
            "Expected PVConfig, dict, or pd.Series."
        )

    def calculate(self, irradiance: float, cell_temperature: float) -> PVOutput:
        """Calculate electrical operating characteristics under specified conditions.

        Parameters
        ----------
        irradiance : float
            Incident effective solar irradiance [W/m^2]. Must be non-negative.
        cell_temperature : float
            Operating junction temperature of the solar cells [°C].

        Returns
        -------
        PVOutput
            Structured dataclass containing short-circuit, open-circuit, and
            maximum power point current, voltage, power, and internal diode parameters.

        Raises
        ------
        TypeError
            If inputs are not numerical.
        ValueError
            If irradiance is negative or exceeds realistic terrestrial bounds,
            or if temperature falls below absolute zero or exceeds thermal limits.
        """
        # Type and NaN validation
        if not isinstance(irradiance, (int, float, np.number)) or math.isnan(float(irradiance)):
            raise TypeError(
                f"Irradiance must be a finite numerical value. Received: {irradiance}"
            )
        if not isinstance(cell_temperature, (int, float, np.number)) or math.isnan(float(cell_temperature)):
            raise TypeError(
                f"Cell temperature must be a finite numerical value. Received: {cell_temperature}"
            )

        g = float(irradiance)
        t_cell = float(cell_temperature)

        # Domain boundary validation
        if g < 0.0:
            raise ValueError(f"Irradiance cannot be negative. Received: {g} W/m^2")
        if g > MAX_SUPPORTED_IRRADIANCE:
            raise ValueError(
                f"Irradiance {g} W/m^2 exceeds upper terrestrial limit ({MAX_SUPPORTED_IRRADIANCE} W/m^2)."
            )
        if t_cell < -273.15:
            raise ValueError(
                f"Cell temperature {t_cell}°C is below absolute zero (-273.15°C)."
            )
        if t_cell < MIN_SUPPORTED_TEMP_C or t_cell > MAX_SUPPORTED_TEMP_C:
            raise ValueError(
                f"Cell temperature {t_cell}°C is outside supported operational bounds "
                f"[{MIN_SUPPORTED_TEMP_C}°C, {MAX_SUPPORTED_TEMP_C}°C]."
            )

        # Physical boundary condition: Complete darkness (Zero Irradiance)
        if g == 0.0:
            logger.debug("Operating at zero irradiance (darkness). Returning zero electrical power.")
            return PVOutput(
                i_sc=0.0,
                v_oc=0.0,
                v_mp=0.0,
                i_mp=0.0,
                p_mp=0.0,
                i_ph=0.0,
                i_o=self._cec_params["I_o_ref"],
                r_s=self._cec_params["R_s"],
                r_sh=float("inf"),
                a=self._cec_params["a_ref"] * ((t_cell + 273.15) / 298.15),
                irradiance=0.0,
                temperature=t_cell,
            )

        # Translate reference parameters to operating condition (De Soto model)
        i_l, i_o, r_s, r_sh, a_mod = pvlib.pvsystem.calcparams_cec(
            effective_irradiance=g,
            temp_cell=t_cell,
            alpha_sc=self._cec_params["alpha_sc"],
            a_ref=self._cec_params["a_ref"],
            I_L_ref=self._cec_params["I_L_ref"],
            I_o_ref=self._cec_params["I_o_ref"],
            R_sh_ref=self._cec_params["R_sh_ref"],
            R_s=self._cec_params["R_s"],
            Adjust=self._cec_params["Adjust"],
        )

        # Solve single-diode implicit equation via Lambert W function
        sd_result = pvlib.pvsystem.singlediode(
            photocurrent=i_l,
            saturation_current=i_o,
            resistance_series=r_s,
            resistance_shunt=r_sh,
            nNsVth=a_mod,
            method="lambertw",
        )

        i_sc = float(sd_result["i_sc"])
        v_oc = float(sd_result["v_oc"])
        v_mp = float(sd_result["v_mp"])
        i_mp = float(sd_result["i_mp"])
        p_mp = float(sd_result["p_mp"])

        # Enforce physical positivity on terminal quantities
        i_sc = max(0.0, i_sc)
        v_oc = max(0.0, v_oc)
        v_mp = max(0.0, v_mp)
        i_mp = max(0.0, i_mp)
        p_mp = max(0.0, p_mp)

        return PVOutput(
            i_sc=i_sc,
            v_oc=v_oc,
            v_mp=v_mp,
            i_mp=i_mp,
            p_mp=p_mp,
            i_ph=float(i_l),
            i_o=float(i_o),
            r_s=float(r_s),
            r_sh=float(r_sh),
            a=float(a_mod),
            irradiance=g,
            temperature=t_cell,
        )

    def get_stc_parameters(self) -> PVOutput:
        """Calculate and return module operating characteristics at STC (1000 W/m^2, 25°C)."""
        return self.calculate(irradiance=1000.0, cell_temperature=25.0)

    def get_module_information(self) -> Dict[str, Any]:
        """Return comprehensive module metadata, reference ratings, and CEC parameters."""
        return {
            "manufacturer": self.manufacturer,
            "model_name": self.model_name,
            "technology": self.technology,
            "num_cells_in_series": self.num_cells_in_series,
            "reference_stc": self.reference_stc,
            "cec_parameters": self._cec_params,
        }
