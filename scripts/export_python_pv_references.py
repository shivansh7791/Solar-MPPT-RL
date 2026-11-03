"""Export Python PV Reference Data for MATLAB Validation.

Executes the validated Python SolarPVModel and PVCharacteristicAnalyzer to produce
canonical CSV datasets for cross-platform MATLAB validation.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from app.pv.pv_model import SolarPVModel
from app.pv.pv_characteristics import PVCharacteristicAnalyzer
from config.default_config import get_default_config

OUT_DIR = Path("Solar_MPPT_MATLAB/reference_data/python_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def export_references():
    cfg = get_default_config()
    pv_model = SolarPVModel()
    analyzer = PVCharacteristicAnalyzer(pv_model)

    # 1. CEC Parameters
    cec_dict = pv_model.get_module_information()["cec_parameters"]
    cec_df = pd.DataFrame([cec_dict])
    cec_df.to_csv(OUT_DIR / "cec_parameters.csv", index=False)
    print("Exported cec_parameters.csv")

    # 2. STC Operating Point
    stc_out = pv_model.get_stc_parameters()
    stc_df = pd.DataFrame([stc_out.to_dict()])
    stc_df.to_csv(OUT_DIR / "stc_reference.csv", index=False)
    print("Exported stc_reference.csv")

    # 3. STC I-V & P-V Curve (500 points)
    stc_curve = analyzer.generate_curve(irradiance=1000.0, cell_temperature=25.0, num_points=500)
    stc_curve_df = pd.DataFrame({
        "voltage": stc_curve.voltage,
        "current": stc_curve.current,
        "power": stc_curve.power,
    })
    stc_curve_df.to_csv(OUT_DIR / "stc_iv_curve.csv", index=False)
    print("Exported stc_iv_curve.csv")

    # 4. Irradiance Sweep (200, 400, 600, 800, 1000 W/m^2 at 25°C)
    irradiances = [200.0, 400.0, 600.0, 800.0, 1000.0]
    irr_mpp_records = []
    irr_curve_dfs = []

    for g in irradiances:
        curve = analyzer.generate_curve(irradiance=g, cell_temperature=25.0, num_points=500)
        pvo = curve.pv_output
        irr_mpp_records.append({
            "irradiance": g,
            "cell_temperature": 25.0,
            "i_sc": pvo.i_sc,
            "v_oc": pvo.v_oc,
            "v_mp": pvo.v_mp,
            "i_mp": pvo.i_mp,
            "p_mp": pvo.p_mp,
            "fill_factor": curve.fill_factor,
            "i_ph": pvo.i_ph,
            "i_o": pvo.i_o,
            "r_s": pvo.r_s,
            "r_sh": pvo.r_sh,
            "a": pvo.a,
        })
        for v, i, p in zip(curve.voltage, curve.current, curve.power):
            irr_curve_dfs.append({
                "irradiance": g,
                "voltage": v,
                "current": i,
                "power": p,
            })

    pd.DataFrame(irr_mpp_records).to_csv(OUT_DIR / "irradiance_sweep_mpp.csv", index=False)
    pd.DataFrame(irr_curve_dfs).to_csv(OUT_DIR / "irradiance_curves.csv", index=False)
    print("Exported irradiance_sweep_mpp.csv and irradiance_curves.csv")

    # 5. Temperature Sweep (15, 25, 35, 45, 55°C at 1000 W/m^2) & (10, 25, 40, 55, 70, 75°C)
    temperatures = [15.0, 25.0, 35.0, 45.0, 55.0, 70.0, 75.0]
    temp_mpp_records = []
    temp_curve_dfs = []

    for t in temperatures:
        curve = analyzer.generate_curve(irradiance=1000.0, cell_temperature=t, num_points=500)
        pvo = curve.pv_output
        temp_mpp_records.append({
            "irradiance": 1000.0,
            "cell_temperature": t,
            "i_sc": pvo.i_sc,
            "v_oc": pvo.v_oc,
            "v_mp": pvo.v_mp,
            "i_mp": pvo.i_mp,
            "p_mp": pvo.p_mp,
            "fill_factor": curve.fill_factor,
            "i_ph": pvo.i_ph,
            "i_o": pvo.i_o,
            "r_s": pvo.r_s,
            "r_sh": pvo.r_sh,
            "a": pvo.a,
        })
        for v, i, p in zip(curve.voltage, curve.current, curve.power):
            temp_curve_dfs.append({
                "cell_temperature": t,
                "voltage": v,
                "current": i,
                "power": p,
            })

    pd.DataFrame(temp_mpp_records).to_csv(OUT_DIR / "temperature_sweep_mpp.csv", index=False)
    pd.DataFrame(temp_curve_dfs).to_csv(OUT_DIR / "temperature_curves.csv", index=False)
    print("Exported temperature_sweep_mpp.csv and temperature_curves.csv")

    print(f"All reference outputs exported successfully to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    export_references()
