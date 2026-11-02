"""Phase 1 Demonstration: Single-Diode PV Model Validation.

Demonstrates initializing the Canadian Solar CS6P-240P model, running STC validation
against reference values, and computing electrical characteristics under varying
environmental conditions.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pv import SolarPVModel
from app.utils import setup_logger



def run_demo() -> None:
    logger = setup_logger("pv_demo")
    logger.info("Initializing SolarPVModel with default Canadian Solar CS6P-240P configuration...")

    model = SolarPVModel()
    info = model.get_module_information()

    print("=" * 80)
    print(f"PV MODULE: {info['manufacturer']} - {info['model_name']} ({info['technology']})")
    print(f"Cells in Series: {info['num_cells_in_series']}")
    print("=" * 80)

    # 1. Standard Test Conditions (STC) Evaluation
    print("\n" + "-" * 80)
    print("1. EVALUATION AT STANDARD TEST CONDITIONS (STC: G = 1000 W/m², T = 25°C)")
    print("-" * 80)

    stc_calc = model.get_stc_parameters()
    ref = info["reference_stc"]

    headers = ["Parameter", "Calculated", "Reference STC", "Absolute Deviation", "Percentage Error (%)"]
    row_fmt = "{:<16} | {:<12.4f} | {:<14.4f} | {:<18.6f} | {:<16.4f}%"

    print(f"{headers[0]:<16} | {headers[1]:<12} | {headers[2]:<14} | {headers[3]:<18} | {headers[4]:<16}")
    print("-" * 85)

    params_to_compare = [
        ("Isc [A]", stc_calc.i_sc, ref["i_sc"]),
        ("Voc [V]", stc_calc.v_oc, ref["v_oc"]),
        ("Vmp [V]", stc_calc.v_mp, ref["v_mp"]),
        ("Imp [A]", stc_calc.i_mp, ref["i_mp"]),
        ("Pmp [W]", stc_calc.p_mp, ref["p_mp"]),
    ]

    for label, calc, reference in params_to_compare:
        abs_err = abs(calc - reference)
        pct_err = (abs_err / abs(reference)) * 100.0 if reference != 0 else 0.0
        print(row_fmt.format(label, calc, reference, abs_err, pct_err))

    # Verify Pmp = Vmp * Imp
    recalc_pmp = stc_calc.v_mp * stc_calc.i_mp
    print(f"\nConsistency Check: Vmp * Imp = {recalc_pmp:.4f} W (Difference from Pmp: {abs(stc_calc.p_mp - recalc_pmp):.6e} W)")

    # 2. Additional Environmental Operating Conditions
    print("\n" + "-" * 80)
    print("2. ADDITIONAL ENVIRONMENTAL OPERATING CONDITIONS")
    print("-" * 80)

    test_conditions = [
        ("Condition A: Low Irradiance", 400.0, 25.0),
        ("Condition B: High Temperature (Summer)", 1000.0, 55.0),
        ("Condition C: Cold Sunny Day (Winter)", 1000.0, 5.0),
        ("Condition D: Cloudy Cold Day", 250.0, 10.0),
    ]

    summary_header = "{:<36} | {:<8} | {:<6} | {:<8} | {:<8} | {:<8} | {:<8}"
    print(summary_header.format("Scenario", "G [W/m²]", "T [°C]", "Isc [A]", "Voc [V]", "Vmp [V]", "Pmp [W]"))
    print("-" * 92)

    for desc, g, t in test_conditions:
        out = model.calculate(irradiance=g, cell_temperature=t)
        print(summary_header.format(
            desc,
            f"{g:.0f}",
            f"{t:.0f}",
            f"{out.i_sc:.3f}",
            f"{out.v_oc:.2f}",
            f"{out.v_mp:.2f}",
            f"{out.p_mp:.2f}",
        ))

    print("=" * 80)
    print("Phase 1 Demonstration successfully completed.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
