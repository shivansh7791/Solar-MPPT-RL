"""Phase 0 Verification Script (Python-Side Audit).

Verifies that the MATLAB project configuration and reference datasets match
the original Python project specifications exactly.
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np

from config.default_config import get_default_config

MATLAB_ROOT = Path("Solar_MPPT_MATLAB")


def test_phase0_integrity():
    py_cfg = get_default_config()

    print("=====================================================")
    print("  VERIFYING PHASE 0 MATLAB PROJECT FOUNDATION        ")
    print("=====================================================")

    # 1. Check directory structure
    required_dirs = [
        "config",
        "models/pv_model",
        "models/converter",
        "models/controllers",
        "models/rl_agent",
        "simulink",
        "experiments",
        "reference_data/python_outputs",
        "results/csv",
        "results/figures",
        "utils",
        "tests",
        "docs/phase_reports",
    ]
    for d in required_dirs:
        dir_path = MATLAB_ROOT / d
        assert dir_path.exists(), f"Missing required directory: {d}"
        print(f"  [PASS] Directory exists: {d}")

    # 2. Check reference files
    ref_dir = MATLAB_ROOT / "reference_data/python_outputs"
    required_refs = [
        "cec_parameters.csv",
        "stc_reference.csv",
        "stc_iv_curve.csv",
        "irradiance_sweep_mpp.csv",
        "irradiance_curves.csv",
        "temperature_sweep_mpp.csv",
        "temperature_curves.csv",
    ]
    for r in required_refs:
        f_path = ref_dir / r
        assert f_path.exists() and f_path.stat().st_size > 0, f"Missing or empty reference: {r}"
        print(f"  [PASS] Reference file valid: {r}")

    # 3. Check STC Reference vs Config
    stc_df = pd.read_csv(ref_dir / "stc_reference.csv")
    row = stc_df.iloc[0]
    np.testing.assert_allclose(row["v_oc"], py_cfg.pv.stc_v_oc, rtol=1e-5)
    np.testing.assert_allclose(row["i_sc"], py_cfg.pv.stc_i_sc, rtol=1e-5)
    np.testing.assert_allclose(row["v_mp"], py_cfg.pv.stc_v_mp, rtol=1e-4)
    np.testing.assert_allclose(row["i_mp"], py_cfg.pv.stc_i_mp, rtol=1e-4)
    np.testing.assert_allclose(row["p_mp"], py_cfg.pv.stc_p_mp, rtol=1e-4)
    print("  [PASS] STC reference data perfectly aligns with Python PV model ratings.")

    # 4. Check MATLAB project_config.m parsing
    cfg_text = (MATLAB_ROOT / "config/project_config.m").read_text()
    assert "CS6P-240P" in cfg_text
    assert "240.097" in cfg_text
    assert "1.577654" in cfg_text
    assert "8.599262" in cfg_text
    assert "0.310448" in cfg_text
    assert "287.92276" in cfg_text
    print("  [PASS] project_config.m contains verified Canadian Solar CS6P-240P parameters.")

    # 5. Check Multirate timing parameters
    assert "1.0e-5" in cfg_text  # 10 us plant dt
    assert "5.0e-3" in cfg_text  # 5 ms control dt
    print("  [PASS] Multirate timing parameters (10 us / 5 ms) verified.")

    print("=====================================================")
    print("  ALL PHASE 0 VERIFICATION CHECKS PASSED (100%)       ")
    print("=====================================================\n")


if __name__ == "__main__":
    test_phase0_integrity()
