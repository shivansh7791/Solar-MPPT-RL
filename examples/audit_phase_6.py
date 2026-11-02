"""Phase 6 Readiness Audit Script for Phase 7 Comparative MPPT Evaluation.

Verifies:
1. Trained realistic SAC model checkpoint exists and loads.
2. Observation dimension matches realistic environment mode (8-dim).
3. Action dimension and bounds match [-1.0, 1.0].
4. Hyperparameters and seed (42) are documented in artifacts.
5. Training report documents actual training timesteps (10,000).
6. Independent deterministic evaluation executes without errors.
7. Physical environment and equilibrium reset remain fully functional.

Saves audit results to results/metrics/phase_7/phase_6_readiness_audit.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
from stable_baselines3 import SAC

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.rl.mppt_env import SolarMPPTEnv
from app.rl.sac_agent import load_sac_agent
from app.utils.logger import get_logger
from config.default_config import RLEnvConfig, get_default_config

logger = get_logger("phase_6_audit")


def run_phase_6_audit() -> Dict[str, Any]:
    """Execute all Phase 6 readiness gate checks."""
    audit_results: Dict[str, Any] = {
        "gate_status": "PENDING",
        "checkpoint_checks": {},
        "model_checks": {},
        "environment_checks": {},
        "training_report_checks": {},
        "independent_evaluation_checks": {},
        "all_passed": False,
    }

    # 1. Checkpoint verification
    best_model_path = project_root / "artifacts/rl/realistic/best_model/best_mppt_model.zip"
    final_model_path = project_root / "artifacts/rl/realistic/final_model/sac_mppt_final.zip"
    config_path = project_root / "artifacts/rl/realistic/config.json"
    summary_path = project_root / "artifacts/rl/realistic/training_summary.json"

    chk_ok = best_model_path.is_file() and final_model_path.is_file()
    audit_results["checkpoint_checks"] = {
        "best_model_exists": best_model_path.is_file(),
        "best_model_path": str(best_model_path),
        "best_model_size_bytes": best_model_path.stat().st_size if best_model_path.is_file() else 0,
        "final_model_exists": final_model_path.is_file(),
        "final_model_path": str(final_model_path),
        "config_json_exists": config_path.is_file(),
        "training_summary_exists": summary_path.is_file(),
        "status": "PASS" if chk_ok else "FAIL",
    }

    if not chk_ok:
        audit_results["gate_status"] = "FAILED"
        return audit_results

    # 2. Model loading and architecture checks
    try:
        model = load_sac_agent(best_model_path, device="cpu")
        obs_shape = model.observation_space.shape
        act_shape = model.action_space.shape
        act_low = float(model.action_space.low[0])
        act_high = float(model.action_space.high[0])

        model_ok = (
            obs_shape == (8,)
            and act_shape == (1,)
            and np.isclose(act_low, -1.0)
            and np.isclose(act_high, 1.0)
        )
        audit_results["model_checks"] = {
            "load_successful": True,
            "observation_dim": int(obs_shape[0]),
            "expected_observation_dim": 8,
            "observation_mode": "realistic_8dim",
            "action_dim": int(act_shape[0]),
            "action_low": act_low,
            "action_high": act_high,
            "policy_type": str(model.policy.__class__.__name__),
            "status": "PASS" if model_ok else "FAIL",
        }
    except Exception as e:
        audit_results["model_checks"] = {
            "load_successful": False,
            "error": str(e),
            "status": "FAIL",
        }
        audit_results["gate_status"] = "FAILED"
        return audit_results

    # 3. Training report and configuration check
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        is_realistic = not cfg_data.get("include_mpp_in_obs", True)
        timesteps = summary_data.get("total_timesteps_completed", 0)
        seed = cfg_data.get("seed", None)
        episodes = summary_data.get("total_episodes_completed", 0)
        best_eff = summary_data.get("best_tracking_efficiency_achieved", 0.0)

        report_ok = is_realistic and timesteps >= 10000 and seed == 42
        audit_results["training_report_checks"] = {
            "is_realistic_mode": is_realistic,
            "seed": seed,
            "total_timesteps_completed": timesteps,
            "total_episodes_completed": episodes,
            "best_tracking_efficiency_achieved": best_eff,
            "mean_evaluation_tracking_efficiency": summary_data.get("mean_evaluation_tracking_efficiency", 0.0),
            "status": "PASS" if report_ok else "FAIL",
        }
    except Exception as e:
        audit_results["training_report_checks"] = {
            "error": str(e),
            "status": "FAIL",
        }
        audit_results["gate_status"] = "FAILED"
        return audit_results

    # 4. Environment functionality and equilibrium check
    try:
        master_cfg = get_default_config()
        rl_env_cfg = RLEnvConfig(
            include_mpp_in_obs=False,
            profile_type="stc",
            max_steps_per_episode=300,
        )
        env = SolarMPPTEnv(config=master_cfg, rl_env_config=rl_env_cfg)
        obs, info = env.reset(seed=42, options={"profile_type": "stc", "randomize_duty": False, "initial_duty": 0.35})

        env_obs_dim = obs.shape[0]
        v_pv_init = info["pv_voltage"]
        d_init = info["duty_cycle"]

        # Check equilibrium state derivative
        derivatives = env.converter.compute_derivatives(env.converter.state, d_init, 1000.0, 25.0)
        norm_deriv = float(np.linalg.norm(derivatives))

        env_ok = (
            env_obs_dim == 8
            and np.all(np.isfinite(obs))
            and norm_deriv < 1e-6
        )
        audit_results["environment_checks"] = {
            "env_reset_successful": True,
            "env_observation_dim": env_obs_dim,
            "initial_duty": d_init,
            "initial_pv_voltage": v_pv_init,
            "equilibrium_derivative_norm": norm_deriv,
            "status": "PASS" if env_ok else "FAIL",
        }
    except Exception as e:
        audit_results["environment_checks"] = {
            "env_reset_successful": False,
            "error": str(e),
            "status": "FAIL",
        }
        audit_results["gate_status"] = "FAILED"
        return audit_results

    # 5. Independent deterministic evaluation check
    try:
        actions_sampled = []
        rewards_sampled = []
        curr_obs = obs
        for _ in range(10):
            action, _ = model.predict(curr_obs, deterministic=True)
            curr_obs, reward, terminated, truncated, step_info = env.step(action)
            actions_sampled.append(float(action[0]))
            rewards_sampled.append(float(reward))
            if terminated or truncated:
                break

        eval_ok = (
            len(actions_sampled) == 10
            and all(np.isfinite(actions_sampled))
            and all(np.isfinite(rewards_sampled))
        )
        audit_results["independent_evaluation_checks"] = {
            "steps_evaluated": len(actions_sampled),
            "all_actions_finite": all(np.isfinite(actions_sampled)),
            "all_rewards_finite": all(np.isfinite(rewards_sampled)),
            "action_mean": float(np.mean(actions_sampled)),
            "action_std": float(np.std(actions_sampled)),
            "status": "PASS" if eval_ok else "FAIL",
        }
    except Exception as e:
        audit_results["independent_evaluation_checks"] = {
            "error": str(e),
            "status": "FAIL",
        }
        audit_results["gate_status"] = "FAILED"
        return audit_results

    # Final readiness assessment
    all_passed = (
        audit_results["checkpoint_checks"]["status"] == "PASS"
        and audit_results["model_checks"]["status"] == "PASS"
        and audit_results["training_report_checks"]["status"] == "PASS"
        and audit_results["environment_checks"]["status"] == "PASS"
        and audit_results["independent_evaluation_checks"]["status"] == "PASS"
    )

    audit_results["all_passed"] = all_passed
    audit_results["gate_status"] = "GATE_PASSED_READY_FOR_PHASE_7" if all_passed else "GATE_FAILED"

    return audit_results


if __name__ == "__main__":
    results = run_phase_6_audit()
    output_dir = project_root / "results/metrics/phase_7"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "phase_6_readiness_audit.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=2))
    print(f"\nAudit complete. Gate status: {results['gate_status']}")
    if not results["all_passed"]:
        sys.exit(1)
