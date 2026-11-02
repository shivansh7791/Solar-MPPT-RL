"""Audit script for Phase 5 environment qualifications before Phase 6 training.

Audits:
1. Reset equilibrium: computes state derivatives immediately after reset across
   randomized (G, T, D) combinations, comparing Phase 5 initial state vs exact steady-state.
2. Observation space bounds: checks bounds across all 5 environmental profiles for both modes.
3. Reward behavior: validates monotonicity, action penalty scaling, fault penalties.

Outputs results to results/metrics/phase_6/phase_5_audit.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import brentq

from app.converter.boost_converter import BoostConverter
from app.pv.pv_model import SolarPVModel
from app.rl.episode_profiles import EpisodeProfileGenerator, ProfileType
from app.rl.mppt_env import SolarMPPTEnv
from app.rl.reward import RewardCalculator, RewardType
from config.default_config import MasterConfig, RLEnvConfig, get_default_config


def compute_exact_equilibrium(
    pv_model: SolarPVModel,
    converter: BoostConverter,
    duty_cycle: float,
    irradiance: float,
    cell_temperature: float,
) -> tuple[float, float, float]:
    """Compute exact steady-state operating point (V_pv*, i_L*, V_out*) where all derivatives are 0."""
    d = float(duty_cycle)
    r_in = ((1.0 - d) ** 2) * converter.R_load + converter.r_L

    # Find V_pv such that V_pv = r_in * I_pv(V_pv)
    v_oc = pv_model.calculate(irradiance, cell_temperature).v_oc

    def residual(v: float) -> float:
        i_pv = pv_model.current_at_voltage(v, irradiance, cell_temperature, clip_non_negative=True)
        return v - r_in * i_pv

    v_pv_star = brentq(residual, 0.0, v_oc * 1.05)
    i_l_star = pv_model.current_at_voltage(v_pv_star, irradiance, cell_temperature, clip_non_negative=True)
    v_out_star = (1.0 - d) * converter.R_load * i_l_star

    return float(v_pv_star), float(i_l_star), float(v_out_star)


def audit_reset_equilibrium(num_trials: int = 10, seed: int = 42) -> dict:
    """Audit converter state derivatives immediately after reset."""
    cfg = get_default_config()
    pv_model = SolarPVModel(cfg.pv)
    converter = BoostConverter(pv_model, cfg.converter)

    rng = np.random.default_rng(seed)
    trials = []

    for i in range(num_trials):
        g = float(rng.uniform(400.0, 1000.0))
        t_cell = float(rng.uniform(15.0, 55.0))
        d0 = float(rng.uniform(0.20, 0.60))

        ref_mpp = pv_model.calculate(g, t_cell)

        # 1. Phase 5 heuristic initialization:
        # V_pv = V_mp, i_L = I_mp, V_out = V_mp / (1 - D)
        v_pv_p5 = float(ref_mpp.v_mp)
        i_l_p5 = float(ref_mpp.i_mp)
        v_out_p5 = float(v_pv_p5 / max(0.1, 1.0 - d0))
        state_p5 = np.array([v_pv_p5, i_l_p5, v_out_p5], dtype=float)

        derivs_p5 = converter.compute_derivatives(state_p5, d0, g, t_cell)

        # 2. Exact physical equilibrium initialization:
        v_pv_eq, i_l_eq, v_out_eq = compute_exact_equilibrium(pv_model, converter, d0, g, t_cell)
        state_eq = np.array([v_pv_eq, i_l_eq, v_out_eq], dtype=float)
        derivs_eq = converter.compute_derivatives(state_eq, d0, g, t_cell)

        trials.append({
            "trial": i + 1,
            "irradiance_W_m2": round(g, 2),
            "temperature_C": round(t_cell, 2),
            "duty_cycle": round(d0, 4),
            "phase_5_heuristic": {
                "initial_state": [round(float(x), 3) for x in state_p5],
                "derivatives": {
                    "dV_pv_dt": round(float(derivs_p5[0]), 3),
                    "di_L_dt": round(float(derivs_p5[1]), 3),
                    "dV_out_dt": round(float(derivs_p5[2]), 3),
                },
                "max_abs_derivative": float(np.max(np.abs(derivs_p5))),
            },
            "exact_equilibrium": {
                "initial_state": [round(float(x), 3) for x in state_eq],
                "derivatives": {
                    "dV_pv_dt": round(float(derivs_eq[0]), 6),
                    "di_L_dt": round(float(derivs_eq[1]), 6),
                    "dV_out_dt": round(float(derivs_eq[2]), 6),
                },
                "max_abs_derivative": float(np.max(np.abs(derivs_eq))),
            },
        })

    p5_max_derivs = [t["phase_5_heuristic"]["max_abs_derivative"] for t in trials]
    eq_max_derivs = [t["exact_equilibrium"]["max_abs_derivative"] for t in trials]

    return {
        "trials_evaluated": num_trials,
        "phase_5_heuristic_mean_max_derivative": float(np.mean(p5_max_derivs)),
        "phase_5_heuristic_worst_max_derivative": float(np.max(p5_max_derivs)),
        "exact_equilibrium_mean_max_derivative": float(np.mean(eq_max_derivs)),
        "exact_equilibrium_worst_max_derivative": float(np.max(eq_max_derivs)),
        "detailed_trials": trials,
    }


def audit_observation_bounds(num_steps: int = 100, seed: int = 42) -> dict:
    """Verify observations remain strictly inside declared bounds for all profiles and both modes."""
    profiles = [ProfileType.STC, ProfileType.RANDOM_CONSTANT, ProfileType.IRRADIANCE_STEP, ProfileType.TEMPERATURE_STEP, ProfileType.COMBINED]
    results = {}

    for include_mpp in [False, True]:
        mode_label = "privileged_10d" if include_mpp else "realistic_8d"
        mode_results = {}

        for p_type in profiles:
            cfg = get_default_config()
            cfg.environment.include_mpp_in_obs = include_mpp
            cfg.environment.profile_type = p_type.value
            cfg.environment.max_steps_per_episode = num_steps

            env = SolarMPPTEnv(config=cfg)
            obs, info = env.reset(seed=seed)

            obs_min = np.copy(obs)
            obs_max = np.copy(obs)
            all_finite = True
            all_in_bounds = True

            rng = np.random.default_rng(seed)
            for _ in range(num_steps):
                action = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
                obs, reward, term, trunc, info = env.step(action)

                if not np.all(np.isfinite(obs)):
                    all_finite = False
                if not env.observation_space.contains(obs):
                    all_in_bounds = False

                obs_min = np.minimum(obs_min, obs)
                obs_max = np.maximum(obs_max, obs)

                if term or trunc:
                    break

            mode_results[p_type.value] = {
                "all_finite": all_finite,
                "strictly_in_bounds": all_in_bounds,
                "observed_min": [round(float(x), 4) for x in obs_min],
                "observed_max": [round(float(x), 4) for x in obs_max],
                "space_low": float(env.observation_space.low[0]),
                "space_high": float(env.observation_space.high[0]),
            }

        results[mode_label] = mode_results

    return results


def audit_reward_behavior() -> dict:
    """Verify reward monotonicity, action penalties, and fault penalties."""
    cfg = RLEnvConfig(reward_type="hybrid_tracking", weight_power=1.0, weight_tracking_error=2.0, weight_action_penalty=0.05, fault_penalty=10.0)
    calc = RewardCalculator(cfg)

    p_mpp = 240.097
    powers = np.linspace(10.0, 240.097, 24)
    rewards = [calc.compute_reward(p, p_mpp, action=0.0)[0] for p in powers]
    is_monotonic = bool(np.all(np.diff(rewards) > 0))

    # Action penalty scaling
    r_act0, _ = calc.compute_reward(240.0, p_mpp, action=0.0)
    r_act1, _ = calc.compute_reward(240.0, p_mpp, action=1.0)
    action_penalty_delta = r_act0 - r_act1

    # Fault penalty
    r_fault, comps_fault = calc.compute_reward(200.0, p_mpp, action=0.0, is_fault=True)

    return {
        "reward_formulation": "hybrid_tracking",
        "power_monotonicity_verified": is_monotonic,
        "reward_at_mpp": round(float(rewards[-1]), 4),
        "reward_at_half_mpp": round(float(calc.compute_reward(p_mpp * 0.5, p_mpp, action=0.0)[0]), 4),
        "action_penalty_at_full_action": round(float(action_penalty_delta), 4),
        "fault_penalty_applied": round(float(r_fault), 4),
        "all_finite": bool(np.all(np.isfinite(rewards))),
    }


def main() -> None:
    print("Executing Phase 5 Audit for Phase 6 Preparation...")
    output_dir = Path("results/metrics/phase_6")
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_data = {
        "reset_equilibrium_audit": audit_reset_equilibrium(num_trials=10, seed=42),
        "observation_bounds_audit": audit_observation_bounds(num_steps=100, seed=42),
        "reward_behavior_audit": audit_reward_behavior(),
    }

    audit_path = output_dir / "phase_5_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit_data, f, indent=4)

    print(f"Audit completed successfully. Results saved to: {audit_path}")
    print("\n--- Summary of Audit Findings ---")
    print(f"1. Reset Equilibrium:")
    print(f"   Phase 5 Heuristic Mean Max Derivative: {audit_data['reset_equilibrium_audit']['phase_5_heuristic_mean_max_derivative']:.2f}")
    print(f"   Exact Equilibrium Mean Max Derivative:  {audit_data['reset_equilibrium_audit']['exact_equilibrium_mean_max_derivative']:.6f}")
    print(f"2. Observation Bounds:")
    for mode, profs in audit_data["observation_bounds_audit"].items():
        all_ok = all(p["strictly_in_bounds"] and p["all_finite"] for p in profs.values())
        print(f"   Mode '{mode}': All profiles strictly in bounds & finite: {all_ok}")
    print(f"3. Reward Behavior:")
    print(f"   Power Monotonicity: {audit_data['reward_behavior_audit']['power_monotonicity_verified']}")
    print(f"   Reward at MPP:      {audit_data['reward_behavior_audit']['reward_at_mpp']}")
    print(f"   Action Penalty:     {audit_data['reward_behavior_audit']['action_penalty_at_full_action']}")
    print(f"   Fault Penalty:      {audit_data['reward_behavior_audit']['fault_penalty_applied']}")


if __name__ == "__main__":
    main()
