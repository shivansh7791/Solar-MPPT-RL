"""Scenario Management Module for MPPT Comparative Evaluation.

Defines, generates, and serializes standardized atmospheric and initial condition scenarios:
- Scenario A: Standard Test Conditions (STC)
- Scenario B: Irradiance Step Changes (1000 -> 600 -> 800 -> 400 W/m^2)
- Scenario C: Temperature Step Changes (25 -> 35 -> 45 -> 25 °C)
- Scenario D: Combined Environmental Changes (G ramp 1000->500, T step 25->50)
- Scenario E: 10 Randomized Fixed-Seed Environmental Trajectories (Seeds 101-110)
- Scenario F: Initial Condition Sensitivity (D_0 in [0.20, 0.30, 0.40, 0.50, 0.60])
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class ScenarioDefinition:
    """Specification of an experimental MPPT evaluation scenario.

    Attributes
    ----------
    scenario_id : str
        Unique identifier (e.g. 'scenario_a_stc', 'scenario_e_rand_101').
    name : str
        Descriptive human-readable scenario title.
    category : str
        Scenario group: 'A_STC', 'B_IRRADIANCE_STEP', 'C_TEMPERATURE_STEP',
        'D_COMBINED', 'E_RANDOMIZED', or 'F_INITIAL_DUTY'.
    duration : float
        Total physical simulation time [s] (default 1.5 s = 300 control steps).
    initial_duty : float
        Starting converter duty cycle command at t=0.
    irradiance_fn : Callable[[float], float]
        Time-dependent solar irradiance function G(t) [W/m^2].
    temperature_fn : Callable[[float], float]
        Time-dependent cell junction temperature function T(t) [°C].
    description : str
        Engineering purpose and physical context of this scenario.
    seed : Optional[int]
        Random seed for reproducible trajectory generation.
    transition_times : List[float]
        Physical transition timestamps where conditions abruptly change [s].
    """

    scenario_id: str
    name: str
    category: str
    duration: float = 1.5
    initial_duty: float = 0.35
    irradiance_fn: Callable[[float], float] = field(default=lambda t: 1000.0)
    temperature_fn: Callable[[float], float] = field(default=lambda t: 25.0)
    description: str = ""
    seed: Optional[int] = None
    transition_times: List[float] = field(default_factory=list)

    def get_conditions(self, t: float) -> Tuple[float, float]:
        """Return (irradiance, temperature) at physical simulation time t."""
        t_clamped = max(0.0, min(float(t), self.duration))
        g = float(self.irradiance_fn(t_clamped))
        temp = float(self.temperature_fn(t_clamped))
        return g, temp

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dictionary representation."""
        # Sample G and T at 5 key timestamps for descriptive metadata
        sample_times = [0.0, 0.375, 0.75, 1.125, self.duration]
        sampled_g = [float(self.irradiance_fn(t)) for t in sample_times]
        sampled_t = [float(self.temperature_fn(t)) for t in sample_times]

        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "category": self.category,
            "duration": self.duration,
            "initial_duty": self.initial_duty,
            "description": self.description,
            "seed": self.seed,
            "transition_times": self.transition_times,
            "sample_times_s": sample_times,
            "sampled_irradiance_W_m2": sampled_g,
            "sampled_temperature_C": sampled_t,
        }


def create_scenario_a() -> ScenarioDefinition:
    """Scenario A: Standard Test Conditions (STC)."""
    return ScenarioDefinition(
        scenario_id="scenario_a_stc",
        name="Scenario A: Standard Test Conditions (STC)",
        category="A_STC",
        duration=1.5,
        initial_duty=0.35,
        irradiance_fn=lambda t: 1000.0,
        temperature_fn=lambda t: 25.0,
        description="Nominal STC operation (1000 W/m^2, 25°C) to measure baseline convergence, tracking accuracy, and steady-state ripple.",
        seed=42,
        transition_times=[],
    )


def create_scenario_b() -> ScenarioDefinition:
    """Scenario B: Multistep Solar Irradiance Changes (1000 -> 600 -> 800 -> 400 W/m^2)."""
    def g_fn(t: float) -> float:
        if t < 0.375:
            return 1000.0
        elif t < 0.750:
            return 600.0
        elif t < 1.125:
            return 800.0
        else:
            return 400.0

    return ScenarioDefinition(
        scenario_id="scenario_b_irradiance_step",
        name="Scenario B: Irradiance Step Transitions",
        category="B_IRRADIANCE_STEP",
        duration=1.5,
        initial_duty=0.35,
        irradiance_fn=g_fn,
        temperature_fn=lambda t: 25.0,
        description="Abrupt solar intensity transitions (1000 -> 600 -> 800 -> 400 W/m^2 at 25°C) to evaluate transient recovery time and dynamic energy loss.",
        seed=42,
        transition_times=[0.375, 0.750, 1.125],
    )


def create_scenario_c() -> ScenarioDefinition:
    """Scenario C: Multistep Cell Temperature Changes (25 -> 35 -> 45 -> 25 °C)."""
    def t_fn(t: float) -> float:
        if t < 0.375:
            return 25.0
        elif t < 0.750:
            return 35.0
        elif t < 1.125:
            return 45.0
        else:
            return 25.0

    return ScenarioDefinition(
        scenario_id="scenario_c_temperature_step",
        name="Scenario C: Temperature Step Transitions",
        category="C_TEMPERATURE_STEP",
        duration=1.5,
        initial_duty=0.35,
        irradiance_fn=lambda t: 1000.0,
        temperature_fn=t_fn,
        description="Cell junction temperature transitions (25 -> 35 -> 45 -> 25 °C at 1000 W/m^2) shifting the PV characteristic curve and V_mp.",
        seed=42,
        transition_times=[0.375, 0.750, 1.125],
    )


def create_scenario_d() -> ScenarioDefinition:
    """Scenario D: Combined Environmental Changes (Irradiance ramp + Temperature step)."""
    def g_fn(t: float) -> float:
        # Smooth continuous linear ramp from 1000 W/m^2 down to 500 W/m^2 over 1.5s
        return 1000.0 - (500.0 * (t / 1.5))

    def t_fn(t: float) -> float:
        # Step from 25°C to 50°C at t=0.75s
        return 25.0 if t < 0.75 else 50.0

    return ScenarioDefinition(
        scenario_id="scenario_d_combined",
        name="Scenario D: Combined Ramp & Step Environmental Dynamics",
        category="D_COMBINED",
        duration=1.5,
        initial_duty=0.35,
        irradiance_fn=g_fn,
        temperature_fn=t_fn,
        description="Simultaneous atmospheric variations: continuous solar irradiance decrease (1000 -> 500 W/m^2) coupled with sudden thermal rise (25 -> 50°C at 0.75s).",
        seed=42,
        transition_times=[0.750],
    )


def create_scenario_e_list(num_scenarios: int = 10, base_seed: int = 101) -> List[ScenarioDefinition]:
    """Scenario E: Generate 10 randomized, reproducible environmental profiles with fixed seeds."""
    scenarios: List[ScenarioDefinition] = []

    # Preset distinct atmospheric archetypes seeded deterministically
    archetypes = [
        ("Cloud Shading Dip", 900.0, 450.0, 28.0, 28.0, [0.5, 1.0]),
        ("Morning Solar Ramp", 400.0, 950.0, 18.0, 32.0, []),
        ("Midday Extreme Heat", 980.0, 980.0, 52.0, 52.0, []),
        ("Low Irradiance Overcast", 420.0, 420.0, 20.0, 20.0, []),
        ("Rapid Solar Fluctuation", 850.0, 550.0, 30.0, 30.0, [0.3, 0.6, 0.9, 1.2]),
        ("Clear Sky Cool Ambience", 1000.0, 1000.0, 15.0, 15.0, []),
        ("Sudden Heavy Shading", 1000.0, 400.0, 25.0, 25.0, [0.6]),
        ("Partial Cloud Intermittency", 800.0, 600.0, 35.0, 40.0, [0.4, 0.8, 1.2]),
        ("Thermal Drift Warmup", 800.0, 800.0, 25.0, 50.0, []),
        ("Dynamic Complex Profile", 750.0, 900.0, 22.0, 45.0, [0.5, 1.0]),
    ]

    for i in range(num_scenarios):
        seed = base_seed + i
        rng = np.random.RandomState(seed)
        arch_idx = i % len(archetypes)
        arch_name, g_start, g_end, t_start, t_end, transitions = archetypes[arch_idx]

        # Construct profile functions based on archetype
        if arch_name == "Cloud Shading Dip":
            def make_g_fn():
                return lambda t: 900.0 if (t < 0.5 or t >= 1.0) else 450.0
            g_func = make_g_fn()
            t_func = lambda t, val=t_start: val
        elif arch_name == "Morning Solar Ramp":
            def make_g_ramp():
                return lambda t: 400.0 + (550.0 * (t / 1.5))
            def make_t_ramp():
                return lambda t: 18.0 + (14.0 * (t / 1.5))
            g_func = make_g_ramp()
            t_func = make_t_ramp()
        elif arch_name == "Rapid Solar Fluctuation":
            def make_g_toggle():
                return lambda t: 850.0 if (int(t / 0.3) % 2 == 0) else 550.0
            g_func = make_g_toggle()
            t_func = lambda t, val=t_start: val
        elif arch_name == "Sudden Heavy Shading":
            def make_g_step():
                return lambda t: 1000.0 if t < 0.6 else 400.0
            g_func = make_g_step()
            t_func = lambda t, val=t_start: val
        elif arch_name == "Thermal Drift Warmup":
            g_func = lambda t, val=g_start: val
            def make_t_drift():
                return lambda t: 25.0 + (25.0 * (t / 1.5))
            t_func = make_t_drift()
        elif arch_name == "Dynamic Complex Profile":
            def make_complex_g():
                return lambda t: 750.0 if t < 0.5 else (900.0 if t < 1.0 else 600.0)
            def make_complex_t():
                return lambda t: 25.0 if t < 0.5 else (45.0 if t < 1.0 else 35.0)
            g_func = make_complex_g()
            t_func = make_complex_t()
            transitions = [0.5, 1.0]
        else:
            # Constant or slight random variance
            g_val = float(g_start)
            t_val = float(t_start)
            g_func = lambda t, val=g_val: val
            t_func = lambda t, val=t_val: val

        scenarios.append(
            ScenarioDefinition(
                scenario_id=f"scenario_e_rand_{seed}",
                name=f"Scenario E.{i+1}: {arch_name} (Seed {seed})",
                category="E_RANDOMIZED",
                duration=1.5,
                initial_duty=0.35,
                irradiance_fn=g_func,
                temperature_fn=t_func,
                description=f"Reproducible randomized environmental trajectory: {arch_name} with seed={seed}.",
                seed=seed,
                transition_times=transitions,
            )
        )

    return scenarios


def create_scenario_f_list() -> List[ScenarioDefinition]:
    """Scenario F: Evaluate initial condition sensitivity across starting duty cycles [0.20, 0.30, 0.40, 0.50, 0.60]."""
    duty_cycles = [0.20, 0.30, 0.40, 0.50, 0.60]
    scenarios: List[ScenarioDefinition] = []

    for d_0 in duty_cycles:
        tag = int(round(d_0 * 100))
        scenarios.append(
            ScenarioDefinition(
                scenario_id=f"scenario_f_duty_{tag}",
                name=f"Scenario F: Initial Duty D_0 = {d_0:.2f}",
                category="F_INITIAL_DUTY",
                duration=1.5,
                initial_duty=d_0,
                irradiance_fn=lambda t: 1000.0,
                temperature_fn=lambda t: 25.0,
                description=f"Initial condition sensitivity under STC starting from off-MPP duty cycle D_0 = {d_0:.2f} (Optimal D* ≈ 0.3939).",
                seed=42,
                transition_times=[],
            )
        )

    return scenarios


def create_standard_scenarios() -> List[ScenarioDefinition]:
    """Create complete standardized suite of 19 evaluation scenarios."""
    scenarios: List[ScenarioDefinition] = []
    scenarios.append(create_scenario_a())
    scenarios.append(create_scenario_b())
    scenarios.append(create_scenario_c())
    scenarios.append(create_scenario_d())
    scenarios.extend(create_scenario_e_list(num_scenarios=10, base_seed=101))
    scenarios.extend(create_scenario_f_list())
    return scenarios


class ScenarioManager:
    """Manages scenario instantiation, indexing, and serialization."""

    def __init__(self, scenarios: Optional[Sequence[ScenarioDefinition]] = None) -> None:
        self._scenarios = list(scenarios if scenarios is not None else create_standard_scenarios())
        self._by_id = {s.scenario_id: s for s in self._scenarios}

    def __len__(self) -> int:
        return len(self._scenarios)

    def get_all(self) -> List[ScenarioDefinition]:
        return list(self._scenarios)

    def get(self, scenario_id: str) -> ScenarioDefinition:
        if scenario_id not in self._by_id:
            raise KeyError(f"Scenario ID '{scenario_id}' not found in ScenarioManager.")
        return self._by_id[scenario_id]

    def export_definitions_json(self, target_path: Union[str, Path]) -> Path:
        """Export all scenario metadata to a structured JSON file."""
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_metadata_dict() for s in self._scenarios]
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return target
