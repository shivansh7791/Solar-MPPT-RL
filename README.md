# Design and Performance Analysis of a Reinforcement Learning-Based Maximum Power Point Tracking Controller for Solar Photovoltaic Systems

## 1. Project Title
**Design and Performance Analysis of a Reinforcement Learning-Based Maximum Power Point Tracking Controller for Solar Photovoltaic Systems**  
*Electrical and Electronics Engineering Minor Project*

---

## 2. Project Overview
Solar photovoltaic (PV) generation exhibits non-linear current-voltage ($I$-$V$) and power-voltage ($P$-$V$) characteristics that vary dynamically with solar irradiance ($G$) and cell operating temperature ($T$). To maximize conversion efficiency, a Maximum Power Point Tracking (MPPT) controller continuously regulates the duty cycle ($d$) of an interfacing power electronic DC-DC boost converter, maintaining operation at the optimal electrical operating point $(V_{mp}, I_{mp})$.

This academic engineering project develops a physics-grounded solar PV generation model, couples it with a dynamic state-space DC-DC boost converter, implements conventional Perturb and Observe (P&O) MPPT as an engineering baseline, and trains a state-of-the-art Reinforcement Learning (RL) agent using Soft Actor-Critic (SAC) within a custom Gymnasium environment. The ultimate objective is a comparative performance evaluation under steady-state, fast ramp, and partial shading irradiance profiles.

---

## 3. Problem Statement
Conventional MPPT techniques, predominantly Perturb and Observe (P&O) and Incremental Conductance (IncCond), suffer from fundamental trade-offs:
1. **Steady-State Oscillations**: Fixed perturbation step sizes induce steady-state power oscillations around the Maximum Power Point (MPP), resulting in avoidable energy losses.
2. **Dynamic Tracking Divergence**: Under rapidly changing atmospheric irradiance, conventional gradient-based perturbation algorithms can misinterpret irradiance-induced power shifts as duty cycle responses, leading to tracking disorientation and transient loss of efficiency.
3. **Local Extrema Trapping**: Under partial shading conditions (PSC), the $P$-$V$ curve exhibits multiple local power peaks alongside a single Global Maximum Power Point (GMPP), where classical hill-climbing algorithms frequently become trapped in suboptimal local maxima.

Reinforcement Learning, specifically off-policy actor-critic architectures like Soft Actor-Critic (SAC), offers continuous action-space control, adaptive non-linear mapping, and entropy-regularized exploration capable of mitigating steady-state ripple while improving dynamic tracking speed.

---

## 4. Project Objectives
1. **Physical PV Array Modeling**: Formulate and validate a physics-based single-diode 5-parameter PV module model using `pvlib`, accounting for temperature and irradiance variations.
2. **Power Electronic Converter Dynamics**: Model a DC-DC boost converter incorporating switched state-space equations, continuous conduction mode (CCM) boundary dynamics, component parasitics, and switching ripple.
3. **Baseline Classical Controller**: Implement a mathematically parameterized Perturb and Observe (P&O) MPPT controller to serve as a rigorous engineering benchmark.
4. **Gymnasium Simulation Environment**: Engineer a high-fidelity Gymnasium-compliant simulation environment bridging the physical PV-converter plant with RL observation spaces, action spaces, and reward formulations.
5. **Continuous RL MPPT Controller**: Formulate and train a Soft Actor-Critic (SAC) reinforcement learning agent optimizing duty cycle adjustments with high sample efficiency and stability.
6. **Rigorous Comparative Benchmark**: Perform quantitative evaluation across standardized test profiles (EN 50530 dynamic efficiency, fast irradiance steps, temperature variations) analyzing tracking efficiency ($\eta_{MPPT}$), response time ($t_r$), and steady-state ripple ($\Delta P$).
7. **Interactive Analytics Dashboard**: Build a Streamlit dashboard visualizing $I$-$V$/$P$-$V$ curves, converter transient waveforms, duty cycle trajectories, and comparative tracking performance.

---

## 5. Proposed Methodology

```
+---------------------------------------------------------------------------------+
|                               Physical PV Array                                  |
|   - 5-Parameter Single-Diode Model (I_ph, I_0, R_s, R_sh, a)                    |
|   - Inputs: Solar Irradiance G [W/m^2], Ambient Temperature T [°C]              |
+---------------------------------------+-----------------------------------------+
                                        | V_pv, I_pv
                                        v
+---------------------------------------------------------------------------------+
|                           DC-DC Boost Converter Plant                            |
|   - State Variables: Inductor Current i_L(t), Capacitor Voltage v_C(t)          |
|   - Actuation: Duty Cycle d(t) in [0.05, 0.95] at Switching Frequency f_sw     |
+---------------------------------------+-----------------------------------------+
                                        | State Feedback: (v_pv, i_pv, P_pv, d)
                                        v
+---------------------------------------------------------------------------------+
|                       Gymnasium RL Environment & MPPT Control                    |
|   - Observation Vector: s_t = [v_pv, i_pv, P_pv, d_t, ΔP_pv]                   |
|   - Action: a_t in [-Δd_max, +Δd_max] -> d_{t+1} = clip(d_t + a_t, d_min, d_max) |
|   - Reward: r_t = P_pv(t) / P_mpp(t) - penalty(Δd)                              |
+-------------------+-----------------------------------------+-------------------+
                    |                                         |
                    v                                         v
         +--------------------+                    +---------------------+
         |  Conventional P&O  |                    |   Soft Actor-Critic |
         |  MPPT Controller   |                    |       (SAC) Agent   |
         +--------------------+                    +---------------------+
                    \                                         /
                     +-------------------+-------------------+
                                         |
                                         v
                         +-------------------------------+
                         |  Comparative Evaluation &     |
                         |  Streamlit Dashboard Metrics  |
                         +-------------------------------+
```

---

## 6. Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.12 / 3.14 | Core programming language |
| **PV Modeling** | `pvlib` | Physical PV cell and diode parameter calculations |
| **Numerical Engine** | `NumPy`, `SciPy` | State-space integration and numerical solvers |
| **Data Handling** | `Pandas` | Irradiance profiles, logging, and metrics processing |
| **RL Environment** | `Gymnasium` (Farama) | Custom MDP environment interface for MPPT control |
| **RL Algorithm** | `Stable-Baselines3` | Off-policy Soft Actor-Critic (SAC) implementation |
| **Deep Learning** | `PyTorch` | Neural network backbones and tensor computation |
| **Visualization** | `Matplotlib`, `Plotly` | Academic figures, transient plots, and $P$-$V$ curves |
| **Dashboard** | `Streamlit` | Interactive UI for waveform inspection and analytics |
| **Testing** | `pytest` | Automated unit, regression, and smoke test suite |
| **Version Control** | `Git` | Repository and experiment configuration tracking |

---

## 7. Project Architecture

The codebase is organized into modular packages adhering to separation of concerns:

```
RL-Based-Solar-MPPT/
├── .gitignore                      # Git exclusion rules for artifacts, venv, caches
├── pytest.ini                      # Centralized pytest configuration
├── requirements.txt                # Pinned production and development dependencies
├── README.md                       # Comprehensive project documentation
├── main.py                         # Top-level entry point and environment validator
│
├── app/                            # Primary application packages
│   ├── __init__.py                 # Application root metadata
│   ├── pv/                         # Physics-based PV modeling subpackage (Phase 1)
│   │   └── __init__.py
│   ├── converter/                  # DC-DC Boost Converter modeling (Phase 2)
│   │   └── __init__.py
│   ├── controllers/                # MPPT Controllers (P&O and RL agents) (Phase 3/5)
│   │   └── __init__.py
│   ├── environment/                # Gymnasium MPPT environment subpackage (Phase 4)
│   │   └── __init__.py
│   ├── training/                   # SAC training pipeline & callbacks (Phase 5)
│   │   └── __init__.py
│   ├── experiments/                # Benchmarking and comparative pipelines (Phase 6)
│   │   └── __init__.py
│   ├── dashboard/                  # Streamlit visual dashboard (Phase 7)
│   │   └── __init__.py
│   └── utils/                      # Reusable utilities (logging, formatting)
│       ├── __init__.py
│       └── logger.py               # Centralized logging facility
│
├── config/                         # Centralized configuration subpackage
│   ├── __init__.py
│   └── default_config.py           # Typed dataclasses for simulation defaults
│
├── data/                           # Data storage
│   ├── profiles/                   # Solar irradiance and temperature profiles
│   └── models/                     # Trained RL policy checkpoints
│
├── notebooks/                      # Exploratory research notebooks
│
├── results/                        # Output artifacts
│   ├── experiment_logs/            # Execution and training logs
│   ├── graphs/                     # Rendered figures and comparison charts
│   └── metrics/                    # Numerical evaluation metrics (CSV/JSON)
│
├── tests/                          # Automated testing suite
│   ├── __init__.py
│   └── test_smoke.py               # Smoke test verifying imports and configuration
│
└── docs/                           # Technical documentation & phase DoD reports
    └── PHASE_0_DOD.md              # Phase 0 Definition of Done Report
```

---

## 8. Installation Instructions

### Prerequisites
- macOS, Linux, or Windows with Python 3.12+ (tested on Apple Silicon macOS with Python 3.14).
- `git` version control installed.

### Clone the Repository
```bash
git clone <repository-url>
cd "Minor Project"
```

### Create Virtual Environment
```bash
python3 -m venv .venv
```

---

## 9. Environment Activation Instructions

### On macOS / Linux:
```bash
source .venv/bin/activate
```

### On Windows (Command Prompt / PowerShell):
```cmd
.venv\Scripts\activate.bat
# or in PowerShell:
.venv\Scripts\Activate.ps1
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 10. How to Run the Project

### Perform System & Environment Health Check (Phase 0)
```bash
python main.py --check-env
```

### Execute PV Model Demonstration (Phase 1)
```bash
python examples/pv_model_demo.py
# or via CLI flag:
python main.py --test-pv
```

### Execute Full Test Suite
```bash
pytest
```

---

## 11. Development Roadmap

- [x] **Phase 0: Project Foundation & Environment Setup**
  - Environment inspection and validation.
  - Modular project directory structure.
  - Centralized dataclass-based configuration.
  - Unified logging framework.
  - Automated smoke testing suite.
- [x] **Phase 1: Solar Photovoltaic Modeling** (Completed)
  - Mathematical formulation of 5-parameter single-diode model (`docs/PHASE_1_THEORY.md`).
  - Selection and specification of commercial module (`Canadian_Solar_Inc__CS6P_240P`).
  - Implementation of `SolarPVModel` using `pvlib.pvsystem.calcparams_cec` and Lambert W.
  - Numerical STC validation (< 0.001% deviation against CEC flash-test reference).
  - Physical sanity and environmental sensitivity tests (27/27 tests passing).
- [ ] **Phase 2: DC-DC Boost Converter Modeling**
  - Analytical state-space differential equations.
  - Inductor and capacitor sizing for Continuous Conduction Mode (CCM).
  - Switched and averaged numerical models.
  - Duty cycle actuation and efficiency validation.
- [ ] **Phase 3: Classical Perturb & Observe (P&O) MPPT Controller**
  - Classical P&O algorithm with configurable perturbation step ($\Delta d$) and sampling rate.
  - Steady-state oscillation and dynamic ramp response evaluation.
- [ ] **Phase 4: Gymnasium MPPT Simulation Environment**
  - Gymnasium MDP formulation: state observation space, action bounds, step physics integration.
  - Continuous reward engineering balancing tracking accuracy and duty cycle actuation stability.
- [ ] **Phase 5: Reinforcement Learning MPPT Controller (SAC)**
  - Soft Actor-Critic policy configuration and training pipeline.
  - Hyperparameter tuning, policy evaluation, and model checkpointing.
- [ ] **Phase 6: Comparative Benchmarking & Performance Analysis**
  - Quantitative metrics: MPPT efficiency ($\eta_{MPPT}$), tracking speed, energy yield, and steady-state ripple.
  - Evaluation under standardized dynamic irradiance profiles (step changes, trapezoidal, cloudy weather).
- [ ] **Phase 7: Streamlit Interactive Dashboard & Deployment**
  - Interactive web UI displaying $P$-$V$ curves, converter transient waveforms, and live tracking comparisons.

---

## 12. Current Development Status

| Feature / Subsystem | Phase | Status | Notes |
| :--- | :---: | :---: | :--- |
| Project Structure & Packaging | Phase 0 | **Implemented** | Complete modular directory layout with `__init__.py` files |
| Centralized Configuration | Phase 0 | **Implemented** | Typed dataclasses in `config/default_config.py` |
| Logging Infrastructure | Phase 0 | **Implemented** | Timestamped console logging in `app/utils/logger.py` |
| Automated Smoke Testing | Phase 0 | **Implemented** | Pytest verification passing (5/5 smoke tests) |
| PV Cell / Array Physics Model | Phase 1 | **Implemented** | `SolarPVModel` verified (<0.001% STC error, 27/27 tests passing) |
| DC-DC Boost Converter Model | Phase 2 | *Planned* | Reserved for Phase 2 |
| Perturb & Observe (P&O) MPPT | Phase 3 | *Planned* | Reserved for Phase 3 |
| Gymnasium Environment | Phase 4 | *Planned* | Reserved for Phase 4 |
| SAC RL MPPT Controller | Phase 5 | *Planned* | Reserved for Phase 5 |
| Comparative Benchmarks | Phase 6 | *Planned* | Reserved for Phase 6 |
| Streamlit Web Dashboard | Phase 7 | *Planned* | Reserved for Phase 7 |

> [!NOTE]
> In accordance with Phase 1 scope restrictions, boost converter state-space equations, MPPT algorithms, and RL agents are strictly deferred to their designated development phases.

