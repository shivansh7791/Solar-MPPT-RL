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

### Execute PV Characteristics Analysis & Sweeps (Phase 2)
```bash
python examples/pv_characteristics_demo.py
# or via CLI flag:
python main.py --analyze-pv
```

### Execute DC-DC Boost Converter Transient Simulation (Phase 3)
```bash
python examples/boost_converter_demo.py
# or via CLI flag:
python main.py --simulate-converter
```

### Execute Closed-Loop Perturb & Observe (P&O) MPPT Simulation (Phase 4)
```bash
python examples/po_controller_demo.py
# or via CLI flag:
python main.py --simulate-po
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
- [x] **Phase 1: Solar Photovoltaic Modeling**
  - Mathematical formulation of 5-parameter single-diode model (`docs/PHASE_1_THEORY.md`).
  - Selection and specification of commercial module (`Canadian_Solar_Inc__CS6P_240P`).
  - Implementation of `SolarPVModel` using `pvlib.pvsystem.calcparams_cec` and Lambert W.
  - Numerical STC validation (< 0.001% deviation against CEC flash-test reference).
  - Physical sanity and environmental sensitivity tests (27/27 tests passing).
- [x] **Phase 2: Solar PV Characteristics Analysis & MPP Identification**
  - Continuous I-V and P-V characteristic curve synthesis (`app/pv/pv_characteristics.py`).
  - Numerical Maximum Power Point (MPP) identification vs. analytical validation (< 0.0011% power error).
  - Parametric sweeps under irradiance (200–1000 W/m²) and temperature (15–55°C) variations.
  - Publication-quality visualization suite generating 8 scientific figures (`app/pv/pv_visualization.py`).
  - Metrics export to CSV (`results/metrics/phase_2_mpp_analysis.csv`) and technical analysis (`docs/PHASE_2_ANALYSIS.md`).
  - Comprehensive unit test suite (42/42 tests passing).
- [x] **Phase 3: DC-DC Boost Converter Modeling & Simulation**
  - Mathematical formulation of state-space averaged differential equations in CCM (`docs/PHASE_3_THEORY.md`).
  - Inductor, input capacitor, output capacitor, and load resistor physical sizing ($L=1\text{ mH}, C_{pv}=470\ \mu\text{F}, C_{out}=470\ \mu\text{F}, R_{load}=10\ \Omega$).
  - 4th-order Runge-Kutta (RK4) numerical integration engine with coupled physical PV current injection.
  - Instantaneous power conservation ($\int (P_{in} - P_{out} - P_{loss}) dt \approx \Delta E$, error $< 0.05\%$) and CCM verification ($i_L > 0$).
  - Publication-quality transient visualization suite generating 8 scientific figures (`results/graphs/phase_3/`).
  - Comprehensive unit test suite (60/60 converter tests passing, 102/102 project tests passing).
- [x] **Phase 4: Classical Perturb & Observe (P&O) MPPT Controller** (Completed)
  - Mathematical derivation of boost converter inverse sign rule ($\Delta D = -\delta \cdot \operatorname{sgn}(\Delta P \cdot \Delta V)$) in `docs/PHASE_4_THEORY.md`.
  - Implementation of `PerturbAndObserveController` with saturation clamping, deadbands, and diagnostics (`app/controllers/po_controller.py`).
  - Multirate closed-loop simulation runner `MPPTSimulator` ($dt_{conv} = 100\ \mu\text{s}, T_{mppt} = 5\text{ ms}$) in `app/controllers/mppt_simulation.py`.
  - Quantitative performance evaluation metrics (`MPPTMetrics` and `calculate_mppt_metrics`) in `app/controllers/mppt_metrics.py`.
  - Five standardized engineering experiments (STC, irradiance steps, temperature steps, step sizes, initial duty sensitivity).
  - 10 publication-quality figures ($300\text{ DPI}$) in `results/graphs/phase_4/` and CSV exports in `results/metrics/`.
  - Comprehensive test suite (57 new tests passing, 159/159 total tests passing repository-wide).
- [x] **Phase 5: Gymnasium MPPT Simulation Environment** (Completed)
  - Custom Gymnasium MDP environment `SolarMPPTEnv(gymnasium.Env)` in `app/rl/mppt_env.py`.
  - Normalized continuous observation space (10-dim privileged / 8-dim realistic) in `app/rl/observations.py`.
  - Continuous incremental action space ($a \in [-1, 1] \implies \Delta D = a \cdot \Delta D_{max}$) in `[-1.0, 1.0]`.
  - Configurable reward calculator (`HybridTrackingReward`, `NormalizedPowerReward`, `PowerDeltaReward`) in `app/rl/reward.py`.
  - Atmospheric episode profile generator (STC, random constant, irradiance steps, temperature steps, combined) in `app/rl/episode_profiles.py`.
  - Gymnasium official checker and Stable-Baselines3 compatibility verification (`Monitor` wrapper, SAC action compatibility).
  - Validation script `examples/validate_rl_environment.py`, 5 publication-quality figures ($300\text{ DPI}$) in `results/graphs/phase_5/`, and CSV/JSON metrics in `results/metrics/phase_5/`.
  - 47 new unit/integration tests passing (206/206 total tests passing repository-wide).
- [ ] **Phase 6: Reinforcement Learning MPPT Controller (SAC)**
  - Soft Actor-Critic policy configuration and training pipeline.
  - Hyperparameter tuning, policy evaluation, and model checkpointing.
- [ ] **Phase 7: Comparative Benchmarking & Performance Analysis**
  - Quantitative metrics: MPPT efficiency ($\eta_{MPPT}$), tracking speed, energy yield, and steady-state ripple.
  - Evaluation under standardized dynamic irradiance profiles (step changes, trapezoidal, cloudy weather).
- [ ] **Phase 8: Streamlit Interactive Dashboard & Deployment**
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
| PV Characteristics & MPP Analysis | Phase 2 | **Implemented** | I-V/P-V analysis, 8 graphs in `results/graphs/`, 42/42 tests passing |
| DC-DC Boost Converter Model | Phase 3 | **Implemented** | State-space averaged RK4 plant, 8 graphs in `results/graphs/phase_3/`, 60 converter tests passing |
| Perturb & Observe (P&O) MPPT | Phase 4 | **Implemented** | Inverse sign rule, multirate closed loop, 10 graphs, 159/159 total tests passing |
| Gymnasium Environment | Phase 5 | **Implemented** | `SolarMPPTEnv`, Gymnasium/SB3 compatible, 5 graphs, 206/206 total tests passing |
| SAC RL MPPT Controller | Phase 6 | *Planned* | Reserved for Phase 6 |
| Comparative Benchmarks | Phase 7 | *Planned* | Reserved for Phase 7 |
| Streamlit Web Dashboard | Phase 8 | *Planned* | Reserved for Phase 8 |

> [!NOTE]
> In accordance with Phase 5 scope restrictions, SAC neural network training, policy optimization, hyperparameter tuning, and Streamlit dashboard are strictly deferred to future phases.


