# Phase 8: Interactive Streamlit Dashboard — User Guide

## 1. Executive Summary & Purpose

The **Interactive MPPT Dashboard** provides a unified, academic, and engineering-grade graphical interface for simulating, controlling, and analyzing Solar Photovoltaic (PV) systems coupled to DC-DC Boost Converters.

The dashboard integrates all validated engineering modules developed across Phases 0 through 7:
1. **Solar PV Mathematical Model** (Physics-based Single-Diode Model for Canadian Solar CS6P-240P).
2. **DC-DC Boost Converter Simulation** (State-Space Averaged Continuous-Conduction Model with RK4 numerical integration).
3. **Classical Perturb & Observe (P&O) Controller** (Fixed step $\Delta D = 0.005$, sampling period $T_s = 5.0\text{ ms}$).
4. **Soft Actor-Critic (SAC) Deep RL Agent** (Continuous policy $\pi_\theta(a \mid s)$, entropy-regularized with clipped Gaussian actions).
5. **Multi-Controller Paired Evaluation & Comparative Metrics** (Exact numerical tracking efficiency $\eta$, energy harvested $E_{harv}$, deficit $E_{loss}$, ripple, settling time).

---

## 2. System Requirements & Quick Start

### 2.1 Prerequisites
- **Python**: Version 3.10+ (Tested on Python 3.14.6 macOS / Linux / Windows).
- **Virtual Environment**: Active virtual environment with project dependencies installed.
- **Dependencies**: `streamlit`, `plotly`, `pandas`, `numpy`, `torch`, `stable-baselines3`, `pvlib`, `scipy`.

### 2.2 Launch Instructions

From the project root directory (`/Users/shivanshupadhyay/Documents/Minor Project`):

```bash
# 1. Activate the Python virtual environment
source .venv/bin/activate

# 2. Launch the Streamlit application
streamlit run app/dashboard/streamlit_app.py
```

Streamlit will start the local HTTP server and automatically launch your default browser at:
```
  Local URL: http://localhost:8501
  Network URL: http://<ip>:8501
```

> **Note on Custom Port**:
> If port `8501` is already in use by another service, specify an alternate port:
> ```bash
> streamlit run app/dashboard/streamlit_app.py --server.port 8502
> ```

---

## 3. Dashboard Structure & Navigation

The application is structured into a persistent **Sidebar Control Panel** on the left and a 5-tab main workspace:

```
┌─────────────────────────┬────────────────────────────────────────────────────────────────┐
│ SIDEBAR CONTROLS        │ TABS NAVIGATION                                                │
│                         │ [1. Overview] [2. PV Curves] [3. MPPT Simulation] [4. Benchmark] [5. Export]
├─────────────────────────┼────────────────────────────────────────────────────────────────┤
│ • Execution Mode        │ Active Tab Content View                                        │
│ • Model Checkpoint      │ • Interactive Plots (Plotly)                                   │
│ • Operating Conditions  │ • Engineering KPI Cards                                        │
│ • Converter Specs       │ • Detailed Data Tables                                         │
│ • Profile Selector      │ • Download Triggers                                            │
└─────────────────────────┴────────────────────────────────────────────────────────────────┘
```

---

## 4. Sidebar Controls & Parameter Guidelines

| Control | Type | Permissible Range / Options | Default Value | Engineering Description |
|---|---|---|---|---|
| **Simulation Mode** | Radio | `Comparison (P&O vs SAC)`, `Classical P&O`, `Deep RL (SAC)` | `Comparison` | Controls which controllers are evaluated during the simulation run. |
| **SAC Policy Model** | Selectbox | Auto-discovered `.zip` checkpoints | `best_mppt_model.zip` (Realistic 8-dim) | Pre-trained SAC checkpoint to load for inference. |
| **Environmental Profile** | Selectbox | `Static`, `Scenario A (STC)`, `Scenario B (Irradiance Steps)`, `Scenario C (Thermal Steps)`, `Scenario D (Combined)`, `Scenario E (Cloud/Ramp)`, `Scenario F (Duty Mismatch)` | `Scenario B` | Defines ambient solar irradiance $G(t)$ and cell temperature $T(t)$ trajectories. |
| **Solar Irradiance $G$** | Slider | $100.0\text{ to }1200.0\text{ W/m}^2$ | $1000.0\text{ W/m}^2$ | Incident irradiance on PV array (active in Static profile mode). |
| **Cell Temperature $T$** | Slider | $-10.0\text{ to }75.0^\circ\text{C}$ | $25.0^\circ\text{C}$ | Semiconductor junction temperature (active in Static profile mode). |
| **Load Resistance $R_{load}$** | Slider | $5.0\text{ to }100.0\ \Omega$ | $20.0\ \Omega$ | Output resistive load impedance connected across filter capacitor $C_{out}$. |
| **Initial Duty Cycle $D_0$** | Slider | $0.10\text{ to }0.85$ | $0.35$ | Starting converter switch command at $t = 0.0\text{ s}$ ($D^* \approx 0.3939$ at STC). |
| **Simulation Duration** | Slider | $0.5\text{ to }3.0\text{ s}$ | $1.5\text{ s}$ | Total physical simulation duration ($300\text{ control cycles}$ at $1.5\text{ s}$). |

---

## 5. Tab Walkthroughs & Features

### 5.1 Tab 1: System Overview & Physical Architecture
- **Interactive Topology Diagram**: Visualizes the solar PV array (CS6P-240P), boost converter power stage ($L, C_{in}, C_{out}, S, D, R_{load}$), sensor acquisition layer ($V_{pv}, I_{pv}$), and closed-loop MPPT controller.
- **Physical Specifications Table**: Displays rated parameters for both the PV module ($P_{max}=240\text{W}, V_{mp}=29.9\text{V}, I_{mp}=8.03\text{A}, V_{oc}=37.0\text{V}, I_{sc}=8.59\text{A}$) and the DC-DC converter ($L=1\text{mH}, C_{in}=100\mu\text{F}, C_{out}=220\mu\text{F}, f_s=50\text{kHz}$).
- **State-Space Averaged Equations**: Mathematical formulation governing converter dynamics under continuous conduction mode (CCM).

### 5.2 Tab 2: PV Characteristics & Operating Point Analyzer
- **Interactive Characteristic Curves**:
  - **I-V Curve**: Terminal current $I_{pv}$ vs terminal voltage $V_{pv}$.
  - **P-V Curve**: Output power $P_{pv}$ vs terminal voltage $V_{pv}$.
- **Analytical MPP Projection**: Prominently highlights the Maximum Power Point $(V_{mp}, I_{mp}, P_{mp})$ computed via Brent's method on the single-diode implicit equation.
- **Dynamic Operating Point Projection**: Allows the user to select an active converter operating voltage $V_{op}$ and visually projects the resulting current and power onto the non-linear characteristic curve.
- **Parameter Sensitivity Table**: Shows open-circuit voltage $V_{oc}$, short-circuit current $I_{sc}$, fill factor $FF$, and equivalent load impedance $R_{opt} = V_{mp} / I_{mp}$.

### 5.3 Tab 3: Interactive MPPT Simulation & Dynamic Response
- **Single or Comparative Execution**: Runs real-time numerical simulation using the 4th-Order Runge-Kutta (RK4) integration engine at $dt = 10\ \mu\text{s}$ sub-stepping.
- **Interactive Multi-Panel Plotly Charts**:
  - **Power Tracking Dynamics**: Actual generated PV power $P_{pv}(t)$ superimposed directly against theoretical optimum $P_{MPP}(t)$.
  - **Tracking Error**: Instantaneous residual error $e(t) = P_{MPP}(t) - P_{pv}(t)$.
  - **PV Voltage & Current**: Array terminal voltage $V_{pv}(t)$ and current $I_{pv}(t)$ trajectories.
  - **Boost Output Voltage**: Load voltage $V_{out}(t)$ exhibiting converter voltage step-up.
  - **Duty Cycle Dynamics**: Control actuation trajectory $D(t)$ bounded within $[0.10, 0.85]$.
  - **Environmental Profile**: Active ambient solar irradiance $G(t)$ and temperature $T(t)$.
- **Real-Time KPI Cards**:
  - MPPT Tracking Efficiency $\eta_{MPPT} = \frac{\int P_{pv}\,dt}{\int P_{mpp}\,dt} \times 100\%$
  - Total Harvested Energy $E_{harv} = \int P_{pv}\,dt$ [Joule and Watt-hour]
  - Ideal Reference Energy $E_{ideal} = \int P_{mpp}\,dt$ [Joule]
  - Harvesting Energy Deficit $E_{loss} = E_{ideal} - E_{harv}$ [Joule]
  - Steady-State Power Ripple $\sigma_{P}$ [W]
  - Settling Time $t_s$ ($2\%$ error band) [s]

### 5.4 Tab 4: Performance Analytics & Benchmark Scenarios
- **Standardized Benchmark Suite**: Direct access to Scenarios A through F from the Phase 7 experimental evaluation:
  - **Scenario A**: STC baseline steady-state ($1000\text{ W/m}^2, 25^\circ\text{C}$).
  - **Scenario B**: Multistep irradiance transitions ($1000 \to 600 \to 800 \to 400\text{ W/m}^2$).
  - **Scenario C**: Thermal step transitions ($25^\circ\text{C} \to 45^\circ\text{C} \to 35^\circ\text{C}$).
  - **Scenario D**: Combined irradiance ramp and thermal fluctuation.
  - **Scenario E**: Realistic environmental archetypes (Cloud shading dips, morning solar ramps, low irradiance overcast).
  - **Scenario F**: Off-MPP initial duty cycle mismatch ($D_0 \in \{0.20, 0.30, 0.40, 0.50, 0.60\}$).
- **Statistical Aggregation Summary**: Presents arithmetic mean and energy-weighted aggregate metrics across all scenarios, demonstrating the $+2.22\%$ energy advantage and $86.1\%$ energy loss reduction achieved by SAC over P&O.

### 5.5 Tab 5: Data Export & Documentation
- **One-Click CSV Export**:
  - Simulation Timeseries ($t, V_{pv}, I_{pv}, P_{pv}, V_{out}, I_L, D, G, T, P_{mpp}, \dots$).
  - Summary Metrics Table ($\eta, E_{harv}, E_{loss}, \sigma_P, t_s$).
  - Phase 7 Comprehensive 19-Scenario Paired Comparison Dataset.
- **JSON Configuration Export**: Full system configuration payload including PV diode parameters, converter inductances, capacitances, and active controller settings.

---

## 6. Step-by-Step Workflows

### 6.1 Workflow 1: Inspecting PV Curves Under Changing Irradiance
1. Open the dashboard in your browser.
2. Navigate to **Tab 2: PV Characteristics**.
3. In the left sidebar, adjust the **Solar Irradiance** slider from $1000\text{ W/m}^2$ down to $400\text{ W/m}^2$.
4. Observe the instantaneous shift in the I-V and P-V curves, with short-circuit current $I_{sc}$ dropping linearly with irradiance and maximum power point $P_{mp}$ decreasing from $240.0\text{W}$ down to $\approx 93.5\text{W}$.

### 6.2 Workflow 2: Simulating Dynamic Cloud Shading (Scenario B)
1. Select **Comparison (P&O vs SAC)** in the sidebar.
2. In the **Environmental Profile** dropdown, select `Scenario B: Multistep Irradiance Changes`.
3. Set **Simulation Duration** to $1.5\text{ s}$ and **Load Resistance** to $20.0\ \Omega$.
4. Click **Run MPPT Simulation**.
5. Switch to **Tab 3: Interactive MPPT Simulation**.
6. Inspect the paired Plotly overlay:
   - Notice how Classical P&O exhibits large oscillations ($\pm 5\text{W}$) around MPP during steady irradiance and drifts temporarily during abrupt downward steps ($1000 \to 600\text{ W/m}^2$).
   - Notice how SAC smoothly tracks the changing MPP with negligible steady-state ripple and faster convergence.
7. Compare the KPI cards: SAC achieves $\approx 99.4\%$ tracking efficiency vs $\approx 97.4\%$ for P&O.

### 6.3 Workflow 3: Exporting Simulation Data for Research Reports
1. After running the simulation in Workflow 2, navigate to **Tab 5: Data Export**.
2. Click **Download Timeseries CSV (P&O)** or **Download Timeseries CSV (SAC)**.
3. The browser will download `mppt_simulation_po_timeseries.csv` and `mppt_simulation_sac_timeseries.csv` containing full state trajectories sampled at $5.0\text{ ms}$.

---

## 7. Troubleshooting & FAQ

### Q1: The browser shows "Connection refused" or does not open automatically.
- **Cause**: Streamlit may be running in headless mode or the local port is occupied.
- **Fix**: Open your browser manually and visit `http://localhost:8501`. If another service occupies that port, run `streamlit run app/dashboard/streamlit_app.py --server.port 8503`.

### Q2: Why does SAC load the realistic checkpoint rather than the privileged checkpoint?
- **Explanation**: In real hardware deployments, theoretical solar irradiance $G$ and cell junction temperature $T$ cannot be measured with low-cost instantaneous sensors. The **realistic policy** operates exclusively on 8 measurable electrical feedback states ($V_{pv}, I_{pv}, P_{pv}, \Delta V_{pv}, \Delta P_{pv}, V_{out}, I_L, D$). The dashboard defaults to the validated realistic model located at `artifacts/rl/realistic/best_model/best_mppt_model.zip`.

### Q3: How do I verify that the simulation satisfies physical conservation of energy?
- **Answer**: In Tab 3 and Tab 4, for every simulation, the metrics card reports both Harvested Energy $E_{harv}$ and Ideal Reference Energy $E_{ideal}$. The harvesting deficit is strictly defined as $E_{loss} = E_{ideal} - E_{harv}$. The dashboard and simulation engine guarantee that $E_{ideal} \equiv E_{harv} + E_{loss}$ with zero unphysical negative deficit.
