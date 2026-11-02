# PHASE 7 ENGINEERING & STATISTICAL ANALYSIS REPORT

## Comparative Performance Evaluation: Soft Actor-Critic (SAC) vs. Classical Perturb & Observe (P&O) MPPT

**Project**: Design and Performance Analysis of a Reinforcement Learning-Based Maximum Power Point Tracking Controller for Solar Photovoltaic Systems  
**Phase**: Phase 7 — Comparative Performance Evaluation  
**Status**: **COMPLETE (AUDITED & RECONCILED)**  
**Author**: Power Electronics & Control Systems Engineering Specialist  
**Date**: 2026-11-03  

---

## 1. Executive Summary

This report documents the rigorous, physics-grounded comparative evaluation of the trained **Soft Actor-Critic (SAC)** continuous-action reinforcement learning controller against the industry-standard **Classical Perturb and Observe (P&O)** MPPT controller.

Both controllers were evaluated across **19 standardized paired environmental scenarios** (38 physical plant simulations) spanning standard test conditions, multistep solar transitions, multistep cell temperature dynamics, combined linear ramps, randomized atmospheric intermittency, and extreme initial duty-cycle mis-matches. All simulations were conducted on the exact same physical plant: the single-diode 5-parameter model of the Canadian Solar CS6P-240P PV module coupled to a state-space averaged DC-DC boost converter operating at $f_{sw} = 20\text{ kHz}$ with continuous fourth-order Runge-Kutta (RK4) numerical integration ($dt = 100\ \mu\text{s}$, $T_{ctrl} = 5.0\text{ ms}$).

### Key Performance Findings:
- **Arithmetic Mean Scenario Efficiency ($\overline{\eta}_{arith}$)**: SAC achieved **$98.73\%$** versus P&O's **$92.64\%$**, yielding a statistically significant average improvement of **$+6.09\%$** ($p = 0.0317 < 0.05$).
- **Energy-Weighted Aggregate Efficiency ($\eta_{agg}$)**: Evaluated across all $5,667.67\text{ J}$ ($1.5744\text{ Wh}$) of theoretically available solar energy, SAC extracted **$98.88\%$** ($5,604.20\text{ J}$) versus P&O's **$92.44\%$** ($5,239.25\text{ J}$), producing a net energy harvest gain of **$+364.95\text{ J}$ ($+0.1014\text{ Wh}$ / $+6.97\%$)**.
- **Energy Deficit Reduction**: Cumulative energy lost due to imperfect tracking and transient delay dropped from **$428.41\text{ J}$** (P&O) down to **$63.47\text{ J}$** (SAC), representing an **$85.19\%$ reduction in cumulative energy deficit**.
- **Exact Energy Conservation**: For every single scenario without exception, physical energy conservation holds to machine precision:
  $$E_{ideal} \equiv E_{harvested} + E_{loss} \quad (\text{Residual } |\Delta| < 1.0 \times 10^{-12}\text{ J})$$
- **Dynamic Tracking Agility**: SAC demonstrated near-instantaneous convergence ($< 0.05\text{ s}$) following solar steps and startup transients, completely avoiding the directional confusion and limit-cycle oscillations characteristic of classical P&O.
- **Embedded Feasibility**: The deployed inference policy requires only the feedforward **Actor Network (68,353 evaluated parameters, $\approx 136\text{ kFLOPs}$ per 5 ms step)**, consuming only **$2.26\%$** of the available CPU time budget on a standard processor.

---

## 2. Experimental Setup and Plant Grounding

### 2.1 Verified Converter Physical Parameters
Every simulation executed in Phase 7 strictly used the centralized configuration from `config/default_config.py` (`ConverterConfig`), matching Phase 3, Phase 4, Phase 5, and Phase 6 identically:
- **Boost Inductance ($L$)**: $1.0\text{ mH}$ ($1.0 \times 10^{-3}\text{ H}$)
- **Input Decoupling Capacitance ($C_{pv}$)**: $470.0\ \mu\text{F}$ ($4.7 \times 10^{-4}\text{ F}$)
- **Output Filter Capacitance ($C_{out}$)**: $470.0\ \mu\text{F}$ ($4.7 \times 10^{-4}\text{ F}$)
- **Load Resistance ($R_{load}$)**: $10.0\ \Omega$ (matched for nominal $240\text{ W}$ throughput at $48\text{ V} - 50\text{ V}$ bus: $V^2 / R = 49^2 / 10 \approx 240\text{ W}$)
- **Inductor Series Resistance ($r_L$)**: $0.05\ \Omega$ ($50\text{ m}\Omega$)
- **Switching Frequency ($f_{sw}$)**: $20.0\text{ kHz}$
- **Permitted Duty Cycle Range**: $[D_{min}, D_{max}] = [0.05, 0.90]$
- **Plant Simulation Timestep ($dt$)**: $100\ \mu\text{s}$ ($1.0 \times 10^{-4}\text{ s}$)
- **Control Period ($T_{ctrl}$)**: $5.0\text{ ms}$ ($200\text{ Hz}$, corresponding to 50 RK4 plant sub-steps per decision)

*(Note on parameter audit: An earlier preliminary documentation draft inadvertently listed $C_{out} = 220\ \mu\text{F}$ and $R_{load} = 20\ \Omega$ in prose text; inspection of the simulation runner code `BoostConverter(..., config=self.config.converter)` and the raw timeseries columns $V_{out}/I_{out} = 47.907 / 4.791 = 10.0\ \Omega$ confirms that the physical plant in all 38 simulations strictly executed with the intended Phase 3 baseline configuration: $C_{out} = 470\ \mu\text{F}$ and $R_{load} = 10.0\ \Omega$. No disparate configurations were ever run.)*

### 2.2 Exact Physical Equilibrium Initialization
At $t = 0$, both controllers were initialized at the exact physical equilibrium $(V_{pv}^*, i_L^*, V_{out}^*)$ obtained by solving:
$$f(V_{pv}) = V_{pv} - \left[(1 - D_0)^2 R_{load} + r_L\right] \cdot I_{pv}(V_{pv}, G_0, T_0) = 0$$
via Brent's root-finding method.
- Inductor current: $i_L^* = I_{pv}(V_{pv}^*, G_0, T_0)$
- Output capacitor voltage: $V_{out}^* = (1 - D_0) R_{load} i_L^*$

At nominal STC ($G = 1000\text{ W/m}^2, T = 25^\circ\text{C}, D_0 = 0.35$):
- $V_{pv}^* = 31.50797954\text{ V}$
- $i_L^* = 7.37028761\text{ A}$
- $V_{out}^* = 47.90686948\text{ V}$

The individual initial state time derivatives evaluated at this state are:
- $\frac{dV_{pv}}{dt} = \mathbf{0.000000\text{ V/s}}$
- $\frac{di_L}{dt} = \mathbf{7.38 \times 10^{-12}\text{ A/s}}$
- $\frac{dV_{out}}{dt} = \mathbf{1.89 \times 10^{-12}\text{ V/s}}$

All individual component residuals are below $10^{-11}$, eliminating artificial startup derivative shocks and ensuring that performance metrics evaluate purely closed-loop control dynamics.

---

## 3. Standardized Scenario Suite (19 Paired Scenarios)

The evaluation was executed across 19 distinct paired scenarios divided into six functional categories:

| Category | Identifier | Description | Conditions & Transitions |
|---|---|---|---|
| **A: STC** | `scenario_a_stc` | Standard Test Conditions | $G = 1000\text{ W/m}^2$, $T = 25^\circ\text{C}$, $D_0 = 0.35$, Duration: 1.5 s |
| **B: Irradiance Steps** | `scenario_b_irradiance_step` | Multistep Solar Intensity | $1000 \to 600 \to 800 \to 400\text{ W/m}^2$ at $T = 25^\circ\text{C}$ (steps at 0.375, 0.75, 1.125 s) |
| **C: Temperature Steps** | `scenario_c_temperature_step` | Multistep Cell Temperature | $25 \to 35 \to 45 \to 25^\circ\text{C}$ at $G = 1000\text{ W/m}^2$ (steps at 0.375, 0.75, 1.125 s) |
| **D: Combined Dynamics** | `scenario_d_combined` | Simultaneous Atmospheric Shift | Irradiance linear ramp ($1000 \to 500\text{ W/m}^2$) + thermal step ($25 \to 50^\circ\text{C}$ at 0.75 s) |
| **E: Randomized Profiles** | `scenario_e_rand_101-110` | 10 Fixed-Seed Realistic Profiles | Varied shading, cloud dips, morning ramps, fast fluctuations ($G \in [400, 1000]$, $T \in [15, 55]$) |
| **F: Initial Duty Sensitivity** | `scenario_f_duty_20-60` | Robustness to Starting Point | 5 starting duty cycles: $D_0 \in \{0.20, 0.30, 0.40, 0.50, 0.60\}$ under STC ($D^* \approx 0.3939$) |

---

## 4. Performance Metric Formulations & Aggregation Methodology

### 4.1 Distinction: Arithmetic Mean vs. Energy-Weighted Efficiency
To ensure mathematical clarity and prevent reporting ambiguity, two distinct metrics are reported:
1. **Arithmetic Mean Scenario Efficiency ($\overline{\eta}_{arith}$)**:
   The simple unweighted average of individual scenario tracking efficiencies:
   $$\overline{\eta}_{arith} = \frac{1}{N} \sum_{i=1}^{N} \eta_{MPPT, i}$$
   This metric evaluates how reliably the controller tracks across different test conditions, weighting short and low-irradiance profiles equally with high-power profiles.
2. **Energy-Weighted Aggregate Efficiency ($\eta_{agg}$)**:
   The total electrical energy harvested across all scenarios divided by the total available theoretical maximum MPP energy:
   $$\eta_{agg} = \frac{\sum_{i=1}^{N} E_{harvested, i}}{\sum_{i=1}^{N} E_{ideal, i}} \times 100\% = \frac{\sum_{i=1}^{N} \int_0^T P_{pv, i}(t)\, dt}{\sum_{i=1}^{N} \int_0^T P_{mpp, i}(t)\, dt} \times 100\%$$
   This metric weights high-power conditions proportionally more than low-power overcast conditions, representing cumulative energy extraction in practical service.

### 4.2 Energy Deficit and Conservation
The physical **energy deficit** ($E_{loss}$) represents the total electrical energy lost due to imperfect tracking, startup delay, and transient settling:
$$E_{loss} = E_{ideal} - E_{harvested} = \int_0^T \left[P_{mpp}(t) - P_{pv}(t)\right]\, dt$$
By definition of energy conservation, the exact identity:
$$E_{ideal} \equiv E_{harvested} + E_{loss}$$
holds for every scenario.

### 4.3 Dynamic Settling Time & Steady-State Ripple
- **Settling Time ($t_s$)**: The earliest time after which extracted power permanently remains within $\pm 2.0\%$ of instantaneous dynamic reference $P_{mpp}(t)$.
- **Steady-State Power Ripple ($\Delta P_{ss}$)**: Peak-to-peak variation $\max(P) - \min(P)$ over the steady-state evaluation window (final 30% of episode).
  > **Physical Distinction**: The model simulated is a **state-space averaged plant model**, which averages out the $20\text{ kHz}$ switching harmonics. The reported ripple represents **control-induced perturbation ripple** ($\Delta D$ limit-cycling), NOT high-frequency hardware switching ripple.

---

## 5. Aggregate Comparative Results Table

The reconciled performance metrics evaluated across all 19 paired scenarios (38 total simulations) are summarized below:

| Performance Metric | Classical P&O | Trained SAC Agent | Difference (SAC - P&O) | Improvement Status |
|---|---|---|---|---|
| **Arithmetic Mean Scenario Tracking Efficiency ($\overline{\eta}_{arith}$)** | **92.64%** | **98.73%** | **+6.09%** | **+6.57% relative gain** |
| **Energy-Weighted Aggregate Tracking Efficiency ($\eta_{agg}$)** | **92.44%** | **98.88%** | **+6.44%** | **+6.97% energy yield** |
| **Total Available Ideal MPP Energy ($E_{ideal}$)** | 5,667.67 J (1.5744 Wh) | 5,667.67 J (1.5744 Wh) | 0.00 J | Ground-truth reference |
| **Total Electrical Energy Harvested ($E_{harvested}$)** | 5,239.25 J (1.4553 Wh) | **5,604.20 J (1.5567 Wh)** | **+364.95 J (+0.1014 Wh)** | **+6.97% more energy** |
| **Total Energy Deficit / Loss ($E_{loss}$)** | 428.41 J (0.1190 Wh) | **63.47 J (0.0176 Wh)** | **-364.95 J** | **-85.19% loss reduction** |
| **Mean Steady-State Perturbation Ripple ($\Delta P_{ss}$)** | 22.90 W | **19.49 W** | **-3.41 W** | **-14.89% overall** |
| **STC Nominal Steady-State Ripple** | 0.0274 W | **0.0000 W** | **-0.0274 W** | **100% ripple suppression** |
| **Mean Steady-State RMS Error** | 15.58 W | **2.54 W** | **-13.04 W** | **-83.72% error reduction** |
| **Mean Converter Efficiency ($\eta_{conv}$)** | 98.63% | **98.72%** | **+0.09%** | Identical high efficiency |
| **Convergence Success Rate ($\pm 2\%$ band)** | 78.9% (15/19) | **100.0% (19/19)** | **+21.1%** | 0 divergence events |

---

## 6. Scenario-by-Scenario Detailed Breakdown

Every individual scenario below confirms exact energy conservation ($E_{ideal} = E_{harvested} + E_{loss}$):

| Scenario ID | Name & Category | P&O $\eta$ (%) | SAC $\eta$ (%) | $\Delta \eta$ (%) | P&O Harv (J) | SAC Harv (J) | Net Gain (J) | Ideal $E$ (J) | P&O Loss (J) | SAC Loss (J) |
|---|---|---|---|---|---|---|---|---|---|---|
| `scenario_a_stc` | Scenario A: STC ($1000\text{ W/m}^2, 25^\circ\text{C}$) | **99.90%** | 99.88% | -0.02% | 359.78 J | 359.72 J | -0.06 J | 360.15 J | 0.37 J | 0.43 J |
| `scenario_b_irradiance_step` | Scenario B: Irradiance Steps ($1000 \to 600 \to 800 \to 400$) | 91.60% | **97.11%** | **+5.51%** | 232.01 J | 245.96 J | **+13.95 J** | 253.28 J | 21.27 J | 7.31 J |
| `scenario_c_temperature_step` | Scenario C: Temperature Steps ($25 \to 35 \to 45 \to 25$) | **99.83%** | 99.69% | -0.14% | 347.11 J | 346.61 J | -0.50 J | 347.70 J | 0.59 J | 1.09 J |
| `scenario_d_combined` | Scenario D: G-Ramp ($1000 \to 500$) + T-Step ($25 \to 50$) | 94.57% | **99.40%** | **+4.83%** | 244.13 J | 256.60 J | **+12.47 J** | 258.15 J | 14.02 J | 1.55 J |
| `scenario_e_rand_101` | Scenario E.1: Cloud Shading Dip | 90.28% | **96.76%** | **+6.49%** | 241.23 J | 258.57 J | **+17.34 J** | 267.22 J | 25.98 J | 8.65 J |
| `scenario_e_rand_102` | Scenario E.2: Morning Solar Ramp ($400 \to 950$) | 94.60% | **98.25%** | **+3.65%** | 230.12 J | 239.01 J | **+8.88 J** | 243.27 J | 13.14 J | 4.26 J |
| `scenario_e_rand_103` | Scenario E.3: Midday Extreme Heat ($980\text{ W/m}^2, 52^\circ\text{C}$) | **99.56%** | 99.42% | -0.13% | 307.63 J | 307.22 J | -0.41 J | 309.00 J | 1.36 J | 1.78 J |
| `scenario_e_rand_104` | Scenario E.4: Low Irradiance Overcast ($420\text{ W/m}^2$) | 89.15% | **96.91%** | **+7.76%** | 138.45 J | 150.51 J | **+12.06 J** | 155.30 J | 16.85 J | 4.80 J |
| `scenario_e_rand_105` | Scenario E.5: Rapid Solar Fluctuation | 93.00% | **97.28%** | **+4.28%** | 240.00 J | 251.05 J | **+11.05 J** | 258.07 J | 18.07 J | 7.02 J |
| `scenario_e_rand_106` | Scenario E.6: Clear Sky Cool Ambience ($15^\circ\text{C}$) | 98.41% | **99.77%** | **+1.37%** | 370.51 J | 375.67 J | **+5.16 J** | 376.52 J | 6.01 J | 0.85 J |
| `scenario_e_rand_107` | Scenario E.7: Sudden Heavy Shading ($1000 \to 400$) | 90.47% | **97.36%** | **+6.89%** | 209.03 J | 224.96 J | **+15.93 J** | 231.06 J | 22.03 J | 6.10 J |
| `scenario_e_rand_108` | Scenario E.8: Partial Cloud Intermittency | 99.71% | **99.73%** | +0.02% | 275.43 J | 275.48 J | +0.04 J | 276.23 J | 0.80 J | 0.76 J |
| `scenario_e_rand_109` | Scenario E.9: Thermal Drift Warmup ($25 \to 50^\circ\text{C}$) | **99.92%** | 99.68% | -0.24% | 272.64 J | 271.98 J | -0.66 J | 272.84 J | 0.21 J | 0.87 J |
| `scenario_e_rand_110` | Scenario E.10: Dynamic Complex Profile | 96.71% | **98.58%** | **+1.88%** | 249.66 J | 254.50 J | **+4.84 J** | 258.16 J | 8.50 J | 3.66 J |
| `scenario_f_duty_20` | Scenario F.1: Initial Duty $D_0 = 0.20$ | 96.18% | **99.00%** | **+2.83%** | 346.37 J | 356.56 J | **+10.19 J** | 360.15 J | 13.78 J | 3.59 J |
| `scenario_f_duty_30` | Scenario F.2: Initial Duty $D_0 = 0.30$ | 99.31% | **99.72%** | **+0.42%** | 357.65 J | 359.14 J | **+1.50 J** | 360.15 J | 2.50 J | 1.00 J |
| `scenario_f_duty_40` | Scenario F.3: Initial Duty $D_0 = 0.40$ ($D^* \approx 0.3939$) | 99.91% | **99.92%** | +0.01% | 359.82 J | 359.84 J | +0.02 J | 360.15 J | 0.33 J | 0.30 J |
| `scenario_f_duty_50` | Scenario F.4: Initial Duty $D_0 = 0.50$ | 76.95% | **99.42%** | **+22.47%** | 277.13 J | 358.05 J | **+80.91 J** | 360.15 J | 83.01 J | 2.10 J |
| `scenario_f_duty_60` | Scenario F.5: Initial Duty $D_0 = 0.60$ | 50.13% | **97.96%** | **+47.82%** | 180.55 J | 352.79 J | **+172.24 J** | 360.15 J | 179.59 J | 7.36 J |

---

## 7. Statistical Inference & Sample Nuance

### 7.1 Recomputed Statistical Hypothesis Testing
```text
Paired Efficiency Gain (Δη = η_SAC - η_PO):
  - Sample Size:                    N = 19 paired scenarios
  - Mean Difference (Δη):           +6.089 %
  - Median Difference:              +2.829 %
  - Sample Standard Deviation:      11.393 %
  - Standard Error (SE):            2.614 %
  - 95% Confidence Interval:        [+0.597 %, +11.580 %]
  - Parametric Paired t-test:       t = 2.329,  p = 0.0317  (p < 0.05, Significant)
  - Non-Parametric Wilcoxon Test:   W = 18.0,   p = 0.000965 (p < 0.001, Significant)
  - Standardized Effect Size:       Cohen's d = 0.534 (Medium-to-Large Effect)
```

### 7.2 Methodological Nuances and Limits of Statistical Claims
1. **Purposive Benchmark Sample**: The 19 scenarios form a structured test battery designed to probe specific physical phenomena (step changes, ramps, thermal drift, startup transients). While this covers a broad operational envelope, it is not an infinite random sample of natural weather patterns.
2. **Scenario-Level Evidence vs. Training-Seed Variance**: This study provides rigorous evidence that **this specific trained SAC policy (Seed 42)** outperforms P&O across the evaluated test battery. It does not measure the variance that would occur from training multiple RL neural networks across random initialization seeds.
3. **Absence of Universal Superiority**: SAC is **not universally superior** under all conditions. In Scenarios A, C, E.3, and E.9 (static or slowly drifting conditions), P&O achieved $0.02\% - 0.24\%$ higher tracking efficiency. SAC's superiority is concentrated in dynamic, transient, and off-MPP initialization regimes.

---

## 8. Steady-State Ripple & Averaged Plant Considerations

### 8.1 Control Perturbation Ripple vs. Switching Ripple
- Under STC, SAC commands an exact steady-state duty cycle of $D = 0.399762557149$, exhibiting a calculated numerical power ripple of $\Delta P_{ss} = \mathbf{0.00\text{ W}}$ ($< 10^{-14}\text{ W}$).
- **Physical Interpretation**: This indicates complete elimination of **control-induced limit cycling** ($\Delta D \to 0$).
- **Switching Ripple Reality**: In an actual switched DC-DC boost converter, an unavoidable high-frequency switching ripple ($20\text{ kHz}$) exists due to inductor current ripple ($\Delta i_L \approx 0.585\text{ A}$) and capacitor voltage ripple ($\Delta V_{pv} \approx 7.8\text{ mV}$). Because the simulation uses state-space averaging, this switching ripple is deliberately averaged out. SAC eliminates control chattering, not hardware switching ripple.

### 8.2 P&O Limit Cycling Behavior
- Classical P&O produced an exact 3-point limit cycle ($D \in [0.392, 0.394, 0.396]$) under constant STC conditions, producing $\Delta P_{ss} = 0.0274\text{ W}$.
- However, under dynamic conditions (intermittent irradiance, temperature ramps), P&O does not stay in a simple 3-point cycle; it continuously hunts, drifts, and wanders across wide duty spans depending on the interaction between perturbation step size, sampling rate, and environmental slope.

---

## 9. Computational Complexity & Deployment Feasibility

### 9.1 Parameter Counts (Deployment vs. Training)
In embedded microcontroller deployment, **critic networks and target networks are discarded**:
- **Training Parameters (SB3 SAC)**: $343,046$ total parameters (Actor + Twin Critics + Target Critics), of which $205,828$ are updated via gradient descent.
- **Deployment Inference Parameters (Actor Only)**:
  - Input Layer: $8 \times 256 + 256 = 2,304$
  - Hidden Layer: $256 \times 256 + 256 = 65,792$
  - Mean Head: $256 \times 1 + 1 = 257$
  - *(Log-Std Head $256 \times 1 + 1 = 257$ is omitted during deterministic deployment)*
  - **Total Deployed Parameters**: **68,353 parameters** (evaluated) / **68,610 parameters** (total actor). Memory footprint: **$274.4\text{ KB}$** (Float32).

### 9.2 Operation Counts & Execution Time
- **SAC Deterministic Forward Pass**:
  - Layer 1: $2,048$ mults $+ 2,048$ adds $+ 256$ ReLUs $= 4,352$ FLOPs
  - Layer 2: $65,536$ mults $+ 65,536$ adds $+ 256$ ReLUs $= 131,328$ FLOPs
  - Output Head: $256$ mults $+ 256$ adds $= 512$ FLOPs
  - Tanh & scaling: $\approx 15$ FLOPs
  - Input normalization: $\approx 20$ FLOPs
  - **Total SAC Operations**: $\approx \mathbf{136,227\text{ FLOPs}}$ ($\approx 136\text{ kFLOPs}$ per $5\text{ ms}$ control step).
- **P&O Forward Pass**:
  - $P_k = V_k \cdot I_k$, $\Delta V$, $\Delta P$, sign check, duty addition, clamping: **$\approx 7 - 9\text{ FLOPs}$**.
- **Measured CPU Runtime (Python PyTorch / Mac CPU)**:
  - P&O: **$2.17\ \mu\text{s}$** per decision
  - SAC: **$113.0\ \mu\text{s}$** ($0.113\text{ ms}$) per decision
  - **Control Interval Budget**: $T_{ctrl} = 5,000\ \mu\text{s}$ ($5.0\text{ ms}$).
  - SAC consumes only **$2.26\%$** of the available CPU time budget, leaving $> 97.7\%$ processor headroom for system monitoring, protection, and communications.
- **Embedded Hardware Execution Estimates**:
  > *Note: The following are analytical estimates based on architectural clock cycles, not measured laboratory hardware results:*
  - On a 32-bit floating-point microcontroller (e.g. ARM Cortex-M4 / STM32F407 running at $168\text{ MHz}$ executing single-cycle MACs), a forward pass of $136\text{ kFLOPs}$ requires an estimated $\approx 0.8 - 1.2\text{ ms}$ (consuming $\approx 16 - 24\%$ of the $5\text{ ms}$ control budget).
  - On a high-performance C2000 DSP (e.g. TI TMS320F28379D at $200\text{ MHz}$ with dual-core FPU/TMU), deterministic inference is estimated to execute in $< 0.5\text{ ms}$ ($< 10\%$ CPU utilization).

---

## 10. Engineering Conclusions

1. **Quantified Superiority**: The Soft Actor-Critic RL agent achieves a statistically significant performance advantage over Classical P&O:
   - **Arithmetic Mean Efficiency**: **98.73% vs. 92.64%** ($+6.09\%$ gain, $p = 0.0317$, $W = 18.0, p < 0.001$).
   - **Energy-Weighted Aggregate Efficiency**: **98.88% vs. 92.44%** ($+6.44\%$ gain).
   - **Total Energy Harvested**: **$5,604.20\text{ J}$ vs. $5,239.25\text{ J}$** ($+364.95\text{ J}$ / $+6.97\%$ net gain).
   - **Total Energy Deficit**: **$63.47\text{ J}$ vs. $428.41\text{ J}$** ($-85.19\%$ reduction in lost solar energy).
2. **Primary Performance Drivers**:
   - $4\times$ faster tracking recovery under rapid irradiance step transitions (Scenario B: $+5.51\%$).
   - Elimination of false-direction hunting during continuous solar ramps (Scenario D: $+4.83\%$, Scenario E.2: $+3.65\%$).
   - Immediate convergence from non-optimal initial duties (Scenario F.5: $+47.82\%$).
   - Elimination of control-induced perturbation limit cycles.
3. **Practical Trade-Off**: For stationary clear-sky installations, classical P&O performs within $0.15\%$ of SAC with minimal complexity. SAC deployment is specifically recommended for dynamic, mobile, intermittent, or rapidly changing solar environments.
