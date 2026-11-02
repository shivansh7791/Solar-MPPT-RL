# Phase 4 Theory: Classical Perturb & Observe (P&O) MPPT Control for Solar PV Systems

**Project Title**: Design and Performance Analysis of a Reinforcement Learning-Based Maximum Power Point Tracking Controller for Solar Photovoltaic Systems  
**Document**: Phase 4 Mathematical & Control Systems Theory  
**Module**: Canadian Solar CS6P-240P Photovoltaic Module Coupled to DC-DC Boost Converter Plant  
**Author**: EEE Minor Project Research Group  
**Status**: APPROVED  

---

## 1. Maximum Power Point Tracking (MPPT) Fundamentals

### 1.1 Non-Linear Photovoltaic Characteristics
A photovoltaic (PV) cell is a non-linear semiconductor current source whose terminal $I$-$V$ characteristic is governed by the single-diode equivalent circuit equation:

$$I = I_{ph} - I_0 \left[ \exp\left(\frac{V + I R_s}{n N_s V_t}\right) - 1 \right] - \frac{V + I R_s}{R_{sh}}$$

The instantaneous electrical power delivered to the converter is:

$$P = V \cdot I$$

Differentiating power with respect to terminal voltage yields:

$$\frac{dP}{dV} = I + V \frac{dI}{dV}$$

At the Maximum Power Point (MPP):

$$\left. \frac{dP}{dV} \right|_{V = V_{mp}} = 0 \iff \left. \frac{dI}{dV} \right|_{V = V_{mp}} = -\frac{I_{mp}}{V_{mp}} = -\frac{1}{R_{mpp}}$$

where:
- $\frac{dP}{dV} > 0$ for $V < V_{mp}$ (constant-current source region, left of MPP).
- $\frac{dP}{dV} < 0$ for $V > V_{mp}$ (constant-voltage region, right of MPP).
- $R_{mpp} = \frac{V_{mp}}{I_{mp}}$ is the optimal dynamic internal load resistance of the PV generator.

For the selected Canadian Solar CS6P-240P module at Standard Test Conditions (STC: $G = 1000\text{ W/m}^2, T = 25^\circ\text{C}$):
- $V_{mp} = 29.900\text{ V}$
- $I_{mp} = 8.030\text{ A}$
- $P_{mp} = 240.097\text{ W}$
- $R_{mpp} = \frac{29.900\text{ V}}{8.030\text{ A}} \approx 3.7235\ \Omega$

### 1.2 Environmental Sensitivity
Solar irradiance $G$ and cell junction temperature $T$ change continuously in field conditions:
1. **Irradiance Dependency**: Photocurrent $I_{ph} \propto G$. As irradiance drops from $1000\text{ W/m}^2$ to $400\text{ W/m}^2$, $I_{mp}$ scales down from $8.03\text{ A}$ to $3.22\text{ A}$, causing $R_{mpp}$ to expand from $3.72\ \Omega$ to $9.27\ \Omega$.
2. **Temperature Dependency**: Reverse saturation current $I_0$ increases exponentially with temperature, shifting the open-circuit voltage $V_{oc}$ downward according to the negative temperature coefficient $\beta_{Voc} \approx -0.34\%/^\circ\text{C}$. Consequently, $V_{mp}$ decreases by approximately $0.10\text{ V/}^\circ\text{C}$.

Because the optimal operating impedance $R_{mpp}(G, T)$ varies dynamically by hundreds of percent, a static load directly connected to a PV module operates at MPP only under a single environmental coincidence.

### 1.3 Impedance Matching via DC-DC Boost Converter
To continuously extract maximum power under all operating conditions, a DC-DC boost converter is inserted between the PV generator and the load ($R_{load} = 10.0\ \Omega$). In Continuous Conduction Mode (CCM), the time-averaged input voltage $V_{pv}$ and output voltage $V_{out}$ are related by:

$$V_{out} = \frac{V_{pv}}{1 - D}$$

Neglecting parasitic losses, ideal power conservation requires $P_{in} \approx P_{out}$:

$$V_{pv} I_{pv} = V_{out} I_{out} = \frac{V_{out}^2}{R_{load}}$$

Substituting the voltage step-up relationship:

$$V_{pv} I_{pv} = \frac{1}{R_{load}} \left(\frac{V_{pv}}{1 - D}\right)^2$$

Dividing both sides by $V_{pv}^2$:

$$\frac{I_{pv}}{V_{pv}} = \frac{1}{(1 - D)^2 R_{load}} \implies R_{in}(D) = \frac{V_{pv}}{I_{pv}} = (1 - D)^2 R_{load}$$

Accounting for the finite inductor series winding resistance $r_L$:

$$R_{in}(D) = (1 - D)^2 R_{load} + r_L$$

By modulating the switching duty cycle $D \in [0.05, 0.90]$, the reflected impedance emulated by the converter spans:
- At $D = 0.05$: $R_{in} = (0.95)^2 (10) + 0.05 = 9.075\ \Omega$ (forces operating point near open-circuit voltage).
- At $D = 0.3939$: $R_{in} = (0.6061)^2 (10) + 0.05 = 3.7235\ \Omega = R_{mpp}$ (matches MPP at STC).
- At $D = 0.90$: $R_{in} = (0.10)^2 (10) + 0.05 = 0.150\ \Omega$ (forces operating point near short-circuit current).

Thus, the MPPT controller's objective is to continuously adjust the duty cycle $D$ such that $R_{in}(D) \equiv R_{mpp}(G, T)$.

---

## 2. The Classical Perturb & Observe (P&O) Algorithm

### 2.1 Working Principle & Hill-Climbing Logic
Perturb & Observe is a hill-climbing search algorithm. At each sampling instant $k$, the controller measures the instantaneous PV terminal voltage $V(k)$ and current $I(k)$, calculates electrical power:

$$P(k) = V(k) \cdot I(k)$$

and evaluates the finite differences from the previous sampling instant $(k-1)$:

$$\Delta V = V(k) - V(k-1)$$

$$\Delta P = P(k) - P(k-1)$$

The gradient of the power-voltage curve is estimated discretely:

$$\frac{\Delta P}{\Delta V} \approx \left. \frac{dP}{dV} \right|_{V(k)}$$

The classical hill-climbing decision logic is:
1. If $\Delta P > 0$, the operating point moved toward the peak of the $P$-$V$ curve. Therefore, continue perturbing terminal voltage in the same direction.
2. If $\Delta P < 0$, the operating point moved away from the peak (overshot the MPP). Therefore, reverse the direction of voltage perturbation.

### 2.2 Classical Decision Table

| $\Delta P$ | $\Delta V$ | Estimated Position on $P$-$V$ Curve | Desired Voltage Action |
| :---: | :---: | :---: | :---: |
| **Positive ($>0$)** | **Positive ($>0$)** | Left of MPP ($dP/dV > 0$) | Increase Voltage ($V_{pv} \uparrow$) |
| **Positive ($>0$)** | **Negative ($<0$)** | Right of MPP ($dP/dV < 0$) | Decrease Voltage ($V_{pv} \downarrow$) |
| **Negative ($<0$)** | **Positive ($>0$)** | Overshot to Right of MPP | Decrease Voltage ($V_{pv} \downarrow$) |
| **Negative ($<0$)** | **Negative ($<0$)** | Overshot to Left of MPP | Increase Voltage ($V_{pv} \uparrow$) |

---

## 3. Boost Converter Duty Cycle Sign Convention

### 3.1 The Inverse Coupling Relationship
A critical error in naive MPPT implementations is assuming that duty cycle $D$ and PV voltage $V_{pv}$ move in the same direction. In a boost converter, the physical coupling is strictly **INVERSE**:

$$\begin{aligned}
D \uparrow &\implies (1 - D) \downarrow \implies R_{in} \downarrow \implies V_{pv} \downarrow \\
D \downarrow &\implies (1 - D) \uparrow \implies R_{in} \uparrow \implies V_{pv} \uparrow
\end{aligned}$$

- **Increasing $D$** closes the MOSFET switch for a larger fraction of the switching period, conducting more current through the inductor to ground. This discharges the PV decoupling capacitor $C_{pv}$, decreasing $V_{pv}$ and drawing higher current toward short circuit ($I_{sc}$).
- **Decreasing $D$** keeps the MOSFET open longer, allowing current to charge the input capacitor $C_{pv}$ and pushing $V_{pv}$ toward open circuit ($V_{oc}$).

### 3.2 Derived Control Law for Boost Converter Actuation
To translate the desired voltage action into a physical duty cycle command:
- To **Increase $V_{pv}$**, the duty cycle must be **DECREASED**: $D(k) = D(k-1) - \delta$.
- To **Decrease $V_{pv}$**, the duty cycle must be **INCREASED**: $D(k) = D(k-1) + \delta$.

Applying this inverse relationship to the classical P&O decision table produces the **Boost Converter P&O Control Rule**:

| $\Delta P$ | $\Delta V$ | Sign of $(\Delta P \cdot \Delta V)$ | Desired $V_{pv}$ Action | Commanded Duty Cycle Action $\Delta D$ |
| :---: | :---: | :---: | :---: | :---: |
| **$> 0$** | **$> 0$** | **Positive ($+$)** | Increase $V_{pv}$ | **$-\delta$ (Decrease $D$)** |
| **$> 0$** | **$< 0$** | **Negative ($-$)** | Decrease $V_{pv}$ | **$+\delta$ (Increase $D$)** |
| **$< 0$** | **$> 0$** | **Negative ($-$)** | Decrease $V_{pv}$ | **$+\delta$ (Increase $D$)** |
| **$< 0$** | **$< 0$** | **Positive ($+$)** | Increase $V_{pv}$ | **$-\delta$ (Decrease $D$)** |

Compact mathematical formulation:

$$\Delta D(k) = -\delta \cdot \operatorname{sgn}\left(\Delta P \cdot \Delta V\right) = -\delta \cdot \operatorname{sgn}\left(\frac{\Delta P}{\Delta V}\right)$$

$$D(k) = \operatorname{clip}\Big(D(k-1) + \Delta D(k), D_{min}, D_{max}\Big)$$

---

## 4. Steady-State Oscillation & Dynamic Trade-Offs

### 4.1 The Three-Point Steady-State Limit Cycle
Once P&O reaches the Maximum Power Point, $\frac{dP}{dV} \approx 0$. Because the perturbation step size $\delta$ is fixed, the controller cannot remain static at $V_{mp}$. 
1. At step $k$, the operating point is at $V_A = V_{mp} - \Delta V$ ($P_A < P_{mp}$).
2. The controller perturbs to the right: $V_B = V_{mp}$ ($P_B = P_{mp} > P_A$). Since $\Delta P > 0$ and $\Delta V > 0$, it perturbs right again.
3. At step $k+1$, it reaches $V_C = V_{mp} + \Delta V$ ($P_C < P_{mp}$). Now $\Delta P < 0$ and $\Delta V > 0$, so it reverses direction.
4. At step $k+2$, it returns to $V_B = V_{mp}$ ($P_B > P_C$). Since $\Delta P > 0$ and $\Delta V < 0$, it continues perturbing left.
5. At step $k+3$, it returns to $V_A$, repeating the cycle indefinitely.

This mandatory 3-point limit cycle produces steady-state power ripple:

$$\Delta P_{ripple} \approx P_{mp} - \min(P_A, P_C)$$

### 4.2 Perturbation Step Size ($\delta$) Trade-Off
The choice of perturbation step $\delta$ represents a fundamental design trade-off in classical P&O:
- **Large step size ($\delta \ge 0.005$)**: Fast transient tracking speed during irradiance ramps, but large steady-state power oscillations around MPP (reducing steady-state tracking efficiency $\eta_{MPPT}$).
- **Small step size ($\delta \le 0.001$)**: Minimal steady-state oscillation and high equilibrium efficiency, but sluggish tracking response during fast environmental variations, leading to prolonged transient energy loss.

### 4.3 Controller Sampling Interval ($T_{mppt}$) vs. Converter Settling Time
The boost converter is a dynamic 2nd-order reactive plant with natural resonant frequency:

$$\omega_0 = \frac{1}{\sqrt{L C_{pv}}} = \frac{1}{\sqrt{1.0\times 10^{-3} \times 470\times 10^{-6}}} \approx 1458.6\text{ rad/s} \implies f_0 \approx 232\text{ Hz}$$

The corresponding resonant time period is $T_0 \approx 4.3\text{ ms}$, with a $2\%$ settling time $t_s \approx 10 - 20\text{ ms}$ for large step transients.
- **Rule of Thumb**: The MPPT sampling interval $T_{mppt}$ must satisfy:

$$T_{mppt} \ge t_{s,small}$$

where $t_{s,small}$ is the converter's settling time under a small perturbation $\delta$.
If $T_{mppt}$ is set too short (e.g., $100\ \mu\text{s}$ equal to the RK4 simulation time step), the controller measures transient capacitor charging/discharging currents rather than steady-state $I$-$V$ characteristics. This leads to erroneous $\Delta P$ signs, false directional decisions, and chaotic limit cycles.
- **Design Decision**: In this implementation, the converter integrates at $\Delta t = 100\ \mu\text{s}$, while the P&O controller updates at $T_{mppt} = 5.0\text{ ms}$ ($50$ integration steps per MPPT decision).

---

## 5. Fast Irradiance Transients and the P&O Drift Phenomenon

A well-documented drawback of conventional P&O is **divergence or drift** under rapid irradiance changes:
- Suppose irradiance increases abruptly during step $k$ ($\Delta G \gg 0$).
- Photocurrent increases, so $P(k) \gg P(k-1)$, yielding $\Delta P > 0$.
- If the controller had previously perturbed $V_{pv}$ downward ($\Delta V < 0$), the decision table evaluates:
  $$\Delta P > 0 \text{ and } \Delta V < 0 \implies \text{Decrease } V_{pv} \implies \Delta D = +\delta$$
- The controller mistakenly attributes the power increase to its voltage decrease, and continues to increase $D$, driving the operating point further away from MPP toward short-circuit ($I_{sc}$).
- This drift persists until the irradiance stabilizes and the controller reverses direction.

Documenting this limitation is essential, as advanced algorithms (such as Reinforcement Learning in Phase 6) are specifically designed to eliminate this drift by learning full state-space value representations.

---

## 6. Performance Evaluation Metrics Formulation

To objectively evaluate MPPT performance, four standardized engineering metrics are defined:

### 6.1 MPPT Tracking Efficiency ($\eta_{MPPT}$)
The ratio of total electrical energy extracted by the converter to the theoretical maximum energy available from the PV module over simulation duration $T$:

$$\eta_{MPPT} = \frac{\int_0^T P_{actual}(t)\,dt}{\int_0^T P_{MPP}(G(t), T(t))\,dt} \times 100\%$$

where $P_{actual}(t) = V_{pv}(t) \cdot I_{pv}(t)$, and $P_{MPP}(G(t), T(t))$ is calculated using the analytical single-diode model solution.

> [!NOTE]
> MPPT tracking efficiency ($\eta_{MPPT}$) must not be confused with converter power conversion efficiency ($\eta_{conv} = P_{out} / P_{pv}$). A system can have $99\%$ converter efficiency but only $85\%$ MPPT tracking efficiency if the controller operates far from MPP.

### 6.2 Settling Time ($t_s$)
The elapsed time from the start of a simulation or from an environmental step disturbance until the PV power enters and remains within a specified tolerance band around $P_{MPP}$:

$$t_s = \min \left\{ t^* \;\middle|\; \forall t \ge t^*, \; |P_{actual}(t) - P_{MPP}(t)| \le \epsilon_{tol} \cdot P_{MPP}(t) \right\}$$

where $\epsilon_{tol} = 0.02$ ($2\%$ tolerance band).

### 6.3 Steady-State Power Oscillation ($\sigma_{P,ss}$ and $\Delta P_{ripple}$)
Evaluated in the steady-state time window $t \in [t_{start}, T]$ after settling:
- **Sample Standard Deviation**:
  $$\sigma_{P,ss} = \sqrt{\frac{1}{N-1} \sum_{k=1}^N \left(P(k) - \bar{P}_{ss}\right)^2}$$
- **Peak-to-Peak Power Ripple**:
  $$\Delta P_{ripple} = \max_{k} P(k) - \min_{k} P(k)$$

### 6.4 Cumulative Tracking Energy Loss ($E_{loss}$)
The total energy wasted due to controller tracking delay and steady-state oscillation:

$$E_{loss} = \int_0^T \Big[ P_{MPP}(t) - P_{actual}(t) \Big] dt \quad [\text{Joules}]$$

---

## 7. Mathematical Summary of Controller Architecture

$$\begin{aligned}
P(k) &= V(k) \cdot I(k) \\
\Delta V(k) &= V(k) - V(k-1) \\
\Delta P(k) &= P(k) - P(k-1) \\
\Delta D(k) &= \begin{cases}
0, & \text{if } |\Delta P(k)| \le \epsilon_P \text{ or } |\Delta V(k)| \le \epsilon_V \\
-\delta \cdot \operatorname{sgn}(\Delta P(k) \cdot \Delta V(k)), & \text{otherwise}
\end{cases} \\
D(k) &= \min\Big(D_{max}, \max\big(D_{min}, D(k-1) + \Delta D(k)\big)\Big)
\end{aligned}$$

where $\epsilon_P \ge 0$ is the optional power deadband and $\epsilon_V \ge 0$ is the optional voltage deadband.
