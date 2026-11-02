# Phase 3: DC-DC Boost Converter Modelling — Theoretical Foundation

## 1. Introduction and Operating Principle

The Maximum Power Point Tracking (MPPT) system requires a controllable power electronic actuator to decouple the photovoltaic (PV) array operating voltage from the load or DC bus voltage. The DC-DC **Boost Converter** (step-up converter) is the standard power electronic interface topology for solar PV systems where the output voltage must exceed or match the input PV voltage ($V_{out} \ge V_{pv}$).

By actively adjusting the converter duty cycle ($D$), the effective input impedance seen by the PV module ($R_{in}$) is dynamically modulated to match the instantaneous PV internal resistance at the Maximum Power Point ($R_{mpp} = V_{mp} / I_{mp}$), thereby extracting peak electrical power under varying irradiance and temperature conditions.

---

## 2. Converter Circuit Topology

The boost converter circuit interfacing the solar PV source and a resistive load is shown below:

```
           +-----------+--------[ L, r_L ]-------+--------|>|---------+--------+--> (+)
           |           |                         |       Diode        |        |
           |           |                        ---                   |        |
         [PV]         === C_pv             Q  ( MOSFET )             === C_out[ ] R_load
        Module         |                     |   |                    |       [ ]
           |           |                         |                    |        |
           +-----------+-------------------------+--------------------+--------+--> (-)
```

### Components and Roles
1. **PV Source ($V_{pv}, I_{pv}$)**: The non-linear semiconductor generator delivering instantaneous terminal current $I_{pv}(V_{pv}, G, T)$.
2. **Input Decoupling Capacitor ($C_{pv}$)**: Buffers high-frequency switching current ripple, maintains smooth PV terminal voltage, and stabilizes the PV operating point.
3. **Power Inductor ($L, r_L$)**: Energy storage inductor with intrinsic parasitic series resistance $r_L$, smoothing input current and enabling step-up voltage boost.
4. **Controlled Switch ($Q$)**: High-frequency power semiconductor switch (MOSFET/IGBT) actuated by pulse-width modulation (PWM) at switching frequency $f_{sw}$ with duty cycle $D$.
5. **Freewheeling Diode ($D_{f}$)**: Fast-recovery power diode providing unidirectional current flow to the output stage during the switch-OFF interval.
6. **Output Filter Capacitor ($C_{out}$)**: Smooths output switching voltage ripple and buffers load transients.
7. **Load ($R_{load}$)**: Equivalent resistive load or DC distribution bus impedance receiving delivered power.

---

## 3. Switching Sub-Intervals (Continuous Conduction Mode)

In Continuous Conduction Mode (CCM), the inductor current $i_L(t)$ remains strictly positive throughout the entire switching cycle ($i_L(t) > 0$). Each switching period $T_{sw} = 1 / f_{sw}$ divides into two sub-intervals:

### Sub-Interval 1: Switch ON ($0 \le t < D \cdot T_{sw}$)
- **Switch State**: MOSFET $Q$ is closed (conducting); diode $D_f$ is reverse-biased (blocking).
- **Inductor Dynamics**: The inductor is connected directly across the input capacitor $C_{pv}$. Inductor current rises linearly, accumulating magnetic energy:
  $$L \frac{di_L}{dt} = V_{pv} - r_L i_L$$
- **Input Capacitor Dynamics**: $C_{pv}$ discharges current into the inductor:
  $$C_{pv} \frac{dV_{pv}}{dt} = I_{pv}(V_{pv}, G, T) - i_L$$
- **Output Capacitor Dynamics**: Diode blocks; $C_{out}$ supplies the entire load current $I_{out} = V_{out} / R_{load}$:
  $$C_{out} \frac{dV_{out}}{dt} = -\frac{V_{out}}{R_{load}}$$

### Sub-Interval 2: Switch OFF ($D \cdot T_{sw} \le t < T_{sw}$)
- **Switch State**: MOSFET $Q$ is open (non-conducting); diode $D_f$ is forward-biased.
- **Inductor Dynamics**: Inductor voltage reverses polarity ($v_L = -L \frac{di}{dt}$); the inductor acts as a boost voltage source in series with $V_{pv}$, transferring stored energy to the output:
  $$L \frac{di_L}{dt} = V_{pv} - V_{out} - r_L i_L$$
- **Input Capacitor Dynamics**: Same as sub-interval 1:
  $$C_{pv} \frac{dV_{pv}}{dt} = I_{pv}(V_{pv}, G, T) - i_L$$
- **Output Capacitor Dynamics**: Inductor current charges $C_{out}$ and powers the load:
  $$C_{out} \frac{dV_{out}}{dt} = i_L - \frac{V_{out}}{R_{load}}$$

---

## 4. State-Space Averaged Differential Equations

Applying the state-space averaging method over one switching cycle:

$$\dot{\mathbf{x}} = D \dot{\mathbf{x}}_{ON} + (1 - D) \dot{\mathbf{x}}_{OFF}$$

Selecting the physical state vector:

$$\mathbf{x}(t) = \begin{bmatrix} V_{pv}(t) \\ i_L(t) \\ V_{out}(t) \end{bmatrix}$$

yields the continuous averaged non-linear differential equations:

$$\boxed{C_{pv} \frac{dV_{pv}}{dt} = I_{pv}(V_{pv}, G, T) - i_L}$$

$$\boxed{L \frac{di_L}{dt} = V_{pv} - (1 - D) V_{out} - r_L i_L}$$

$$\boxed{C_{out} \frac{dV_{out}}{dt} = (1 - D) i_L - \frac{V_{out}}{R_{load}}}$$

### Variable Definitions and Units

| Symbol | Parameter Description | SI Unit |
| :--- | :--- | :--- |
| $V_{pv}$ | Instantaneous PV terminal / input capacitor voltage | $\text{V}$ (Volts) |
| $i_L$ | Instantaneous inductor current | $\text{A}$ (Amperes) |
| $V_{out}$ | Instantaneous output capacitor / load terminal voltage | $\text{V}$ (Volts) |
| $I_{pv}$ | PV module current generated at operating voltage $V_{pv}$ | $\text{A}$ (Amperes) |
| $D$ | Converter duty cycle ($0.05 \le D \le 0.90$) | Dimensionless |
| $C_{pv}$ | Input buffer capacitance | $\text{F}$ (Farads) |
| $L$ | Boost power inductance | $\text{H}$ (Henries) |
| $C_{out}$ | Output filter capacitance | $\text{F}$ (Farads) |
| $R_{load}$ | External resistive load | $\Omega$ (Ohms) |
| $r_L$ | Inductor equivalent series resistance | $\Omega$ (Ohms) |
| $G$ | Incident solar irradiance | $\text{W/m}^2$ |
| $T$ | Solar cell junction temperature | $^\circ\text{C}$ |

---

## 5. Steady-State Equilibrium and Impedance Matching

Setting all state derivatives to zero ($\dot{V}_{pv} = 0, \dot{i}_L = 0, \dot{V}_{out} = 0$):

1. **Inductor Current Equilibrium**:
   $$i_{L,ss} = I_{pv}(V_{pv,ss}, G, T)$$
2. **Output Voltage Equilibrium**:
   $$V_{out,ss} = (1 - D) i_{L,ss} R_{load} = (1 - D) I_{pv,ss} R_{load}$$
3. **PV Voltage and Reflected Input Resistance**:
   Substituting $V_{out,ss}$ into the inductor equation:
   $$V_{pv,ss} = (1 - D) V_{out,ss} + r_L i_{L,ss} = \left[ (1 - D)^2 R_{load} + r_L \right] i_{L,ss}$$
   Therefore, the **effective input resistance** presented by the converter to the PV terminals is:
   $$R_{in}(D) \equiv \frac{V_{pv,ss}}{i_{L,ss}} = (1 - D)^2 R_{load} + r_L$$

### Physical MPPT Mechanism
To operate at the Maximum Power Point, the converter must match the dynamic internal resistance of the solar module:

$$R_{in}(D^*) = R_{mpp} \equiv \frac{V_{mp}}{I_{mp}}$$

$$(1 - D^*)^2 R_{load} + r_L = \frac{V_{mp}}{I_{mp}} \implies D^* = 1 - \sqrt{\frac{\frac{V_{mp}}{I_{mp}} - r_L}{R_{load}}}$$

This confirms that by controlling duty cycle $D$, the converter directly moves the PV operating point across the entire $I$-$V$ / $P$-$V$ curve.

---

## 6. Energy Storage and Instantaneous Power Balance

The total instantaneous energy stored in the reactive components is:

$$E(t) = \frac{1}{2} C_{pv} V_{pv}^2(t) + \frac{1}{2} L i_L^2(t) + \frac{1}{2} C_{out} V_{out}^2(t)$$

Differentiating with respect to time:

$$\frac{dE}{dt} = C_{pv} V_{pv} \frac{dV_{pv}}{dt} + L i_L \frac{di_L}{dt} + C_{out} V_{out} \frac{dV_{out}}{dt}$$

Substituting the governing differential equations:

$$\frac{dE}{dt} = V_{pv} (I_{pv} - i_L) + i_L \left( V_{pv} - (1 - D) V_{out} - r_L i_L \right) + V_{out} \left( (1 - D) i_L - \frac{V_{out}}{R_{load}} \right)$$

Expanding and collecting terms:

$$\frac{dE}{dt} = \underbrace{V_{pv} I_{pv}}_{P_{in}(t)} - \underbrace{V_{pv} i_L + i_L V_{pv}}_{0} - \underbrace{(1 - D) i_L V_{out} + (1 - D) i_L V_{out}}_{0} - \underbrace{r_L i_L^2}_{P_{loss}(t)} - \underbrace{\frac{V_{out}^2}{R_{load}}}_{P_{out}(t)}$$

$$\boxed{P_{in}(t) - P_{out}(t) - P_{loss}(t) = \frac{dE}{dt}}$$

### Physical Conservation Interpretation
- In steady state ($\frac{dE}{dt} = 0$): Input power equals output power plus inductor copper loss:
  $$P_{in} = P_{out} + r_L i_L^2$$
- During transients: Differences between instantaneous input and output power are stored or released by the reactive magnetic and electric fields of $L$, $C_{pv}$, and $C_{out}$. The converter never fabricates or dissipates energy unaccounted for.

---

## 7. Component Sizing Calculations

The converter is engineered specifically for the **Canadian Solar CS6P-240P** ($P_{mp} = 240.1\text{ W}, V_{mp} = 29.9\text{ V}, I_{mp} = 8.03\text{ A}, V_{oc} = 37.0\text{ V}, I_{sc} = 8.59\text{ A}$):

### 1. Load Resistance ($R_{load}$)
Targeting a nominal $48\text{ V}$ to $50\text{ V}$ DC bus at MPP:
$$R_{load} = \frac{V_{out,nom}^2}{P_{mp}} = \frac{49^2}{240.1} \approx 10.0\ \Omega$$
At $R_{load} = 10.0\ \Omega$, the optimal MPP duty cycle is:
$$(1 - D^*)^2 \cdot 10.0 \approx \frac{29.9}{8.03} \approx 3.72\ \Omega \implies 1 - D^* \approx 0.61 \implies D^* \approx 0.39$$
which falls centrally within the accessible duty cycle range $[0.05, 0.90]$.

### 2. Boost Inductor ($L$)
To maintain Continuous Conduction Mode (CCM) with peak-to-peak inductor current ripple $\Delta i_L \le 20\%$ of nominal current at $f_{sw} = 20\text{ kHz}$:
$$L \ge \frac{V_{in} D}{\Delta i_L f_{sw}} = \frac{29.9 \times 0.40}{(0.20 \times 8.03) \times 20{,}000} \approx 0.372\text{ mH}$$
Selecting **$L = 1.0\text{ mH}$** provides an ample safety margin, keeping the converter firmly in CCM down to low irradiance levels ($G = 200\text{ W/m}^2$, $i_L \approx 1.6\text{ A}$).

### 3. Inductor Series Resistance ($r_L$)
A standard 1 mH, 10A power inductor exhibits wire resistance $r_L \approx 0.05\ \Omega$, yielding high converter efficiency ($\eta \approx 98.7\%$ at MPP).

### 4. Input and Output Capacitors ($C_{pv}, C_{out}$)
- Sizing for voltage ripple $\le 1\%$:
  $$C_{out} \ge \frac{I_{out} D}{f_{sw} \Delta V_{out}} = \frac{4.9 \times 0.39}{20{,}000 \times 0.50} \approx 191\ \mu\text{F}$$
- Selecting **$C_{pv} = 470\ \mu\text{F}$** and **$C_{out} = 470\ \mu\text{F}$** provides excellent voltage stability and smooth averaged dynamic responses.

### 5. Simulation Time Step ($dt$)
The LC natural resonant frequency is:
$$f_0 = \frac{1}{2\pi \sqrt{L C_{out}}} = \frac{1}{2\pi \sqrt{10^{-3} \times 470 \times 10^{-6}}} \approx 232\text{ Hz} \quad (T_0 \approx 4.3\text{ ms})$$
Choosing **$dt = 1 \times 10^{-4}\text{ s}$ ($0.1\text{ ms}$)** provides $> 40$ integration points per resonant cycle, ensuring high numerical stability and rapid execution using 4th-order Runge-Kutta (RK4).

---

## 8. Continuous Conduction Mode Boundary & Limitations

### CCM Criterion
The averaged equations remain strictly valid only while:

$$i_L(t) > 0$$

If operating at extreme low irradiance or near-zero duty cycle causes $i_L \le 0$, the diode ceases conducting and the converter enters Discontinuous Conduction Mode (DCM). The model detects and logs DCM events, clamping $i_L \ge 0$.

### Scope Boundaries
This is an **averaged model**. It models state trajectories, settling times, and power balances without high-frequency switching harmonics ($f_{sw} = 20\text{ kHz}$), diode reverse-recovery spikes, or MOSFET gate charge dissipation. This provides the optimal fidelity and execution speed required for continuous-time MPPT simulation and reinforcement learning agent training.
