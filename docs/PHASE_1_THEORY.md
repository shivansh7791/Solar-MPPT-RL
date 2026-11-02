# Phase 1: Photovoltaic Mathematical Modelling — Theoretical Foundation

## 1. Introduction and Physical Principle

A solar photovoltaic (PV) cell is a semiconductor $p$-$n$ junction device that converts electromagnetic photon energy directly into electrical energy via the photovoltaic effect. When incident photons possess energy greater than the semiconductor bandgap energy ($h\nu > E_g$), electron-hole pairs are generated within and near the depletion region. The internal electric field established by the ionized donor and acceptor atoms sweeps the photogenerated minority carriers across the junction, creating a photogenerated current ($I_{ph}$) directed from the $n$-side to the $p$-side.

When an external load is connected across the terminals, this current creates a forward-bias voltage across the junction, inducing a recombination current (diode diffusion current) flowing in opposition to the photocurrent. Real PV cells also exhibit parasitic resistive losses: series resistance ($R_s$) arising from semiconductor bulk resistivity, metal contacts, and busbars; and shunt resistance ($R_{sh}$) arising from junction leakage paths, crystal defects, and edge shunting.

---

## 2. Equivalent Circuit of the Single-Diode Model

The single-diode 5-parameter model represents the PV cell using the classical equivalent circuit depicted below:

```
          +----------------------+---------------+-------------+--> (+)
          |                      |               |             |
          |                      |               |            [ ] R_s
          |                      |               |            [ ]
         (^) I_ph               --- D           [ ] R_sh       |
          |                     / \             [ ]            |
          |                    -----             |             |
          |                      |               |             |
          +----------------------+---------------+-------------+--> (-)
```

The circuit comprises:
1. **Photocurrent Source ($I_{ph}$)**: A constant current source proportional to incident solar irradiance and cell temperature.
2. **Junction Diode ($D$)**: Represents carrier diffusion, charge recombination across the $p$-$n$ space-charge junction, and forward-bias behavior.
3. **Series Resistance ($R_s$)**: Models the internal ohmic resistance of the semiconductor bulk material, metallic grid contacts, busbar connections, and emitter sheet resistance.
4. **Shunt Resistance ($R_{sh}$)**: Models parallel leakage currents across the $p$-$n$ junction, peripheral cell borders, and crystal lattice defects.

---

## 3. Governing Mathematical Equation

Applying Kirchhoff's Current Law (KCL) at the internal node yields the terminal output current:

$$I = I_{ph} - I_D - I_{sh}$$

Substituting Shockley's diode equation for $I_D$ and Ohm's law across $R_{sh}$:

$$I = I_{ph} - I_0 \left[ \exp\left( \frac{V + I R_s}{n N_s V_t} \right) - 1 \right] - \frac{V + I R_s}{R_{sh}}$$

### Variable Definitions and Units

| Symbol | Definition | SI Unit |
| :--- | :--- | :--- |
| $I$ | Net terminal current delivered to the external load | $\text{A}$ (Amperes) |
| $V$ | Net terminal voltage across module terminals | $\text{V}$ (Volts) |
| $I_{ph}$ | Photogenerated current (light current) | $\text{A}$ (Amperes) |
| $I_0$ | Diode reverse dark saturation current | $\text{A}$ (Amperes) |
| $R_s$ | Module equivalent series resistance | $\Omega$ (Ohms) |
| $R_{sh}$ | Module equivalent shunt (parallel) resistance | $\Omega$ (Ohms) |
| $n$ | Diode ideality factor (dimensionless, typically $1.0 \le n \le 1.8$ for silicon) | Unitless |
| $N_s$ | Number of series-connected individual solar cells in the module | Unitless |
| $V_t$ | Cell thermal voltage: $V_t = \frac{k T_{cell}}{q}$ | $\text{V}$ (Volts) |
| $k$ | Boltzmann constant: $1.380649 \times 10^{-23}$ | $\text{J/K}$ |
| $q$ | Elementary electron charge: $1.602176634 \times 10^{-19}$ | $\text{C}$ |
| $T_{cell}$ | Absolute cell junction temperature | $\text{K}$ (Kelvin) |

In the standard De Soto / CEC formulation implemented in `pvlib`, the modified diode ideality factor in voltage units is defined at the module level as:

$$a \equiv n N_s V_t = \frac{n N_s k T_{cell}}{q}$$

Thus, the governing transcendental equation simplifies to:

$$I = I_{ph} - I_0 \left[ \exp\left( \frac{V + I R_s}{a} \right) - 1 \right] - \frac{V + I R_s}{R_{sh}}$$

---

## 4. Temperature and Irradiance Parameter Dependencies

Under variable environmental conditions—irradiance $G$ [$\text{W/m}^2$] and cell temperature $T_{cell}$ [$\text{K}$]—the five parameters translate according to the De Soto et al. (2006) formulation:

### 1. Photogenerated Current ($I_{ph}$)
Photocurrent is directly proportional to absorbed photon flux and exhibits a small positive temperature coefficient ($\alpha_{sc}$) due to thermal bandgap narrowing:

$$I_{ph}(G, T_{cell}) = \frac{G}{G_{ref}} \left[ I_{L,ref} + \alpha_{sc} (T_{cell} - T_{ref}) \right]$$

where:
- $G_{ref} = 1000\text{ W/m}^2$ (Standard Test Condition irradiance)
- $T_{ref} = 298.15\text{ K}$ ($25^\circ\text{C}$)
- $\alpha_{sc}$ is the short-circuit current temperature coefficient [$\text{A/K}$]

### 2. Diode Reverse Saturation Current ($I_0$)
The reverse saturation current is governed by minority carrier diffusion and intrinsic carrier concentration ($n_i^2 \propto T^3 \exp(-E_g / k T)$):

$$I_0(T_{cell}) = I_{o,ref} \left( \frac{T_{cell}}{T_{ref}} \right)^3 \exp\left[ \frac{q E_g(T_{ref})}{k T_{ref}} - \frac{q E_g(T_{cell})}{k T_{cell}} \right]$$

The temperature-dependent semiconductor bandgap energy $E_g(T)$ for crystalline silicon is modeled as:

$$E_g(T_{cell}) = E_{g,ref} \left[ 1 - 0.0002677 (T_{cell} - T_{ref}) \right]$$

with $E_{g,ref} \approx 1.121\text{ eV}$ for silicon at $298.15\text{ K}$.

### 3. Diode Thermal / Ideality Parameter ($a$)
The thermal voltage scales strictly linearly with absolute temperature:

$$a(T_{cell}) = a_{ref} \left( \frac{T_{cell}}{T_{ref}} \right)$$

### 4. Shunt Resistance ($R_{sh}$)
In real semiconductor junctions, leakage current paths across defect states depend inversely on incident irradiance:

$$R_{sh}(G) = R_{sh,ref} \left( \frac{G_{ref}}{G} \right)$$

As irradiance decreases, shunt resistance increases, reflecting reduced recombination pathway activation.

### 5. Series Resistance ($R_s$)
Series resistance $R_s$ is assumed invariant with respect to irradiance and operating temperature:

$$R_s = R_{s,ref}$$

---

## 5. Key Electrical Benchmark Points

From the transcendental $I$-$V$ equation, five primary operating points characterize the PV module:

1. **Short-Circuit Point ($V = 0, I = I_{sc}$)**:
   Maximum current output when module terminals are directly shorted:
   $$I_{sc} \approx I_{ph} - I_0 \left[ \exp\left(\frac{I_{sc} R_s}{a}\right) - 1 \right] - \frac{I_{sc} R_s}{R_{sh}}$$

2. **Open-Circuit Point ($I = 0, V = V_{oc}$)**:
   Maximum voltage across terminals when net current is zero:
   $$0 = I_{ph} - I_0 \left[ \exp\left(\frac{V_{oc}}{a}\right) - 1 \right] - \frac{V_{oc}}{R_{sh}}$$
   Since $I_{ph} \gg I_0$ and $R_{sh} \gg R_s$, the approximate analytical relation is:
   $$V_{oc} \approx a \ln\left( \frac{I_{ph}}{I_0} + 1 \right)$$

3. **Maximum Power Point ($V_{mp}, I_{mp}, P_{mp}$)**:
   The operating point where electric power output $P(V) = V \cdot I(V)$ is maximized:
   $$\left. \frac{dP}{dV} \right|_{V_{mp}} = I_{mp} + V_{mp} \left. \frac{dI}{dV} \right|_{V_{mp}} = 0 \implies \left. \frac{dI}{dV} \right|_{V_{mp}} = -\frac{I_{mp}}{V_{mp}}$$
   $$P_{mp} = V_{mp} \cdot I_{mp}$$

4. **Fill Factor ($FF$)**:
   The ratio of maximum deliverable power to the theoretical maximum rectangular product:
   $$FF = \frac{P_{mp}}{V_{oc} \cdot I_{sc}} = \frac{V_{mp} \cdot I_{mp}}{V_{oc} \cdot I_{sc}}$$

---

## 6. Analytical Solution via Lambert $W$ Function

Because the single-diode equation is implicit and transcendental in both $I$ and $V$, evaluating $I(V)$ or finding $(V_{mp}, I_{mp})$ historically required iterative numerical root-finding (e.g. Newton-Raphson). However, Jain & Kapoor (2004) proved that the single-diode equation possesses an exact closed-form analytical solution expressed in terms of the **Lambert $W$ function** ($W(z)$, the principal branch of $w e^w = z$):

$$I(V) = \frac{R_{sh}(I_{ph} + I_0) - V}{R_s + R_{sh}} - \frac{a}{R_s} W\left( \frac{R_s R_{sh} I_0}{a (R_s + R_{sh})} \exp\left[ \frac{R_{sh}(V + R_s(I_{ph} + I_0))}{a (R_s + R_{sh})} \right] \right)$$

In `pvlib.pvsystem.singlediode`, the Lambert $W$ formulation is evaluated via SciPy's vectorized `scipy.special.lambertw`. This provides machine-precision root evaluation without divergence, convergence failure, or initial guess sensitivity.

---

## 7. Assumptions and Limitations of the Single-Diode Model

### Assumptions
1. **Uniform Cell Behavior**: All $N_s$ cells in the module are assumed identical and subjected to uniform irradiance and temperature (no localized hot-spots or partial cell mismatch in this basic module formulation).
2. **Constant Ideality Factor**: The diode ideality factor $n$ is assumed constant across the operational voltage range.
3. **Single Recombination Mechanism**: Recombination is approximated via a single equivalent diode, combining band-to-band and trap-assisted recombination into a single effective parameter $a$.
4. **Linear Photocurrent Scaling**: $I_{ph}$ scales strictly linearly with irradiance $G$.

### Limitations
1. **Low-Irradiance Deviation**: Under very low irradiance ($G < 100\text{ W/m}^2$), recombination in the space-charge region can dominate over diffusion, where a two-diode model ($D_1$ diffusion, $D_2$ space-charge recombination) exhibits slightly higher empirical fidelity.
2. **Partial Shading Inability without String Discretization**: A single lumped equivalent circuit represents uniform conditions. Modeling multi-peaked $P$-$V$ curves under partial shading requires sub-string discretization with bypass diodes (to be formulated in Phase 4/6).
3. **High-Temperature Non-Linearities**: Series resistance and recombination parameters may exhibit slight non-linear temperature drifts above $75^\circ\text{C}$ not fully captured by first-order coefficients.

### Justification for Project Selection
The single-diode 5-parameter model provides the optimal balance between physical accuracy and computational efficiency for dynamic simulation and reinforcement learning control. It is backed by verified commercial parameter databases (NREL SAM / California Energy Commission) and standard open-source numerical libraries (`pvlib`), ensuring reproducibility and engineering rigor.
