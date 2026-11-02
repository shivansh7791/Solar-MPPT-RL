import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

os.makedirs("results/diagrams", exist_ok=True)

# Set common font properties
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# ==============================================================================
# DIAGRAM 1: SINGLE-DIODE PV EQUIVALENT CIRCUIT
# ==============================================================================
def create_single_diode_diagram():
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=250)
    fig.patch.set_facecolor("#07182D")
    ax.set_facecolor("#07182D")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Header / Title
    ax.text(6.5, 6.9, "PHYSICAL PV CELL: SINGLE-DIODE 5-PARAMETER MODEL", 
            color="#FFFFFF", fontsize=15, fontweight="bold", ha="center")
    ax.text(6.5, 6.45, "Non-linear Physics Formulation: Canadian Solar CS6P-240P Module", 
            color="#38BDF8", fontsize=11, ha="center")

    # Wire rails
    # Top rail from current source to series resistor
    ax.plot([2.0, 9.2], [5.0, 5.0], color="#94A3B8", lw=3.0)
    # Series resistor Rs between 9.2 and 10.8
    # Bottom rail
    ax.plot([2.0, 11.8], [1.5, 1.5], color="#94A3B8", lw=3.0)
    # Top rail after Rs to output terminal
    ax.plot([10.8, 11.8], [5.0, 5.0], color="#94A3B8", lw=3.0)

    # 1. Photocurrent Source I_ph at x = 2.5
    circle = patches.Circle((2.5, 3.25), 0.75, edgecolor="#FFB800", facecolor="#0F2744", lw=2.5)
    ax.add_patch(circle)
    # Arrow inside circle pointing up
    ax.annotate("", xy=(2.5, 3.75), xytext=(2.5, 2.75),
                arrowprops=dict(arrowstyle="-|>", color="#FFB800", lw=2.5, mutation_scale=15))
    # Sun rays hitting current source
    for angle in [-40, -20, 0]:
        rad = np.radians(angle)
        dx = 0.6 * np.cos(rad)
        dy = 0.6 * np.sin(rad)
        ax.annotate("", xy=(2.0, 3.6), xytext=(2.0 - dx, 3.6 - dy + 0.5),
                    arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=2.0))
    ax.text(1.3, 4.3, "Solar\nPhotons\n(G, T)", color="#F59E0B", fontsize=9, fontweight="bold", ha="center")
    ax.text(2.5, 2.1, r"$I_{ph}$" + "\n(Photo-current)", color="#FFB800", fontsize=11, fontweight="bold", ha="center")
    # Wire connects to rails
    ax.plot([2.5, 2.5], [4.0, 5.0], color="#94A3B8", lw=2.5)
    ax.plot([2.5, 2.5], [1.5, 2.5], color="#94A3B8", lw=2.5)

    # 2. Forward Diode D (I_d) at x = 5.0
    ax.plot([5.0, 5.0], [4.1, 5.0], color="#94A3B8", lw=2.5)
    ax.plot([5.0, 5.0], [1.5, 2.4], color="#94A3B8", lw=2.5)
    # Diode triangle pointing down
    triangle = patches.Polygon([[4.6, 3.7], [5.4, 3.7], [5.0, 2.9]], 
                               closed=True, edgecolor="#38BDF8", facecolor="#0284C7", lw=2.0)
    ax.add_patch(triangle)
    # Diode cathode line
    ax.plot([4.6, 5.4], [2.9, 2.9], color="#38BDF8", lw=3.0)
    # Current arrow
    ax.annotate("", xy=(5.0, 2.0), xytext=(5.0, 2.4),
                arrowprops=dict(arrowstyle="-|>", color="#38BDF8", lw=2.0, mutation_scale=12))
    ax.text(5.0, 4.3, r"$I_d$", color="#38BDF8", fontsize=11, fontweight="bold", ha="center")
    ax.text(5.0, 2.1, r"Diode $D$" + "\n" + r"($I_0, n, V_t$)", color="#38BDF8", fontsize=10, ha="center")

    # 3. Shunt Resistor R_sh at x = 7.5
    ax.plot([7.5, 7.5], [4.1, 5.0], color="#94A3B8", lw=2.5)
    ax.plot([7.5, 7.5], [1.5, 2.4], color="#94A3B8", lw=2.5)
    rect_sh = patches.Rectangle((7.15, 2.4), 0.7, 1.7, edgecolor="#00C9A7", facecolor="#0F2744", lw=2.0)
    ax.add_patch(rect_sh)
    ax.text(7.5, 3.25, r"$R_{sh}$", color="#00C9A7", fontsize=12, fontweight="bold", ha="center", va="center")
    ax.text(7.5, 4.3, r"$I_{sh}$", color="#00C9A7", fontsize=11, fontweight="bold", ha="center")
    ax.text(7.5, 1.9, r"Shunt Loss" + "\n" + r"($R_{sh} \gg 100\ \Omega$)", color="#00C9A7", fontsize=9, ha="center")

    # 4. Series Resistor R_s at y = 5.0, between x = 9.2 and 10.8
    rect_s = patches.Rectangle((9.2, 4.65), 1.6, 0.7, edgecolor="#E2E8F0", facecolor="#0F2744", lw=2.0)
    ax.add_patch(rect_s)
    ax.text(10.0, 5.0, r"$R_s$", color="#FFFFFF", fontsize=12, fontweight="bold", ha="center", va="center")
    ax.text(10.0, 5.55, r"Series Resistance" + "\n" + r"($R_s \approx 0.32\ \Omega$)", color="#CBD5E1", fontsize=9, ha="center")

    # Output terminals (+ and -) at x = 11.8
    term_pos = patches.Circle((11.8, 5.0), 0.12, facecolor="#EF4444", edgecolor="#FFFFFF", lw=1.5)
    term_neg = patches.Circle((11.8, 1.5), 0.12, facecolor="#3B82F6", edgecolor="#FFFFFF", lw=1.5)
    ax.add_patch(term_pos)
    ax.add_patch(term_neg)
    ax.text(12.2, 5.0, "(+) Term", color="#EF4444", fontsize=11, fontweight="bold", va="center")
    ax.text(12.2, 1.5, "(-) Term", color="#3B82F6", fontsize=11, fontweight="bold", va="center")

    # Output Voltage V_pv annotation
    ax.annotate("", xy=(11.8, 4.7), xytext=(11.8, 1.8),
                arrowprops=dict(arrowstyle="<->", color="#F59E0B", lw=2.0))
    ax.text(11.45, 3.25, r"$V_{pv}$", color="#F59E0B", fontsize=13, fontweight="bold", ha="right", va="center")

    # Output Current I_pv arrow
    ax.annotate("", xy=(11.5, 5.0), xytext=(10.9, 5.0),
                arrowprops=dict(arrowstyle="-|>", color="#FFFFFF", lw=2.5, mutation_scale=14))
    ax.text(11.2, 5.35, r"$I_{pv}$", color="#FFFFFF", fontsize=12, fontweight="bold", ha="center")

    # Box for governing equations at bottom
    eq_box = patches.FancyBboxPatch((1.0, 0.25), 11.0, 1.0, boxstyle="round,pad=0.15",
                                    edgecolor="#38BDF8", facecolor="#0B1E36", lw=1.5)
    ax.add_patch(eq_box)
    ax.text(6.5, 0.82, r"$I_{pv} = I_{ph} - I_0 \left[ \exp\left( \frac{V_{pv} + I_{pv} R_s}{n V_t} \right) - 1 \right] - \frac{V_{pv} + I_{pv} R_s}{R_{sh}}$",
            color="#FFFFFF", fontsize=11.5, ha="center", va="center")
    ax.text(6.5, 0.42, r"Parameters: $I_{ph}(G, T)$, Reverse Saturation $I_0(T)$, Diode Ideality $n$, Thermal Voltage $V_t = k T / q$",
            color="#94A3B8", fontsize=9.5, ha="center", va="center")

    plt.tight_layout()
    plt.savefig("results/diagrams/diagram_single_diode.png", facecolor=fig.get_facecolor(), dpi=250)
    plt.close()
    print("Generated results/diagrams/diagram_single_diode.png")

# ==============================================================================
# DIAGRAM 2: SYSTEM ARCHITECTURE BLOCK DIAGRAM
# ==============================================================================
def create_system_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=250)
    fig.patch.set_facecolor("#07182D")
    ax.set_facecolor("#07182D")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(7.0, 7.0, "COMPLETE CLOSED-LOOP SOLAR MPPT CONTROL SYSTEM ARCHITECTURE", 
            color="#FFFFFF", fontsize=15, fontweight="bold", ha="center")
    ax.text(7.0, 6.55, "Physics-Grounded PV Generation, DC-DC Power Stage & Feedback Actuation", 
            color="#38BDF8", fontsize=11, ha="center")

    # Helper function for nice rounded boxes
    def draw_block(x, y, w, h, title, subtitle, border_color, fill_color, text_color="#FFFFFF"):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                     edgecolor=border_color, facecolor=fill_color, lw=2.0)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.62, title, color=text_color, fontsize=11, fontweight="bold", ha="center", va="center")
        ax.text(x + w/2, y + h*0.32, subtitle, color="#94A3B8", fontsize=9, ha="center", va="center")

    # 1. Atmospheric Inputs Box
    draw_block(0.6, 4.4, 2.2, 1.4, "ATMOSPHERE", "G(t) [W/m²]\nT(t) [°C]", "#F59E0B", "#1E293B", "#F59E0B")

    # 2. PV Module Box
    draw_block(3.4, 4.4, 2.5, 1.4, "PV ARRAY", "CS6P-240P\n(5-Param Single Diode)", "#00C9A7", "#0F2744", "#00C9A7")

    # 3. Boost Converter Box
    draw_block(6.8, 4.4, 2.7, 1.4, "BOOST CONVERTER", "L=1mH, C=220uF\nState-Space CCM", "#38BDF8", "#0F2744", "#38BDF8")

    # 4. DC Load Box
    draw_block(10.5, 4.4, 2.2, 1.4, "DC LOAD", "R_load = 20 Ω\nV_out, I_load", "#E2E8F0", "#1E293B", "#FFFFFF")

    # Arrows along power path
    ax.annotate("", xy=(3.4, 5.1), xytext=(2.8, 5.1),
                arrowprops=dict(arrowstyle="-|>", color="#F59E0B", lw=3.0, mutation_scale=16))
    ax.text(3.1, 5.35, "Sunlight", color="#F59E0B", fontsize=9, fontweight="bold", ha="center")

    ax.annotate("", xy=(6.8, 5.1), xytext=(5.9, 5.1),
                arrowprops=dict(arrowstyle="-|>", color="#00C9A7", lw=3.0, mutation_scale=16))
    ax.text(6.35, 5.4, "V_pv, I_pv\nPower Stage", color="#00C9A7", fontsize=9, fontweight="bold", ha="center")

    ax.annotate("", xy=(10.5, 5.1), xytext=(9.5, 5.1),
                arrowprops=dict(arrowstyle="-|>", color="#38BDF8", lw=3.0, mutation_scale=16))
    ax.text(10.0, 5.4, "V_out, P_out\nBoosted DC", color="#38BDF8", fontsize=9, fontweight="bold", ha="center")

    # Sensor Tap from PV line to Controller
    ax.plot([6.35, 6.35], [5.1, 2.5], color="#00C9A7", lw=2.0, ls="--")
    ax.plot([6.35, 4.5], [2.5, 2.5], color="#00C9A7", lw=2.0, ls="--")
    ax.annotate("", xy=(4.5, 2.5), xytext=(5.0, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#00C9A7", lw=2.0, mutation_scale=14))
    ax.text(6.45, 3.6, "Sensors\nV_pv, I_pv", color="#00C9A7", fontsize=9, fontweight="bold")

    # 5. MPPT Controller Box (P&O / SAC)
    box_ctrl = patches.FancyBboxPatch((1.5, 1.6), 3.0, 1.8, boxstyle="round,pad=0.15",
                                      edgecolor="#FFB800", facecolor="#162E4D", lw=2.5)
    ax.add_patch(box_ctrl)
    ax.text(3.0, 2.9, "MPPT CONTROLLER", color="#FFB800", fontsize=12, fontweight="bold", ha="center")
    ax.text(3.0, 2.45, "• Classical P&O Baseline", color="#E2E8F0", fontsize=9.5, ha="center")
    ax.text(3.0, 2.1, "• Soft Actor-Critic (SAC) Agent", color="#38BDF8", fontsize=9.5, fontweight="bold", ha="center")
    ax.text(3.0, 1.8, "T_control = 5.0 ms (200 Hz)", color="#94A3B8", fontsize=8.5, ha="center")

    # 6. PWM Generator Box
    draw_block(6.0, 1.8, 2.3, 1.4, "PWM GENERATOR", "f_sw = 50 kHz\nGate Drive", "#E2E8F0", "#1E293B", "#FFFFFF")

    # Arrow Controller -> PWM
    ax.annotate("", xy=(6.0, 2.5), xytext=(4.5, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#FFB800", lw=2.5, mutation_scale=15))
    ax.text(5.25, 2.75, "Duty Ratio\nD(t)", color="#FFB800", fontsize=9.5, fontweight="bold", ha="center")

    # Arrow PWM -> Boost Converter switch
    ax.plot([7.15, 7.15], [3.2, 4.4], color="#E2E8F0", lw=2.0, ls="-")
    ax.annotate("", xy=(7.15, 4.4), xytext=(7.15, 3.8),
                arrowprops=dict(arrowstyle="-|>", color="#E2E8F0", lw=2.5, mutation_scale=14))
    ax.text(7.25, 3.8, "PWM Pulses\nGate Driver", color="#E2E8F0", fontsize=9, ha="left")

    # Dynamic impedance feedback loop label
    box_fb = patches.FancyBboxPatch((9.0, 1.6), 4.2, 1.8, boxstyle="round,pad=0.15",
                                    edgecolor="#64748B", facecolor="#0B1E36", lw=1.5)
    ax.add_patch(box_fb)
    ax.text(11.1, 3.0, "DYNAMIC IMPEDANCE MATCHING", color="#38BDF8", fontsize=10.5, fontweight="bold", ha="center")
    ax.text(11.1, 2.5, r"$R_{in}(D) = R_{load} \cdot (1 - D)^2$", color="#FFFFFF", fontsize=11, fontweight="bold", ha="center")
    ax.text(11.1, 2.05, "Modulating duty cycle D forces converter input\nresistance to match optimal PV impedance: R_in = V_mp / I_mp",
            color="#94A3B8", fontsize=8.5, ha="center")

    # Bottom summary bar
    sum_bar = patches.FancyBboxPatch((0.6, 0.3), 12.8, 0.85, boxstyle="round,pad=0.12",
                                     edgecolor="#38BDF8", facecolor="#0F2744", lw=1.5)
    ax.add_patch(sum_bar)
    ax.text(7.0, 0.82, "CLOSED-LOOP OPERATION: Sensed (V_pv, I_pv) -> Control Policy selects ΔD -> Converter sets R_in -> PV operating point driven to MPP",
            color="#38BDF8", fontsize=10, fontweight="bold", ha="center")
    ax.text(7.0, 0.52, "SAC Reinforcement Learning achieves smooth, zero-ripple continuous tracking vs. Classical P&O 3-point limit cycle oscillation.",
            color="#E2E8F0", fontsize=9, ha="center")

    plt.tight_layout()
    plt.savefig("results/diagrams/diagram_system_architecture.png", facecolor=fig.get_facecolor(), dpi=250)
    plt.close()
    print("Generated results/diagrams/diagram_system_architecture.png")

# ==============================================================================
# DIAGRAM 3: BOOST CONVERTER POWER STAGE SCHEMATIC & EQUATIONS
# ==============================================================================
def create_boost_converter_diagram():
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=250)
    fig.patch.set_facecolor("#07182D")
    ax.set_facecolor("#07182D")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(6.5, 6.9, "DC-DC BOOST CONVERTER: CIRCUIT SCHEMATIC & DYNAMIC MODEL", 
            color="#FFFFFF", fontsize=15, fontweight="bold", ha="center")
    ax.text(6.5, 6.45, "State-Space Averaged CCM Power Stage with Component Parasitics", 
            color="#38BDF8", fontsize=11, ha="center")

    # Power Circuit Wiring
    # Input rails
    ax.plot([1.2, 3.2], [4.8, 4.8], color="#94A3B8", lw=3.0)
    ax.plot([1.2, 11.2], [1.8, 1.8], color="#94A3B8", lw=3.0) # Bottom ground rail

    # PV input capacitor C_pv at x = 2.2
    ax.plot([2.2, 2.2], [4.8, 3.8], color="#94A3B8", lw=2.5)
    ax.plot([2.2, 2.2], [1.8, 2.8], color="#94A3B8", lw=2.5)
    ax.plot([1.8, 2.6], [3.8, 3.8], color="#38BDF8", lw=3.5) # Top plate
    ax.plot([1.8, 2.6], [2.8, 2.8], color="#38BDF8", lw=3.5) # Bottom plate
    ax.text(2.2, 3.3, r"$C_{pv}$", color="#38BDF8", fontsize=12, fontweight="bold", ha="center", va="center")
    ax.text(2.2, 1.35, r"$470\ \mu\text{F}$", color="#94A3B8", fontsize=9, ha="center")

    # Inductor L with ESR r_L between x = 3.2 and 5.8
    # Inductor loops (arc representations)
    for cx in [3.6, 4.1, 4.6]:
        arc = patches.Arc((cx, 4.8), 0.5, 0.6, angle=0, theta1=0, theta2=180, color="#FFB800", lw=3.0)
        ax.add_patch(arc)
    ax.plot([3.2, 3.35], [4.8, 4.8], color="#94A3B8", lw=2.5)
    ax.plot([4.85, 5.0], [4.8, 4.8], color="#94A3B8", lw=2.5)
    # ESR r_L
    rect_rl = patches.Rectangle((5.0, 4.5), 0.8, 0.6, edgecolor="#FFB800", facecolor="#0F2744", lw=2.0)
    ax.add_patch(rect_rl)
    ax.text(5.4, 4.8, r"$r_L$", color="#FFB800", fontsize=11, fontweight="bold", ha="center", va="center")
    ax.plot([5.8, 6.5], [4.8, 4.8], color="#94A3B8", lw=3.0)
    ax.text(4.1, 5.4, r"Inductor $L = 1.0\text{ mH}$ ($r_L = 50\text{ m}\Omega$)", color="#FFB800", fontsize=10, fontweight="bold", ha="center")
    ax.annotate("", xy=(4.6, 5.05), xytext=(3.6, 5.05),
                arrowprops=dict(arrowstyle="-|>", color="#FFB800", lw=2.5, mutation_scale=14))
    ax.text(4.1, 4.4, r"$i_L(t)$", color="#FFB800", fontsize=11, fontweight="bold", ha="center")

    # MOSFET Switch S at x = 6.5
    ax.plot([6.5, 6.5], [4.8, 3.8], color="#94A3B8", lw=2.5)
    ax.plot([6.5, 6.5], [1.8, 2.8], color="#94A3B8", lw=2.5)
    rect_sw = patches.Rectangle((6.1, 2.8), 0.8, 1.0, edgecolor="#00C9A7", facecolor="#0F2744", lw=2.0)
    ax.add_patch(rect_sw)
    ax.text(6.5, 3.3, r"Switch $S$" + "\n" + r"(MOSFET)", color="#00C9A7", fontsize=9, fontweight="bold", ha="center", va="center")
    # Gate control signal
    ax.annotate("", xy=(6.1, 3.3), xytext=(5.3, 3.3),
                arrowprops=dict(arrowstyle="-|>", color="#00C9A7", lw=2.5, mutation_scale=14))
    ax.text(5.1, 3.6, "PWM (D)", color="#00C9A7", fontsize=10, fontweight="bold", ha="center")

    # Diode D at y = 4.8 between x = 6.5 and 8.5
    ax.plot([6.5, 7.3], [4.8, 4.8], color="#94A3B8", lw=3.0)
    triangle_d = patches.Polygon([[7.3, 4.4], [7.3, 5.2], [8.0, 4.8]], 
                                 closed=True, edgecolor="#38BDF8", facecolor="#0284C7", lw=2.0)
    ax.add_patch(triangle_d)
    ax.plot([8.0, 8.0], [4.4, 5.2], color="#38BDF8", lw=3.5)
    ax.plot([8.0, 11.2], [4.8, 4.8], color="#94A3B8", lw=3.0)
    ax.text(7.65, 5.4, r"Diode $D_1$", color="#38BDF8", fontsize=11, fontweight="bold", ha="center")

    # Output Capacitor C_out at x = 9.2
    ax.plot([9.2, 9.2], [4.8, 3.8], color="#94A3B8", lw=2.5)
    ax.plot([9.2, 9.2], [1.8, 2.8], color="#94A3B8", lw=2.5)
    ax.plot([8.8, 9.6], [3.8, 3.8], color="#38BDF8", lw=3.5)
    ax.plot([8.8, 9.6], [2.8, 2.8], color="#38BDF8", lw=3.5)
    ax.text(9.2, 3.3, r"$C_{out}$", color="#38BDF8", fontsize=12, fontweight="bold", ha="center", va="center")
    ax.text(9.2, 1.35, r"$220\ \mu\text{F}$", color="#94A3B8", fontsize=9, ha="center")

    # Load Resistor R_load at x = 11.2
    ax.plot([11.2, 11.2], [4.8, 4.1], color="#94A3B8", lw=2.5)
    ax.plot([11.2, 11.2], [1.8, 2.5], color="#94A3B8", lw=2.5)
    rect_ld = patches.Rectangle((10.85, 2.5), 0.7, 1.6, edgecolor="#E2E8F0", facecolor="#0F2744", lw=2.0)
    ax.add_patch(rect_ld)
    ax.text(11.2, 3.3, r"$R_{load}$" + "\n" + r"$20\ \Omega$", color="#FFFFFF", fontsize=11, fontweight="bold", ha="center", va="center")

    # Voltage indicators
    ax.annotate("", xy=(1.2, 4.6), xytext=(1.2, 2.0),
                arrowprops=dict(arrowstyle="<->", color="#00C9A7", lw=2.0))
    ax.text(0.9, 3.3, r"$V_{pv}$", color="#00C9A7", fontsize=12, fontweight="bold", ha="right", va="center")

    ax.annotate("", xy=(11.9, 4.6), xytext=(11.9, 2.0),
                arrowprops=dict(arrowstyle="<->", color="#38BDF8", lw=2.0))
    ax.text(12.2, 3.3, r"$V_{out}$", color="#38BDF8", fontsize=12, fontweight="bold", ha="left", va="center")

    # Mathematical State Equations box at bottom
    box_math = patches.FancyBboxPatch((0.8, 0.2), 11.4, 1.05, boxstyle="round,pad=0.15",
                                      edgecolor="#38BDF8", facecolor="#0B1E36", lw=1.5)
    ax.add_patch(box_math)
    ax.text(3.5, 0.82, r"$\frac{di_L}{dt} = \frac{1}{L} \left[ v_{pv} - i_L r_L - (1-D) v_{out} \right]$", 
            color="#FFFFFF", fontsize=11, ha="center")
    ax.text(8.5, 0.82, r"$\frac{dv_{out}}{dt} = \frac{1}{C_{out}} \left[ (1-D) i_L - \frac{v_{out}}{R_{load}} \right]$", 
            color="#FFFFFF", fontsize=11, ha="center")
    ax.text(6.0, 0.42, r"Continuous Conduction Mode (CCM): $V_{out} = \frac{V_{pv}}{1 - D}, \quad I_{out} = (1 - D) I_{pv}, \quad R_{in}(D) = R_{load}(1 - D)^2$",
            color="#38BDF8", fontsize=10, fontweight="bold", ha="center")

    plt.tight_layout()
    plt.savefig("results/diagrams/diagram_boost_converter.png", facecolor=fig.get_facecolor(), dpi=250)
    plt.close()
    print("Generated results/diagrams/diagram_boost_converter.png")

# ==============================================================================
# DIAGRAM 4: CLASSICAL PERTURB & OBSERVE (P&O) FLOWCHART
# ==============================================================================
def create_po_flowchart_diagram():
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=250)
    fig.patch.set_facecolor("#07182D")
    ax.set_facecolor("#07182D")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(6.5, 7.0, "PERTURB & OBSERVE (P&O) MPPT CONTROL ALGORITHM FLOWCHART", 
            color="#FFFFFF", fontsize=15, fontweight="bold", ha="center")
    ax.text(6.5, 6.55, "Classical Gradient Hill-Climbing Logic and Decision Tree", 
            color="#38BDF8", fontsize=11, ha="center")

    # Helper function for decision diamonds
    def draw_diamond(cx, cy, w, h, text, border_color="#FFB800", fill_color="#0F2744"):
        diamond = patches.Polygon([[cx, cy + h/2], [cx + w/2, cy], [cx, cy - h/2], [cx - w/2, cy]],
                                  closed=True, edgecolor=border_color, facecolor=fill_color, lw=2.0)
        ax.add_patch(diamond)
        ax.text(cx, cy, text, color="#FFFFFF", fontsize=10, fontweight="bold", ha="center", va="center")

    # Helper function for process rectangles
    def draw_proc(x, y, w, h, title, subtitle="", border_color="#38BDF8", fill_color="#1E293B"):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                     edgecolor=border_color, facecolor=fill_color, lw=1.8)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.6, title, color="#FFFFFF", fontsize=10, fontweight="bold", ha="center", va="center")
            ax.text(x + w/2, y + h*0.3, subtitle, color="#94A3B8", fontsize=8.5, ha="center", va="center")
        else:
            ax.text(x + w/2, y + h*0.5, title, color="#FFFFFF", fontsize=10, fontweight="bold", ha="center", va="center")

    # 1. Start Box
    draw_proc(5.2, 5.65, 2.6, 0.65, "START / SAMPLE SENSORS", "Measure V(k), I(k)", "#00C9A7", "#0F2744")

    # Arrow Down
    ax.annotate("", xy=(6.5, 4.8), xytext=(6.5, 5.65),
                arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=2.0, mutation_scale=12))

    # 2. Power Calculation Box
    draw_proc(4.9, 4.15, 3.2, 0.65, "COMPUTE POWER & DELTAS", "P(k)=V·I, ΔP=P(k)-P(k-1), ΔV=V(k)-V(k-1)", "#38BDF8", "#0F2744")

    # Arrow Down
    ax.annotate("", xy=(6.5, 3.35), xytext=(6.5, 4.15),
                arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=2.0, mutation_scale=12))

    # 3. Main Decision Diamond: ΔP > 0 ?
    draw_diamond(6.5, 2.85, 2.4, 1.0, "ΔP > 0 ?", "#FFB800", "#162E4D")

    # Branches from ΔP > 0: Left is YES (x=3.5), Right is NO (x=9.5)
    ax.annotate("", xy=(3.5, 2.85), xytext=(5.3, 2.85),
                arrowprops=dict(arrowstyle="-|>", color="#00C9A7", lw=2.0, mutation_scale=12))
    ax.text(4.4, 3.05, "YES (Power Up)", color="#00C9A7", fontsize=9, fontweight="bold", ha="center")

    ax.annotate("", xy=(9.5, 2.85), xytext=(7.7, 2.85),
                arrowprops=dict(arrowstyle="-|>", color="#EF4444", lw=2.0, mutation_scale=12))
    ax.text(8.6, 3.05, "NO (Power Down)", color="#EF4444", fontsize=9, fontweight="bold", ha="center")

    # 4. Sub-decisions: ΔV > 0 ?
    draw_diamond(3.5, 1.9, 2.0, 0.9, "ΔV > 0 ?", "#38BDF8", "#0F2744")
    draw_diamond(9.5, 1.9, 2.0, 0.9, "ΔV > 0 ?", "#38BDF8", "#0F2744")

    ax.plot([3.5, 3.5], [2.85, 2.35], color="#00C9A7", lw=2.0)
    ax.plot([9.5, 9.5], [2.85, 2.35], color="#EF4444", lw=2.0)

    # 4 Action Boxes at bottom
    # Action 1: Left-Left (ΔP>0, ΔV>0) -> Decrement Duty D = D - ΔD
    draw_proc(1.0, 0.45, 2.2, 0.65, "D = D - ΔD", "Reduce Duty", "#00C9A7", "#0F2744")
    # Action 2: Left-Right (ΔP>0, ΔV<=0) -> Increment Duty D = D + ΔD
    draw_proc(3.6, 0.45, 2.2, 0.65, "D = D + ΔD", "Increase Duty", "#00C9A7", "#0F2744")
    # Action 3: Right-Left (ΔP<=0, ΔV>0) -> Increment Duty D = D + ΔD
    draw_proc(7.2, 0.45, 2.2, 0.65, "D = D + ΔD", "Reverse: Up", "#EF4444", "#0F2744")
    # Action 4: Right-Right (ΔP<=0, ΔV<=0) -> Decrement Duty D = D - ΔD
    draw_proc(9.8, 0.45, 2.2, 0.65, "D = D - ΔD", "Reverse: Down", "#EF4444", "#0F2744")

    # Connecting arrows to action boxes
    # Left Diamond branches
    ax.plot([3.5, 2.1], [1.45, 1.45], color="#94A3B8", lw=1.8)
    ax.annotate("", xy=(2.1, 1.1), xytext=(2.1, 1.45),
                arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=1.8, mutation_scale=10))
    ax.text(2.6, 1.55, "YES", color="#94A3B8", fontsize=8.5, fontweight="bold")

    ax.plot([3.5, 4.7], [1.45, 1.45], color="#94A3B8", lw=1.8)
    ax.annotate("", xy=(4.7, 1.1), xytext=(4.7, 1.45),
                arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=1.8, mutation_scale=10))
    ax.text(4.2, 1.55, "NO", color="#94A3B8", fontsize=8.5, fontweight="bold")

    # Right Diamond branches
    ax.plot([9.5, 8.3], [1.45, 1.45], color="#94A3B8", lw=1.8)
    ax.annotate("", xy=(8.3, 1.1), xytext=(8.3, 1.45),
                arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=1.8, mutation_scale=10))
    ax.text(8.7, 1.55, "YES", color="#94A3B8", fontsize=8.5, fontweight="bold")

    ax.plot([9.5, 10.9], [1.45, 1.45], color="#94A3B8", lw=1.8)
    ax.annotate("", xy=(10.9, 1.1), xytext=(10.9, 1.45),
                arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=1.8, mutation_scale=10))
    ax.text(10.3, 1.55, "NO", color="#94A3B8", fontsize=8.5, fontweight="bold")

    # Inherent Limitations Callout Box (Right Side / Top)
    callout = patches.FancyBboxPatch((9.5, 4.3), 3.2, 1.9, boxstyle="round,pad=0.12",
                                     edgecolor="#EF4444", facecolor="#1F1520", lw=1.8)
    ax.add_patch(callout)
    ax.text(11.1, 5.85, "CRITICAL P&O LIMITATIONS", color="#EF4444", fontsize=10, fontweight="bold", ha="center")
    ax.text(11.1, 5.35, "1. Steady-State Ripple:\n   Never settles; oscillates in a\n   3-point limit cycle around MPP.",
            color="#E2E8F0", fontsize=8.5, ha="center")
    ax.text(11.1, 4.65, "2. Ramp Disorientation:\n   Fast irradiance drops cause false\n   perturbations away from MPP.",
            color="#FCA5A5", fontsize=8.5, ha="center")

    plt.tight_layout()
    plt.savefig("results/diagrams/diagram_po_flowchart.png", facecolor=fig.get_facecolor(), dpi=250)
    plt.close()
    print("Generated results/diagrams/diagram_po_flowchart.png")

# ==============================================================================
# DIAGRAM 5: REINFORCEMENT LEARNING (SAC) ARCHITECTURE
# ==============================================================================
def create_rl_sac_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=250)
    fig.patch.set_facecolor("#07182D")
    ax.set_facecolor("#07182D")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(7.0, 7.0, "SOFT ACTOR-CRITIC (SAC) CONTINUOUS RL MPPT ARCHITECTURE", 
            color="#FFFFFF", fontsize=15, fontweight="bold", ha="center")
    ax.text(7.0, 6.55, "Off-Policy Maximum Entropy Actor-Critic Framework with Twin Q-Networks", 
            color="#38BDF8", fontsize=11, ha="center")

    def draw_card(x, y, w, h, title, items, border_color, fill_color, title_color="#FFFFFF"):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                     edgecolor=border_color, facecolor=fill_color, lw=2.0)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.35, title, color=title_color, fontsize=11, fontweight="bold", ha="center")
        y_text = y + h - 0.7
        for itm in items:
            ax.text(x + 0.2, y_text, itm, color="#CBD5E1", fontsize=9, va="center")
            y_text -= 0.32

    # 1. State / Observation Card (Left)
    draw_card(0.8, 1.8, 3.4, 4.2, "OBSERVATION VECTOR s_t", [
        "Normalized State Features in ℝ⁸:",
        "1. v_pv / V_scale  (Scaled Voltage)",
        "2. i_pv / I_scale  (Scaled Current)",
        "3. P_pv / P_scale  (Scaled Power)",
        "4. d_t             (Current Duty Cycle)",
        "5. Δv_pv           (Voltage Gradient)",
        "6. Δi_pv           (Current Gradient)",
        "7. ΔP_pv           (Power Gradient)",
        "8. slew            (Prior Action Memory)",
        "Control Step: T_ctrl = 5.0 ms (200 Hz)"
    ], "#38BDF8", "#0F2744", "#38BDF8")

    # 2. Actor Network Card (Center Top)
    draw_card(4.9, 3.8, 4.2, 2.2, "ACTOR NETWORK π_ϕ(a|s)", [
        "Continuous Multi-Layer Perceptron (MLP):",
        "• 2 Hidden Layers (256 units, ReLU)",
        "• Output: Mean μ_ϕ(s) and Std σ_ϕ(s)",
        "• Action Sample: a_t ~ tanh(N(μ, σ)) ∈ [-1, +1]",
        "• Modulated Duty Slew: ΔD = a_t · ΔD_max"
    ], "#00C9A7", "#132D48", "#00C9A7")

    # 3. Twin Critic Networks Card (Center Bottom)
    draw_card(4.9, 1.3, 4.2, 2.2, "TWIN CRITIC NETWORKS Q_θ1, Q_θ2", [
        "Clipped Double Q-Learning Architecture:",
        "• Q_θ1(s, a), Q_θ2(s, a) (256x256 MLPs)",
        "• Target Value: y = r + γ(min Q_target - α log π)",
        "• Prevents Overestimation Bias in Q-values",
        "• Temperature α auto-tuned for max entropy"
    ], "#FFB800", "#1E2A38", "#FFB800")

    # 4. Plant & Environment Card (Right Top)
    draw_card(9.8, 3.8, 3.5, 2.2, "GYMNASIUM PLANT ENV", [
        "Physics PV-Converter Simulation:",
        "• 5-Parameter Single-Diode Model",
        "• Boost State-Space Integration (RK4)",
        "• Commanded Duty: D_{t+1} = clip(D_t + ΔD)",
        "• Next State s_{t+1} and Reward r_t"
    ], "#F59E0B", "#1C2D42", "#F59E0B")

    # 5. Experience Replay & Reward Card (Right Bottom)
    draw_card(9.8, 1.3, 3.5, 2.2, "REWARD & REPLAY BUFFER D", [
        "Entropy-Regularized Optimization:",
        "• Reward: r_t = P_pv / P_mpp - λ (ΔD)²",
        "• Replay Buffer: 100,000 transitions",
        "• Batch Size: 256 transitions per step",
        "• Discount Factor: γ = 0.99"
    ], "#E2E8F0", "#162334", "#E2E8F0")

    # Connecting Signal Arrows
    # State Arrow to Actor
    ax.annotate("", xy=(4.9, 5.0), xytext=(4.2, 5.0),
                arrowprops=dict(arrowstyle="-|>", color="#38BDF8", lw=2.5, mutation_scale=14))
    ax.text(4.55, 5.25, "s_t", color="#38BDF8", fontsize=11, fontweight="bold", ha="center")

    # State Arrow to Critics
    ax.annotate("", xy=(4.9, 2.4), xytext=(4.2, 2.4),
                arrowprops=dict(arrowstyle="-|>", color="#38BDF8", lw=2.5, mutation_scale=14))
    ax.text(4.55, 2.65, "s_t", color="#38BDF8", fontsize=11, fontweight="bold", ha="center")

    # Action Arrow from Actor to Environment
    ax.annotate("", xy=(9.8, 5.0), xytext=(9.1, 5.0),
                arrowprops=dict(arrowstyle="-|>", color="#00C9A7", lw=2.5, mutation_scale=14))
    ax.text(9.45, 5.25, "a_t (ΔD)", color="#00C9A7", fontsize=11, fontweight="bold", ha="center")

    # Feedback Loop: Next State & Reward from Env to Buffer / State
    ax.plot([11.5, 11.5], [3.8, 3.5], color="#F59E0B", lw=2.0)
    ax.plot([11.5, 9.8], [3.5, 3.5], color="#F59E0B", lw=2.0)
    ax.annotate("", xy=(9.8, 2.8), xytext=(9.8, 3.5),
                arrowprops=dict(arrowstyle="-|>", color="#F59E0B", lw=2.0, mutation_scale=12))

    # Bottom summary
    ax.text(7.0, 0.55, "KEY ADVANTAGE: Continuous action output smoothly drives ΔD -> 0 at the MPP, achieving ZERO steady-state ripple,\nwhile full action slew (±ΔD_max) enables 4x faster dynamic recovery during rapid irradiance shifts.",
            color="#38BDF8", fontsize=9.5, fontweight="bold", ha="center")

    plt.tight_layout()
    plt.savefig("results/diagrams/diagram_rl_sac_architecture.png", facecolor=fig.get_facecolor(), dpi=250)
    plt.close()
    print("Generated results/diagrams/diagram_rl_sac_architecture.png")

if __name__ == "__main__":
    create_single_diode_diagram()
    create_system_architecture_diagram()
    create_boost_converter_diagram()
    create_po_flowchart_diagram()
    create_rl_sac_architecture_diagram()
    print("All 5 presentation diagrams generated successfully.")
