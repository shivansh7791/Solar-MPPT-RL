# PHASE 6 THEORY: Soft Actor-Critic (SAC) Reinforcement Learning Agent for Solar MPPT

## 1. Executive Summary & Purpose

This document provides the theoretical, physical, and algorithmic foundations governing the **Soft Actor-Critic (SAC)** reinforcement learning agent designed to achieve Maximum Power Point Tracking (MPPT) for a solar photovoltaic (PV) array connected to a DC-DC boost converter.

The objective of Phase 6 is to transition from environment definition (Phase 5) to **autonomous policy optimization**, where a stochastic, entropy-regularized continuous-control actor-critic agent learns an optimal duty-cycle adjustment policy $\pi_\theta(a \mid s)$ through active physical interaction with the non-linear PV-converter plant.

---

## 2. Markov Decision Process (MDP) for Solar MPPT

The continuous-time coupled non-linear system consisting of the single-diode PV model and the state-space averaged boost converter is discretized over the RL control period $T_{RL} = 5.0\text{ ms}$, forming a discrete-time Markov Decision Process:

$$\mathcal{M} = \left\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \right\rangle$$

- **Time Horizon:** $k \in \{0, 1, \dots, K-1\}$, corresponding to physical time $t_k = k \cdot T_{RL}$.
- **State Space $\mathcal{S} \subseteq \mathbb{R}^{d_s}$:** Bounded continuous observation vector.
  - Primary Realistic Mode ($d_s = 8$):
    $$s_t = \left[ \frac{V_{pv}}{V_{scale}}, \frac{I_{pv}}{I_{scale}}, \frac{P_{pv}}{P_{scale}}, D_k, \frac{\Delta V_{pv}}{\Delta V_{scale}}, \frac{\Delta P_{pv}}{\Delta P_{scale}}, \frac{G}{G_{scale}}, \frac{T - T_{ref}}{T_{scale}} \right]^T$$
  - Privileged Mode ($d_s = 10$): Includes theoretical ground-truth $V_{mp} / V_{scale}$ and $P_{mp} / P_{scale}$.
- **Action Space $\mathcal{A} = [-1.0, 1.0]$:** Normalized duty-cycle slew command:
  $$D_{k} = \operatorname{clip}\left(D_{k-1} + a_k \cdot \Delta D_{max},\ D_{min},\ D_{max}\right)$$
  with $\Delta D_{max} = 0.01$, $[D_{min}, D_{max}] = [0.05, 0.90]$.
- **State Transition Probability $\mathcal{P}(s_{k+1} \mid s_k, a_k)$:** Deterministic physical mapping governed by $N_{sub} = 50$ Runge-Kutta 4th-order (RK4) integration steps of the converter differential equations under Zero-Order Hold (ZOH) duty actuation:
  $$\begin{aligned}
  C_{pv} \frac{dV_{pv}}{dt} &= I_{pv}(V_{pv}, G, T) - i_L \\
  L \frac{di_L}{dt} &= V_{pv} - (1 - D) V_{out} - r_L i_L \\
  C_{out} \frac{dV_{out}}{dt} &= (1 - D) i_L - \frac{V_{out}}{R_{load}}
  \end{aligned}$$
- **Reward Function $\mathcal{R}(s_k, a_k, s_{k+1})$:** Inherited from Phase 5 Hybrid Tracking formulation:
  $$r_t = w_p \left( \frac{P_{pv}}{P_{mpp} + \epsilon} \right) - w_e \left( \frac{P_{mpp} - P_{pv}}{P_{mpp} + \epsilon} \right)^2 - w_d (a_k)^2 - r_{fault} \cdot \mathbb{I}_{fault}$$
- **Discount Factor:** $\gamma = 0.99$.

---

## 3. Why Soft Actor-Critic (SAC)?

Classical MPPT algorithms (such as P&O) suffer from an unavoidable trade-off between tracking speed and steady-state oscillation. Standard off-policy RL algorithms (such as DDPG) suffer from extreme brittleness, hyperparameter sensitivity, and policy collapse in continuous power electronics control.

SAC is uniquely suited for solar MPPT due to three key properties:
1. **Off-Policy Sample Efficiency:** Reuses past experience via a large replay buffer ($\mathcal{D} = 100,000$ transitions), critical because physical simulation steps carry numerical integration cost.
2. **Maximum Entropy Reinforcement Learning:** Optimizes a trade-off between expected return and policy entropy $\mathcal{H}(\pi(\cdot \mid s))$, preventing the agent from collapsing prematurely into sub-optimal local duty cycles.
3. **Clipped Double-Q Learning (Twin Critics):** Mitigates positive value estimation bias, ensuring stable Q-value convergence on the non-linear $P$-$V$ curve.

---

## 4. Maximum Entropy Objective

In standard RL, the policy maximizes expected cumulative return $\mathbb{E}_{\pi}[\sum_t \gamma^t r_t]$. In contrast, Soft Actor-Critic maximizes the **entropy-augmented objective**:

$$J(\pi) = \sum_{t=0}^{\infty} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ \gamma^t \left( r(s_t, a_t) + \alpha \mathcal{H}\left(\pi(\cdot \mid s_t)\right) \right) \right]$$

where:
- $\mathcal{H}(\pi(\cdot \mid s_t)) = \mathbb{E}_{a_t \sim \pi} [-\log \pi(a_t \mid s_t)]$ is the Shannon entropy of the stochastic policy at state $s_t$.
- $\alpha > 0$ is the **temperature (entropy coefficient)** determining the relative importance of exploration versus power exploitation.
- In Stable-Baselines3, $\alpha$ is **automatically tuned** via dual gradient descent to maintain target entropy $\overline{\mathcal{H}} = -\dim(\mathcal{A}) = -1.0$:
  $$J(\alpha) = \mathbb{E}_{a_t \sim \pi_t} \left[ -\alpha \log \pi_t(a_t \mid s_t) - \alpha \overline{\mathcal{H}} \right]$$

```mermaid
graph LR
    A["State s_t"] --> B["Actor Network pi_theta"]
    B --> C["Gaussian Mean mu(s) & Std sigma(s)"]
    C --> D["Tanh Squashing: a_t in [-1, 1]"]
    D --> E["Twin Critics Q_phi1, Q_phi2"]
    E --> F["Min Q-Value: min(Q1, Q2)"]
    F --> G["Soft Bellman Target Update"]
```

---

## 5. Network Architecture and Component Specifications

### 5.1 Actor (Policy) Network $\pi_\theta(a \mid s)$
- Input layer: $d_s = 8$ (realistic mode) or $10$ (privileged mode).
- Hidden layers: Multi-Layer Perceptron (MLP) with two fully connected hidden layers of 256 units each:
  $$\text{FC}(d_s, 256) \to \text{ReLU} \to \text{FC}(256, 256) \to \text{ReLU}$$
- Output heads: Two parallel linear heads outputting mean $\mu_\theta(s)$ and log standard deviation $\log \sigma_\theta(s)$.
- Bounded Continuous Action Sampling: Uses the reparameterization trick with a squashing hyperbolic tangent function:
  $$u \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s)), \quad a = \tanh(u) \in (-1, 1)$$
- Log-Probability Correction:
  $$\log \pi_\theta(a \mid s) = \log \mu(u \mid s) - \sum_{i=1}^{d_a} \log \left( 1 - \tanh^2(u_i) + \delta \right)$$

### 5.2 Twin Critic Networks $Q_{\phi_1}(s, a)$ and $Q_{\phi_2}(s, a)$
- Input layer: Concatenation of state and action $[s, a] \in \mathbb{R}^{d_s + 1}$.
- Hidden layers: Two independent MLPs, each with $[256, 256]$ hidden units with ReLU activations.
- Output: Scalar state-action value $Q(s, a)$.
- Target Critic Networks: Maintained as Polyak-averaged copies $Q_{\overline{\phi}_1}, Q_{\overline{\phi}_2}$.

---

## 6. SAC Loss Formulations and Optimization

### 6.1 Critic Loss Function
The twin critics are trained to minimize the mean squared Bellman error over mini-batches $\mathcal{B} \sim \mathcal{D}$:

$$L_Q(\phi_i) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{B}} \left[ \frac{1}{2} \left( Q_{\phi_i}(s, a) - y(r, s', d) \right)^2 \right], \quad i \in \{1, 2\}$$

where the soft Bellman target value $y$ is computed using the target networks and entropy bonus:

$$y(r, s', d) = r + \gamma (1 - d) \left( \min_{j \in \{1, 2\}} Q_{\overline{\phi}_j}(s', a') - \alpha \log \pi_\theta(a' \mid s') \right), \quad a' \sim \pi_\theta(\cdot \mid s')$$

### 6.2 Actor Loss Function
The actor parameters $\theta$ are updated to maximize expected return and entropy:

$$L_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{B}, a \sim \pi_\theta(\cdot \mid s)} \left[ \alpha \log \pi_\theta(a \mid s) - \min_{j \in \{1, 2\}} Q_{\phi_j}(s, a) \right]$$

### 6.3 Soft Target Updates (Polyak Averaging)
Target critic parameters are smoothly tracked using exponential moving average parameter $\tau = 0.005$:

$$\overline{\phi}_j \leftarrow \tau \phi_j + (1 - \tau) \overline{\phi}_j, \quad j \in \{1, 2\}$$

---

## 7. Hyperparameter Justification

| Hyperparameter | Value | Technical Justification |
|---|---:|---|
| **Learning Rate** | $3.0 \times 10^{-4}$ | Standard Adam learning rate ensuring stable gradient descent without divergence on the steep $P$-$V$ curve. |
| **Replay Buffer Size** | $100,000$ | Retains $\approx 333$ complete episodes, ensuring diverse decorrelated experiences across all atmospheric profiles. |
| **Batch Size** | $256$ | Balances GPU/CPU vectorization efficiency with stochastic gradient noise reduction. |
| **Discount Factor ($\gamma$)** | $0.99$ | Far-sighted horizon ($1 / (1 - \gamma) = 100$ steps $\approx 0.5\text{ s}$), matching converter dynamic settling time. |
| **Polyak Factor ($\tau$)** | $0.005$ | Slow, smooth target network updates preventing destabilizing value feedback loops. |
| **Learning Starts** | $2,000$ | Collects initial random exploration samples to seed the replay buffer before gradient updates commence. |
| **Target Entropy** | $-1.0$ | Automatically set to $-\dim(\mathcal{A}) = -1.0$, maintaining adequate exploration around the MPP peak. |
| **Hidden Layers** | $[256, 256]$ | Sufficient representational capacity to capture the non-linear Lambert-W PV characteristic and converter impedance. |
| **Activation** | ReLU | Non-saturating piecewise linear activation preventing vanishing gradients during deep backpropagation. |

---

## 8. Reproducibility & Computing Architecture

- **Random Seeds:** Explicit seeding applied to Python `random`, `numpy.random`, `torch.manual_seed`, Gymnasium environment, and SB3 algorithm.
- **Hardware Device:** CPU selected for simulation-in-the-loop training. Because each control step integrates 50 RK4 sub-steps of the physical converter model on the CPU, running neural network updates on CPU avoids host-to-device PCI-e tensor transfer bottlenecks.
- **Artifact Management:** Checkpoints, TensorBoard telemetry, and evaluation metrics are isolated in `artifacts/rl/realistic/` and `artifacts/rl/privileged/`.
