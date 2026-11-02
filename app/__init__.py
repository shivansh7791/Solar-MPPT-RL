"""RL-Based Solar MPPT Application Package.

This package contains modular subpackages for:
- pv: Physics-based Solar PV array modeling (pvlib / single-diode model)
- converter: DC-DC Boost Converter state-space and averaged models
- controllers: MPPT algorithms including Perturb & Observe (P&O) and RL agents
- environment: Gymnasium custom environment for solar MPPT control
- training: Reinforcement learning training workflows (Soft Actor-Critic)
- experiments: Evaluation suites, benchmarks, and comparison pipelines
- dashboard: Interactive Streamlit analytics dashboard
- utils: Cross-cutting utilities (logging, plotting, metrics calculation)
"""

__version__ = "0.1.0"
