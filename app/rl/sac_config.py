"""Centralized Configuration for Soft Actor-Critic (SAC) Reinforcement Learning Agent.

Defines typed dataclasses for SAC algorithm hyperparameters, network architecture,
entropy regularization, experience replay buffer, training workflow, and checkpointing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


@dataclass
class SACConfig:
    """Hyperparameter and architecture configuration for Stable-Baselines3 SAC."""

    # Algorithm and Policy Specification
    algorithm: str = "SAC"
    policy: str = "MlpPolicy"
    net_arch_pi: List[int] = field(default_factory=lambda: [256, 256])
    net_arch_qf: List[int] = field(default_factory=lambda: [256, 256])
    activation_fn_name: str = "relu"  # 'relu', 'tanh', 'elu'

    # Optimization Hyperparameters
    learning_rate: float = 3.0e-4
    buffer_size: int = 100_000
    learning_starts: int = 2_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1

    # Entropy Regularization
    ent_coef: Union[str, float] = "auto"
    target_entropy: Union[str, float] = "auto"
    target_update_interval: int = 1

    # Training Horizon & Randomization
    total_timesteps: int = 100_000
    seed: int = 42
    device: str = "auto"  # 'auto', 'cpu', 'mps', 'cuda'

    # Environment & Observation Mode
    include_mpp_in_obs: bool = False  # False: Realistic 8-dim; True: Privileged 10-dim
    profile_type: str = "random_constant"  # Training environment atmospheric profile

    # Checkpointing & Evaluation Cadence
    checkpoint_interval: int = 10_000
    eval_interval: int = 5_000
    eval_episodes: int = 5
    verbose: int = 1

    # Directory Paths
    artifacts_dir: Optional[str] = None
    tensorboard_log: Optional[str] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate hyperparameter physical and mathematical sanity."""
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be strictly positive. Received: {self.learning_rate}")
        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive integer. Received: {self.buffer_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive integer. Received: {self.batch_size}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1]. Received: {self.tau}")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1]. Received: {self.gamma}")
        if self.learning_starts < 0:
            raise ValueError(f"learning_starts must be non-negative. Received: {self.learning_starts}")
        if self.total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be strictly positive. Received: {self.total_timesteps}")

    def get_activation_fn(self) -> type[nn.Module]:
        """Resolve PyTorch activation function type."""
        name = self.activation_fn_name.lower().strip()
        if name == "relu":
            return nn.ReLU
        elif name == "tanh":
            return nn.Tanh
        elif name == "elu":
            return nn.ELU
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn_name}. Choose relu, tanh, or elu.")

    def get_policy_kwargs(self) -> Dict[str, Any]:
        """Construct policy_kwargs dictionary formatted for Stable-Baselines3 SAC."""
        return {
            "net_arch": {
                "pi": list(self.net_arch_pi),
                "qf": list(self.net_arch_qf),
            },
            "activation_fn": self.get_activation_fn(),
        }

    def resolve_device(self) -> str:
        """Determine optimal computation device for training."""
        if self.device != "auto":
            return self.device

        # Check MPS (Apple Silicon) or CUDA
        if torch.cuda.is_available():
            return "cuda"
        # Note: Simulation step execution runs in CPU Python/NumPy.
        # For lightweight MLP architectures, CPU typically matches or exceeds MPS
        # due to zero tensor host-device transfer overhead.
        return "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SACConfig:
        """Instantiate SACConfig from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save_json(self, file_path: Path | str) -> None:
        """Save configuration dictionary to JSON file."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
