"""Soft Actor-Critic (SAC) Agent Factory and Architecture Specification.

Initializes Stable-Baselines3 SAC agents with configured actor-critic networks,
entropy regularization, and experience replay buffers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import torch

from app.rl.sac_config import SACConfig
from app.utils.logger import get_logger

logger = get_logger("sac_agent")


def create_sac_agent(
    env: gym.Env,
    config: Optional[SACConfig] = None,
    seed: Optional[int] = None,
    tensorboard_log: Optional[str] = None,
    device: Optional[str] = None,
) -> SAC:
    """Instantiate a Stable-Baselines3 SAC agent configured for the solar MPPT environment.

    Parameters
    ----------
    env : gym.Env
        Gymnasium solar MPPT environment (typically wrapped with Monitor).
    config : Optional[SACConfig], optional
        SAC configuration dataclass. If None, default SACConfig is used.
    seed : Optional[int], optional
        Random seed for reproducibility. If None, uses config.seed.
    tensorboard_log : Optional[str], optional
        Directory for TensorBoard logging. If None, uses config.tensorboard_log.
    device : Optional[str], optional
        Target execution device ('cpu', 'mps', 'cuda'). If None, resolved from config.

    Returns
    -------
    SAC
        Initialized Stable-Baselines3 SAC agent.
    """
    cfg = config if config is not None else SACConfig()
    effective_seed = seed if seed is not None else cfg.seed
    effective_device = device if device is not None else cfg.resolve_device()
    effective_tb_log = tensorboard_log if tensorboard_log is not None else cfg.tensorboard_log
    if effective_tb_log is not None:
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            logger.warning("Tensorboard not installed; disabling TensorBoard logging.")
            effective_tb_log = None

    policy_kwargs = cfg.get_policy_kwargs()

    logger.info(
        f"Initializing SAC Agent: Policy={cfg.policy}, LR={cfg.learning_rate}, "
        f"BatchSize={cfg.batch_size}, BufferSize={cfg.buffer_size}, Gamma={cfg.gamma}, "
        f"Tau={cfg.tau}, NetArch Pi={cfg.net_arch_pi}, QF={cfg.net_arch_qf}, "
        f"Device={effective_device}, Seed={effective_seed}"
    )

    agent = SAC(
        policy=cfg.policy,
        env=env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        ent_coef=cfg.ent_coef,
        target_entropy=cfg.target_entropy,
        target_update_interval=cfg.target_update_interval,
        policy_kwargs=policy_kwargs,
        verbose=cfg.verbose,
        seed=effective_seed,
        device=effective_device,
        tensorboard_log=effective_tb_log,
    )

    return agent


def load_sac_agent(
    model_path: Union[Path, str],
    env: Optional[gym.Env] = None,
    device: str = "auto",
) -> SAC:
    """Load a trained SAC checkpoint from disk.

    Parameters
    ----------
    model_path : Union[Path, str]
        Path to the saved zip file.
    env : Optional[gym.Env], optional
        Optional environment to attach to the model.
    device : str, optional
        Target device for model inference.

    Returns
    -------
    SAC
        Loaded model instance.
    """
    path = Path(model_path)
    if not path.is_file() and not Path(f"{model_path}.zip").is_file():
        raise FileNotFoundError(f"SAC model checkpoint not found at: {model_path}")

    logger.info(f"Loading SAC model checkpoint from: {model_path}")
    model = SAC.load(path, env=env, device=device)
    return model


def predict_mppt_action(
    model: SAC,
    observation: np.ndarray,
    deterministic: bool = True,
) -> float:
    """Perform policy inference to obtain normalized duty-cycle adjustment action in [-1, 1].

    Parameters
    ----------
    model : SAC
        Trained or initialized SAC model.
    observation : np.ndarray
        Observation vector matching environment observation space.
    deterministic : bool, optional
        Whether to use deterministic action (mean of Gaussian policy), by default True.

    Returns
    -------
    float
        Scalar action in [-1.0, 1.0].
    """
    action, _states = model.predict(observation, deterministic=deterministic)
    return float(action[0])


def get_agent_summary(model: SAC) -> Dict[str, Any]:
    """Extract model architecture and parameter count diagnostics."""
    actor_params = sum(p.numel() for p in model.actor.parameters() if p.requires_grad)
    critic_params = sum(p.numel() for p in model.critic.parameters() if p.requires_grad)
    target_critic_params = sum(p.numel() for p in model.critic_target.parameters() if p.requires_grad)

    return {
        "algorithm": "SAC",
        "actor_parameters": actor_params,
        "critic_parameters": critic_params,
        "target_critic_parameters": target_critic_params,
        "total_trainable_parameters": actor_params + critic_params,
        "device": str(model.device),
        "learning_rate": float(model.learning_rate),
        "batch_size": int(model.batch_size),
        "buffer_size": int(model.buffer_size),
        "gamma": float(model.gamma),
        "tau": float(model.tau),
    }
