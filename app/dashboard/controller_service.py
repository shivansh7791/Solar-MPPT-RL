"""Controller Service for Streamlit Dashboard.

Manages checkpoint loading, model validation, observation dimension checking,
and adapter instantiation for Classical P&O and Trained Soft Actor-Critic (SAC).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st

from app.evaluation.controller_adapters import (
    BaseMPPTController,
    POControllerAdapter,
    SACControllerAdapter,
)
from app.rl.sac_agent import load_sac_agent
from app.utils.logger import get_logger
from config.default_config import MasterConfig, POConfig, get_default_config

logger = get_logger("controller_service")


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata describing a validated SAC checkpoint."""

    name: str
    path: Path
    mode: str  # 'realistic' (8-dim) or 'privileged' (10-dim)
    obs_dim: int
    total_actor_params: int
    active_eval_params: int
    description: str


class ControllerService:
    """Service managing controller instantiation and neural network validation."""

    DEFAULT_REALISTIC_REL_PATH = "artifacts/rl/realistic/best_model/best_mppt_model.zip"
    FALLBACK_REALISTIC_REL_PATH = "artifacts/rl/realistic/final_model/sac_mppt_final.zip"
    PRIVILEGED_REL_PATH = "artifacts/rl/privileged/best_model/best_mppt_model.zip"

    def __init__(self, config: Optional[MasterConfig] = None) -> None:
        self.config = config if config is not None else get_default_config()
        self.root_dir = self.config.paths.root_dir

    def get_available_checkpoints(self) -> Dict[str, CheckpointMetadata]:
        """Discover and return all validated SAC model checkpoints on disk."""
        checkpoints: Dict[str, CheckpointMetadata] = {}

        # 1. Realistic 8-feature checkpoint (Production Default)
        p_real_best = self.root_dir / self.DEFAULT_REALISTIC_REL_PATH
        if p_real_best.is_file():
            checkpoints["Realistic SAC (Best - 8 Obs) [Recommended]"] = CheckpointMetadata(
                name="Realistic SAC (Best - 8 Obs) [Recommended]",
                path=p_real_best,
                mode="realistic",
                obs_dim=8,
                total_actor_params=68610,
                active_eval_params=68353,
                description="Audited production checkpoint trained on 8 realistic measurements without privileged converter states.",
            )

        p_real_final = self.root_dir / self.FALLBACK_REALISTIC_REL_PATH
        if p_real_final.is_file():
            checkpoints["Realistic SAC (Final Step 10k - 8 Obs)"] = CheckpointMetadata(
                name="Realistic SAC (Final Step 10k - 8 Obs)",
                path=p_real_final,
                mode="realistic",
                obs_dim=8,
                total_actor_params=68610,
                active_eval_params=68353,
                description="Final training step checkpoint with 8 realistic observation dimensions.",
            )

        # 2. Privileged 10-feature checkpoint (Experimental)
        p_priv_best = self.root_dir / self.PRIVILEGED_REL_PATH
        if p_priv_best.is_file():
            checkpoints["Privileged SAC (10 Obs) [Experimental]"] = CheckpointMetadata(
                name="Privileged SAC (10 Obs) [Experimental]",
                path=p_priv_best,
                mode="privileged",
                obs_dim=10,
                total_actor_params=69122,
                active_eval_params=68865,
                description="Experimental oracle policy requiring privileged inductor current and output voltage observations.",
            )

        return checkpoints

    def get_default_checkpoint_path(self) -> Path:
        """Return absolute path to the default realistic SAC checkpoint."""
        best = self.root_dir / self.DEFAULT_REALISTIC_REL_PATH
        if best.is_file():
            return best
        fallback = self.root_dir / self.FALLBACK_REALISTIC_REL_PATH
        if fallback.is_file():
            return fallback
        raise FileNotFoundError(
            f"No realistic SAC checkpoint found at {best} or {fallback}."
        )

    def load_sac_model(self, checkpoint_path: Optional[Path] = None) -> Any:
        """Load SAC model checkpoint with resource caching."""
        target_path = checkpoint_path if checkpoint_path is not None else self.get_default_checkpoint_path()
        return _cached_load_model(str(target_path))

    def validate_sac_observation_dim(self, model: Any, expected_dim: int = 8) -> None:
        """Verify model observation space dimension matches expected configuration."""
        obs_shape = model.observation_space.shape
        if not obs_shape or obs_shape[0] != expected_dim:
            raise ValueError(
                f"Observation dimension mismatch: Model has observation dimension {obs_shape}, "
                f"but controller service expected {expected_dim}. "
                f"Realistic mode requires exactly 8 dimensions; Privileged mode requires 10 dimensions."
            )

    def create_po_adapter(self, po_config: Optional[POConfig] = None) -> POControllerAdapter:
        """Instantiate Classical P&O controller adapter."""
        cfg = po_config if po_config is not None else self.config.mppt.po
        return POControllerAdapter(config=cfg)

    def create_sac_adapter(
        self,
        checkpoint_path: Optional[Path] = None,
        deterministic: bool = True,
        expected_dim: int = 8,
    ) -> SACControllerAdapter:
        """Instantiate Trained SAC controller adapter with validated observation space."""
        model = self.load_sac_model(checkpoint_path)
        self.validate_sac_observation_dim(model, expected_dim=expected_dim)
        return SACControllerAdapter(model=model, deterministic=deterministic)


@st.cache_resource(show_spinner="Loading Soft Actor-Critic Checkpoint...")
def _cached_load_model(path_str: str) -> Any:
    """Streamlit cached helper to prevent reloading weights on every UI interaction."""
    p = Path(path_str)
    if not p.is_file():
        raise FileNotFoundError(f"SAC model file not found at path: {path_str}")
    logger.info(f"Loading SAC model into memory from: {p}")
    return load_sac_agent(p, device="cpu")
