"""Unit Tests for Atmospheric Episode Profiles."""

import numpy as np
import pytest

from app.rl.episode_profiles import EpisodeProfileGenerator, ProfileType
from config.default_config import RLEnvConfig


class TestEpisodeProfiles:
    """Test atmospheric trajectories and reproducible seeding."""

    def test_stc_profile_constancy(self) -> None:
        profile = EpisodeProfileGenerator.create(ProfileType.STC)
        g0, t0 = profile.reset()
        assert g0 == 1000.0
        assert t0 == 25.0

        for t in [0.0, 0.25, 0.5, 1.0, 1.5]:
            g, temp = profile.get_conditions(t)
            assert g == 1000.0
            assert temp == 25.0

    def test_random_constant_profile_within_bounds(self) -> None:
        cfg = RLEnvConfig(irradiance_min=400.0, irradiance_max=900.0, temperature_min=20.0, temperature_max=40.0)
        profile = EpisodeProfileGenerator.create(ProfileType.RANDOM_CONSTANT, config=cfg)

        g0, t0 = profile.reset(rng=np.random.default_rng(101))
        assert 400.0 <= g0 <= 900.0
        assert 20.0 <= t0 <= 40.0

        # Remains constant throughout episode
        for t in [0.1, 0.5, 1.2]:
            g, temp = profile.get_conditions(t)
            assert g == g0
            assert temp == t0

    def test_irradiance_step_profile_transitions(self) -> None:
        profile = EpisodeProfileGenerator.create(ProfileType.IRRADIANCE_STEP)
        profile.reset()

        # [0, 0.4s) -> 1000
        g_early, t_early = profile.get_conditions(0.2)
        assert g_early == 1000.0
        assert t_early == 25.0

        # [0.4, 0.9s) -> 600
        g_mid, t_mid = profile.get_conditions(0.6)
        assert g_mid == 600.0
        assert t_mid == 25.0

        # [0.9s, inf) -> 800
        g_late, t_late = profile.get_conditions(1.1)
        assert g_late == 800.0
        assert t_late == 25.0

    def test_temperature_step_profile_transitions(self) -> None:
        profile = EpisodeProfileGenerator.create(ProfileType.TEMPERATURE_STEP)
        profile.reset()

        # [0, 0.4s) -> 25°C
        g_early, t_early = profile.get_conditions(0.1)
        assert g_early == 1000.0
        assert t_early == 25.0

        # [0.4, 0.9s) -> 45°C
        g_mid, t_mid = profile.get_conditions(0.5)
        assert t_mid == 45.0

        # [0.9s, inf) -> 35°C
        g_late, t_late = profile.get_conditions(1.0)
        assert t_late == 35.0

    def test_combined_profile_dynamics(self) -> None:
        profile = EpisodeProfileGenerator.create(ProfileType.COMBINED)
        profile.reset()

        # t < 0.3: G=1000, T=25
        g_init, t_init = profile.get_conditions(0.1)
        assert g_init == 1000.0
        assert t_init == 25.0

        # t = 0.5: linear ramp halfway between 1000 and 500 = 750; temp stepped to 50
        g_mid, t_mid = profile.get_conditions(0.5)
        assert abs(g_mid - 750.0) < 1.0
        assert t_mid == 50.0

        # t > 0.7: G=500, T=50
        g_late, t_late = profile.get_conditions(1.0)
        assert g_late == 500.0
        assert t_late == 50.0

    def test_deterministic_seeding_reproducibility(self) -> None:
        prof1 = EpisodeProfileGenerator.create(ProfileType.RANDOM_CONSTANT)
        prof2 = EpisodeProfileGenerator.create(ProfileType.RANDOM_CONSTANT)

        g1, t1 = prof1.reset(rng=np.random.default_rng(42))
        g2, t2 = prof2.reset(rng=np.random.default_rng(42))
        assert g1 == g2
        assert t1 == t2

        g3, t3 = prof2.reset(rng=np.random.default_rng(999))
        assert (g1 != g3) or (t1 != t3)
