"""Numerical contracts specific to the policy encoder and PPO serialization."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from fleetrl.config import FleetConfig, TrainConfig
from fleetrl.rl import (
    MaskedEntityEncoder,
    SeededTrainingEnv,
    curriculum_stages,
    curriculum_variants,
    policy_kwargs,
    validation_rank,
)


def sample_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "robots": spaces.Box(-1, 1, (20, 14), np.float32),
            "robot_mask": spaces.Box(0, 1, (20,), np.float32),
            "tasks": spaces.Box(-1, 1, (40, 10), np.float32),
            "task_mask": spaces.Box(0, 1, (40,), np.float32),
            "ports": spaces.Box(-1, 1, (4, 6), np.float32),
            "port_mask": spaces.Box(0, 1, (4,), np.float32),
            "global": spaces.Box(-1, 1, (30,), np.float32),
        }
    )


def tensor_observation(space: spaces.Dict) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(7)
    observations = {
        key: torch.tensor(rng.uniform(-1, 1, (2, *subspace.shape)), dtype=torch.float32)
        for key, subspace in space.spaces.items()
    }
    for key in ("robot_mask", "task_mask", "port_mask"):
        observations[key].zero_()
        observations[key][0, :2] = 1
        observations[key][1, :1] = 1
    return observations


def test_entity_encoder_is_permutation_invariant_and_ignores_padding() -> None:
    torch.manual_seed(21)
    space = sample_space()
    extractor = MaskedEntityEncoder(space).eval()
    original = tensor_observation(space)
    expected = extractor(original)
    modified = {key: value.clone() for key, value in original.items()}
    for entities, mask in extractor.entity_pairs:
        modified[entities][~modified[mask].bool()] = 123456.0
        permutation = torch.randperm(modified[entities].shape[1])
        modified[entities] = modified[entities][:, permutation]
        modified[mask] = modified[mask][:, permutation]
    actual = extractor(modified)
    assert actual.shape == (2, 256)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_empty_entities_have_finite_forward_and_backward() -> None:
    torch.manual_seed(13)
    space = sample_space()
    extractor = MaskedEntityEncoder(space)
    observation = tensor_observation(space)
    for _, mask in extractor.entity_pairs:
        observation[mask].zero_()
    output = extractor(observation)
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in extractor.parameters()
    )
    pooled = extractor.pool(torch.ones(2, 4, 64), torch.zeros(2, 4), torch.ones(2, 64))
    assert torch.count_nonzero(pooled) == 0


def test_policy_architecture_switch_preserves_proposal_mlp() -> None:
    assert policy_kwargs("mlp")["net_arch"] == {"pi": [128, 128], "vf": [128, 128]}
    assert policy_kwargs("entity")["features_extractor_class"] is MaskedEntityEncoder
    with pytest.raises(ValueError, match="architecture"):
        policy_kwargs("graph")


def test_validation_prioritizes_safety_and_rejects_partial_episodes() -> None:
    safe = {
        "collisions": 0,
        "edge_conflicts": 0,
        "energy_emergencies": 0,
        "reserve_violations": 0,
        "completed": 5,
        "pending": 3,
        "total_lateness_s": 10,
        "terminated": True,
        "truncated": False,
    }
    unsafe = {**safe, "completed": 20, "collisions": 1}
    assert validation_rank([safe]) > validation_rank([unsafe])
    assert validation_rank([{**safe, "safety_incidents": 0}]) > validation_rank(
        [{**safe, "completed": 100, "safety_incidents": 1}]
    )
    assert validation_rank([{**safe, "completed": 6}]) > validation_rank([safe])
    with pytest.raises(RuntimeError, match="complete"):
        validation_rank([{**safe, "truncated": True}])
    with pytest.raises(ValueError, match="schema"):
        validation_rank([{}])


def test_training_reset_cannot_escape_training_seed_range() -> None:
    class SeedEchoEnv(gym.Env):
        observation_space = spaces.Box(0, 1, (1,), np.float32)
        action_space = spaces.Discrete(6)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, np.float32), {"actual_seed": seed}

    first = SeededTrainingEnv(SeedEchoEnv(), 11, 4, 6)
    second = SeededTrainingEnv(SeedEchoEnv(), 11, 4, 6)
    seeds = [first.reset(seed=1000)[1]["actual_seed"] for _ in range(30)]
    assert seeds == [second.reset(seed=2000)[1]["actual_seed"] for _ in range(30)]
    assert set(seeds) == {4, 5, 6}


def test_curriculum_keeps_target_config_and_disables_short_warmup() -> None:
    target = FleetConfig(n_robots=15, demand_per_hour=90)
    options = TrainConfig(total_timesteps=300000)
    stages = curriculum_stages(target, options)
    assert [stage[1].n_robots for stage in stages] == [5, 10, 15]
    assert stages[-1][1] == target
    assert stages[0][1].demand_per_hour == pytest.approx(21)
    assert not stages[0][1].disturbances
    assert target.disturbances  # No in-place mutation of validation config.
    options.total_timesteps = 8
    assert [stage[0] for stage in curriculum_stages(target, options)] == ["target"]


def test_episode_mixing_is_reproducible_masked_and_isolated_from_validation() -> None:
    from fleetrl.env import FleetEnv

    target = FleetConfig(
        n_robots=15, horizon_s=20, initial_tasks=0, demand_per_hour=90, disturbances=False
    )
    unchanged = target.to_dict()
    options = TrainConfig(curriculum=True)
    variants = curriculum_variants(target, options)
    assert len(variants) == 9
    assert {c.n_robots for c in variants} == {10, 15, 20}
    assert all(c.horizon_s == target.horizon_s for c in variants)
    first = SeededTrainingEnv(FleetEnv(target), 7, 4, 6, variants)
    second = SeededTrainingEnv(FleetEnv(target), 7, 4, 6, variants)
    fixed = SeededTrainingEnv(FleetEnv(target), 7, 4, 6)
    seen_sizes = set()
    try:
        for _ in range(18):
            observation, info = first.reset(seed=1000)
            _, second_info = second.reset(seed=2000)
            _, fixed_info = fixed.reset(seed=3000)
            assert (
                info["training_episode_seed"]
                == second_info["training_episode_seed"]
                == fixed_info["training_episode_seed"]
            )
            assert info["training_episode_seed"] in {4, 5, 6}
            size = info["training_n_robots"]
            seen_sizes.add(size)
            assert size == second_info["training_n_robots"]
            assert info["training_demand_per_hour"] == second_info["training_demand_per_hour"]
            assert observation["robot_mask"].sum() == size
            assert np.count_nonzero(observation["robots"][size:]) == 0
            assert first.observation_space.contains(observation)
        assert seen_sizes == {10, 15, 20}
        assert target.to_dict() == unchanged
        assert fixed.env.config.n_robots == 15
        assert curriculum_variants(target.copy(n_robots=5), options) == []
    finally:
        first.close()
        second.close()
        fixed.close()


@pytest.mark.parametrize("encoder", ["entities", "mlp"])
def test_real_env_ppo_update_checkpoint_reload_and_resume(tmp_path, encoder) -> None:
    """Two tiny updates verify executable plumbing, not learned task quality."""
    from fleetrl.env import FleetEnv
    from fleetrl.rl import evaluate_checkpoint, load_policy, train

    env_config = FleetConfig(
        n_robots=3, horizon_s=20, initial_tasks=2, demand_per_hour=0, disturbances=False
    )
    train_config = TrainConfig(
        total_timesteps=8,
        n_envs=1,
        encoder=encoder,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        eval_freq=8,
        checkpoint_freq=8,
        eval_seeds=(1000,),
        curriculum=False,
        tensorboard=False,
    )
    result = train(env_config, train_config, tmp_path / encoder)
    assert result["status"] == "completed"
    assert result["actual_additional_timesteps"] == 8
    final = load_policy(result["final_model"])
    assert final._n_updates >= 2
    env = FleetEnv(env_config)
    try:
        observation, _ = env.reset(seed=1001)
        before, _ = final.predict(observation, deterministic=True)
        clone_path = tmp_path / f"{encoder}_clone.zip"
        final.save(clone_path)
        clone = load_policy(clone_path)
        after, _ = clone.predict(observation, deterministic=True)
        assert int(before) == int(after)
        for key, parameter in final.policy.state_dict().items():
            torch.testing.assert_close(parameter, clone.policy.state_dict()[key], rtol=0, atol=0)
    finally:
        env.close()
    evaluation = evaluate_checkpoint(result["final_model"], env_config, [1001])
    assert evaluation[0]["terminated"] and not evaluation[0]["truncated"]
    assert np.isfinite(evaluation[0]["episode_reward"])
    # One encoder covers resume serialization; both cover rollout/update/reload.
    if encoder == "entities":
        resumed = train(
            env_config, train_config, tmp_path / "resumed", resume=result["final_model"]
        )
        assert resumed["actual_total_timesteps"] == 16
        assert resumed["actual_additional_timesteps"] == 8
        assert resumed["stages"][0]["name"] == "resume_target"
