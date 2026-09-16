"""Centralized PPO: masked entity encoding, reproducible training and validation.

The learned action is a solver cost-profile index, never a robot motion command.
This module provides the shared entity encoder and training helpers. Registered
methods use learning/training.py for their lifecycle; the PPO-only trainer below
supports configurations without a method identity.
"""

from __future__ import annotations

import copy
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan
from torch import nn

from fleetrl.config import FleetConfig, TrainConfig, save_config


class MaskedEntityEncoder(BaseFeaturesExtractor):
    """Encode padded entity sets without learning arbitrary row identities.

    Each entity group uses shared row weights, followed by masked mean, max and
    global-query attention pooling. All-empty sets contribute exactly zero.
    Additional non-entity observations are flattened as global context.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        entity_dim: int = 64,
        entity_pairs: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("MaskedEntityEncoder requires a Dict observation space")
        if features_dim < 1 or entity_dim < 1:
            raise ValueError("Encoder dimensions must be positive")
        super().__init__(observation_space, features_dim)
        available = observation_space.spaces
        if entity_pairs is None:
            pairs = []
            for key, space in available.items():
                if len(space.shape or ()) != 2:
                    continue
                candidates = (f"{key}_mask", f"{key.rstrip('s')}_mask")
                mask_key = next((name for name in candidates if name in available), None)
                if mask_key is not None:
                    pairs.append((key, mask_key))
            entity_pairs = pairs
        self.entity_pairs = tuple(tuple(pair) for pair in entity_pairs)
        if not self.entity_pairs:
            raise ValueError("No entity/mask pairs found in the observation space")
        paired_keys: set[str] = set()
        self.encoders = nn.ModuleDict()
        self.queries = nn.ModuleDict()
        for entity_key, mask_key in self.entity_pairs:
            shape = available[entity_key].shape
            mask_shape = available[mask_key].shape
            if len(shape) != 2 or int(np.prod(mask_shape)) != shape[0]:
                raise ValueError(f"Incompatible entity/mask shapes for {entity_key}")
            if entity_key in paired_keys or mask_key in paired_keys:
                raise ValueError("Entity and mask keys must be unique")
            paired_keys.update((entity_key, mask_key))
            self.encoders[entity_key] = nn.Sequential(
                nn.Linear(shape[1], entity_dim),
                nn.LayerNorm(entity_dim),
                nn.Tanh(),
                nn.Linear(entity_dim, entity_dim),
                nn.Tanh(),
            )
        self.global_keys = tuple(key for key in available if key not in paired_keys)
        global_size = sum(int(np.prod(available[key].shape)) for key in self.global_keys)
        self.global_encoder = nn.Sequential(
            nn.Linear(max(1, global_size), entity_dim),
            nn.LayerNorm(entity_dim),
            nn.Tanh(),
        )
        for entity_key, _ in self.entity_pairs:
            self.queries[entity_key] = nn.Linear(entity_dim, entity_dim, bias=False)
        fusion_size = entity_dim * (1 + 3 * len(self.entity_pairs))
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh(),
        )
        self.entity_dim = entity_dim

    @staticmethod
    def pool(encoded: torch.Tensor, mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Return concatenated mean/max/attention; ignore every padded row."""
        present = mask.reshape(mask.shape[0], -1).bool()
        expanded = present.unsqueeze(-1)
        count = present.sum(1, keepdim=True).clamp_min(1)
        masked = torch.where(expanded, encoded, torch.zeros_like(encoded))
        mean = masked.sum(1) / count
        maximum = encoded.masked_fill(~expanded, torch.finfo(encoded.dtype).min).max(1).values
        maximum = torch.where(present.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        logits = (masked * query.unsqueeze(1)).sum(-1) / math.sqrt(encoded.shape[-1])
        logits = logits.masked_fill(~present, torch.finfo(encoded.dtype).min)
        weights = torch.softmax(logits, dim=1) * present.to(encoded.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        attention = (masked * weights.unsqueeze(-1)).sum(1)
        return torch.cat((mean, maximum, attention), dim=1)

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode a batched observation dictionary to fixed-size policy features."""
        first = observations[self.entity_pairs[0][0]]
        if self.global_keys:
            global_input = torch.cat(
                [observations[key].float().reshape(first.shape[0], -1) for key in self.global_keys],
                dim=1,
            )
        else:
            global_input = first.new_zeros((first.shape[0], 1))
        context = self.global_encoder(global_input)
        features = [context]
        for entity_key, mask_key in self.entity_pairs:
            # Zero first, preventing padded garbage from reaching the row MLP.
            mask = observations[mask_key].reshape(first.shape[0], -1).bool()
            rows = torch.where(mask.unsqueeze(-1), observations[entity_key].float(), 0.0)
            encoded = self.encoders[entity_key](rows)
            features.append(self.pool(encoded, mask, self.queries[entity_key](context)))
        return self.fusion(torch.cat(features, dim=1))


def policy_kwargs(architecture: str = "entities") -> dict[str, Any]:
    """Return explicit, comparable actor/critic configuration for SB3 PPO."""
    common: dict[str, Any] = {
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        "activation_fn": nn.Tanh,
        "ortho_init": True,
        "optimizer_kwargs": {"eps": 1e-5},
    }
    if architecture in {"entity", "entities"}:
        common.update(
            features_extractor_class=MaskedEntityEncoder,
            features_extractor_kwargs={"features_dim": 256, "entity_dim": 64},
        )
    elif architecture != "mlp":
        raise ValueError("architecture must be 'entities' or 'mlp'")
    return common


class SeededTrainingEnv(gym.Wrapper):
    """Sample train-only seeds and optional episode-level curriculum variants."""

    def __init__(
        self,
        env: gym.Env,
        stream_seed: int,
        seed_min: int,
        seed_max: int,
        variant_configs: Sequence[FleetConfig] | None = None,
    ):
        super().__init__(env)
        self.rng = np.random.default_rng(stream_seed)
        # Separate streams keep episode seed draws identical if mixing is toggled.
        self.variant_rng = np.random.default_rng(stream_seed + 71417)
        self.seed_min = seed_min
        self.seed_max = seed_max
        self.variant_configs = tuple(copy.deepcopy(config) for config in (variant_configs or ()))
        self.episode_seed: int | None = None

    def _episode_metadata(self) -> dict[str, Any]:
        config = getattr(self.env, "config", None)
        return {
            "training_episode_seed": self.episode_seed,
            "training_n_robots": getattr(config, "n_robots", 0),
            "training_demand_per_hour": getattr(config, "demand_per_hour", 0.0),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # SB3's initial vector-env seed must not bypass our split restriction.
        self.episode_seed = int(self.rng.integers(self.seed_min, self.seed_max + 1))
        if self.variant_configs:
            from fleetrl.env import FleetEnv

            index = int(self.variant_rng.integers(len(self.variant_configs)))
            replacement = FleetEnv(copy.deepcopy(self.variant_configs[index]))
            if (
                replacement.observation_space != self.env.observation_space
                or replacement.action_space != self.env.action_space
            ):
                replacement.close()
                raise ValueError("Curriculum variants must preserve observation and action spaces")
            self.env.close()
            self.env = replacement
        observation, info = self.env.reset(seed=self.episode_seed, options=options)
        info = dict(info)
        info.update(self._episode_metadata())
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.update(self._episode_metadata())
        return observation, reward, terminated, truncated, info


def curriculum_variants(env_config: FleetConfig, train_config: TrainConfig) -> list[FleetConfig]:
    """Build the target fleet-size/workload cross-product without changing a shift."""
    if not train_config.curriculum or env_config.n_robots < 10:
        return []
    return [
        env_config.copy(
            n_robots=count,
            demand_per_hour=env_config.demand_per_hour * count / env_config.n_robots * factor,
        )
        for count in train_config.curriculum_robot_counts
        for factor in train_config.curriculum_demand_factors
    ]


def make_training_env(
    env_config: FleetConfig,
    train_config: TrainConfig,
    output_dir: str | Path,
    stage_index: int = 0,
    mix_curriculum: bool = False,
) -> VecCheckNan:
    """Create independent monitored workers; use spawn for portable processes."""
    from fleetrl.env import FleetEnv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = curriculum_variants(env_config, train_config) if mix_curriculum else []

    def factory(rank: int) -> Callable[[], gym.Env]:
        config = copy.deepcopy(env_config)
        stream_seed = train_config.seed + stage_index * 100003 + rank * 1009

        def create() -> gym.Env:
            env = SeededTrainingEnv(
                FleetEnv(config),
                stream_seed,
                train_config.train_seed_min,
                train_config.train_seed_max,
                variant_configs=variants,
            )
            return Monitor(
                env,
                str(output_dir / f"stage_{stage_index}_worker_{rank}"),
                info_keywords=(
                    "training_episode_seed",
                    "training_n_robots",
                    "training_demand_per_hour",
                ),
            )

        return create

    factories = [factory(rank) for rank in range(train_config.n_envs)]
    vec = (
        DummyVecEnv(factories)
        if train_config.n_envs == 1
        else SubprocVecEnv(factories, start_method="spawn")
    )
    return VecCheckNan(vec, raise_exception=True, check_inf=True)


def _json_write(path: Path, data: Any) -> None:
    """Atomically replace a UTF-8 JSON record; reject non-finite metadata."""

    def default(value: Any):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Unsupported JSON value {type(value).__name__}")

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=default),
        encoding="utf-8",
    )
    temporary.replace(path)


def ensure_finite_parameters(model: PPO) -> None:
    """Fail immediately if a learned parameter or optimizer tensor is non-finite."""
    for name, parameter in model.policy.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Non-finite policy parameter: {name}")
    for state in model.policy.optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite optimizer state: {name}")


def evaluate_model(
    model: PPO,
    env_config: FleetConfig,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Run deterministic complete episodes without changing learning state."""
    from fleetrl.env import FleetEnv
    from fleetrl.evaluation import account_policy_latency

    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    if env_config.method is not None or hasattr(model, "_fleetrl_metadata"):
        from .evaluation import run_episode

        method = env_config.method or model._fleetrl_metadata["method"]
        return [
            {
                **run_episode(env_config, method, int(seed), model)["metrics"],
                "seed": int(seed),
                "terminated": True,
                "truncated": False,
            }
            for seed in seeds
        ]
    records: list[dict[str, Any]] = []
    env = FleetEnv(copy.deepcopy(env_config))
    maximum_steps = math.ceil(env_config.horizon_s / env_config.decision_s) + 2
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=int(seed))
            total_reward = 0.0
            action_counts = [0] * 6
            started = time.perf_counter()
            for steps in range(1, maximum_steps + 1):
                inference_started = time.perf_counter()
                action, _ = model.predict(observation, deterministic=True)
                inference_ms = (time.perf_counter() - inference_started) * 1000
                selected = int(np.asarray(action).item())
                if not 0 <= selected < 6:
                    raise ValueError(f"Policy returned invalid action {selected}")
                action_counts[selected] += 1
                observation, reward, terminated, truncated, info = env.step(selected)
                account_policy_latency(env, info, inference_ms)
                if not np.isfinite(reward):
                    raise FloatingPointError("Non-finite evaluation reward")
                total_reward += float(reward)
                if terminated or truncated:
                    metrics = dict(env.metrics())
                    records.append(
                        {
                            **metrics,
                            "seed": int(seed),
                            "episode_reward": total_reward,
                            "episode_steps": steps,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "action_counts": action_counts,
                            "evaluation_wall_s": time.perf_counter() - started,
                        }
                    )
                    break
            else:
                raise RuntimeError("Evaluation exceeded the configured episode horizon")
    finally:
        env.close()
    return records


def validation_rank(records: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    """Rank safety first, then service output, lateness, and unfinished work.

    This criterion is fixed before training, independent of training rewards.
    The metrics schema deliberately uses direct operational counts.
    """
    if not records:
        raise ValueError("Cannot rank an empty validation set")
    safety_keys = ("collisions", "edge_conflicts", "energy_emergencies", "reserve_violations")
    required = (*safety_keys, "completed", "pending", "total_lateness_s")
    for record in records:
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"Validation metric schema missing: {missing}")
        if any(not math.isfinite(float(record[key])) for key in required):
            raise FloatingPointError("Non-finite validation metrics")
        if "safety_incidents" in record and not math.isfinite(float(record["safety_incidents"])):
            raise FloatingPointError("Non-finite validation safety incidents")
        if record.get("truncated", False) or not record.get("terminated", True):
            raise RuntimeError("Checkpoint selection requires complete simulated shifts")
    return (
        -sum(
            float(record["safety_incidents"])
            if "safety_incidents" in record
            else sum(float(record[key]) for key in safety_keys)
            for record in records
        ),
        float(np.mean([record["completed"] for record in records])),
        -float(np.mean([record["total_lateness_s"] for record in records])),
        -float(np.mean([record["pending"] for record in records])),
    )


class ValidationCheckpointCallback(BaseCallback):
    """Record fixed-seed validation and save the strongest measured checkpoint."""

    def __init__(self, env_config: FleetConfig, train_config: TrainConfig, output_dir: Path):
        super().__init__()
        self.env_config = copy.deepcopy(env_config)
        self.train_config = copy.deepcopy(train_config)
        self.output_dir = Path(output_dir)
        self.best_rank: tuple[float, ...] | None = None
        self.best_step = 0
        self.evaluations: list[dict[str, Any]] = []
        self.next_evaluation = 0
        self.next_checkpoint = 0
        self.last_evaluation_step = -1

    def _on_training_start(self) -> None:
        self.next_evaluation = self.num_timesteps + self.train_config.eval_freq
        self.next_checkpoint = self.num_timesteps + self.train_config.checkpoint_freq
        if self.best_rank is None:
            self.run_validation("initial_or_resumed")

    def _on_rollout_start(self) -> None:
        ensure_finite_parameters(self.model)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_evaluation:
            self.run_validation("periodic")
            self.next_evaluation = self.num_timesteps + self.train_config.eval_freq
        if self.num_timesteps >= self.next_checkpoint:
            self.model.save(self.output_dir / "checkpoints" / f"step_{self.num_timesteps}")
            self.next_checkpoint = self.num_timesteps + self.train_config.checkpoint_freq
        for info in self.locals.get("infos", []):
            for key, value in info.get("reward_components", {}).items():
                if isinstance(value, (int, float, np.number)):
                    self.logger.record_mean(f"reward_components/{key}", float(value))
        return True

    def _on_training_end(self) -> None:
        # This runs after PPO's final optimizer update, unlike _on_rollout_end.
        ensure_finite_parameters(self.model)
        self.run_validation("stage_final")
        # SB3 normally dumps before training; flush the final update's losses
        # as well, including when a smoke run contains only one rollout.
        self.logger.dump(step=int(self.num_timesteps))

    def run_validation(self, reason: str) -> None:
        ensure_finite_parameters(self.model)
        self.logger.info(
            f"Validation: {reason}, step={self.num_timesteps}, "
            f"seeds={list(self.train_config.eval_seeds)}"
        )
        records = evaluate_model(self.model, self.env_config, self.train_config.eval_seeds)
        rank = validation_rank(records)
        improved = self.best_rank is None or rank > self.best_rank
        summary = {
            "timesteps": int(self.num_timesteps),
            "reason": reason,
            "rank": list(rank),
            "improved": improved,
            "episodes": records,
        }
        self.evaluations.append(summary)
        _json_write(self.output_dir / "validation.json", self.evaluations)
        self.logger.record("validation/completed_mean", rank[1])
        self.logger.record("validation/safety_events", -rank[0])
        self.logger.record("validation/lateness_s_mean", -rank[2])
        self.logger.record(
            "validation/reward_mean", float(np.mean([r["episode_reward"] for r in records]))
        )
        if improved:
            self.best_rank = rank
            self.best_step = int(self.num_timesteps)
            self.model.save(self.output_dir / "best_model")
            _json_write(self.output_dir / "best_validation.json", summary)
        self.last_evaluation_step = int(self.num_timesteps)
        self.logger.info(f"Validation rank={rank}; best_step={self.best_step}")


def curriculum_stages(
    env_config: FleetConfig, train_config: TrainConfig
) -> list[tuple[str, FleetConfig, int]]:
    """Return training-only stages; keep validation at the exact target config."""
    rollout_size = train_config.n_steps * train_config.n_envs
    if not train_config.curriculum or train_config.total_timesteps < 3 * rollout_size:
        return [("target", copy.deepcopy(env_config), train_config.total_timesteps)]
    stages = []
    allocations = (0.15, 0.25, 0.60)
    names = ("warmup", "intermediate", "target")
    for index, (name, fraction) in enumerate(zip(names, allocations)):
        count = min(env_config.n_robots, (5, 10, env_config.n_robots)[index])
        ratio = count / env_config.n_robots
        if name == "target":
            config = copy.deepcopy(env_config)
        else:
            config = env_config.copy(
                n_robots=count,
                demand_per_hour=env_config.demand_per_hour * ratio * (0.7 if index == 0 else 0.9),
                initial_tasks=max(1, int(round(env_config.initial_tasks * ratio)))
                if env_config.initial_tasks
                else 0,
                burst_multiplier=1.0 if index == 0 else env_config.burst_multiplier,
                disturbances=False,
            )
        steps = max(rollout_size, int(round(train_config.total_timesteps * fraction)))
        stages.append((name, config, steps))
    return stages


def load_policy(path: str | Path, device: str = "cpu") -> PPO:
    """Load a trusted local SB3 checkpoint, then check learned tensors."""
    from .learning.training import load_checkpoint

    resolved = Path(path) if Path(path).exists() else Path(str(path) + ".zip")
    return load_checkpoint(resolved, device=device)


def evaluate_checkpoint(
    path: str | Path,
    env_config: FleetConfig,
    seeds: Sequence[int],
    output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one checkpoint; never feed these results back into training."""
    torch.set_num_threads(1)
    records = evaluate_model(load_policy(path), env_config, seeds)
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        _json_write(folder / "evaluation.json", {"checkpoint": str(path), "episodes": records})
    return records


def train(
    env_config: FleetConfig,
    train_config: TrainConfig,
    output_dir: str | Path,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Train PPO and return checkpoint/log paths plus a measured run manifest.

    total_timesteps is an additional interaction budget when resuming. PPO
    finishes full rollouts, so actual steps may exceed the requested budget.
    Resume restarts episodes, not the exact simulator or rollout-buffer state.
    """
    if env_config.method is not None:
        from .learning.training import train_method

        return train_method(env_config, train_config, output_dir, resume)
    env_config = copy.deepcopy(env_config).validate()
    train_config = copy.deepcopy(train_config).validate()
    if env_config.controller != "hybrid":
        raise ValueError("RL training requires controller='hybrid' so actions affect optimization")
    if not train_config.eval_seeds or len(set(train_config.eval_seeds)) != len(
        train_config.eval_seeds
    ):
        raise ValueError("Validation seeds must be nonempty and unique")
    if (
        train_config.seed < 0
        or train_config.train_seed_min < 0
        or any(seed < 0 for seed in train_config.eval_seeds)
    ):
        raise ValueError("Random seeds must be nonnegative")
    if train_config.eval_freq < 1 or train_config.checkpoint_freq < 1:
        raise ValueError("Evaluation and checkpoint frequencies must be positive")
    if train_config.torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    if (train_config.n_steps * train_config.n_envs) % train_config.batch_size:
        raise ValueError("batch_size must divide n_steps * n_envs for complete minibatches")
    numeric = (
        train_config.learning_rate,
        train_config.gamma,
        train_config.gae_lambda,
        train_config.clip_range,
        train_config.ent_coef,
        train_config.vf_coef,
        train_config.max_grad_norm,
        train_config.target_kl,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("PPO hyperparameters must be finite")
    if not (
        train_config.learning_rate > 0
        and 0 < train_config.gamma <= 1
        and 0 <= train_config.gae_lambda <= 1
        and 0 < train_config.clip_range < 1
        and train_config.ent_coef >= 0
        and train_config.vf_coef >= 0
        and train_config.max_grad_norm > 0
        and train_config.target_kl > 0
        and train_config.n_epochs >= 1
    ):
        raise ValueError("PPO hyperparameters are outside their valid ranges")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # A fresh output folder preserves historical evidence and resume inputs.
    if (output_dir / "manifest.json").exists() or (output_dir / "best_model.zip").exists():
        raise FileExistsError("Output contains a prior run; choose a new --output directory")
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    save_config(output_dir / "config.yaml", env_config, train_config)
    torch.set_num_threads(train_config.torch_threads)
    formats = ["stdout", "csv"] + (["tensorboard"] if train_config.tensorboard else [])
    logger = configure(str(output_dir / "logs"), formats)
    stages = curriculum_stages(env_config, train_config)
    # Continuation trains the target distribution (including configured mixing), avoiding a return
    # to easy warmup whenever a user adds training steps to a checkpoint.
    if resume is not None:
        stages = [("resume_target", env_config, train_config.total_timesteps)]
    callback = ValidationCheckpointCallback(env_config, train_config, output_dir)
    model: PPO | None = None
    current_env = None
    manifest: dict[str, Any] = {
        "status": "running",
        "algorithm": "PPO",
        "encoder": train_config.encoder,
        "requested_additional_timesteps": train_config.total_timesteps,
        "env": asdict(env_config),
        "train": asdict(train_config),
        "train_seed_range": [train_config.train_seed_min, train_config.train_seed_max],
        "validation_seeds": list(train_config.eval_seeds),
        "test_used_for_selection": False,
        "selection_order": [
            "min_safety_events",
            "max_completed",
            "min_total_lateness_s",
            "min_pending",
        ],
        "observation_normalization": "fixed_physical_scaling_in_environment",
        "reward_normalization": "none",
        "resume_from": str(resume) if resume else None,
        "resume_exact_trajectory": False,
        "stages": [],
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gym.__version__,
        },
        "cuda_available": torch.cuda.is_available(),
        "started_unix_s": time.time(),
    }
    _json_write(output_dir / "manifest.json", manifest)
    started = time.perf_counter()
    initial_timesteps = 0
    try:
        for index, (name, stage_config, budget) in enumerate(stages):
            if current_env is not None:
                current_env.close()
                current_env = None
            mix_curriculum = bool(
                train_config.curriculum
                and stage_config.n_robots >= 10
                and name in {"target", "resume_target"}
            )
            next_env = make_training_env(
                stage_config,
                train_config,
                output_dir / "monitor",
                index,
                mix_curriculum=mix_curriculum,
            )
            current_env = next_env
            if model is None:
                if resume is None:
                    model = PPO(
                        "MultiInputPolicy",
                        next_env,
                        policy_kwargs=policy_kwargs(train_config.encoder),
                        learning_rate=train_config.learning_rate,
                        n_steps=train_config.n_steps,
                        batch_size=train_config.batch_size,
                        n_epochs=train_config.n_epochs,
                        gamma=train_config.gamma,
                        gae_lambda=train_config.gae_lambda,
                        clip_range=train_config.clip_range,
                        ent_coef=train_config.ent_coef,
                        vf_coef=train_config.vf_coef,
                        max_grad_norm=train_config.max_grad_norm,
                        target_kl=train_config.target_kl,
                        seed=train_config.seed,
                        device=train_config.device,
                        verbose=1,
                    )
                else:
                    model = PPO.load(
                        str(resume),
                        env=next_env,
                        device=train_config.device,
                        learning_rate=train_config.learning_rate,
                        n_steps=train_config.n_steps,
                        batch_size=train_config.batch_size,
                        n_epochs=train_config.n_epochs,
                        gamma=train_config.gamma,
                        gae_lambda=train_config.gae_lambda,
                        clip_range=train_config.clip_range,
                        ent_coef=train_config.ent_coef,
                        vf_coef=train_config.vf_coef,
                        max_grad_norm=train_config.max_grad_norm,
                        target_kl=train_config.target_kl,
                        seed=train_config.seed,
                    )
                    actual_encoder = (
                        "entities"
                        if isinstance(model.policy.features_extractor, MaskedEntityEncoder)
                        else "mlp"
                    )
                    if actual_encoder != train_config.encoder:
                        raise ValueError(
                            f"Resume encoder is {actual_encoder}; config requests {train_config.encoder}"
                        )
                    initial_timesteps = int(model.num_timesteps)
                model.set_logger(logger)
            else:
                model.set_env(next_env, force_reset=True)
            before = int(model.num_timesteps)
            manifest["current_stage"] = {
                "name": name,
                "start_timestep": before,
                "requested_timesteps": budget,
                "episode_mixing": mix_curriculum,
            }
            _json_write(output_dir / "manifest.json", manifest)
            model.learn(
                total_timesteps=budget,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            manifest["stages"].append(
                {
                    "name": name,
                    "config": asdict(stage_config),
                    "requested_timesteps": budget,
                    "actual_timesteps": int(model.num_timesteps) - before,
                    "episode_mixing": mix_curriculum,
                    "variants": [asdict(c) for c in curriculum_variants(stage_config, train_config)]
                    if mix_curriculum
                    else [],
                }
            )
            _json_write(output_dir / "manifest.json", manifest)
        assert model is not None
        ensure_finite_parameters(model)
        model.save(output_dir / "final_model")
        manifest.update(
            {
                "status": "completed",
                "actual_total_timesteps": int(model.num_timesteps),
                "actual_additional_timesteps": int(model.num_timesteps) - initial_timesteps,
                "best_validation_step": callback.best_step,
                "best_validation_rank": list(callback.best_rank),
                "policy_parameters": sum(
                    parameter.numel() for parameter in model.policy.parameters()
                ),
                "actual_device": str(model.device),
                "ppo_updates": int(model._n_updates),
                "wall_s": time.perf_counter() - started,
                "best_model": str(output_dir / "best_model.zip"),
                "final_model": str(output_dir / "final_model.zip"),
                "output_dir": str(output_dir),
            }
        )
        _json_write(output_dir / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
            wall_s=time.perf_counter() - started,
        )
        if model is not None:
            try:
                ensure_finite_parameters(model)
                model.save(output_dir / "interrupted_model")
            except Exception:
                pass
        _json_write(output_dir / "manifest.json", manifest)
        raise
    finally:
        if current_env is not None:
            current_env.close()
        logger.close()
