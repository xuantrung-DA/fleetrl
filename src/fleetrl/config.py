"""Validated portable dataclass configuration; unknown keys are errors."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml


@dataclass
class FleetConfig:
    n_robots: int = 15
    horizon_s: float = 3600.0
    tick_s: float = 0.5  # Cập nhật mỗi tick_s giây
    decision_s: float = 5.0  # Cập nhật quyết định mỗi decision_s giây
    map_name: str = "A"
    map_path: str | None = None
    demand_per_hour: float = 90.0
    burst_multiplier: float = 2.0
    burst_start_s: float = 900.0
    burst_end_s: float = 1500.0
    burst_zone: int = 1
    initial_tasks: int = 8
    initial_soc_min: float = 0.25
    initial_soc_max: float = 0.90
    low_soc_fraction: float = 0.0
    low_soc_max: float = 0.40
    shift_stations: bool = False
    heavy_fraction: float = 0.20
    charger_power_w: float = 600.0
    reserve_soc: float = 0.10
    charge_targets: tuple[float, ...] = (0.60, 0.80, 0.95)
    charge_threshold: float = 0.30
    charge_efficiency: float = 0.90
    auxiliary_power_w: float = 12.0
    load_factor: float = 0.50
    service_s: float = 10.0
    docking_s: float = 5.0
    route_wait_bound_s: float = 300.0
    charger_wait_bound_s: float = 300.0
    charging_horizon_s: float = 900.0
    deadlock_s: float = 30.0
    observation_delay_s: float = 0.0
    stale_after_s: float = 10.0
    max_tasks_observed: int = 40
    task_candidates_per_robot: int = 6
    max_candidates: int = 500
    solver_time_limit_s: float = 0.10
    decision_budget_s: float = 0.25
    solver_workers: int = 1
    fixed_profile: int = 0
    controller: str = "hybrid"  # hybrid | fixed | heuristic
    advanced_charging: bool = True
    disturbances: bool = True
    seed: int = 0
    log_decisions: bool = False
    energy_multiplier: float = 1.0
    method: str | None = None
    backend: str = "cpsat"
    action_mode: str = "discrete"
    observation_mode: str = "entities"
    mpc_horizon_s: float = 300.0
    forecast_alpha: float = 0.25
    # Reward fixed across profiles, includes all arrived tasks.
    reward_backlog: float = 0.2
    reward_late: float = 0.4
    reward_energy: float = 0.02
    reward_violation: float = 10.0
    reward_terminal_backlog: float = 1.0
    reward_terminal_late: float = 1.0

    def validate(self) -> "FleetConfig":
        for k, v in asdict(self).items():
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"{k} must be finite")
        if not 1 <= self.n_robots <= 20:
            raise ValueError("n_robots must be 1..20")
        if self.horizon_s <= 0 or self.tick_s <= 0 or self.decision_s <= 0:
            raise ValueError("time values must be positive")
        if abs(self.decision_s / self.tick_s - round(self.decision_s / self.tick_s)) > 1e-8:
            raise ValueError("decision_s must be a multiple of tick_s")
        if abs(self.horizon_s / self.tick_s - round(self.horizon_s / self.tick_s)) > 1e-8:
            raise ValueError("horizon_s must be a multiple of tick_s")
        if self.map_name not in {"A", "B"}:
            raise ValueError("map_name must be A or B")
        if not 0 < self.reserve_soc < self.initial_soc_min <= self.initial_soc_max <= 1:
            raise ValueError("require reserve_soc < initial_soc_min <= initial_soc_max <= 1")
        if not 0 <= self.heavy_fraction <= 1:
            raise ValueError("heavy_fraction must be 0..1")
        if not 0 <= self.low_soc_fraction <= 1 or not 0 < self.low_soc_max <= 1:
            raise ValueError("invalid low_soc configuration")
        if self.low_soc_fraction > 0 and self.low_soc_max < self.initial_soc_min:
            raise ValueError("low_soc_max must exceed initial_soc_min when enabled")
        if self.demand_per_hour < 0 or self.initial_tasks < 0:
            raise ValueError("demand/initial_tasks cannot be negative")
        if self.burst_multiplier < 0 or not 0 <= self.burst_zone < 4:
            raise ValueError("invalid burst configuration")
        if not 0 < self.charge_threshold < 1 or not 0 < self.charge_efficiency <= 1:
            raise ValueError("invalid charge threshold/efficiency")
        if self.energy_multiplier <= 0 or self.auxiliary_power_w < 0 or self.load_factor < 0:
            raise ValueError("invalid energy model values")
        if (
            min(
                self.service_s,
                self.docking_s,
                self.route_wait_bound_s,
                self.charger_wait_bound_s,
                self.charging_horizon_s,
                self.deadlock_s,
            )
            <= 0
        ):
            raise ValueError("operational durations must be positive")
        if self.observation_delay_s < 0 or self.stale_after_s < 0:
            raise ValueError("observation delays cannot be negative")
        if any(
            getattr(self, k) < 0
            for k in (
                "reward_backlog",
                "reward_late",
                "reward_energy",
                "reward_violation",
                "reward_terminal_backlog",
                "reward_terminal_late",
            )
        ):
            raise ValueError("reward penalties must be nonnegative")
        if not 1 <= self.max_tasks_observed <= 40 or not 1 <= self.task_candidates_per_robot <= 40:
            raise ValueError("task limits must be 1..40")
        if self.solver_workers != 1:
            raise ValueError("use one solver worker for reproducibility and fair budgets")
        if not 0 < self.solver_time_limit_s <= self.decision_budget_s:
            raise ValueError("solver limit must be positive and <= decision budget")
        if any(not 0 < q <= 1 for q in self.charge_targets):
            raise ValueError("charge targets must be fractions in (0,1]")
        if self.charger_power_w * self.charge_efficiency * 0.5 <= self.auxiliary_power_w:
            raise ValueError("charger power cannot overcome auxiliary draw above80%")
        if self.controller not in {"hybrid", "fixed", "heuristic"}:
            raise ValueError("unknown controller")
        if not 0 <= self.fixed_profile < 6:
            raise ValueError("fixed_profile must be 0..5")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.max_candidates < 1 or not self.charge_targets:
            raise ValueError("candidate and charging target sets must be nonempty")
        if self.backend not in {"cpsat", "milp", "mpc", "arbiter"}:
            raise ValueError("unknown backend")
        if self.action_mode not in {"discrete", "continuous", "multiagent"}:
            raise ValueError("unknown action mode")
        if self.observation_mode not in {"entities", "compact"}:
            raise ValueError("unknown observation mode")
        if self.mpc_horizon_s <= 0 or not 0 < self.forecast_alpha <= 1:
            raise ValueError("invalid MPC/forecast settings")
        return self

    def copy(self, **updates) -> "FleetConfig":
        return replace(self, **updates).validate()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    total_timesteps: int = 300000
    n_envs: int = 2
    seed: int = 11
    encoder: str = "entities"
    device: str = "cpu"
    learning_rate: float = 0.0003
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    eval_freq: int = 10000
    checkpoint_freq: int = 10000
    eval_seeds: tuple[int, ...] = (1000, 1001, 1002)
    train_seed_min: int = 0
    train_seed_max: int = 199
    curriculum: bool = True
    curriculum_robot_counts: tuple[int, ...] = (10, 15, 20)
    curriculum_demand_factors: tuple[float, ...] = (0.7, 1.0, 1.3)
    torch_threads: int = 1
    tensorboard: bool = True
    algorithm: str = "ppo"
    buffer_size: int = 20000
    learning_starts: int = 1000
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1000
    exploration_fraction: float = 0.2
    exploration_final_eps: float = 0.05
    tau: float = 0.005
    q_alpha: float = 0.1
    max_replay_memory_mb: float = 2048.0

    def validate(self) -> "TrainConfig":
        for key, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.encoder not in {"entities", "mlp"}:
            raise ValueError("encoder must be entities or mlp")
        if self.total_timesteps < 1 or self.n_envs < 1 or self.n_steps < 2:
            raise ValueError("invalid rollout/training size")
        if self.algorithm not in {
            "ppo",
            "a2c",
            "dqn",
            "qrdqn",
            "recurrent_ppo",
            "sac",
            "td3",
            "qlearning",
            "mappo",
        }:
            raise ValueError("unknown learning algorithm")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >=2")
        if self.n_epochs < 1 or min(self.eval_freq, self.checkpoint_freq, self.torch_threads) < 1:
            raise ValueError("update and logging frequencies must be positive")
        if self.seed < 0 or self.train_seed_min < 0:
            raise ValueError("seeds must be nonnegative")
        if not 0 <= self.gae_lambda <= 1 or not 0 < self.clip_range < 1 or self.max_grad_norm <= 0:
            raise ValueError("invalid policy update coefficients")
        if (
            self.ent_coef < 0
            or self.vf_coef < 0
            or self.target_kl <= 0
            or self.max_replay_memory_mb <= 0
        ):
            raise ValueError("invalid loss/memory coefficients")
        if not self.eval_seeds or len(set(self.eval_seeds)) != len(self.eval_seeds):
            raise ValueError("validation seeds must be nonempty and unique")
        if (
            self.algorithm in {"ppo", "recurrent_ppo", "mappo"}
            and self.batch_size > self.n_envs * self.n_steps
        ):
            raise ValueError("batch_size must fit rollout")
        if self.algorithm in {"qlearning", "mappo"} and self.n_envs != 1:
            raise ValueError(
                "custom learners currently require n_envs=1; steps count fleet transitions"
            )
        if (
            min(self.buffer_size, self.train_freq, self.gradient_steps, self.target_update_interval)
            < 1
            or self.learning_starts < 0
        ):
            raise ValueError("invalid off-policy configuration")
        if (
            not 0 < self.q_alpha <= 1
            or not 0 <= self.exploration_final_eps <= 1
            or not 0 < self.exploration_fraction <= 1
        ):
            raise ValueError("invalid exploration/tabular settings")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1 or self.learning_rate <= 0:
            raise ValueError("invalid learner coefficients")
        if self.train_seed_min > self.train_seed_max:
            raise ValueError("invalid train seed range")
        if not self.curriculum_robot_counts or any(
            not 1 <= n <= 20 for n in self.curriculum_robot_counts
        ):
            raise ValueError("curriculum counts must be1..20")
        if not self.curriculum_demand_factors or any(
            not math.isfinite(f) or f <= 0 for f in self.curriculum_demand_factors
        ):
            raise ValueError("invalid curriculum demand factors")
        if any(self.train_seed_min <= s <= self.train_seed_max for s in self.eval_seeds):
            raise ValueError("validation seeds overlap train seeds")
        return self


def load_config(path: str | Path | None = None) -> tuple[FleetConfig, TrainConfig]:
    data = {} if path is None else yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(data) - {
        "env",
        "train",
        "schema_version",
        "method",
        "optimizer",
        "evaluation",
        "study",
    }
    if unknown:
        raise ValueError(f"Unknown top-level keys: {sorted(unknown)}")
    if data.get("schema_version", 1) not in {1, 2}:
        raise ValueError("unsupported config schema")
    if "study" in data:
        raise ValueError("use fleetrl study with a dedicated study config")
    env = dict(data.get("env", {}))
    train = dict(data.get("train", {}))
    evaluation = dict(data.get("evaluation", {}))
    if set(evaluation) - {"validation_seeds", "eval_freq"}:
        raise ValueError("unknown evaluation key")
    if "validation_seeds" in evaluation:
        train["eval_seeds"] = evaluation["validation_seeds"]
    if "eval_freq" in evaluation:
        train["eval_freq"] = evaluation["eval_freq"]
    method = data.get("method")
    if method is not None:
        from .methods.registry import get_method

        if isinstance(method, dict):
            extra = set(method) - {"name", "observation_mode", "action_mode"}
            if extra:
                raise ValueError(f"Unknown method keys: {sorted(extra)}")
            env.update({k: v for k, v in method.items() if k != "name"})
            method = method["name"]
        spec = get_method(method)
        env.update(
            method=spec.name,
            backend=spec.backend,
            action_mode=env.get("action_mode", spec.action_mode),
            controller=spec.controller,
            advanced_charging=spec.advanced_charging,
        )
        if spec.algorithm:
            train["algorithm"] = spec.algorithm
    optimizer = dict(data.get("optimizer", {}))
    if set(optimizer) - {
        "solver_time_limit_s",
        "decision_budget_s",
        "mpc_horizon_s",
        "forecast_alpha",
        "max_candidates",
    }:
        raise ValueError("unknown optimizer key")
    env.update(optimizer)
    if "charge_targets" in env:
        env["charge_targets"] = tuple(env["charge_targets"])
    if "eval_seeds" in train:
        train["eval_seeds"] = tuple(train["eval_seeds"])
    for k in ("curriculum_robot_counts", "curriculum_demand_factors"):
        if k in train:
            train[k] = tuple(train[k])
    return FleetConfig(**env).validate(), TrainConfig(**train).validate()


def save_config(path: str | Path, env: FleetConfig, train: TrainConfig | None = None) -> None:
    data = {"env": asdict(env)}
    if train is not None:
        data["train"] = asdict(train)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
