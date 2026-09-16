"""Frozen study definition and expected run grid; holdout seeds never train/tune."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from ..methods.registry import METHODS, get_method


@dataclass
class Study:
    methods: tuple = tuple(METHODS)
    training_seeds: tuple = (11, 12, 13)
    tape_seeds: tuple = (2000, 2001, 2002, 2003, 2004)
    validation_seeds: tuple = (1000, 1001, 1002)
    scenarios: tuple = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9")
    train_seed_min: int = 0
    train_seed_max: int = 199
    total_timesteps: int = 300000
    horizon_s: float = 3600.0
    config_dir: str = "configs/methods"
    bootstrap_samples: int = 2000
    search_steps: int = 10000
    search_trials: tuple = (0.0001, 0.0003, 0.001)
    search_q_alphas: tuple = (0.05, 0.1, 0.2)
    search_thresholds: tuple = (0.2, 0.3, 0.4)
    matched_ood: bool = True
    pretrained_study: str | None = None

    def validate(self):
        self.methods = tuple(get_method(m).name for m in self.methods)
        for name in ("methods", "training_seeds", "tape_seeds", "validation_seeds", "scenarios"):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be nonempty and unique")
        if any(not 1000 <= s < 2000 for s in self.validation_seeds):
            raise ValueError("validation seeds must be 1000..1999")
        if any(s < 2000 for s in self.tape_seeds):
            raise ValueError("test tape seeds must be >=2000")
        if not 0 <= self.train_seed_min <= self.train_seed_max < 1000:
            raise ValueError("training tape seeds must be 0..999")
        if self.total_timesteps < 1 or not math.isfinite(self.horizon_s) or self.horizon_s <= 0:
            raise ValueError("invalid study budget")
        if self.bootstrap_samples < 1 or self.search_steps < 1 or not self.search_trials:
            raise ValueError("search/bootstrap budgets must be positive")
        if any(not isinstance(s, int) or s < 0 for s in self.training_seeds):
            raise ValueError("training seeds must be nonnegative integers")
        if any(s not in {f"S{i}" for i in range(1, 10)} for s in self.scenarios):
            raise ValueError("study scenarios must be S1..S9; OOD controls use matched_ood")
        if any(not math.isfinite(x) or x <= 0 for x in self.search_trials):
            raise ValueError("invalid learning-rate search grid")
        if any(not 0 < x <= 1 for x in self.search_q_alphas) or any(
            not 0 < x < 1 for x in self.search_thresholds
        ):
            raise ValueError("invalid alpha/threshold search grid")
        if not len(self.search_trials) == len(self.search_q_alphas) == len(self.search_thresholds):
            raise ValueError("search grids must have equal trial counts")
        return self

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def scenarios_with_controls(self):
        return list(self.scenarios) + (["OOD_A", "OOD_B"] if self.matched_ood else [])

    def expected_grid(self):
        return {
            name: {
                "training_seeds": list(self.training_seeds)
                if get_method(name).learnable
                else ["deterministic"],
                "tape_seeds": list(self.tape_seeds),
            }
            for name in self.methods
        }


def load_study(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if set(data) - {"schema_version", "study"}:
        raise ValueError("study file accepts schema_version and study only")
    return Study(**data.get("study", {})).validate()
