"""Sparse tabular Q-learning over a frozen causal discretization and six profiles."""

import json
import zipfile
from collections import defaultdict

import numpy as np

from ..learning.compact_state import SCHEMA, state_key


class QLearning:
    def __init__(self, tc):
        self.tc = tc
        self.rng = np.random.default_rng(tc.seed)
        self.q = defaultdict(lambda: np.zeros(6, dtype=np.float64))
        self.visits = defaultdict(int)
        self.num_timesteps = 0
        self._n_updates = 0
        self.unseen_predictions = 0
        self.predictions = 0

    def epsilon(self):
        fraction = min(
            1, self.num_timesteps / max(1, self.tc.total_timesteps * self.tc.exploration_fraction)
        )
        return 1 - fraction * (1 - self.tc.exploration_final_eps)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        key = state_key(observation)
        self.predictions += 1
        if key not in self.q:
            self.unseen_predictions += 1
        if not deterministic and self.rng.random() < self.epsilon():
            return int(self.rng.integers(6)), None
        return int(np.argmax(self.q.get(key, np.zeros(6)))), None

    def update(self, observation, action, reward, next_observation, done):
        key = state_key(observation)
        next_key = state_key(next_observation)
        target = float(reward) + (0 if done else self.tc.gamma * float(np.max(self.q[next_key])))
        self.q[key][action] += self.tc.q_alpha * (target - self.q[key][action])
        self.visits[key] += 1
        self._n_updates += 1

    def learn_custom(self, env, steps, callback):
        obs, _ = env.reset()
        for _ in range(steps):
            action, _ = self.predict(obs, deterministic=False)
            nxt, reward, terminated, truncated, info = env.step(action)
            self.update(obs, action, reward, nxt, terminated or truncated)
            self.num_timesteps += 1
            callback(self, info)
            obs = env.reset()[0] if terminated or truncated else nxt
        return self

    def save(self, path):
        from dataclasses import asdict

        payload = {
            "schema": SCHEMA,
            "config": asdict(self.tc),
            "steps": self.num_timesteps,
            "updates": self._n_updates,
            "rng": self.rng.bit_generator.state,
            "table": [[list(k), v.tolist(), self.visits[k]] for k, v in self.q.items()],
        }
        with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as out:
            out.writestr("qlearning.json", json.dumps(payload, allow_nan=False))

    @classmethod
    def load(cls, path, **kwargs):
        from ..config import TrainConfig

        with zipfile.ZipFile(path) as stream:
            data = json.loads(stream.read("qlearning.json"))
        if data["schema"] != SCHEMA:
            raise ValueError("incompatible compact-state schema")
        obj = cls(TrainConfig(**data["config"]))
        obj.num_timesteps = data["steps"]
        obj._n_updates = data["updates"]
        obj.rng.bit_generator.state = data["rng"]
        for key, values, visits in data["table"]:
            if len(values) != 6 or not np.isfinite(values).all():
                raise ValueError("invalid Q table")
            obj.q[tuple(key)] = np.asarray(values)
            obj.visits[tuple(key)] = visits
        return obj


def build(env, tc):
    return QLearning(tc)


def load(path, **kwargs):
    return QLearning.load(path, **kwargs)
