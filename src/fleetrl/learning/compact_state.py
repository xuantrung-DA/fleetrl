"""Frozen, causal compact representation shared by Q-learning and controls."""

import numpy as np
from gymnasium import spaces

SCHEMA = "compact-bins-v1"
INDICES = (0, 4, 7, 8, 9, 10, 16, 17, 18, 19)
BINS = np.array([0.05, 0.15, 0.3, 0.6, 1.0], dtype=np.float32)


def compact(observation):
    values = np.asarray(observation["global"])[list(INDICES)]
    return np.digitize(values, BINS).astype(np.float32)


def compact_space():
    return spaces.Dict({"compact": spaces.Box(0, len(BINS), (len(INDICES),), dtype=np.float32)})


def state_key(observation):
    values = observation["compact"] if "compact" in observation else compact(observation)
    return tuple(int(v) for v in values)
