"""Gymnasium integration: PPO profile -> CP-SAT -> guarded simulator execution."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import FleetConfig
from .metrics import fleet_metrics
from .observations import encode_observation, observation_space
from .optimizer import FleetOptimizer, ObjectiveProfile
from .simulator import Simulator
from .types import EventTape


class FleetEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, config: FleetConfig | None = None, render_mode=None):
        self.config = (config or FleetConfig()).validate()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(6)
        self.observation_space = observation_space()
        if self.config.action_mode == "continuous":
            self.action_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
        if self.config.observation_mode == "compact":
            from .learning.compact_state import compact_space

            self.observation_space = compact_space()
        if self.config.action_mode == "multiagent":
            from .methods.mappo_dispatch import MAX_CHOICES, dispatch_space

            self.action_space = spaces.MultiDiscrete([MAX_CHOICES] * 20)
            self.observation_space = dispatch_space()
        self.sim = Simulator(self.config)
        self.optimizer = FleetOptimizer(self.config)
        if self.config.backend == "mpc":
            from .methods.forecast_mpc_milp import MPCController

            self.optimizer = MPCController(self.config)
        self.decision_logs = []
        self.initial_battery_wh = 0.0
        self._done = True
        self._last_frame = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        tape = (options or {}).get("tape")
        if isinstance(tape, dict):
            tape = EventTape.from_dict(tape)
        self.sim.reset(seed=int(seed), tape=tape)
        self.decision_logs = []
        self._done = False
        self.initial_battery_wh = sum(r.battery_wh for r in self.sim.robots.values())
        if hasattr(self.optimizer, "reset"):
            self.optimizer.reset()
        self._inference_ms = 0.0
        self._inference_failure = None
        return self._observe(), {"seed": int(seed), "tape_hash": self.sim.tape.tape_hash}

    def snapshot(self):
        return self.sim.snapshot(delay_s=self.config.observation_delay_s)

    def _observe(self):
        started = time.perf_counter()
        self._decision_snapshot = self.snapshot()
        obs = encode_observation(self._decision_snapshot, self.config)
        if self.config.observation_mode == "compact":
            from .learning.compact_state import compact

            obs = {"compact": compact(obs)}
        if self.config.action_mode == "multiagent":
            from .methods.mappo_dispatch import dispatch_observation

            obs, self._choices = dispatch_observation(self.optimizer, self._decision_snapshot, obs)
        self._preparation_ms = (time.perf_counter() - started) * 1000
        return obs

    def set_inference_context(self, elapsed_ms=0.0, failure=None):
        self._inference_ms = float(elapsed_ms)
        self._inference_failure = failure

    def set_tick_observer(self, observer: Callable[[], None] | None = None) -> None:
        """Register a read-only callback invoked after every physical simulator tick."""
        self._tick_observer = observer

    def step(self, action):
        if self._done:
            raise RuntimeError("episode finished; call reset before step")
        start = time.perf_counter()
        invalid = False
        weights = None
        deadline = (
            start
            + self.config.decision_budget_s
            - (self._preparation_ms + self._inference_ms) / 1000
        )
        try:
            value = np.asarray(action)
            if not np.isfinite(value).all():
                raise ValueError
            if self.config.action_mode == "continuous":
                if value.shape != (5,) or (np.abs(value) > 1).any():
                    raise ValueError
                decoded = np.array([0.5, 0.5, 0.5, 0.5, 0.2]) + (value + 1) / 2 * np.array(
                    [2.5, 2.5, 2.5, 2.5, 0.8]
                )
                weights = ObjectiveProfile("continuous", *decoded.tolist())
                profile = 0
            elif self.config.action_mode == "multiagent":
                if value.shape != (20,) or not np.equal(value, np.floor(value)).all():
                    raise ValueError
                profile = 0
            else:
                if value.size != 1:
                    raise ValueError
                profile = int(value.item())
                if float(value.item()) != profile or not 0 <= profile < 6:
                    raise ValueError
        except (TypeError, ValueError, OverflowError):
            profile = self.config.fixed_profile
            invalid = True
        pre = dict(self.sim.counters)
        before = self.sim.time_s
        snapshot = self._decision_snapshot
        if time.perf_counter() >= deadline or self._inference_failure:
            plan = self.optimizer._hold(
                snapshot, profile, self._inference_failure or "decision_preparation_budget"
            )
        elif self.config.action_mode == "multiagent" and not invalid:
            from .methods.mappo_dispatch import arbitrate

            plan = arbitrate(self.optimizer, snapshot, self._choices, value, deadline)
        else:
            plan = self.optimizer.decide(snapshot, profile, weights=weights, deadline=deadline)
        if invalid:
            plan.fallback_reason = "invalid_policy_action"
        if self.config.action_mode != "discrete":
            plan.profile_id = None
        plan.metadata.update(
            action_mode=self.config.action_mode,
            raw_action=np.asarray(action).tolist(),
            objective_weights=asdict(weights) if weights else None,
        )
        commit_start = time.perf_counter()
        result = self.sim.submit_plan(plan)
        # Online latency ends after guarding/committing plan, excludes subsequent simulation.
        commit_ms = (time.perf_counter() - commit_start) * 1000
        plan.total_ms = (
            (time.perf_counter() - start) * 1000 + self._preparation_ms + self._inference_ms
        )
        record = asdict(plan)
        record["commit"] = result
        record["sim_time_s"] = before
        record.update(
            preparation_ms=self._preparation_ms,
            policy_inference_ms=self._inference_ms,
            commit_ms=commit_ms,
            latency_accounted=True,
        )
        self._inference_ms = 0.0
        self._inference_failure = None
        record["budget_exceeded"] = plan.total_ms > self.config.decision_budget_s * 1000
        self.decision_logs.append(record)
        dt = min(self.config.decision_s, self.config.horizon_s - self.sim.time_s)
        advance_start = time.perf_counter()
        if getattr(self, "_tick_observer", None) is None:
            self.sim.advance(dt)
        else:
            remaining = dt
            while remaining > 1e-9:
                tick = min(self.config.tick_s, remaining)
                self.sim.advance(tick)
                self._tick_observer()
                remaining -= tick
        record["advance_ms"] = (time.perf_counter() - advance_start) * 1000
        after = self.sim.time_s
        self._done = after >= self.config.horizon_s - 1e-8
        q_after = sum(t.completed_at_s is None for t in self.sim.tasks.values())

        def delta(k):
            return max(0, self.sim.counters.get(k, 0) - pre.get(k, 0))

        c = delta("completed")
        q = delta("backlog_integral_s") / max(dt, 1e-9)
        late_backlog = delta("late_integral_s") / max(dt, 1e-9)
        violations = sum(
            delta(k)
            for k in ("collisions", "edge_conflicts", "energy_emergencies", "reserve_violations")
        )
        scale = dt / 5.0
        components = {
            "completed": c,
            "backlog": -self.config.reward_backlog * q / 40 * scale,
            "late": -self.config.reward_late * late_backlog / 40 * scale,
            "energy": -self.config.reward_energy * delta("consumed_wh") / 10,
            "violations": -self.config.reward_violation * violations,
            "terminal": 0.0,
        }
        if self._done:
            lateness = sum(
                max(0, (t.completed_at_s if t.completed_at_s is not None else after) - t.deadline_s)
                for t in self.sim.tasks.values()
            )
            components["terminal"] = (
                -self.config.reward_terminal_backlog * q_after / 40
                - self.config.reward_terminal_late * lateness / 600
            )
        reward = float(sum(components.values()))
        if not np.isfinite(reward):
            raise RuntimeError("non-finite reward")
        info = {"reward_components": components, "decision": record, "sim_time_s": after}
        if self._done:
            info["metrics"] = self.metrics()
        return self._observe(), reward, self._done, False, info

    def metrics(self):
        return fleet_metrics(self.sim, self.decision_logs, self.initial_battery_wh)

    def render(self):
        from .ui import render_frame

        return render_frame(self.sim.snapshot(), self.metrics())

    def close(self):
        self._last_frame = None
