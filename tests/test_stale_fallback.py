"""Local dispatch must preserve delayed task visibility and fresh commit guards."""

from dataclasses import asdict
from time import perf_counter

import numpy as np
import pytest

from fleetrl.config import FleetConfig
from fleetrl.env import FleetEnv
from fleetrl.types import Decision, DecisionPlan, ScenarioEvent, Task


def delayed_env(action_mode: str = "discrete") -> FleetEnv:
    env = FleetEnv(
        FleetConfig(
            n_robots=3,
            horizon_s=60,
            initial_tasks=1,
            demand_per_hour=0,
            disturbances=False,
            observation_delay_s=15,
            initial_soc_min=0.99,
            initial_soc_max=0.99,
            controller="heuristic",
            action_mode=action_mode,
        )
    )
    env.reset(seed=1005)
    env.sim.advance(15)
    env._observe()
    return env


@pytest.mark.parametrize("mode", ["discrete", "continuous", "multiagent"])
def test_stale_fallback_dispatches_known_tasks_without_using_policy(mode: str) -> None:
    env = delayed_env(mode)
    action = 0 if mode == "discrete" else np.zeros(5 if mode == "continuous" else 20)
    _, _, _, _, info = env.step(action)
    plan = info["decision"]
    assert plan["fallback_reason"] == "stale_snapshot_local_fallback"
    assert plan["metadata"]["policy_snapshot_age_s"] == 15
    assert plan["metadata"]["execution_observed_at_s"] == 15
    assert plan["metadata"]["policy_action_used"] is False
    assert any(d["kind"] == "task" for d in plan["commit"]["accepted"])
    assert not plan["commit"]["rejected"]


def test_local_fallback_does_not_admit_unobserved_task_or_refresh_policy() -> None:
    env = delayed_env()
    task = Task(999, env.sim.warehouse.pickups[0], env.sim.warehouse.dropoffs[0], 5, 16, 100)
    env.sim.inject_event(ScenarioEvent(16, "task", asdict(task)))
    env.sim.advance(5)
    env._observe()
    assert 999 not in {t.id for t in env._decision_snapshot.tasks}
    observed_at = env._decision_snapshot.observed_at_s
    plan = env._stale_fallback(env._decision_snapshot, perf_counter() + 1)
    assert 999 not in plan.metadata["known_task_ids"]
    assert all(d.task_id != 999 for d in plan.decisions)
    assert env._decision_snapshot.observed_at_s == observed_at
    assert env.sim.tasks[999].status == "waiting"


def test_local_fallback_respects_live_pause_and_preserves_stale_commit_rejection() -> None:
    env = delayed_env()
    old = env._decision_snapshot
    for robot in env.sim.robots.values():
        robot.paused_until_s = 100
    plan = env._stale_fallback(old, perf_counter() + 1)
    assert all(d.kind == "wait" for d in plan.decisions)
    result = env.sim.submit_plan(DecisionPlan(old.version, 0, [Decision(0, "wait")]))
    assert result["rejected"][0]["reason"] == "stale_snapshot"


def test_stale_fallback_does_not_override_exhausted_inference_budget() -> None:
    env = delayed_env()
    env.set_inference_context(300, "policy_inference_budget_exceeded")
    _, _, _, _, info = env.step(0)
    assert info["decision"]["fallback_reason"] == "policy_inference_budget_exceeded"
    assert all(d["kind"] == "wait" for d in info["decision"]["decisions"])
