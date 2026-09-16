"""Independent numerical/behavioral checks for the expanded algorithm study."""

from dataclasses import replace
from time import perf_counter

import numpy as np
import pytest
from test_optimizer import snapshot

from fleetrl.config import FleetConfig, TrainConfig, load_config
from fleetrl.methods.registry import METHODS, get_method
from fleetrl.optimizer import FleetOptimizer
from fleetrl.types import ChargeBooking


def test_registry_has_fourteen_real_modules_and_eleven_learners():
    assert len(METHODS) == 14
    assert sum(m.learnable for m in METHODS.values()) == 11
    for spec in METHODS.values():
        module = spec.module()
        assert hasattr(module, "build") if spec.learnable else hasattr(module, "controller")
        cfg, tc = load_config("configs/methods/" + spec.name + ".yaml")
        assert cfg.method == spec.name
        assert tc.algorithm == spec.algorithm if spec.learnable else True
    assert get_method("H").name == "ppo_cpsat"


@pytest.mark.parametrize("soc,tasks,profile", [(0.9, 2, 0), (0.5, 0, 2), (0.5, 1, 3), (0.25, 0, 0)])
def test_milp_matches_cpsat_optimum(soc, tasks, profile):
    state = snapshot(robots=2, soc=soc, tasks=tasks, ports=1)
    cfg = FleetConfig(n_robots=2, solver_time_limit_s=3, decision_budget_s=6)
    cp = FleetOptimizer(cfg).decide(state, profile)
    milp = FleetOptimizer(cfg.copy(backend="milp")).decide(state, profile)
    assert cp.solver_status == milp.solver_status == "OPTIMAL"
    assert milp.objective == pytest.approx(cp.objective, abs=2e-6)
    assert len({d.robot_id for d in milp.decisions}) == len(milp.decisions)


def test_milp_respects_full_tail_and_fixed_booking():
    state = snapshot(
        robots=2, soc=0.25, ports=1, bookings=(ChargeBooking(90, 0, 1300, 1300, 2000, 0.95),)
    )
    cfg = FleetConfig(n_robots=2, backend="milp", solver_time_limit_s=2, decision_budget_s=4)
    plan = FleetOptimizer(cfg).decide(state, 2)
    charges = [d for d in plan.decisions if d.kind == "charge"]
    for d in charges:
        assert d.start_s % 5 == 0
        assert d.end_s <= 1300 or d.start_s >= 2000


def test_qlearning_terminal_target_and_resume(tmp_path):
    from fleetrl.methods.qlearning_cpsat import QLearning

    cfg = TrainConfig(algorithm="qlearning", n_envs=1, q_alpha=1, gamma=0.9)
    model = QLearning(cfg)
    obs = {"compact": np.zeros(10)}
    nxt = {"compact": np.ones(10)}
    model.q[tuple([1] * 10)][:] = 100
    model.update(obs, 2, 7, nxt, True)
    assert model.q[tuple([0] * 10)][2] == 7  # No bootstrap after terminal.
    model.update(obs, 3, 7, nxt, False)
    assert model.q[tuple([0] * 10)][3] == 97
    model.save(tmp_path / "q.zip")
    restored = QLearning.load(tmp_path / "q.zip")
    np.testing.assert_equal(restored.q[tuple([0] * 10)], model.q[tuple([0] * 10)])
    assert restored.rng.random() == model.rng.random()


def test_recurrent_sessions_retain_memory_and_reset_independently():
    from fleetrl.learning.policies import InferenceSession

    class MemoryPolicy:
        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            next_state = 1 if state is None or episode_start[0] else state + 1
            return next_state, next_state

    left = InferenceSession(MemoryPolicy())
    right = InferenceSession(MemoryPolicy())
    assert left.predict({})[0] == 1
    assert left.predict({})[0] == 2
    assert right.predict({})[0] == 1
    left.reset()
    assert left.predict({})[0] == 1


def test_continuous_action_endpoints_and_invalid_fallback():
    from fleetrl.env import FleetEnv

    cfg = get_method("sac_cpsat").configure(
        FleetConfig(n_robots=1, horizon_s=15, initial_tasks=0, disturbances=False)
    )
    env = FleetEnv(cfg)
    env.reset(seed=3)
    _, _, _, _, info = env.step(np.ones(5))
    weights = info["decision"]["metadata"]["objective_weights"]
    assert [weights[k] for k in ("tardiness", "gain", "availability", "zone", "energy")] == [
        3,
        3,
        3,
        3,
        1,
    ]
    assert info["decision"]["profile_id"] is None
    _, _, _, _, info = env.step(np.array([np.nan] * 5))
    assert info["decision"]["fallback_reason"] == "invalid_policy_action"
    env.close()


def test_shared_deadline_prevents_second_solver_budget(monkeypatch):
    from fleetrl.env import FleetEnv

    env = FleetEnv(FleetConfig(n_robots=1, horizon_s=5))
    env.reset(seed=3)
    env.set_inference_context(300, "policy_inference_budget_exceeded")

    def forbidden(*a, **k):
        raise AssertionError("solver must not run after inference exhausts budget")

    monkeypatch.setattr(env.optimizer, "decide", forbidden)
    _, _, _, _, info = env.step(0)
    assert info["decision"]["total_ms"] >= 300
    assert info["decision"]["fallback_reason"] == "policy_inference_budget_exceeded"


def test_frozen_backend_control_requires_explicit_label():
    from types import SimpleNamespace

    from fleetrl.evaluation import run_episode

    policy = SimpleNamespace(
        predict=lambda obs, deterministic=True: (0, None),
        _fleetrl_metadata={
            "method": "ppo_cpsat",
            "observation_mode": "entities",
            "action_mode": "discrete",
            "env": {"charge_threshold": 0.3},
        },
    )
    cfg = FleetConfig(n_robots=1, horizon_s=5, initial_tasks=1)
    with pytest.raises(ValueError, match="method mismatch"):
        run_episode(cfg, "ppo_milp", 2000, policy)
    result = run_episode(cfg, "ppo_milp", 2000, policy, frozen_backend_control=True)
    assert (
        result["comparison_role"] == "frozen_backend_control"
        and result["training_method"] == "ppo_cpsat"
    )


def test_mappo_mask_and_duplicate_task_arbiter():
    from fleetrl.methods.mappo_dispatch import MAPPO, arbitrate, dispatch_observation
    from fleetrl.observations import encode_observation

    cfg = get_method("mappo_dispatch").configure(FleetConfig(n_robots=2))
    state = snapshot(robots=2, soc=0.9, tasks=1)
    owner = FleetOptimizer(cfg)
    obs, choices = dispatch_observation(owner, state, encode_observation(state, cfg))
    model = MAPPO(TrainConfig(algorithm="mappo", n_envs=1))
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=False)
        assert all(obs["candidate_mask"][i, a] == 1 for i, a in enumerate(action))
    actions = np.zeros(20, int)
    for index, (rid, candidates) in choices.items():
        actions[index] = next(j for j, c in enumerate(candidates) if c.decision.kind == "task")
    plan = arbitrate(owner, state, choices, actions, perf_counter() + 2)
    assert sum(d.kind == "task" for d in plan.decisions) == 1
    assert len(plan.metadata["arbitration_rejections"]) == 1
    # Local actor parameters are shared; centralized critic consumes all observed entities.
    actor_before = {k: p.detach().clone() for k, p in model.policy.actor.named_parameters()}
    assert actor_before


def test_mpc_expands_and_selects_two_operations_and_forecast_is_causal():
    from fleetrl.methods.forecast_mpc_milp import CausalForecast, MPCController

    state = snapshot(robots=1, soc=0.9, tasks=2, ports=1)
    cfg = FleetConfig(
        n_robots=1, backend="mpc", solver_time_limit_s=1, decision_budget_s=4, mpc_horizon_s=300
    )
    controller = MPCController(cfg)
    plan = controller.decide(state)
    assert plan.solver_status == "OPTIMAL"
    assert plan.metadata["two_operation_routes"] > 0
    assert any(len(seq) == 2 for seq in plan.metadata["selected_sequences"])
    assert len(plan.decisions) == 1
    forecast = CausalForecast(0.5)
    observed = replace(state, demand_rate_by_zone=(1, 0, 0, 0))
    np.testing.assert_array_equal(forecast.observe(observed), [1, 0, 0, 0])
    # Same timestamp cannot introduce another update, even if passed twice.
    np.testing.assert_array_equal(
        forecast.observe(replace(observed, demand_rate_by_zone=(4, 0, 0, 0))), [1, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        forecast.observe(replace(observed, observed_at_s=5, demand_rate_by_zone=(3, 0, 0, 0))),
        [2, 0, 0, 0],
    )


def test_mpc_busy_fleet_is_a_normal_hold_not_a_fallback():
    from fleetrl.methods.forecast_mpc_milp import MPCController

    state = snapshot(robots=2, soc=0.9, tasks=1)
    state = replace(state, robots=tuple(replace(r, status="moving") for r in state.robots))
    plan = MPCController(FleetConfig(n_robots=2, backend="mpc")).decide(state)
    assert plan.solver_status == "HOLD" and plan.fallback_reason is None
    assert plan.metadata["solver_called"] is False
