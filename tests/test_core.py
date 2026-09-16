import math
from dataclasses import asdict

import numpy as np
import pytest

from fleetrl.config import FleetConfig, TrainConfig, load_config, save_config
from fleetrl.energy import charge_duration, charge_step, energy_for, travel_time
from fleetrl.env import FleetEnv
from fleetrl.maps import distance, load_map, make_map, save_map
from fleetrl.scenario import generate_tape, make_robots, scenario_config
from fleetrl.types import EventTape, Robot, ScenarioEvent, Task


def test_energy_worked_example_and_tick_speeds():
    c = FleetConfig()
    r = Robot(0, "M", (1, 1), 300, 90, 1, 50, 0.055)
    assert energy_for(r, 20, 20, 0, c) + energy_for(r, 50, 50, 40, c) + energy_for(
        r, 30, 50, 0, c
    ) == pytest.approx(7)
    assert energy_for(r, 0, 600, 0, c) == pytest.approx(2)
    assert travel_time(Robot(1, "L", (0, 0), 180, 100, 2, 20, 0.035), 1, c) == 0.5
    assert travel_time(r, 1, c) == 1
    assert travel_time(Robot(2, "H", (0, 0), 450, 200, 0.75, 100, 0.085), 1, c) == 1.5


def test_charge_crosses_80_without_wrong_power_or_overcharge():
    c = FleetConfig()
    r = Robot(0, "M", (0, 0), 300, 239.99, 1, 50, 0.055)
    net, grid = charge_step(r, 600, 1, c)
    first = 0.01 / ((540 - 12) / 3600)
    expected = 0.01 + (1 - first) * (270 - 12) / 3600
    assert net == pytest.approx(expected)
    assert grid == pytest.approx(first * 600 / 3600 + (1 - first) * 300 / 3600)
    r.battery_wh = 299.99
    assert charge_step(r, 600, 100, c)[0] == pytest.approx(0.01)
    assert charge_duration(r, 299.99, 0.95, 600, c) == 0


def test_configuration_roundtrip_and_unknown_key_rejection(tmp_path):
    c = FleetConfig()
    t = TrainConfig()
    p = tmp_path / "cfg.yaml"
    save_config(p, c, t)
    a, b = load_config(p)
    assert asdict(a) == asdict(c) and asdict(b) == asdict(t)
    p.write_text("env:\n  n_robot: 14\n")
    with pytest.raises(TypeError):
        load_config(p)
    with pytest.raises(ValueError):
        FleetConfig(decision_s=0.7).validate()
    with pytest.raises(ValueError):
        TrainConfig(eval_seeds=(50,)).validate()


@pytest.mark.parametrize("name", [f"S{i}" for i in range(1, 10)])
def test_every_test_scenario_valid_reproducible(name, tmp_path):
    c = scenario_config(name)
    m = make_map(c)
    t = generate_tape(c, m, 2000)
    assert t.tape_hash == generate_tape(c, m, 2000).tape_hash
    assert t.tape_hash == EventTape.from_dict(t.to_dict()).tape_hash
    assert all(e.time_s < c.horizon_s for e in t.events)
    path = tmp_path / "map.json"
    save_map(path, m)
    other = load_map(path)
    assert asdict(m) == asdict(other)
    assert math.isfinite(distance(m, m.pickups[0], m.dropoffs[-1]))


def test_s4_s6_match_low_soc_subset_and_moved_station_intent():
    c = scenario_config("S4")
    robots = make_robots(c, make_map(c), 2000)
    assert sum(r.soc <= 0.4 for r in robots) >= round(0.4 * c.n_robots)
    assert any(r.soc > 0.4 for r in robots)
    a = make_map(scenario_config("S2"))
    b = make_map(scenario_config("S6"))
    assert a.ports[0].position != b.ports[0].position
    assert all(p.power_w == 300 for p in b.ports)


def test_all_pending_tasks_counted_even_outside_observation():
    c = FleetConfig(
        n_robots=5, horizon_s=10, initial_tasks=0, demand_per_hour=0, disturbances=False
    )
    tasks = [Task(i, (7, 6), (3, 6), 80, 0, 1, 3, 0) for i in range(60)]
    tape = EventTape(2000, [ScenarioEvent(0, "task", asdict(t)) for t in tasks])
    env = FleetEnv(c)
    obs, _ = env.reset(seed=2000, options={"tape": tape})
    assert obs["task_mask"].sum() == 40 and len(env.sim.tasks) == 60
    while True:
        obs, r, done, _, info = env.step(0)
        if done:
            break
    m = info["metrics"]
    assert m["arrived"] == 60 and m["completed"] + m["pending"] == 60
    assert m["total_lateness_s"] >= m["pending"] * 9
    assert info["reward_components"]["terminal"] < 0
    env.close()


def test_tape_future_not_observed_and_invalid_action_has_fallback():
    c = FleetConfig(
        n_robots=3, horizon_s=20, initial_tasks=0, demand_per_hour=0, disturbances=False
    )
    t = Task(99, (7, 6), (3, 6), 5, 15, 100)
    env = FleetEnv(c)
    obs, _ = env.reset(
        seed=1, options={"tape": EventTape(1, [ScenarioEvent(15, "task", asdict(t))])}
    )
    assert obs["task_mask"].sum() == 0
    obs, r, done, trunc, info = env.step(float("nan"))
    assert info["decision"]["fallback_reason"] == "invalid_policy_action"
    assert obs["task_mask"].sum() == 0 and np.isfinite(r)
    env.close()


def test_snapshot_mutation_cannot_mutate_engine():
    env = FleetEnv(FleetConfig(n_robots=3, horizon_s=10))
    env.reset(seed=2)
    snap = env.snapshot()
    snap.robots[0].battery_wh = 0
    assert env.sim.robots[snap.robots[0].id].battery_wh > 0
    env.close()


def test_episode_requires_reset_after_termination():
    env = FleetEnv(FleetConfig(n_robots=3, horizon_s=5))
    env.reset(seed=1)
    _, _, term, trunc, _ = env.step(0)
    assert term and not trunc
    with pytest.raises(RuntimeError):
        env.step(0)
    env.close()
