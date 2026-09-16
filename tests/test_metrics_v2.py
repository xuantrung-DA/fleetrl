import numpy as np
from test_simulator import make_sim, robot, set_robots

from fleetrl.metrics import METRIC_GROUPS, _paired_ci, fleet_metrics


def test_sixteen_groups_and_undefined_denominators(tmp_path):
    sim = make_sim(tmp_path)
    metrics = fleet_metrics(sim, [])
    assert len(METRIC_GROUPS) == 16
    assert metrics["throughput_per_hour"] is None
    assert metrics["completion_rate"] is None
    assert metrics["mean_wait_s"] is None
    assert metrics["mean_charge_wait_s"] is None
    assert metrics["fallback_rate"] is None


def test_feasible_is_not_automatically_a_timeout(tmp_path):
    sim = make_sim(tmp_path)
    logs = [{"solver_status": "FEASIBLE", "metadata": {"solver_called": True, "timed_out": None}}]
    metrics = fleet_metrics(sim, logs)
    assert metrics["solver_nonoptimal_rate"] == 1
    assert metrics["solver_timeout_rate"] is None
    assert metrics["solver_timeout_unknown_calls"] == 1


def test_missing_whole_training_replica_withholds_ci():
    candidate = []
    reference = []
    for seed in [2000, 2001]:
        reference.append({"seed": seed, "tape_hash": str(seed), "metrics": {"completed": 2}})
        for training_seed in [11, 12]:
            candidate.append(
                {
                    "seed": seed,
                    "tape_hash": str(seed),
                    "training_seed": training_seed,
                    "metrics": {"completed": 3},
                }
            )
    result = _paired_ci(
        candidate, reference, "completed", 100, np.random.default_rng(1), [2000, 2001], [11, 12, 13]
    )
    assert result["complete_crossed_grid"] is False
    assert result["ci95"] is None


def test_physical_audit_catches_payload_and_task_ownership(tmp_path):
    sim = make_sim(tmp_path)
    r = robot()
    r.load_kg = 90
    r.task_id = 999
    set_robots(sim, [r])
    sim._audit_operations()
    sim._audit_operations()
    metrics = fleet_metrics(sim)
    assert metrics["payload_violations"] == 1
    assert metrics["task_ownership_violations"] == 1


def test_fractional_charge_time_energy_conservation(tmp_path):
    sim = make_sim(tmp_path)
    r = robot(position=(7, 5), battery=239.999)
    r.target_soc = 0.8
    r.port_id = 0
    r.status = "charging"
    set_robots(sim, [r])
    initial = r.battery_wh
    sim._charge(r)
    assert 0 < sim.port_usage[0]["charging_s"] < sim.config.tick_s
    metrics = fleet_metrics(sim, initial_battery_wh=initial)
    assert abs(metrics["battery_balance_error_wh"]) < 1e-8


def test_first_movement_does_not_confirm_deadlock_recovery(tmp_path):
    sim = make_sim(tmp_path)
    r = robot()
    set_robots(sim, [r])
    sim.deadlock_incidents = [
        {
            "incident_id": 1,
            "robot_id": r.id,
            "detected_s": 0,
            "first_progress_s": 1,
            "progress_since_s": 1,
            "last_progress_s": 1,
            "resolved_s": None,
            "resolution": None,
        }
    ]
    sim.time_s = 2
    metrics = fleet_metrics(sim)
    assert metrics["deadlock_confirmed_resolved"] == 0 and metrics["deadlock_pending"] == 1
    sim._resolve_deadlock(r, "operation_completed")
    assert fleet_metrics(sim)["deadlock_confirmed_resolved"] == 1
