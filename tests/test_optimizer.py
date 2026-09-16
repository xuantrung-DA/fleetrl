"""Optimizer resource, safety and behavioral regression tests."""

from dataclasses import replace

import pytest

from fleetrl.config import FleetConfig
from fleetrl.optimizer import FleetOptimizer, earliest_slot, queue_fits, validate_profile
from fleetrl.types import ChargeBooking, GridMap, Port, Robot, Snapshot, Task


def snapshot(robots=2, soc=0.5, tasks=0, bookings=(), ports=2):
    """Small connected warehouse keeps CP-SAT tests independent of map tuning."""
    queue = ((2, 1), (3, 1), (4, 1), (5, 1))
    ps = tuple(Port(i, 0, (i + 2, 2), queue_cells=queue) for i in range(ports))
    grid = GridMap(
        12,
        12,
        set(),
        ((4, 4), (8, 8)),
        ((7, 4), (9, 8)),
        ((1, 1), (1, 10), (10, 1), (10, 10), (6, 1), (6, 10)),
        ps,
    )
    rs = tuple(Robot(i, "L", (i + 1, 3), 180, 180 * soc, 2, 20, 0.035) for i in range(robots))
    ts = tuple(Task(i, (4, 4), (7, 4), 5, 0, 200, zone=0) for i in range(tasks))
    return Snapshot(0, 0, 0, rs, ts, ps, tuple(bookings), grid)


def optimizer(**updates):
    return FleetOptimizer(
        FleetConfig(n_robots=4, solver_time_limit_s=0.2, decision_budget_s=1.0, **updates)
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 6, 2.5, None, "bad"])
def test_invalid_action_has_explicit_fallback(value):
    assert validate_profile(value) == (0, "invalid_policy_action")


def test_charge_tail_can_extend_beyond_start_horizon():
    assert earliest_slot(890, 900, 2000, [], 10) == 890
    assert earliest_slot(891, 900, 2000, [], 10) == 900


def test_half_open_port_intervals_and_grid_rounding():
    assert earliest_slot(0, 20, 10, [(10, 20)]) == 0
    assert earliest_slot(0, 40, 10, [(0, 13)], 10) == 20
    assert earliest_slot(0, 15, 10, [(0, 13)], 10) is None


def test_queue_capacity_counts_only_overlapping_waits():
    assert queue_fits(0, 10, [(10, 20)], 1)
    assert not queue_fits(0, 10, [(9, 20)], 1)
    assert queue_fits(0, 10, [(0, 10), (0, 10), (0, 10)], 4)
    assert not queue_fits(0, 10, [(0, 10)] * 4, 4)
    assert queue_fits(10, 10, [(0, 20)] * 4, 0)


def test_busy_fleet_serves_work_instead_of_all_charge_or_wait():
    state = snapshot(robots=4, soc=0.45, tasks=4)
    plan = optimizer().decide(state, 0)
    assert plan.solver_status in {"OPTIMAL", "FEASIBLE"}
    assignments = [d for d in plan.decisions if d.kind == "task"]
    assert len(assignments) == 4
    assert len({d.task_id for d in assignments}) == 4


def test_same_soc_different_observed_demand_changes_work_or_charge():
    opt = optimizer()
    idle = opt.decide(snapshot(robots=1, soc=0.5, tasks=0, ports=1), 0)
    busy = opt.decide(snapshot(robots=1, soc=0.5, tasks=1, ports=1), 0)
    assert idle.decisions[0].kind == "charge"
    assert busy.decisions[0].kind == "task"


def test_candidate_generation_enforces_payload_and_energy_reserve():
    state = snapshot(robots=1, soc=0.12, tasks=1)
    state.tasks[0].weight_kg = 40
    opt = optimizer()
    assert not any(c.decision.kind == "task" for c in opt.generate_candidates(state))
    assert opt.last_rejections["payload"] == 1
    state.tasks[0].weight_kg = 5
    state.robots[0].battery_wh = 18.1
    assert not any(c.decision.kind == "task" for c in opt.generate_candidates(state))
    assert opt.last_rejections["insufficient_energy_or_return_calendar"] == 1


def test_charge_plans_have_nonoverlapping_ports_and_keep_long_tail():
    state = snapshot(robots=4, soc=0.25, ports=2)
    plan = optimizer().decide(state, 2)
    charges = [d for d in plan.decisions if d.kind == "charge"]
    assert charges
    for decision in charges:
        assert decision.start_s <= min(900, decision.arrival_s + 300)
        assert decision.start_s % 5 == 0
    assert any(d.end_s > 900 for d in charges)
    for port_id in (0, 1):
        times = sorted((d.start_s, d.end_s) for d in charges if d.port_id == port_id)
        assert all(a[1] <= b[0] for a, b in zip(times, times[1:]))


def test_station_queue_cumulative_blocks_a_fifth_waiting_robot():
    bookings = tuple(ChargeBooking(90 + i, i, 0, 250, 2000, 0.95) for i in range(4))
    state = snapshot(robots=1, soc=0.5, bookings=bookings, ports=4)
    opt = optimizer()
    # A60% session could fit a port's early free gap, but ingress rounding
    # would require waiting while all four physical queue cells are reserved.
    assert any(c.decision.kind == "charge" for c in opt.generate_candidates(state))
    plan = opt.decide(state, 2)
    assert all(d.kind != "charge" for d in plan.decisions)


def test_task_is_rejected_when_return_calendar_is_unavailable():
    booking = ChargeBooking(99, 0, 0, 0, 2000, 0.95)
    state = snapshot(robots=1, soc=0.9, tasks=1, bookings=(booking,), ports=1)
    candidates = optimizer().generate_candidates(state)
    assert not any(c.decision.kind == "task" for c in candidates)


def test_partial60_fits_a_gap_while80_and95_do_not():
    # Light robot50%→60 needs~140s, to80%~380s. Existing reservation
    # [250,2000) leaves only the60% operation eligible within300s.
    booking = ChargeBooking(99, 0, 250, 250, 2000, 0.95)
    state = snapshot(robots=1, soc=0.5, bookings=(booking,), ports=1)
    candidates = optimizer().generate_candidates(state)
    targets = {c.decision.target_soc for c in candidates if c.decision.kind == "charge"}
    assert targets == {0.6}
    plan = optimizer().decide(state, 2)
    assert [(d.kind, d.target_soc) for d in plan.decisions] == [("charge", 0.6)]


def test_profiles_change_partial_charge_target_without_changing_candidates():
    state = snapshot(robots=1, soc=0.5, ports=1)
    opt = optimizer()
    available = {
        c.decision.target_soc for c in opt.generate_candidates(state) if c.decision.kind == "charge"
    }
    assert available == {0.6, 0.8, 0.95}
    balanced = opt.decide(state, 2)
    preservation = opt.decide(state, 0)
    assert balanced.decisions[0].kind == preservation.decisions[0].kind == "charge"
    assert balanced.decisions[0].target_soc == 0.95
    assert preservation.decisions[0].target_soc == 0.8


def test_critical_battery_gets_port_before_healthy_elective_charging():
    state = snapshot(robots=2, soc=0.70, ports=1)
    state.robots[0].battery_wh = 0.14 * 180
    opt = optimizer()
    plan = opt.decide(state, 2)  # Even the store-energy profile respects urgency.
    assert plan.metadata["energy_urgent_robot_ids"] == [0]
    charges = [d for d in plan.decisions if d.kind == "charge"]
    assert len(charges) == 1
    assert charges[0].robot_id == 0
    assert charges[0].target_soc == 0.60
    assert charges[0].estimated_energy_wh + 0.10 * 180 <= state.robots[0].battery_wh


def test_subfive_percent_topups_are_not_charge_candidates():
    state = snapshot(robots=1, soc=0.799, ports=1)
    opt = optimizer()
    options = opt.generate_candidates(state)
    assert not any(c.decision.kind == "charge" and c.decision.target_soc == 0.80 for c in options)
    assert opt.last_rejections["charge_gain_below_5pct"] >= 1
    assert opt.decide(state, 0).decisions[0].kind != "charge"


def test_threshold_ablation_locks_timing_and95_target():
    state = snapshot(robots=1, soc=0.29, tasks=1, ports=1)
    plan = optimizer(advanced_charging=False).decide(state, 0)
    assert [(d.kind, d.target_soc) for d in plan.decisions] == [("charge", 0.95)]
    state.robots[0].battery_wh = 0.6 * 180
    assert not any(
        c.decision.kind == "charge"
        for c in optimizer(advanced_charging=False).generate_candidates(state)
    )


def test_stale_snapshot_and_invalid_policy_are_logged():
    state = snapshot(robots=1, tasks=1)
    stale = replace(state, sim_time_s=20)
    assert optimizer().decide(stale).fallback_reason == "stale_snapshot"
    plan = optimizer().decide(state, float("nan"))
    assert plan.profile_id == 0
    assert plan.fallback_reason == "invalid_policy_action"


def test_fixed_baseline_uses_exact_same_candidate_set():
    state = snapshot(robots=3, tasks=3)
    full = optimizer().generate_candidates(state)
    fixed = optimizer(controller="fixed", fixed_profile=3).generate_candidates(state)

    def key(c):
        return (
            c.decision.robot_id,
            c.decision.kind,
            c.decision.task_id,
            c.decision.port_id,
            c.decision.target_soc,
            c.decision.target_position,
        )

    assert [key(c) for c in full] == [key(c) for c in fixed]


def test_solver_failure_never_reads_a_nonexistent_solution(monkeypatch):
    from fleetrl.optimizer import cp_model

    monkeypatch.setattr(cp_model.CpSolver, "Solve", lambda *args: cp_model.UNKNOWN)
    state = snapshot(robots=2, tasks=2)
    plan = optimizer().decide(state)
    assert plan.solver_status == "FALLBACK"
    assert plan.fallback_reason == "solver_unknown"
    assert len({d.task_id for d in plan.decisions if d.kind == "task"}) == 2
