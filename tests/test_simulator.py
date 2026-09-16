"""Simulator regression tests use tiny maps with checkable physical answers."""

from dataclasses import asdict
from math import ceil

import pytest

from fleetrl.config import FleetConfig
from fleetrl.energy import charge_duration, energy_for, travel_time
from fleetrl.maps import save_map
from fleetrl.planner import ReservationTable
from fleetrl.simulator import Simulator
from fleetrl.types import (
    ChargeBooking,
    Decision,
    DecisionPlan,
    EventTape,
    GridMap,
    Port,
    Robot,
    ScenarioEvent,
    Task,
)


def make_sim(tmp_path, n=1, **updates):
    warehouse = GridMap(
        9,
        7,
        set(),
        ((2, 2), (5, 2)),
        ((5, 3),),
        ((1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)),
        (Port(0, 0, (7, 5), 600.0, ("L", "M", "H"), ((5, 5), (6, 5), (5, 6), (6, 6))),),
    )
    path = tmp_path / "map.json"
    save_map(path, warehouse)
    config = FleetConfig(
        n_robots=n,
        map_path=str(path),
        initial_tasks=0,
        demand_per_hour=0.0,
        disturbances=False,
        horizon_s=2000.0,
        initial_soc_min=0.8,
        initial_soc_max=0.8,
        **updates,
    )
    sim = Simulator(config)
    sim.reset(3, EventTape(3, []))
    return sim


def set_robots(sim, robots):
    sim.robots = {r.id: r for r in robots}
    sim.reservations = ReservationTable({r.id: r.position for r in robots})


def robot(id=0, position=(1, 1), kind="M", battery=240.0):
    specs = {
        "L": (180.0, 2.0, 20.0, 0.035),
        "M": (300.0, 1.0, 50.0, 0.055),
        "H": (450.0, 0.75, 100.0, 0.085),
    }
    capacity, speed, payload, coefficient = specs[kind]
    return Robot(id, kind, position, capacity, min(battery, capacity), speed, payload, coefficient)


def commit(sim, decisions):
    snapshot = sim.snapshot()
    return sim.submit_plan(DecisionPlan(snapshot.version, 0, decisions, "TEST"))


def add_task(sim, id=0, pickup=(2, 2), dropoff=(5, 3), weight=15.0):
    task = Task(id, pickup, dropoff, weight, sim.time_s, 200.0)
    sim._apply_event(ScenarioEvent(sim.time_s, "task", asdict(task)))
    return sim.tasks[id]


def test_complete_task_has_real_transit_service_and_energy(tmp_path):
    sim = make_sim(tmp_path)
    r = robot(position=(1, 2))
    set_robots(sim, [r])
    task = add_task(sim)
    assert commit(sim, [Decision(0, "task", task_id=task.id)])["accepted"]
    initial = r.battery_wh
    sim.advance(0.5)
    assert r.position == (1, 2) and r.move_to == (2, 2)
    assert task.status == "assigned"
    sim.advance(0.5)
    assert r.position == (1, 2)  # medium edge needs a full second
    sim.advance(0.5)
    assert r.position == (2, 2) and task.status == "picking"
    sim.advance(10.0)
    assert r.load_kg == 15.0 and task.status == "carrying"
    sim.advance(20.0)
    assert task.status == "completed" and task.completed_at_s >= 25.0
    assert sim.counters["distance_m"] == 5.0
    expected = energy_for(r, 1.0, sim.time_s, 0.0, sim.config) + energy_for(
        r, 4.0, 0.0, 15.0, sim.config
    )
    assert initial - r.battery_wh == pytest.approx(expected)
    assert sim.counters["consumed_wh"] == pytest.approx(expected)
    assert sim.counters["completed"] == 1


@pytest.mark.parametrize("kind,edge_seconds", [("L", 0.5), ("M", 1.0), ("H", 1.5)])
def test_heterogeneous_speed_ceil_tick(tmp_path, kind, edge_seconds):
    sim = make_sim(tmp_path)
    r = robot(position=(1, 1), kind=kind)
    set_robots(sim, [r])
    assert commit(sim, [Decision(0, "reposition", target_position=(2, 1))])["accepted"]
    sim.advance(0.5)
    assert r.move_to == (2, 1)
    if edge_seconds > 0.5:
        sim.advance(edge_seconds - 0.5)
        assert r.position == (1, 1)
    sim.advance(0.5)
    assert r.position == (2, 1)


def test_duplicate_assignment_payload_and_energy_guards_are_atomic(tmp_path):
    sim = make_sim(tmp_path, 2)
    set_robots(sim, [robot(0, (1, 1), "L", 100), robot(1, (3, 1), "M", 32)])
    task = add_task(sim, weight=40.0)
    result = commit(
        sim, [Decision(0, "task", task_id=task.id), Decision(1, "task", task_id=task.id)]
    )
    assert len(result["rejected"]) == 2
    assert {r["reason"] for r in result["rejected"]} == {
        "payload_exceeded",
        "insufficient_energy_or_charge_access",
    }
    assert task.status == "waiting" and task.robot_id is None
    sim.robots[1].battery_wh = 200.0
    result = commit(
        sim, [Decision(1, "task", task_id=task.id), Decision(0, "task", task_id=task.id)]
    )
    assert len(result["accepted"]) == 1 and len(result["rejected"]) == 1


def test_delayed_snapshot_contains_no_future_tasks_and_rejects_stale_plan(tmp_path):
    sim = make_sim(tmp_path)
    initial = sim.snapshot()
    sim.inject_event(ScenarioEvent(5.0, "task", asdict(Task(7, (2, 2), (5, 3), 5.0, 5.0, 200.0))))
    sim.advance(15.0)
    delayed = sim.snapshot(delay_s=15.0)
    assert delayed.observed_at_s == 0.0 and delayed.sim_time_s == 15.0
    assert len(delayed.tasks) == 0 and len(sim.snapshot().tasks) == 1
    delayed.robots[0].battery_wh = 1.0
    assert sim.robots[0].battery_wh != 1.0  # snapshot isolation
    result = sim.submit_plan(DecisionPlan(initial.version, 0, [Decision(0, "wait")]))
    assert result["rejected"][0]["reason"] == "stale_snapshot"


def test_plan_replay_is_rejected(tmp_path):
    sim = make_sim(tmp_path)
    snapshot = sim.snapshot()
    plan = DecisionPlan(snapshot.version, 0, [Decision(0, "wait")])
    assert sim.submit_plan(plan)["accepted"]
    assert sim.submit_plan(plan)["rejected"][0]["reason"] == "duplicate_plan"


def test_pause_cancels_unpicked_work_but_preserves_carried_goods(tmp_path):
    sim = make_sim(tmp_path)
    r = robot(position=(1, 2))
    set_robots(sim, [r])
    task = add_task(sim)
    commit(sim, [Decision(0, "task", task_id=task.id)])
    sim.inject_event(ScenarioEvent(0.0, "pause", {"robot_id": 0, "duration_s": 3.0}))
    sim.advance(0.5)
    assert task.status == "waiting" and task.robot_id is None and r.task_id is None
    sim.advance(3.0)
    commit(sim, [Decision(0, "task", task_id=task.id)])
    sim.advance(12.0)
    assert r.load_kg == task.weight_kg
    sim.inject_event(ScenarioEvent(sim.time_s, "pause", {"robot_id": 0, "duration_s": 4.0}))
    sim.advance(0.5)
    task_id = r.task_id
    sim.advance(1.5)
    held_position = r.position
    sim.advance(1.0)
    assert r.position == held_position and r.task_id == task_id and r.load_kg == task.weight_kg
    sim.advance(30.0)
    assert task.status == "completed"


def test_block_occupied_cell_rejected_and_empty_block_routes_around(tmp_path):
    sim = make_sim(tmp_path)
    r = robot(position=(1, 1))
    set_robots(sim, [r])
    sim._apply_event(ScenarioEvent(0.0, "block", {"cell": r.position, "duration_s": 10.0}))
    assert r.position not in sim.blocked_cells
    assert sim.counters["blocked_incidents_rejected"] == 1
    sim._apply_event(ScenarioEvent(0.0, "block", {"cell": (2, 1), "duration_s": 20.0}))
    assert commit(sim, [Decision(0, "reposition", target_position=(3, 1))])["accepted"]
    visited = []
    for _ in range(24):
        sim.advance(0.5)
        visited.append(r.position)
    assert (2, 1) not in visited and r.position == (3, 1)
    assert sim.counters["distance_m"] == 4


def charge_decision(sim, r, target=0.6, start=None):
    port = sim.ports[0]
    distance = sim._distance(r.position, port.position)
    arrival = sim.time_s + travel_time(r, distance, sim.config)
    start = (
        ceil(arrival / sim.config.decision_s) * sim.config.decision_s if start is None else start
    )
    projected = max(
        0.0,
        r.battery_wh
        - energy_for(
            r, distance, travel_time(r, distance, sim.config) + 600.0 + 5.0, 0.0, sim.config
        ),
    )
    end = (
        start
        + 2 * sim.config.docking_s
        + charge_duration(r, projected, target, port.power_w, sim.config)
    )
    return Decision(
        r.id,
        "charge",
        port_id=port.id,
        target_soc=target,
        arrival_s=arrival,
        start_s=start,
        end_s=end,
    )


@pytest.mark.parametrize("target", [0.6, 0.8, 0.95])
def test_partial_charge_real_queue_docking_and_target_release(tmp_path, target):
    sim = make_sim(tmp_path)
    r = robot(position=(5, 5), battery=150.0)
    set_robots(sim, [r])
    initial = r.battery_wh
    decision = charge_decision(sim, r, target)
    result = commit(sim, [decision])
    assert result["accepted"], result
    for _ in range(3500):
        sim.advance(0.5)
        if sim.booking_history[0].actual_end_s is not None and not sim.bookings:
            break
    booking = sim.booking_history[0]
    assert booking.actual_start_s is not None and booking.actual_end_s is not None
    assert booking.actual_start_s >= sim.config.docking_s
    assert r.battery_wh == pytest.approx(target * r.capacity_wh, abs=0.15)
    assert r.position != sim.ports[0].position
    assert sim.counters["grid_wh"] > 0 and sim.counters["charge_sessions"] == 1
    # Auxiliary consumption is included once; charging efficiency accounts
    # separately for grid loss and measured physical battery change.
    balance = (
        initial
        + sim.counters["grid_wh"] * sim.config.charge_efficiency
        - sim.counters["consumed_wh"]
    )
    assert r.battery_wh == pytest.approx(balance, abs=0.002)


def test_booking_overlap_and_finite_queue_rejected(tmp_path):
    sim = make_sim(tmp_path, 2)
    a, b = robot(0, (5, 5), battery=100.0), robot(1, (6, 5), battery=100.0)
    set_robots(sim, [a, b])
    first = charge_decision(sim, a)
    second = charge_decision(sim, b, start=first.start_s)
    result = commit(sim, [first, second])
    assert len(result["accepted"]) == 1
    assert result["rejected"][0]["reason"] == "port_overlap"
    assert len(sim.bookings) == 1


def test_late_arrival_extends_interval_tail_without_overlap(tmp_path):
    sim = make_sim(tmp_path, 2)
    a, b = robot(0, (1, 1), battery=150.0), robot(1, (3, 1), battery=150.0)
    set_robots(sim, [a, b])
    first = charge_decision(sim, a)
    charge_decision(sim, b, start=first.end_s)
    # The second slot may be outside the 300s bound at this SoC; choose a
    # manually scheduled history to directly validate runtime repair.
    sim.bookings = [
        ChargeBooking(0, 0, 10.0, 10.0, 30.0, 0.6),
        ChargeBooking(1, 0, 20.0, 30.0, 50.0, 0.6),
    ]
    sim.booking_history = list(sim.bookings)
    a.port_id = b.port_id = 0
    a.status = b.status = "to_charge_queue"
    a.paused_until_s = 100.0
    sim.advance(20.0)
    ordered = sorted(sim.bookings, key=lambda x: x.start_s)
    assert ordered[1].start_s >= ordered[0].end_s
    assert any(e["kind"] == "booking_rescheduled" for e in sim.events)


def test_two_robot_motion_never_collides(tmp_path):
    sim = make_sim(tmp_path, 2)
    a, b = robot(0, (1, 2)), robot(1, (5, 2))
    set_robots(sim, [a, b])
    add_task(sim, 0, pickup=(5, 2), dropoff=(5, 3), weight=5.0)
    add_task(sim, 1, pickup=(1, 2), dropoff=(5, 3), weight=5.0)
    commit(sim, [Decision(0, "task", task_id=0), Decision(1, "task", task_id=1)])
    for _ in range(200):
        sim.advance(0.5)
        assert a.position != b.position
        assert not (a.move_to == b.position and b.move_to == a.position)
    assert sim.counters["collisions"] == sim.counters["edge_conflicts"] == 0


def test_deadlock_attempts_and_unresolved_hold_are_logged(tmp_path):
    sim = make_sim(tmp_path, 2, deadlock_s=2.0)
    a, b = robot(0, (1, 2)), robot(1, (2, 2))
    set_robots(sim, [a, b])
    # Loaded robot in a one-cell passage cannot yield through its neighbour.
    for y in range(sim.warehouse.height):
        for x in range(sim.warehouse.width):
            if y != 2:
                sim.planner.set_blocked((x, y), True)
    a.status = "to_dropoff"
    a.load_kg = 5.0
    a.target_position = (3, 2)
    sim.advance(8.0)
    assert sim.counters["deadlocks"] == 1
    assert sim.counters["deadlocks_unresolved"] == 1
    assert a.status == "hold" and a.position == (1, 2)
    assert sum(e["kind"] == "deadlock_recovery_attempt" for e in sim.events) == 2


def test_energy_fault_keeps_task_and_nonnegative_battery(tmp_path):
    sim = make_sim(tmp_path)
    r = robot(position=(1, 2))
    set_robots(sim, [r])
    task = add_task(sim)
    commit(sim, [Decision(0, "task", task_id=task.id)])
    r.battery_wh = 0.0  # explicit fault injection, never a normal scenario event
    sim.advance(5.0)
    assert r.battery_wh == 0.0 and r.status == "hold"
    assert task.id in sim.tasks and task.status == "waiting"
    assert sim.counters["energy_emergencies"] == 1


def test_idle_robot_yields_delivery_endpoint_before_congestion(tmp_path):
    sim = make_sim(tmp_path, 2, deadlock_s=2.0)
    a, b = robot(0, (4, 3)), robot(1, (5, 3))
    set_robots(sim, [a, b])
    task = add_task(sim, pickup=(4, 3), dropoff=(5, 3), weight=5.0)
    commit(sim, [Decision(0, "task", task_id=task.id)])
    sim.advance(35.0)
    assert task.status == "completed"
    assert any(e["kind"] == "endpoint_blocker_yield" for e in sim.events)
    assert sim.counters["deadlocks_unresolved"] == 0


def test_unblock_cannot_remove_static_map_wall(tmp_path):
    sim = make_sim(tmp_path)
    sim.planner.set_blocked((4, 4), True)
    sim.warehouse.walls.add((4, 4))
    sim._apply_event(ScenarioEvent(0.0, "unblock", {"cell": (4, 4)}))
    assert sim._path((4, 3), (4, 4)) is None


def test_reward_integrals_include_fractional_deadline_crossing(tmp_path):
    sim = make_sim(tmp_path)
    task = add_task(sim)
    task.deadline_s = 0.75
    task.priority = 3
    sim.advance(2.0)
    assert sim.counters["backlog_integral_s"] == pytest.approx(2.0)
    assert sim.counters["late_integral_s"] == pytest.approx((2.0 - 0.75) * 3.0)
    task.completed_at_s = 2.0
    sim.advance(1.0)
    assert sim.counters["backlog_integral_s"] == pytest.approx(2.0)
    assert sim.counters["late_integral_s"] == pytest.approx(3.75)


def test_full_station_queue_rejects_charge_without_side_effect(tmp_path):
    sim = make_sim(tmp_path)
    r = robot(position=(1, 1), battery=150.0)
    set_robots(sim, [r])
    for index, cell in enumerate(sim.ports[0].queue_cells):
        sim._queue_owners[cell] = 100 + index
    result = commit(sim, [charge_decision(sim, r)])
    assert result["rejected"][0]["reason"] == "queue_full"
    assert r.status == "idle" and not sim.bookings and not sim.booking_history


def test_multiple_carriers_complete_without_surrounding_delivery_cell(tmp_path):
    sim = make_sim(tmp_path, 3, deadlock_s=3.0)
    set_robots(sim, [robot(0, (1, 1)), robot(1, (2, 1)), robot(2, (3, 1))])
    for index in range(3):
        add_task(sim, index, pickup=(index + 1, 2), dropoff=(5, 3), weight=5.0)
    result = commit(sim, [Decision(index, "task", task_id=index) for index in range(3)])
    assert len(result["accepted"]) == 3
    sim.advance(150.0)
    assert sim.counters["completed"] == 3
    assert sim.counters["collisions"] == sim.counters["deadlocks_unresolved"] == 0


def test_exceeded_queue_wait_is_visible_emergency_and_cancels_booking(tmp_path):
    sim = make_sim(tmp_path, charger_wait_bound_s=2.0)
    r = robot(position=(5, 5), battery=150.0)
    set_robots(sim, [r])
    booking = ChargeBooking(0, 0, 0.0, 20.0, 100.0, 0.6)
    sim.bookings = [booking]
    sim.booking_history = [booking]
    sim._queue_owners[(5, 5)] = 0
    r.port_id = 0
    r.target_soc = 0.6
    r.status = "waiting_charge"
    r.metadata.update(queue_cell=(5, 5), charge_queue_arrived_s=0.0)
    sim.advance(3.0)
    assert sim.counters["energy_emergencies"] == 1
    assert booking.cancelled_at_s is not None and not sim.bookings
    assert (5, 5) not in sim._queue_owners
    assert any(
        e["kind"] == "energy_emergency" and e["reason"] == "charge_queue_wait_bound_exceeded"
        for e in sim.events
    )


def test_predictably_late_booking_cancelled_before_wait_bound(tmp_path):
    sim = make_sim(tmp_path, charger_wait_bound_s=30.0)
    r = robot(position=(5, 5), battery=150.0)
    set_robots(sim, [r])
    booking = ChargeBooking(0, 0, 0.0, 100.0, 200.0, 0.6)
    sim.bookings = [booking]
    sim.booking_history = [booking]
    sim._queue_owners[(5, 5)] = 0
    r.port_id = 0
    r.target_soc = 0.6
    r.status = "waiting_charge"
    r.metadata.update(queue_cell=(5, 5), charge_queue_arrived_s=0.0)
    sim.advance(5.0)
    assert booking.cancelled_at_s == 5.0 and not sim.bookings
    assert sim.counters["energy_emergencies"] == 0
    assert r.status == "reposition" and r.target_position in sim.warehouse.parking
    assert any(e["kind"] == "charge_booking_cancelled" for e in sim.events)
