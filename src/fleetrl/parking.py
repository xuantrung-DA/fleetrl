"""Shared causal parking benefit and minimum residence guard."""

from __future__ import annotations

from .config import FleetConfig
from .energy import energy_for, reserve_wh, travel_time
from .types import Cell, Robot, Snapshot


def interruptible_parking(robot: Robot) -> bool:
    """Parking intent may yield to work; loaded, booked and emergency intents may not."""
    return (
        robot.status == "reposition"
        and robot.task_id is None
        and robot.port_id is None
        and robot.load_kg == 0
        and not robot.metadata.get("energy_emergency")
    )


def task_dispatchable(robot: Robot) -> bool:
    return robot.task_id is None and (
        (robot.status == "idle" and robot.move_to is None) or interruptible_parking(robot)
    )


def parking_rejection(
    snapshot: Snapshot, robot: Robot, target: Cell, distance: float, config: FleetConfig
) -> str | None:
    """Allow clearance to a bay; require net observed benefit to leave a legal bay."""
    if target == robot.position:
        return "already_parked"
    if robot.position not in snapshot.warehouse.parking:
        return None
    changed_at = robot.metadata.get("parking_changed_at_s")
    if changed_at is not None and snapshot.sim_time_s - changed_at < config.parking_min_residence_s:
        return "parking_min_residence"
    demand = [
        sum(t.zone == z and t.status in {"waiting", "blocked"} for t in snapshot.tasks)
        + snapshot.demand_rate_by_zone[z] * 60
        for z in range(4)
    ]
    total = sum(demand)
    if total <= 0:
        return "parking_no_benefit"
    robots = [r for r in snapshot.robots if r.port_id is None]
    n = max(1, len(snapshot.robots))
    target_total = sum(r.status in {"idle", "reposition"} or r.task_id is not None for r in robots)
    counts = [0] * 4
    for other in robots:
        ready = other.battery_wh > (
            reserve_wh(other, config) + config.auxiliary_power_w * 600 / 3600
        )
        if (
            other.task_id is None
            and other.status != "reposition"
            and not (other.status == "idle" and ready)
        ):
            continue
        # Committed parking destinations already contribute to their future zone.
        position = (
            other.target_position
            if other.status in {"reposition", "recovering"} and other.target_position is not None
            else other.position
        )
        counts[snapshot.warehouse.zone_of(position)] += 1
    desired = [target_total * d / total for d in demand]
    before = sum(abs(count - want) for count, want in zip(counts, desired))
    counts[snapshot.warehouse.zone_of(robot.position)] -= 1
    counts[snapshot.warehouse.zone_of(target)] += 1
    after = sum(abs(count - want) for count, want in zip(counts, desired))
    duration = travel_time(robot, distance, config)
    movement_cost = 0.3 * duration / 900 + 0.2 * energy_for(robot, distance, 0, 0, config) / 100
    gain = (before - after - movement_cost) / n
    return None if gain > config.parking_min_improvement + 1e-9 else "parking_no_benefit"
