"""Consistent Wh/s/metre model, shared by admission, execution and tests."""

from __future__ import annotations

import math

from .config import FleetConfig
from .types import Robot


def reserve_wh(robot: Robot, config: FleetConfig) -> float:
    return robot.capacity_wh * config.reserve_soc


def travel_time(robot: Robot, distance_m: float, config: FleetConfig) -> float:
    """Per 1m edge rounding preserves discrete simulator speed semantics."""
    if distance_m < 0:
        raise ValueError("distance cannot be negative")
    if not math.isfinite(distance_m):
        return math.inf
    per_edge = math.ceil((1.0 / robot.speed_mps) / config.tick_s - 1e-12) * config.tick_s
    return distance_m * per_edge


def energy_for(
    robot: Robot, distance_m: float, elapsed_s: float, load_kg: float, config: FleetConfig
) -> float:
    if min(distance_m, elapsed_s, load_kg) < 0:
        raise ValueError("energy inputs must be nonnegative")
    traction = (
        robot.movement_wh_m
        * (1 + config.load_factor * load_kg / robot.payload_capacity_kg)
        * distance_m
    )
    return traction * config.energy_multiplier + config.auxiliary_power_w * elapsed_s / 3600.0


def charge_duration(
    robot: Robot, energy_wh: float, target_soc: float, power_w: float, config: FleetConfig
) -> float:
    """Net battery charging time, rounded to tick; docking is separate."""
    if not 0 <= energy_wh <= robot.capacity_wh + 1e-6 or not 0 < target_soc <= 1:
        raise ValueError("invalid battery energy or target")
    target = robot.capacity_wh * target_soc
    if target <= energy_wh:
        return 0.0
    r1 = (config.charge_efficiency * power_w - config.auxiliary_power_w) / 3600
    r2 = (config.charge_efficiency * power_w * 0.5 - config.auxiliary_power_w) / 3600
    if min(r1, r2) <= 0:
        raise ValueError("charger cannot overcome auxiliary load")
    mid = 0.8 * robot.capacity_wh
    low = max(0, min(target, mid) - energy_wh)
    high = max(0, target - max(mid, energy_wh))
    seconds = low / r1 + high / r2
    return math.ceil(seconds / config.tick_s - 1e-12) * config.tick_s


def charge_step(
    robot: Robot, power_w: float, dt: float, config: FleetConfig
) -> tuple[float, float]:
    """Return (net battery Wh, grid Wh) without mutating robot. Split at80%."""
    net, grid, _ = charge_step_details(robot, power_w, dt, config)
    return net, grid


def charge_step_details(
    robot: Robot, power_w: float, dt: float, config: FleetConfig, target_soc: float = 1.0
):
    """Net battery gain, grid energy and actual active seconds, split at 80%/target."""
    if dt < 0 or not 0 < target_soc <= 1:
        raise ValueError("invalid charging interval or target")
    energy = max(0.0, min(robot.battery_wh, robot.capacity_wh))
    initial = energy
    grid = 0.0
    remaining = dt
    target = target_soc * robot.capacity_wh
    for limit, factor in ((min(0.8 * robot.capacity_wh, target), 1.0), (target, 0.5)):
        if energy >= limit - 1e-12:
            continue
        rate = (config.charge_efficiency * power_w * factor - config.auxiliary_power_w) / 3600
        if rate <= 0:
            raise ValueError("charger net rate must be positive")
        seconds = min(remaining, (limit - energy) / rate)
        energy += seconds * rate
        grid += power_w * factor * seconds / 3600
        remaining -= seconds
        if remaining <= 1e-12:
            break
    return max(0, energy - initial), grid, dt - remaining
