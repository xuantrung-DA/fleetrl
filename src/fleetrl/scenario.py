"""External event tapes, independent robot initialization and held-out scenarios."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import FleetConfig
from .maps import distance
from .types import EventTape, GridMap, Robot, ScenarioEvent, Task

ROBOT_SPECS = {
    "L": (180.0, 2.0, 20.0, 0.035),
    "M": (300.0, 1.0, 50.0, 0.055),
    "H": (450.0, 0.75, 100.0, 0.085),
}


def make_robots(config: FleetConfig, warehouse: GridMap, seed: int) -> list[Robot]:
    if len(warehouse.parking) < config.n_robots:
        raise ValueError("not enough unique starting parking cells")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 871]))
    heavy = max(1, round(config.n_robots * config.heavy_fraction)) if config.n_robots >= 3 else 0
    heavy = min(config.n_robots, heavy)
    light = (config.n_robots - heavy) // 2
    medium = config.n_robots - heavy - light
    kinds = ["L"] * light + ["M"] * medium + ["H"] * heavy
    positions = rng.permutation(len(warehouse.parking))[: config.n_robots]
    low_ids = set(
        rng.choice(
            config.n_robots, round(config.n_robots * config.low_soc_fraction), replace=False
        ).tolist()
    )
    robots = []
    for i, kind in enumerate(kinds):
        cap, v, payload, a = ROBOT_SPECS[kind]
        upper = (
            min(config.initial_soc_max, config.low_soc_max)
            if i in low_ids
            else config.initial_soc_max
        )
        robots.append(
            Robot(
                i,
                kind,
                warehouse.parking[int(positions[i])],
                cap,
                float(rng.uniform(config.initial_soc_min, upper)) * cap,
                v,
                payload,
                a,
            )
        )
    return robots


def generate_tape(config: FleetConfig, warehouse: GridMap, seed: int) -> EventTape:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 137]))
    events = []
    next_id = 0

    def add_task(t, zone):
        nonlocal next_id
        choices = [c for c in warehouse.pickups if warehouse.zone_of(c) == zone] or list(
            warehouse.pickups
        )
        pickup = choices[int(rng.integers(len(choices)))]
        drop = warehouse.dropoffs[int(rng.integers(len(warehouse.dropoffs)))]
        mass = float(rng.choice([5, 15, 40, 80], p=[0.30, 0.25, 0.30, 0.15]))
        priority = 3 if rng.random() < 0.2 else 1
        nominal = distance(warehouse, pickup, drop) + 2 * config.service_s + 15
        deadline = t + nominal * (1.5 if priority == 3 else float(rng.uniform(2, 3)))
        task = Task(next_id, pickup, drop, mass, float(t), float(deadline), priority, zone)
        events.append(ScenarioEvent(float(t), "task", asdict(task)))
        next_id += 1

    for _ in range(config.initial_tasks):
        add_task(0.0, int(rng.integers(4)))
    # Piecewise Poisson generated per-zone, never revealed to controller in advance.
    cuts = sorted(
        set(
            [
                0.0,
                config.horizon_s,
                max(0, min(config.horizon_s, config.burst_start_s)),
                max(0, min(config.horizon_s, config.burst_end_s)),
            ]
        )
    )
    for zone in range(4):
        for begin, end in zip(cuts[:-1], cuts[1:]):
            rate = config.demand_per_hour / 4 / 3600
            if config.burst_start_s <= begin < config.burst_end_s and zone == config.burst_zone:
                rate *= config.burst_multiplier
            if rate <= 0:
                continue
            t = begin + float(rng.exponential(1 / rate))
            while t < end:
                add_task(t, zone)
                t += float(rng.exponential(1 / rate))
    if config.disturbances:
        t = config.horizon_s * 0.45
        cell = (20, 14)
        if warehouse.is_free(cell):
            events += [
                ScenarioEvent(
                    t, "block", {"cell": cell, "duration_s": min(300.0, config.horizon_s * 0.1)}
                ),
                ScenarioEvent(t + min(300.0, config.horizon_s * 0.1), "unblock", {"cell": cell}),
            ]
        if config.n_robots > 1:
            events.append(
                ScenarioEvent(
                    config.horizon_s * 0.65,
                    "pause",
                    {"robot_id": 0, "duration_s": min(60.0, config.horizon_s * 0.1)},
                )
            )
    events.sort(key=lambda e: (e.time_s, 0 if e.kind == "task" else 1))
    return EventTape(
        seed, events, {"config": config.to_dict(), "map_name": warehouse.name, "schema_version": 1}
    )


def save_tape(path: str | Path, tape: EventTape) -> None:
    Path(path).write_text(json.dumps(tape.to_dict(), indent=2), encoding="utf-8")


def load_tape(path: str | Path) -> EventTape:
    return EventTape.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def scenario_config(name: str, base: FleetConfig | None = None) -> FleetConfig:
    c = (base or FleetConfig()).copy()
    name = name.upper()
    options = {
        "S1": dict(n_robots=10, demand_per_hour=60, burst_multiplier=1, disturbances=False),
        "S2": dict(n_robots=15, demand_per_hour=90, burst_multiplier=1, disturbances=False),
        "S3": dict(n_robots=20, demand_per_hour=120, burst_multiplier=1, disturbances=False),
        "S4": dict(
            n_robots=15,
            demand_per_hour=90,
            burst_multiplier=2,
            initial_soc_min=0.25,
            initial_soc_max=0.9,
            low_soc_fraction=0.4,
            disturbances=False,
        ),
        "S5": dict(n_robots=20, demand_per_hour=120, heavy_fraction=0.4, disturbances=False),
        "S6": dict(
            n_robots=15,
            demand_per_hour=90,
            charger_power_w=300,
            shift_stations=True,
            disturbances=False,
        ),
        "S7": dict(n_robots=20, demand_per_hour=120, disturbances=True),
        "S8": dict(n_robots=15, demand_per_hour=90, observation_delay_s=15, disturbances=False),
        "S9": dict(
            n_robots=20,
            demand_per_hour=120,
            map_name="B",
            burst_multiplier=2.5,
            burst_zone=2,
            disturbances=True,
        ),
    }
    if name not in options:
        raise ValueError(f"unknown scenario {name}; expected S1..S9")
    return c.copy(**options[name])
