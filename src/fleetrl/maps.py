"""Warehouse JSON roundtrip and cached geometry (not collision avoidance)."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from .config import FleetConfig
from .types import Cell, GridMap, Port


def make_map(config: FleetConfig) -> GridMap:
    if config.map_path:
        return load_map(config.map_path)
    walls = set()
    bands = ((5, 10), (16, 23)) if config.map_name == "A" else ((5, 12), (18, 24))
    for x0 in (8, 16, 24, 32):
        for y0, y1 in bands:
            walls.update((x, y) for x in range(x0, x0 + 3) for y in range(y0, y1))
    pickups = (
        (7, 6),
        (15, 8),
        (11, 18),
        (19, 20),
        (23, 6),
        (31, 8),
        (27, 18),
        (35, 20),
        (7, 22),
        (15, 22),
        (23, 22),
        (31, 22),
    )
    dropoffs = ((3, 6), (36, 6), (3, 24), (36, 24))
    parking = tuple((x, y) for y in (1, 28) for x in (5, 6, 7, 11, 12, 13, 19, 20, 21, 28, 29, 30))
    qa = ((1, 3), (2, 3), (3, 3), (4, 3))
    qb = ((35, 26), (36, 26), (37, 26), (38, 26))
    ports = (
        Port(0, 0, (2, 2), config.charger_power_w, ("L", "M"), qa),
        Port(1, 0, (3, 2), config.charger_power_w, ("L", "M"), qa),
        Port(2, 1, (36, 27), config.charger_power_w, ("L", "M", "H"), qb),
        Port(3, 1, (37, 27), config.charger_power_w, ("L", "M", "H"), qb),
    )
    if config.map_name == "B":
        ports = tuple(
            Port(
                p.id,
                p.station_id,
                (p.position[0], 14),
                p.power_w,
                p.compatible_kinds,
                tuple((x, 13 if p.station_id == 0 else 15) for x, _ in p.queue_cells),
            )
            for p in ports
        )
        pickups = tuple((x, 15 if y == 18 else y) for x, y in pickups)
    elif config.shift_stations:
        positions = ((4, 14), (5, 14), (34, 14), (35, 14))
        queues = (((2, 13), (3, 13), (4, 13), (5, 13)), ((34, 15), (35, 15), (36, 15), (37, 15)))
        ports = tuple(
            Port(
                p.id,
                p.station_id,
                positions[p.id],
                p.power_w,
                p.compatible_kinds,
                queues[p.station_id],
            )
            for p in ports
        )
    m = GridMap(40, 30, walls, pickups, dropoffs, parking, ports, config.map_name)
    validate_map(m)
    return m


def validate_map(warehouse: GridMap) -> None:
    if warehouse.width < 3 or warehouse.height < 3:
        raise ValueError("map too small")
    if not warehouse.pickups or not warehouse.dropoffs or not warehouse.parking:
        raise ValueError("map needs pickup, dropoff and parking cells")
    if not 1 <= len(warehouse.ports) <= 4:
        raise ValueError("map requires 1..4 ports")
    ids = [p.id for p in warehouse.ports]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate port id")
    points = list(warehouse.pickups) + list(warehouse.dropoffs) + list(warehouse.parking)
    points += [p.position for p in warehouse.ports]
    points += [c for p in warehouse.ports for c in p.queue_cells]
    if any(not warehouse.is_free(p) for p in points):
        raise ValueError("map endpoint in wall or outside grid")
    if len(set(warehouse.parking)) != len(warehouse.parking):
        raise ValueError("duplicate parking cell")
    if any(not p.queue_cells for p in warehouse.ports):
        raise ValueError("port station must have queue cells")
    start = warehouse.parking[0]
    if any(not math.isfinite(distance(warehouse, start, c)) for c in points):
        raise ValueError("map endpoints must be connected")


@lru_cache(maxsize=256)
def _bfs(width: int, height: int, walls: frozenset, blocked: frozenset, start: Cell):
    parents = {start: None}
    dist = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for p in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (
                not (0 <= p[0] < width and 0 <= p[1] < height)
                or p in walls
                or p in blocked
                or p in parents
            ):
                continue
            parents[p] = (x, y)
            dist[p] = dist[(x, y)] + 1
            queue.append(p)
    return parents, dist


def shortest_path(
    warehouse: GridMap, start: Cell, goal: Cell, blocked: set[Cell] | None = None
) -> list[Cell] | None:
    start = tuple(start)
    goal = tuple(goal)
    if not warehouse.is_free(goal, blocked):
        return None
    parents, _ = _bfs(
        warehouse.width,
        warehouse.height,
        frozenset(warehouse.walls),
        frozenset(blocked or ()),
        start,
    )
    if goal not in parents:
        return None
    path = []
    p = goal
    while p is not None:
        path.append(p)
        p = parents[p]
    return path[::-1]


def distance(
    warehouse: GridMap, start: Cell, goal: Cell, blocked: set[Cell] | None = None
) -> float:
    start = tuple(start)
    goal = tuple(goal)
    if not warehouse.is_free(goal, blocked):
        return math.inf
    _, dist = _bfs(
        warehouse.width,
        warehouse.height,
        frozenset(warehouse.walls),
        frozenset(blocked or ()),
        start,
    )
    return float(dist.get(goal, math.inf))


def save_map(path: str | Path, warehouse: GridMap) -> None:
    body = asdict(warehouse)
    body["walls"] = sorted(warehouse.walls)
    Path(path).write_text(json.dumps(body, indent=2), encoding="utf-8")


def load_map(path: str | Path) -> GridMap:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("pickups", "dropoffs", "parking"):
        d[k] = tuple(tuple(c) for c in d[k])
    d["walls"] = {tuple(c) for c in d["walls"]}
    d["ports"] = tuple(
        Port(
            **{
                **p,
                "position": tuple(p["position"]),
                "compatible_kinds": tuple(p["compatible_kinds"]),
                "queue_cells": tuple(tuple(c) for c in p["queue_cells"]),
            }
        )
        for p in d["ports"]
    )
    m = GridMap(**d)
    validate_map(m)
    return m
