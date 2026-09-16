"""Fixed-size, masked, causally observable inputs; no future event-tape access."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .config import FleetConfig
from .types import Snapshot

STATUS_NAMES = (
    "idle",
    "to_pickup",
    "pickup",
    "to_dropoff",
    "dropoff",
    "to_charge_queue",
    "waiting_charge",
    "to_charge_port",
    "docking",
    "charging",
    "charge_egress",
    "reposition",
    "recovering",
    "hold",
)


def observation_space() -> spaces.Dict:
    def box(shape):
        return spaces.Box(-5.0, 5.0, shape=shape, dtype=np.float32)

    return spaces.Dict(
        {
            "robots": box((20, 14)),
            "robot_mask": spaces.Box(0, 1, (20,), np.float32),
            "tasks": box((40, 10)),
            "task_mask": spaces.Box(0, 1, (40,), np.float32),
            "ports": box((4, 6)),
            "port_mask": spaces.Box(0, 1, (4,), np.float32),
            "global": box((40,)),
        }
    )


def select_tasks(snapshot: Snapshot, limit: int = 40):
    pending = [t for t in snapshot.tasks if t.completed_at_s is None]
    # Half oldest, half priority/deadline: starvation protection independent of RL.
    old = sorted(pending, key=lambda t: (t.created_s, t.id))[: max(1, limit // 2)]
    ids = {t.id for t in old}
    urgent = sorted(
        (t for t in pending if t.id not in ids), key=lambda t: (-t.priority, t.deadline_s, t.id)
    )
    return sorted(
        (old + urgent[: limit - len(old)]), key=lambda t: (t.deadline_s, t.created_s, t.id)
    )


def encode_observation(snapshot: Snapshot, config: FleetConfig) -> dict[str, np.ndarray]:
    out = {k: np.zeros(v.shape, dtype=np.float32) for k, v in observation_space().spaces.items()}
    now = snapshot.sim_time_s
    w = max(1, snapshot.warehouse.width - 1)
    h = max(1, snapshot.warehouse.height - 1)
    H = max(1, config.horizon_s)
    robots = sorted(snapshot.robots, key=lambda r: r.id)
    for i, r in enumerate(robots[:20]):
        status = STATUS_NAMES.index(r.status) if r.status in STATUS_NAMES else len(STATUS_NAMES) - 1
        out["robots"][i] = [
            r.position[0] / w,
            r.position[1] / h,
            r.soc,
            r.speed_mps / 2,
            r.payload_capacity_kg / 100,
            r.capacity_wh / 450,
            r.movement_wh_m / 0.085,
            r.load_kg / max(1, r.payload_capacity_kg),
            status / (len(STATUS_NAMES) - 1),
            r.target_soc or 0,
            (r.move_remaining_s + r.service_remaining_s) / H,
            max(0, now - r.observed_at_s) / 15,
            snapshot.warehouse.zone_of(r.position) / 3,
            float(r.task_id is not None),
        ]
        out["robot_mask"][i] = 1
    for i, t in enumerate(select_tasks(snapshot, config.max_tasks_observed)):
        out["tasks"][i] = [
            t.pickup[0] / w,
            t.pickup[1] / h,
            t.dropoff[0] / w,
            t.dropoff[1] / h,
            t.weight_kg / 100,
            max(0, now - t.created_s) / H,
            (t.deadline_s - now) / H,
            t.priority / 3,
            t.zone / 3,
            sum(r.payload_capacity_kg >= t.weight_kg for r in robots) / max(1, len(robots)),
        ]
        out["task_mask"][i] = 1
    live = [
        b
        for b in snapshot.bookings
        if b.cancelled_at_s is None and b.actual_end_s is None and b.end_s > now
    ]
    for i, p in enumerate(sorted(snapshot.ports, key=lambda p: p.id)[:4]):
        bookings = [b for b in live if b.port_id == p.id]
        out["ports"][i] = [
            p.position[0] / w,
            p.position[1] / h,
            p.power_w / 600,
            max([max(0, b.end_s - now) for b in bookings] + [0]) / 900,
            sum(b.actual_start_s is None for b in bookings) / 4,
            float("H" in p.compatible_kinds),
        ]
        out["port_mask"][i] = 1
    tasks = list(snapshot.tasks)
    pending = [t for t in tasks if t.completed_at_s is None]
    soc = [r.soc for r in robots] or [0]
    g = out["global"]
    g[:16] = [
        now / H,
        len(robots) / 20,
        len(tasks) / 200,
        (len(tasks) - len(pending)) / 200,
        len(pending) / 40,
        sum(r.task_id is not None for r in robots) / 20,
        sum(max(0, now - t.created_s) for t in pending) / (40 * H),
        sum(max(0, now - t.deadline_s) for t in pending) / (40 * H),
        sum(s < 0.3 for s in soc) / 20,
        sum(r.status == "idle" for r in robots) / 20,
        np.mean(soc),
        min(soc),
        np.mean([p.power_w for p in snapshot.ports] or [0]) / 600,
        float(snapshot.warehouse.name == "B"),
        max(0, now - snapshot.observed_at_s) / 15,
        len(snapshot.blocked_cells) / 10,
    ]
    for zone in range(4):
        g[16 + zone] = sum(now - 60 < t.created_s <= now and t.zone == zone for t in tasks) / 10
        g[20 + zone] = sum(t.zone == zone for t in pending) / 40
        g[24 + zone] = sum(snapshot.warehouse.zone_of(r.position) == zone for r in robots) / 20
        g[28 + zone] = sum(now - (zone + 1) * 60 < t.created_s <= now for t in tasks) / 40
        g[32 + zone] = sum(t.weight_kg == [5, 15, 40, 80][zone] for t in pending) / 40
        g[36 + zone] = sum(t.priority == 3 and t.zone == zone for t in pending) / 40
    for arr in out.values():
        np.clip(arr, -5, 5, out=arr)
    if any(not np.isfinite(a).all() for a in out.values()):
        raise RuntimeError("non-finite observation")
    return out
