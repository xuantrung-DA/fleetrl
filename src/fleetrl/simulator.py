"""Deterministic physical fleet simulator with independently checked execution.

The optimizer never mutates this module's state. Plans are revalidated against
live state, physical moves acquire conservative vertex/edge reservations, and
charging uses real travel, docking, finite queue cells and piecewise power.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from copy import deepcopy
from dataclasses import asdict
from math import ceil, isfinite

from .config import FleetConfig
from .energy import charge_duration, energy_for, reserve_wh, travel_time
from .maps import make_map
from .planner import GridPlanner, ReservationTable
from .scenario import generate_tape, make_robots
from .types import (
    ChargeBooking,
    Decision,
    DecisionPlan,
    EventTape,
    Robot,
    ScenarioEvent,
    Snapshot,
    Task,
)


class Simulator:
    """Own live state; ``snapshot`` returns an isolated historical observation."""

    def __init__(self, config: FleetConfig):
        self.config = config.validate()
        self.reset(config.seed)

    def reset(self, seed: int | None = None, tape: EventTape | None = None) -> Snapshot:
        self.seed = self.config.seed if seed is None else int(seed)
        self.warehouse = make_map(self.config)
        self.planner = GridPlanner(
            self.warehouse.width, self.warehouse.height, self.warehouse.walls
        )
        self.robots = {r.id: r for r in make_robots(self.config, self.warehouse, self.seed)}
        self.tasks: dict[int, Task] = {}
        self.ports = {p.id: deepcopy(p) for p in self.warehouse.ports}
        self.bookings: list[ChargeBooking] = []
        self.booking_history: list[ChargeBooking] = []
        self.reservations = ReservationTable({r.id: r.position for r in self.robots.values()})
        self.blocked_cells: set[tuple[int, int]] = set()
        self._blocked_until: dict[tuple[int, int], float] = {}
        self._queue_owners: dict[tuple[int, int], int] = {}
        self._endpoint_leases: dict[tuple[int, int], int] = {}
        self.time_s = 0.0
        self.tick = 0
        self.version = 0
        self.map_version = 0
        self.events: list[dict] = []
        self._event_serial = 0
        self.deadlock_incidents = []
        self._invariant_seen = set()
        self.invariant_incidents = []
        self.port_usage = {
            p.id: {"charging_s": 0.0, "occupied_s": 0.0} for p in self.ports.values()
        }
        self.counters = {
            k: 0.0
            for k in (
                "consumed_wh",
                "grid_wh",
                "completed",
                "collisions",
                "edge_conflicts",
                "energy_emergencies",
                "reserve_violations",
                "deadlocks",
                "deadlocks_resolved",
                "deadlocks_unresolved",
                "distance_m",
                "empty_distance_m",
                "loaded_distance_m",
                "charge_sessions",
                "rejected_decisions",
                "blocked_incidents_rejected",
                "backlog_integral_s",
                "late_integral_s",
                "charged_battery_wh",
            )
        }
        for kind in {"L", "M", "H"} | {r.kind for r in self.robots.values()}:
            for metric in (
                "consumed_wh",
                "grid_wh",
                "distance_m",
                "empty_distance_m",
                "loaded_distance_m",
            ):
                self.counters[f"robot_kind_{kind}_{metric}"] = 0.0
        self.tape = (
            deepcopy(tape)
            if tape is not None
            else generate_tape(self.config, self.warehouse, self.seed)
        )
        self.tape.events.sort(key=lambda e: e.time_s)
        self._event_cursor = 0
        self._history: deque[Snapshot] = deque(
            maxlen=max(
                8,
                ceil((max(30.0, self.config.observation_delay_s) + 10.0) / self.config.decision_s)
                + 2,
            )
        )
        self._submitted: set[str] = set()
        self._known_versions: dict[int, float] = {}
        self._process_events()
        self._history.append(self._live_snapshot())
        return self.snapshot()

    def _log(self, kind: str, **payload) -> None:
        self._event_serial += 1
        self.events.append(
            {"event_id": self._event_serial, "time_s": self.time_s, "kind": kind, **payload}
        )

    def _live_snapshot(self) -> Snapshot:
        recent = [t for t in self.tasks.values() if t.created_s >= self.time_s - 60.0]
        # Tasks/second, only the trailing observed 60 s. A full-window
        # denominator prevents an artificial infinite rate at episode start.
        rates = tuple(sum(t.zone == z for t in recent) / 60.0 for z in range(4))
        return Snapshot(
            self.time_s,
            self.time_s,
            self.version,
            tuple(deepcopy(list(self.robots.values()))),
            tuple(deepcopy(list(self.tasks.values()))),
            tuple(deepcopy(list(self.ports.values()))),
            tuple(deepcopy(self.bookings)),
            deepcopy(self.warehouse),
            frozenset(self.blocked_cells),
            rates,
            self.map_version,
        )

    def snapshot(self, delay_s: float = 0.0) -> Snapshot:
        """Return only information observed at/before ``now-delay``.

        Historical frames are captured on dispatch boundaries. An arbitrary
        non-grid delay rounds conservatively to an older frame, never forward.
        ``sim_time_s`` is now while ``observed_at_s`` identifies the frame age.
        """
        if delay_s < 0 or not isfinite(delay_s):
            raise ValueError("delay_s must be nonnegative and finite")
        if delay_s == 0:
            result = self._live_snapshot()
        else:
            target = self.time_s - delay_s
            eligible = [s for s in self._history if s.observed_at_s <= target + 1e-9]
            historical = eligible[-1] if eligible else self._history[0]
            result = deepcopy(historical)
            object.__setattr__(result, "sim_time_s", self.time_s)
        self._known_versions[result.version] = result.observed_at_s
        return result

    get_snapshot = snapshot

    def inject_event(self, event: ScenarioEvent) -> None:
        """Insert a recorded event; past requests take effect at the next tick."""
        event = deepcopy(event)
        event.time_s = max(float(event.time_s), self.time_s + self.config.tick_s)
        self.tape.events.append(event)
        self.tape.events[self._event_cursor :] = sorted(
            self.tape.events[self._event_cursor :], key=lambda e: e.time_s
        )
        self._log("event_injected", event=asdict(event))

    def _process_events(self) -> None:
        while (
            self._event_cursor < len(self.tape.events)
            and self.tape.events[self._event_cursor].time_s <= self.time_s + 1e-9
        ):
            event = self.tape.events[self._event_cursor]
            self._event_cursor += 1
            self._apply_event(event)
        for cell, until in list(self._blocked_until.items()):
            if self.time_s >= until:
                self._apply_event(ScenarioEvent(self.time_s, "unblock", {"cell": cell}))

    def _apply_event(self, event: ScenarioEvent) -> None:
        p = event.payload
        if event.kind == "task":
            data = dict(p.get("task", p))
            for key in ("pickup", "dropoff"):
                data[key] = tuple(data[key])
            data.setdefault("created_s", event.time_s)
            task = Task(**data)
            if task.id in self.tasks:
                self._log("event_rejected", reason="duplicate_task", task_id=task.id)
                return
            if task.weight_kg <= 0 or not isfinite(task.weight_kg):
                task.status, task.blocked_reason = "blocked", "invalid_weight"
            elif task.weight_kg > max(r.payload_capacity_kg for r in self.robots.values()):
                task.status, task.blocked_reason = "blocked", "payload_unavailable"
            elif self.planner.shortest_path(task.pickup, task.dropoff) is None:
                task.status, task.blocked_reason = "blocked", "disconnected"
            self.tasks[task.id] = task
            self._log("task_arrived", task_id=task.id, status=task.status)
        elif event.kind == "block":
            cell = tuple(p.get("cell", p.get("position", ())))
            occupied = {r.position for r in self.robots.values()} | {
                r.move_to for r in self.robots.values() if r.move_to is not None
            }
            if not self.warehouse.is_free(cell) or cell in occupied:
                self.counters["blocked_incidents_rejected"] += 1
                self._log("event_rejected", reason="block_occupied_or_invalid", cell=cell)
                return
            self.blocked_cells.add(cell)
            self.planner.set_blocked(cell, True)
            if "duration_s" in p:
                self._blocked_until[cell] = self.time_s + float(p["duration_s"])
            self.map_version += 1
            for r in self.robots.values():
                r.path.clear()
            self._log("blocked", cell=cell)
        elif event.kind == "unblock":
            cell = tuple(p.get("cell", p.get("position", ())))
            if cell not in self.blocked_cells:
                self._log("event_rejected", reason="unblock_not_temporary_obstacle", cell=cell)
                return
            self.blocked_cells.discard(cell)
            self._blocked_until.pop(cell, None)
            self.planner.set_blocked(cell, False)
            self.map_version += 1
            for task in self.tasks.values():
                if task.status == "blocked" and task.blocked_reason == "disconnected":
                    task.status, task.blocked_reason = "waiting", None
            self._log("unblocked", cell=cell)
        elif event.kind == "pause":
            robot = self.robots.get(int(p["robot_id"]))
            if robot is None:
                self._log("event_rejected", reason="unknown_robot")
                return
            # If an edge is already owned, complete that edge then stop at its
            # endpoint; stopping half a cell never releases the reservations.
            robot.paused_until_s = max(
                robot.paused_until_s, self.time_s + float(p.get("duration_s", 60.0))
            )
            if robot.task_id is not None and robot.load_kg == 0:
                task = self.tasks[robot.task_id]
                task.status, task.robot_id, task.assigned_at_s = "waiting", None, None
                robot.task_id = None
                robot.status = "idle"
                robot.target_position = None
                robot.path.clear()
                self._log("task_released", task_id=task.id, reason="robot_paused_before_pickup")
            self._log("robot_paused", robot_id=robot.id, until_s=robot.paused_until_s)
        else:
            self._log("event_rejected", reason="unknown_event", event_kind=event.kind)

    def _path(self, a, b):
        return self.planner.shortest_path(a, b)

    def _distance(self, a, b) -> float:
        path = self._path(a, b)
        return float("inf") if path is None else len(path) - 1

    def _task_required_energy(self, robot: Robot, task: Task) -> float:
        c = self.config
        a, b = (
            self._distance(robot.position, task.pickup),
            self._distance(task.pickup, task.dropoff),
        )
        if not isfinite(a + b):
            return float("inf")
        service_end = self.time_s + travel_time(robot, a + b, c) + 2 * c.service_s
        available = []
        for port in self.ports.values():
            if robot.kind not in port.compatible_kinds:
                continue
            d = self._distance(task.dropoff, port.position)
            if not isfinite(d):
                continue
            arrival = service_end + travel_time(robot, d, c)
            occupied = sorted((x.start_s, x.end_s) for x in self.bookings if x.port_id == port.id)
            first = arrival
            for left, right in occupied:
                if right <= first:
                    continue
                if left > first:
                    break
                first = right
            if first - arrival > c.charger_wait_bound_s + 1e-8:
                continue
            needed = (
                energy_for(robot, a, travel_time(robot, a, c), 0.0, c)
                + energy_for(robot, b, travel_time(robot, b, c), task.weight_kg, c)
                + energy_for(robot, d, travel_time(robot, d, c), 0.0, c)
                + energy_for(
                    robot,
                    0.0,
                    2 * c.service_s + c.route_wait_bound_s + c.charger_wait_bound_s,
                    0.0,
                    c,
                )
                + reserve_wh(robot, c)
            )
            available.append(needed)
        return min(available, default=float("inf"))

    def _valid_charge(
        self, robot: Robot, decision: Decision, pending: list[ChargeBooking]
    ) -> str | None:
        c = self.config
        port = self.ports.get(decision.port_id)
        if port is None or robot.kind not in port.compatible_kinds:
            return "incompatible_port"
        if decision.target_soc not in c.charge_targets or decision.target_soc <= robot.soc + 1e-8:
            return "invalid_charge_target"
        values = (decision.arrival_s, decision.start_s, decision.end_s)
        if any(v is None or not isfinite(float(v)) for v in values):
            return "invalid_charge_interval"
        arrival, start, end = map(float, values)
        if (
            arrival < self.time_s - 1e-8
            or start < arrival - 1e-8
            or end <= start
            or start - arrival > c.charger_wait_bound_s + 1e-8
        ):
            return "invalid_charge_interval"
        distance = self._distance(robot.position, port.position)
        if not isfinite(distance):
            return "no_route"
        travel = travel_time(robot, distance, c)
        needed = energy_for(
            robot, distance, travel + c.route_wait_bound_s + c.charger_wait_bound_s, 0.0, c
        ) + reserve_wh(robot, c)
        if robot.battery_wh + 1e-8 < needed:
            return "insufficient_energy"
        minimum = (
            charge_duration(robot, robot.battery_wh, decision.target_soc, port.power_w, c)
            + 2 * c.docking_s
        )
        if end - start + c.tick_s < minimum:
            return "charge_interval_too_short"
        for booking in self.bookings + pending:
            if (
                booking.port_id == port.id
                and start < booking.end_s - 1e-8
                and end > booking.start_s + 1e-8
            ):
                return "port_overlap"
        queue = self._choose_queue(robot, port)
        if queue is None and robot.position != port.position:
            return "queue_full"
        if queue is not None:
            queue_arrival = self.time_s + travel_time(
                robot, self._distance(robot.position, queue), c
            )
            if start - queue_arrival > c.charger_wait_bound_s + 1e-8:
                return "physical_queue_wait_limit"
        return None

    def submit_plan(self, plan: DecisionPlan) -> dict:
        """Revalidate and atomically commit each operation (task/route/port).

        A rejected operation has no side effects. Independent accepted robot
        operations remain committed. Reusing an identical plan is idempotently
        rejected. A delayed <=10 s snapshot still receives live revalidation.
        """
        digest = hashlib.sha256(
            json.dumps(
                {"v": plan.snapshot_version, "d": [asdict(d) for d in plan.decisions]},
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        report = {"accepted": [], "rejected": [], "snapshot_version": plan.snapshot_version}
        observed = self._known_versions.get(plan.snapshot_version)
        global_reason = (
            "duplicate_plan"
            if digest in self._submitted
            else "unknown_snapshot"
            if observed is None
            else "stale_snapshot"
            if self.time_s - observed > self.config.stale_after_s + 1e-8
            else None
        )
        self._submitted.add(digest)
        seen_robots: set[int] = set()
        seen_tasks: set[int] = set()
        for decision in plan.decisions:
            robot = self.robots.get(decision.robot_id)
            reason = global_reason
            if reason is None and (robot is None or robot.id in seen_robots):
                reason = "unknown_or_duplicate_robot"
            if reason is None and (
                robot.status != "idle" or robot.move_to is not None or robot.task_id is not None
            ):
                reason = "robot_busy"
            if reason is None and robot.paused_until_s > self.time_s:
                reason = "robot_paused"
            if reason is None and decision.kind == "task":
                task = self.tasks.get(decision.task_id)
                if (
                    task is None
                    or task.id in seen_tasks
                    or task.status not in {"waiting", "blocked"}
                ):
                    reason = "task_unavailable"
                elif task.weight_kg > robot.payload_capacity_kg:
                    reason = "payload_exceeded"
                elif task.weight_kg <= 0:
                    reason = "invalid_weight"
                elif self._task_required_energy(robot, task) > robot.battery_wh + 1e-8:
                    reason = "insufficient_energy_or_charge_access"
                elif self._path(robot.position, task.pickup) is None:
                    reason = "no_route"
            elif reason is None and decision.kind == "charge":
                reason = self._valid_charge(robot, decision, [])
            elif reason is None and decision.kind == "reposition":
                target = decision.target_position
                if target not in self.warehouse.parking:
                    reason = "invalid_parking"
                elif any(
                    r.id != robot.id and (r.position == target or r.target_position == target)
                    for r in self.robots.values()
                ):
                    reason = "parking_occupied"
                else:
                    distance = self._distance(robot.position, target)
                    if not isfinite(distance):
                        reason = "no_route"
                    elif (
                        energy_for(
                            robot,
                            distance,
                            travel_time(robot, distance, self.config)
                            + self.config.route_wait_bound_s,
                            0.0,
                            self.config,
                        )
                        + reserve_wh(robot, self.config)
                        > robot.battery_wh
                    ):
                        reason = "insufficient_energy"
            elif reason is None and decision.kind != "wait":
                reason = "unknown_decision_kind"
            if reason is not None:
                record = {"robot_id": decision.robot_id, "kind": decision.kind, "reason": reason}
                report["rejected"].append(record)
                self.counters["rejected_decisions"] += 1
                continue
            seen_robots.add(robot.id)
            if decision.kind != "wait":
                robot.metadata["route_wait_s"] = 0.0
                robot.metadata.pop("charge_queue_arrived_s", None)
            if decision.kind == "task":
                task = self.tasks[decision.task_id]
                seen_tasks.add(task.id)
                task.status, task.robot_id, task.assigned_at_s, task.blocked_reason = (
                    "assigned",
                    robot.id,
                    self.time_s,
                    None,
                )
                robot.task_id, robot.status = task.id, "to_pickup"
                self._set_goal(robot, task.pickup)
            elif decision.kind == "charge":
                port = self.ports[decision.port_id]
                booking = ChargeBooking(
                    robot.id,
                    port.id,
                    float(decision.arrival_s),
                    float(decision.start_s),
                    float(decision.end_s),
                    float(decision.target_soc),
                    self.time_s,
                )
                self.bookings.append(booking)
                self.booking_history.append(booking)
                robot.port_id, robot.target_soc, robot.status = (
                    port.id,
                    decision.target_soc,
                    "to_charge_queue",
                )
                queue = self._choose_queue(robot, port)
                if robot.position == port.position:
                    self._set_goal(robot, port.position)
                    robot.status = "to_charge_port"
                else:
                    self._queue_owners[queue] = robot.id
                    robot.metadata["queue_cell"] = queue
                    self._set_goal(robot, queue)
            elif decision.kind == "reposition":
                robot.status = "reposition"
                self._set_goal(robot, decision.target_position)
            record = {
                "robot_id": robot.id,
                "kind": decision.kind,
                "task_id": decision.task_id,
                "port_id": decision.port_id,
            }
            report["accepted"].append(record)
        self._log(
            "plan_commit",
            profile_id=plan.profile_id,
            solver_status=plan.solver_status,
            fallback_reason=plan.fallback_reason,
            **report,
        )
        return report

    def _choose_queue(self, robot: Robot, port):
        occupied = {r.position for r in self.robots.values() if r.id != robot.id}
        candidates = [
            cell
            for cell in port.queue_cells
            if cell not in occupied
            and self._queue_owners.get(cell, robot.id) == robot.id
            and cell not in self.blocked_cells
        ]
        candidates.sort(key=lambda cell: (self._distance(robot.position, cell), cell))
        return candidates[0] if candidates else None

    def _set_goal(self, robot: Robot, goal) -> None:
        robot.target_position = tuple(goal) if goal is not None else None
        robot.path = []
        robot.blocked_since_s = None

    def _booking(self, robot: Robot) -> ChargeBooking | None:
        return next((b for b in self.bookings if b.robot_id == robot.id), None)

    def advance(self, seconds: float) -> dict:
        """Advance an integral number of physical ticks and return KPI deltas."""
        if (
            not isfinite(seconds)
            or seconds < 0
            or abs(seconds / self.config.tick_s - round(seconds / self.config.tick_s)) > 1e-8
        ):
            raise ValueError("advance seconds must be nonnegative and tick-aligned")
        previous = dict(self.counters)
        for _ in range(round(seconds / self.config.tick_s)):
            if self.time_s >= self.config.horizon_s - 1e-9:
                break
            self._tick()
        return {key: self.counters[key] - value for key, value in previous.items()}

    def _tick(self) -> None:
        c = self.config
        for port in self.ports.values():
            if any(r.position == port.position for r in self.robots.values()):
                self.port_usage[port.id]["occupied_s"] += c.tick_s
        pending = [task for task in self.tasks.values() if task.completed_at_s is None]
        self.counters["backlog_integral_s"] += len(pending) * c.tick_s
        self.counters["late_integral_s"] += sum(
            task.priority * max(0.0, self.time_s + c.tick_s - max(self.time_s, task.deadline_s))
            for task in pending
        )
        self.tick += 1
        self.time_s = self.tick * c.tick_s
        self.version = self.tick
        arrivals = self.reservations.advance(self.tick)
        for robot_id, target in arrivals.items():
            robot = self.robots[robot_id]
            robot.position = target
            robot.move_to = None
            robot.move_remaining_s = 0.0
            self.counters["distance_m"] += 1.0
            self.counters["loaded_distance_m" if robot.load_kg else "empty_distance_m"] += 1.0
            self.counters[f"robot_kind_{robot.kind}_distance_m"] += 1.0
            self.counters[
                f"robot_kind_{robot.kind}_"
                + ("loaded_distance_m" if robot.load_kg else "empty_distance_m")
            ] += 1.0
            incident = self._deadlock_incident(robot)
            if incident:
                if incident["first_progress_s"] is None:
                    incident["first_progress_s"] = self.time_s
                if incident["progress_since_s"] is None:
                    incident["progress_since_s"] = self.time_s
                incident["last_progress_s"] = self.time_s
            if robot.metadata.pop("recovering", False):
                self.counters["deadlocks_resolved"] += 1.0
                robot.metadata.pop("deadlock_attempts", None)
                self._log("deadlock_recovered", robot_id=robot.id)
            robot.blocked_since_s = None
            if (
                robot.status == "charge_egress"
                and robot.port_id is not None
                and target != self.ports[robot.port_id].position
            ):
                self._release_port(robot)
        self._process_events()
        for robot in self.robots.values():
            robot.observed_at_s = self.time_s
            if robot.move_to is not None:
                robot.move_remaining_s = max(0.0, robot.move_remaining_s - c.tick_s)
            if robot.status == "charging" and robot.paused_until_s <= self.time_s:
                self._charge(robot)
            else:
                self._consume(robot, energy_for(robot, 0.0, c.tick_s, 0.0, c))
            if robot.paused_until_s > self.time_s:
                continue
            if robot.move_to is None:
                if robot.status in {"pickup", "dropoff", "docking"}:
                    robot.service_remaining_s = max(0.0, robot.service_remaining_s - c.tick_s)
                    if robot.service_remaining_s <= 1e-9:
                        self._finish_service(robot)
                elif robot.target_position == robot.position:
                    self._arrived(robot)
                if robot.status == "waiting_charge":
                    self._admit_charge(robot)
                if robot.status == "charge_egress" and robot.target_position is None:
                    self._choose_egress(robot)
        priority = sorted(
            self.robots.values(),
            key=lambda r: (
                -int(r.load_kg > 0),
                r.soc >= 0.15,
                -(self.time_s - (r.blocked_since_s or self.time_s)),
                r.id,
            ),
        )
        for robot in priority:
            if (
                robot.paused_until_s <= self.time_s
                and robot.move_to is None
                and robot.target_position is not None
                and robot.target_position != robot.position
                and robot.status
                not in {"charging", "docking", "pickup", "dropoff", "waiting_charge", "hold"}
            ):
                self._move(robot)
        self._audit_positions()
        self._audit_operations()
        for robot in self.robots.values():
            incident = self._deadlock_incident(robot)
            if incident and incident["progress_since_s"] is not None:
                if (
                    robot.blocked_since_s is not None
                    and self.time_s - robot.blocked_since_s > c.decision_s
                ):
                    incident["progress_since_s"] = None
                elif (
                    self.time_s - incident["progress_since_s"] >= 30
                    and incident["last_progress_s"] is not None
                    and self.time_s - incident["last_progress_s"] <= c.decision_s
                ):
                    self._resolve_deadlock(robot, "sustained_progress_30s")
        if self.tick % max(1, round(c.decision_s / c.tick_s)) == 0:
            self._refresh_bookings()
            self._history.append(self._live_snapshot())
            # Bounded state for million-step training; event tape persists full
            # input, optional JSONL runner drains operational events.
            if not c.log_decisions and len(self.events) > 2000:
                self.events = self.events[-1000:]
            if len(self._submitted) > 5000:
                self._submitted = set()
            self._known_versions = {
                v: t
                for v, t in self._known_versions.items()
                if self.time_s - t < max(60.0, c.observation_delay_s + 10.0)
            }

    def _consume(self, robot: Robot, amount: float) -> None:
        used = min(robot.battery_wh, max(0.0, amount))
        robot.battery_wh = max(0.0, robot.battery_wh - used)
        self.counters["consumed_wh"] += used
        self.counters[f"robot_kind_{robot.kind}_consumed_wh"] += used
        if robot.battery_wh < reserve_wh(robot, self.config) - 1e-8 and not robot.metadata.get(
            "reserve_reported"
        ):
            robot.metadata["reserve_reported"] = True
            self.counters["reserve_violations"] += 1
            self._log("reserve_violation", robot_id=robot.id, battery_wh=robot.battery_wh)
            self._energy_emergency(robot, "reserve_crossed")
        if used + 1e-10 < amount and not robot.metadata.get("energy_emergency"):
            self._energy_emergency(robot, "battery_exhausted")

    def _energy_emergency(self, robot: Robot, reason: str) -> None:
        if robot.metadata.get("energy_emergency"):
            return
        robot.metadata["energy_emergency"] = True
        self.counters["energy_emergencies"] += 1
        self._log("energy_emergency", robot_id=robot.id, reason=reason)
        if robot.port_id is not None and robot.status not in {
            "charging",
            "docking",
            "charge_egress",
        }:
            booking = self._booking(robot)
            if booking:
                booking.cancelled_at_s = self.time_s
                self.bookings.remove(booking)
            queue = robot.metadata.pop("queue_cell", None)
            if queue is not None:
                self._queue_owners.pop(tuple(queue), None)
            robot.port_id = None
            robot.target_soc = None
        # Keep carried goods and every task in the KPI. An unpicked task can
        # re-enter backlog while a nonnegative-energy robot holds its cell.
        if robot.task_id is not None and robot.load_kg == 0:
            task = self.tasks[robot.task_id]
            task.status, task.robot_id, task.assigned_at_s = "waiting", None, None
            robot.task_id = None
        robot.path = []
        robot.target_position = None
        robot.status = "hold"
        # Attempt a physical retreat to an off-aisle bay. Carried goods remain
        # attached to the robot; no task is declared completed or discarded.
        occupied = {r.position for r in self.robots.values()} | {
            r.target_position for r in self.robots.values()
        }
        candidates = []
        for bay in self.warehouse.parking:
            if bay in occupied:
                continue
            d = self._distance(robot.position, bay)
            if (
                isfinite(d)
                and energy_for(
                    robot,
                    d,
                    travel_time(robot, d, self.config) + self.config.tick_s,
                    robot.load_kg,
                    self.config,
                )
                <= robot.battery_wh
            ):
                candidates.append((d, bay))
        if candidates and robot.move_to is None:
            _, bay = min(candidates)
            robot.status = "emergency_parking"
            self._set_goal(robot, bay)
            self._log("energy_recovery_attempt", robot_id=robot.id, bay=bay)

    def _move(self, robot: Robot) -> None:
        c = self.config
        goal = robot.target_position
        forbidden = {
            p.position for p in self.ports.values() if p.position not in {robot.position, goal}
        }
        if robot.status in {"to_pickup", "to_dropoff"}:
            # Serialize the final approach to service cells and keep one-cell
            # egress space. Other carriers wait outside the approach buffer;
            # this is an explicit service queue, not an unexplained deadlock.
            owner_id = self._endpoint_leases.get(goal)
            owner = self.robots.get(owner_id)
            if owner is None or (
                owner.target_position != goal
                and abs(owner.position[0] - goal[0]) + abs(owner.position[1] - goal[1]) > 1
            ):
                contenders = [
                    r
                    for r in self.robots.values()
                    if r.target_position == goal and r.status in {"to_pickup", "to_dropoff"}
                ]
                owner = min(
                    contenders,
                    key=lambda r: (self._distance(r.position, goal), -int(r.load_kg > 0), r.id),
                )
                self._endpoint_leases[goal] = owner.id
            for blocker in self.robots.values():
                if (
                    blocker.id != robot.id
                    and blocker.position == goal
                    and blocker.status == "idle"
                    and blocker.move_to is None
                    and blocker.paused_until_s <= self.time_s
                ):
                    self._park_idle_blocker(blocker, robot.id)
            if (
                owner.id != robot.id
                and abs(robot.position[0] - goal[0]) + abs(robot.position[1] - goal[1]) <= 3
            ):
                robot.metadata["endpoint_wait"] = True
                robot.blocked_since_s = None
                self._route_wait(robot)
                return
            robot.metadata.pop("endpoint_wait", None)
            # Proactively vacate an idle endpoint before approaching carriers
            # can surround it during the 30s deadlock detection window.
        if (
            len(robot.path) < 2
            or robot.path[0] != robot.position
            or robot.metadata.get("path_goal") != goal
        ):
            path = self.planner.shortest_path(robot.position, goal, blocked_extra=forbidden)
            robot.path = path or []
            robot.metadata["path_goal"] = goal
        step_ticks = max(1, round(travel_time(robot, 1.0, c) / c.tick_s))
        next_cell = robot.path[1] if len(robot.path) > 1 else None
        started = False
        if (
            next_cell is not None
            and next_cell not in self.blocked_cells
            and next_cell not in forbidden
            and self.planner.corridor_move_allowed(
                robot.id, robot.position, next_cell, self.reservations
            )
        ):
            started = self.reservations.try_reserve_move(
                robot.id, robot.position, next_cell, self.tick, step_ticks
            )
        if not started and next_cell is not None:
            # Time-expanded A* is bounded to the proposal's 60 s planning
            # horizon. Its first physical leg is rechecked at execution.
            timed = self.planner.timed_path(
                robot.position,
                goal,
                self.tick,
                step_ticks,
                self.reservations,
                robot.id,
                horizon_ticks=round(60.0 / c.tick_s),
                blocked_extra=forbidden,
            )
            if timed and len(timed) > 1 and timed[1][1] != robot.position:
                alternative = timed[1][1]
                if self.planner.corridor_move_allowed(
                    robot.id, robot.position, alternative, self.reservations
                ) and self.reservations.try_reserve_move(
                    robot.id, robot.position, alternative, self.tick, step_ticks
                ):
                    next_cell, started = alternative, True
                    # Retain the detour prefix across ticks. Reverting to a
                    # static shortest path after each first leg oscillates
                    # around temporary blockers and can exhaust the battery.
                    robot.path = []
                    for _, cell in timed:
                        if not robot.path or robot.path[-1] != cell:
                            robot.path.append(cell)
        if started:
            cost = energy_for(robot, 1.0, 0.0, robot.load_kg, c)
            safety_floor = 0.0 if robot.metadata.get("energy_emergency") else reserve_wh(robot, c)
            if robot.battery_wh + 1e-8 < cost + safety_floor:
                # An edge has been reserved but not entered; rollback only
                # this robot's future movement and keep its current position.
                self.reservations.cancel_move(robot.id)
                self._energy_emergency(robot, "edge_would_violate_reserve")
                return
            self._consume(robot, cost)
            robot.move_to = next_cell
            robot.move_remaining_s = step_ticks * c.tick_s
            robot.metadata["last_move_s"] = self.time_s
            if robot.path and len(robot.path) > 1 and robot.path[1] == next_cell:
                robot.path = robot.path[1:]
            return
        if robot.blocked_since_s is None:
            robot.blocked_since_s = self.time_s
        self._route_wait(robot)
        if robot.metadata.get("energy_emergency"):
            return
        if self.time_s - robot.blocked_since_s >= c.deadlock_s:
            self._recover_deadlock(robot)

    def _route_wait(self, robot: Robot) -> None:
        robot.metadata["route_wait_s"] = (
            float(robot.metadata.get("route_wait_s", 0.0)) + self.config.tick_s
        )
        if robot.metadata["route_wait_s"] > self.config.route_wait_bound_s + 1e-8:
            self._energy_emergency(robot, "route_wait_bound_exceeded")

    def _recover_deadlock(self, robot: Robot) -> None:
        if self._deadlock_incident(robot) is None:
            self.deadlock_incidents.append(
                {
                    "incident_id": len(self.deadlock_incidents) + 1,
                    "robot_id": robot.id,
                    "detected_s": self.time_s,
                    "first_progress_s": None,
                    "last_progress_s": None,
                    "progress_since_s": None,
                    "resolved_s": None,
                    "resolution": None,
                }
            )
        attempts = int(robot.metadata.get("deadlock_attempts", 0))
        if attempts == 0:
            self.counters["deadlocks"] += 1
        if attempts >= 2:
            if not robot.metadata.get("deadlock_unresolved"):
                robot.metadata["deadlock_unresolved"] = True
                self.counters["deadlocks_unresolved"] += 1
                self._log("deadlock_unresolved", robot_id=robot.id, goal=robot.target_position)
            robot.status = "hold"
            return
        robot.metadata["deadlock_attempts"] = attempts + 1
        robot.metadata["recovering"] = True
        robot.path = []
        robot.blocked_since_s = self.time_s
        # A completed robot may be idling on a delivery endpoint. Move that
        # unloaded blocker to a legal off-aisle bay instead of leaving the
        # loaded requester permanently waiting on an otherwise open map.
        static = self._path(robot.position, robot.target_position)
        wanted = {robot.target_position}
        if static and len(static) > 1:
            wanted.add(static[1])
        for blocker in self.robots.values():
            if (
                blocker.id == robot.id
                or blocker.position not in wanted
                or blocker.status != "idle"
                or blocker.move_to is not None
                or blocker.paused_until_s > self.time_s
            ):
                continue
            occupied = {r.position for r in self.robots.values()} | {
                r.target_position for r in self.robots.values()
            }
            bays = [
                p
                for p in self.warehouse.parking
                if p not in occupied and self._path(blocker.position, p)
            ]
            if bays:
                bay = min(bays, key=lambda p: (self._distance(blocker.position, p), p))
                d = self._distance(blocker.position, bay)
                if (
                    energy_for(blocker, d, travel_time(blocker, d, self.config), 0.0, self.config)
                    + reserve_wh(blocker, self.config)
                    <= blocker.battery_wh
                ):
                    blocker.status = "reposition"
                    self._set_goal(blocker, bay)
                    self._log(
                        "deadlock_blocker_yield",
                        robot_id=blocker.id,
                        for_robot_id=robot.id,
                        bay=bay,
                    )
        # Reorder/replan first. Second attempt diverts an unloaded robot to a
        # free off-aisle bay and then restores the original operation.
        if (
            attempts == 1
            and robot.load_kg == 0
            and robot.status not in {"to_charge_port", "charge_egress", "recovering"}
        ):
            occupied = {r.position for r in self.robots.values()} | {
                r.target_position for r in self.robots.values()
            }
            choices = [
                p
                for p in self.warehouse.parking
                if p not in occupied and self._path(robot.position, p)
            ]
            if choices:
                bay = min(choices, key=lambda p: (self._distance(robot.position, p), p))
                robot.metadata["recovery_goal"] = (robot.target_position, robot.status)
                self._set_goal(robot, bay)
                robot.status = "recovering"
        self._log(
            "deadlock_recovery_attempt",
            robot_id=robot.id,
            attempt=attempts + 1,
            goal=robot.target_position,
        )

    def _park_idle_blocker(self, blocker: Robot, for_robot_id: int) -> bool:
        occupied = {r.position for r in self.robots.values()} | {
            r.target_position for r in self.robots.values()
        }
        bays = [
            p
            for p in self.warehouse.parking
            if p not in occupied and self._path(blocker.position, p)
        ]
        if not bays:
            return False
        bay = min(bays, key=lambda p: (self._distance(blocker.position, p), p))
        d = self._distance(blocker.position, bay)
        if (
            energy_for(blocker, d, travel_time(blocker, d, self.config), 0.0, self.config)
            + reserve_wh(blocker, self.config)
            > blocker.battery_wh
        ):
            return False
        blocker.status = "reposition"
        self._set_goal(blocker, bay)
        self._log("endpoint_blocker_yield", robot_id=blocker.id, for_robot_id=for_robot_id, bay=bay)
        return True

    def _arrived(self, robot: Robot) -> None:
        robot.path = []
        robot.target_position = None
        if robot.status == "to_pickup":
            task = self.tasks[robot.task_id]
            task.picked_at_s = self.time_s
            task.status = "picking"
            robot.status = "pickup"
            robot.service_remaining_s = self.config.service_s
            self._log("pickup_started", robot_id=robot.id, task_id=task.id)
        elif robot.status == "to_dropoff":
            robot.status = "dropoff"
            robot.service_remaining_s = self.config.service_s
        elif robot.status == "to_charge_queue":
            robot.status = "waiting_charge"
            robot.metadata["charge_queue_arrived_s"] = self.time_s
            self._log("charge_queue_arrived", robot_id=robot.id, port_id=robot.port_id)
        elif robot.status == "to_charge_port":
            robot.status = "docking"
            robot.service_remaining_s = self.config.docking_s
            queue = robot.metadata.pop("queue_cell", None)
            if queue is not None:
                self._queue_owners.pop(tuple(queue), None)
            self._log("port_arrived", robot_id=robot.id, port_id=robot.port_id)
        elif robot.status == "recovering":
            goal, status = robot.metadata.pop("recovery_goal")
            robot.status = status
            self._set_goal(robot, goal)
        elif robot.status == "emergency_parking":
            robot.status = "hold"
            self._log("energy_recovery_parked", robot_id=robot.id)
        elif robot.status in {"reposition", "charge_egress"}:
            if robot.status == "charge_egress":
                self._release_port(robot)
            robot.status = "idle"

    def _finish_service(self, robot: Robot) -> None:
        if robot.status == "pickup":
            task = self.tasks[robot.task_id]
            robot.load_kg = task.weight_kg
            task.status = "carrying"
            robot.status = "to_dropoff"
            self._set_goal(robot, task.dropoff)
            self._log("pickup_finished", robot_id=robot.id, task_id=task.id)
        elif robot.status == "dropoff":
            task = self.tasks[robot.task_id]
            task.status = "completed"
            task.completed_at_s = self.time_s
            self.counters["completed"] += 1
            robot.task_id = None
            robot.load_kg = 0.0
            robot.status = "idle"
            robot.metadata.pop("deadlock_attempts", None)
            self._log("task_completed", robot_id=robot.id, task_id=task.id)
            self._resolve_deadlock(robot, "operation_completed")
            contender = next(
                (
                    r
                    for r in self.robots.values()
                    if r.id != robot.id and r.target_position == robot.position
                ),
                None,
            )
            if contender:
                self._park_idle_blocker(robot, contender.id)
        elif robot.status == "docking":
            robot.status = "charging"
            booking = self._booking(robot)
            if booking:
                booking.actual_start_s = self.time_s
                self.counters["charge_sessions"] += 1
                self._log(
                    "charge_started",
                    robot_id=robot.id,
                    port_id=robot.port_id,
                    target_soc=robot.target_soc,
                )

    def _admit_charge(self, robot: Robot) -> None:
        booking = self._booking(robot)
        if booking is None:
            robot.status = "idle"
            return
        queue_since = float(robot.metadata.get("charge_queue_arrived_s", self.time_s))
        if self.time_s - queue_since > self.config.charger_wait_bound_s + 1e-8:
            self._energy_emergency(robot, "charge_queue_wait_bound_exceeded")
            return
        if self.time_s < booking.start_s - 1e-8:
            return
        port = self.ports[booking.port_id]
        earlier = [
            b
            for b in self.bookings
            if b.port_id == booking.port_id
            and b.robot_id != robot.id
            and (b.start_s, b.requested_at_s, b.robot_id)
            < (booking.start_s, booking.requested_at_s, booking.robot_id)
        ]
        port_busy = any(
            r.id != robot.id
            and (
                r.position == port.position
                or r.move_to == port.position
                or (
                    r.port_id == port.id
                    and r.status in {"to_charge_port", "docking", "charging", "charge_egress"}
                )
            )
            for r in self.robots.values()
        )
        if not earlier and not port_busy:
            robot.status = "to_charge_port"
            self._set_goal(robot, port.position)
        elif (
            self.time_s - booking.requested_at_s
            > self.config.charger_wait_bound_s + self.config.route_wait_bound_s
            and not robot.metadata.get("queue_bound_reported")
        ):
            robot.metadata["queue_bound_reported"] = True
            self._log("charge_wait_bound_exceeded", robot_id=robot.id, port_id=port.id)
            if robot.battery_wh < reserve_wh(robot, self.config):
                self._energy_emergency(robot, "charge_wait_bound_exceeded")

    def _charge(self, robot: Robot) -> None:
        port = self.ports[robot.port_id]
        from .energy import charge_step_details

        net, grid, active_s = charge_step_details(
            robot, port.power_w, self.config.tick_s, self.config, float(robot.target_soc)
        )
        target = float(robot.target_soc) * robot.capacity_wh
        actual = min(max(0.0, target - robot.battery_wh), net)
        fraction = active_s / self.config.tick_s
        robot.battery_wh = min(robot.capacity_wh, robot.battery_wh + actual)
        reached = robot.battery_wh >= target - 1e-8
        auxiliary = self.config.auxiliary_power_w * self.config.tick_s / 3600.0
        # After hitting a target mid-tick, charging stops but the controller's
        # 12W auxiliary draw continues for the unused fraction of that tick.
        robot.battery_wh = max(0.0, robot.battery_wh - auxiliary * (1.0 - fraction))
        self.counters["grid_wh"] += grid
        self.counters["consumed_wh"] += auxiliary
        self.counters["charged_battery_wh"] += grid * self.config.charge_efficiency
        self.counters[f"robot_kind_{robot.kind}_grid_wh"] += grid
        self.counters[f"robot_kind_{robot.kind}_consumed_wh"] += auxiliary
        self.port_usage[port.id]["charging_s"] += self.config.tick_s * fraction
        if reached:
            booking = self._booking(robot)
            if booking:
                booking.actual_end_s = self.time_s - self.config.tick_s * (1 - fraction)
            self._resolve_deadlock(robot, "charge_completed")
            robot.status = "charge_egress"
            self._choose_egress(robot)
            self._log(
                "charge_target_reached",
                robot_id=robot.id,
                soc=robot.soc,
                target_soc=robot.target_soc,
            )

    def _choose_egress(self, robot: Robot) -> None:
        """Retry a blocked departure every tick; release only after movement."""
        choices = [
            p
            for p in self.warehouse.parking
            if not any(
                r.id != robot.id and (r.position == p or r.target_position == p)
                for r in self.robots.values()
            )
            and self._path(robot.position, p)
        ]
        if choices:
            self._set_goal(
                robot, min(choices, key=lambda p: (self._distance(robot.position, p), p))
            )
        else:
            robot.target_position = next(
                (
                    p
                    for p in self.warehouse.neighbors(robot.position, self.blocked_cells)
                    if self.reservations.is_available(p, robot.id)
                ),
                None,
            )
            if robot.target_position is None and not robot.metadata.get("egress_block_reported"):
                robot.metadata["egress_block_reported"] = True
                self._log("charge_egress_blocked", robot_id=robot.id, port_id=robot.port_id)

    def _deadlock_incident(self, robot):
        return next(
            (
                i
                for i in reversed(self.deadlock_incidents)
                if i["robot_id"] == robot.id and i["resolved_s"] is None
            ),
            None,
        )

    def _resolve_deadlock(self, robot, reason):
        incident = self._deadlock_incident(robot)
        if incident:
            incident.update(resolved_s=self.time_s, resolution=reason)
            self._log(
                "deadlock_resolution_confirmed",
                incident_id=incident["incident_id"],
                robot_id=robot.id,
                reason=reason,
            )

    def _audit_operations(self):
        """Independent checks on committed state, counted once per incident identity."""
        violations = []
        owners = {}
        for robot in self.robots.values():
            if robot.load_kg > robot.payload_capacity_kg + 1e-8:
                violations.append(("payload", robot.id, robot.task_id))
            if (
                not isfinite(robot.battery_wh)
                or not -1e-8 <= robot.battery_wh <= robot.capacity_wh + 1e-8
            ):
                violations.append(("battery", robot.id, None))
            if robot.task_id is not None:
                if robot.task_id in owners:
                    violations.append(("duplicate_assignment", robot.task_id, robot.id))
                owners[robot.task_id] = robot.id
                task = self.tasks.get(robot.task_id)
                if task is None or task.robot_id != robot.id:
                    violations.append(("task_ownership", robot.id, robot.task_id))
            if robot.load_kg > 0 and robot.task_id is None:
                violations.append(("lost_payload", robot.id, None))
        for task in self.tasks.values():
            if (
                task.completed_at_s is None
                and task.robot_id is not None
                and owners.get(task.id) != task.robot_id
            ):
                violations.append(("task_ownership", task.robot_id, task.id))
        for violation in violations:
            if violation not in self._invariant_seen:
                self._invariant_seen.add(violation)
                self.invariant_incidents.append(
                    {"time_s": self.time_s, "kind": violation[0], "identity": violation[1:]}
                )

    def _release_port(self, robot: Robot) -> None:
        booking = self._booking(robot)
        if booking:
            booking.end_s = self.time_s
            self.bookings.remove(booking)
        self._log("port_released", robot_id=robot.id, port_id=robot.port_id)
        robot.port_id = None
        robot.target_soc = None
        queue = robot.metadata.pop("queue_cell", None)
        if queue is not None:
            self._queue_owners.pop(tuple(queue), None)
        robot.metadata.pop("reserve_reported", None)

    def _refresh_bookings(self) -> None:
        """Repair late arrival and interval tails without double-booking ports."""
        c = self.config
        for port in self.ports.values():
            ordered = sorted(
                (b for b in self.bookings if b.port_id == port.id),
                key=lambda b: (b.start_s, b.requested_at_s, b.robot_id),
            )
            cursor = self.time_s
            for booking in ordered:
                robot = self.robots[booking.robot_id]
                old = (booking.start_s, booking.end_s)
                active = robot.status in {"docking", "charging", "charge_egress", "to_charge_port"}
                if active:
                    remaining = charge_duration(
                        robot, robot.battery_wh, booking.target_soc, port.power_w, c
                    )
                    if robot.status == "docking":
                        remaining += robot.service_remaining_s
                    elif robot.status == "to_charge_port":
                        remaining += (
                            travel_time(robot, self._distance(robot.position, port.position), c)
                            + c.docking_s
                        )
                    booking.end_s = max(
                        self.time_s + c.tick_s, self.time_s + remaining + c.docking_s
                    )
                else:
                    distance = self._distance(robot.position, port.position)
                    arrival = (
                        self.time_s + travel_time(robot, distance, c)
                        if isfinite(distance)
                        else self.time_s + c.route_wait_bound_s
                    )
                    duration = max(c.tick_s, booking.end_s - booking.start_s)
                    booking.arrival_s = max(booking.arrival_s, arrival)
                    booking.start_s = max(booking.start_s, cursor, booking.arrival_s)
                    booking.end_s = booking.start_s + duration
                    queue_since = robot.metadata.get("charge_queue_arrived_s")
                    if (
                        queue_since is not None
                        and booking.start_s - float(queue_since) > c.charger_wait_bound_s + 1e-8
                    ):
                        # Cancel an already predictably infeasible slot before
                        # crossing the promised waiting bound. It remains in
                        # booking_history as a censored/cancelled request.
                        self._cancel_pending_charge(robot, "rescheduled_queue_wait_infeasible")
                        continue
                cursor = max(cursor, booking.end_s)
                if old != (booking.start_s, booking.end_s):
                    self._log(
                        "booking_rescheduled",
                        robot_id=robot.id,
                        port_id=port.id,
                        old_start_s=old[0],
                        old_end_s=old[1],
                        start_s=booking.start_s,
                        end_s=booking.end_s,
                    )

    def _cancel_pending_charge(self, robot: Robot, reason: str) -> None:
        booking = self._booking(robot)
        if booking:
            booking.cancelled_at_s = self.time_s
            self.bookings.remove(booking)
        queue = robot.metadata.pop("queue_cell", None)
        if queue is not None:
            self._queue_owners.pop(tuple(queue), None)
        old_port = robot.port_id
        robot.port_id = None
        robot.target_soc = None
        robot.status = "idle"
        robot.target_position = None
        robot.path = []
        robot.metadata.pop("charge_queue_arrived_s", None)
        self._log("charge_booking_cancelled", robot_id=robot.id, port_id=old_port, reason=reason)
        # A physical retreat clears the station queue for other bookings.
        if not self._park_idle_blocker(robot, robot.id):
            self._energy_emergency(robot, "cannot_recover_infeasible_charge_booking")

    def _audit_positions(self) -> None:
        self.reservations.assert_safe()
        table_positions = self.reservations.positions
        for robot in self.robots.values():
            if table_positions.get(robot.id) != robot.position:
                raise RuntimeError("simulator position disagrees with reservation owner")
            move = self.reservations.pending_move(robot.id)
            if (None if move is None else move.target) != robot.move_to:
                raise RuntimeError("simulator transit disagrees with reservation owner")
        positions = [r.position for r in self.robots.values()]
        if len(set(positions)) != len(positions):
            self.counters["collisions"] += 1
            raise RuntimeError("physical vertex collision detected independently of planner")
        moving = [r for r in self.robots.values() if r.move_to is not None]
        for index, a in enumerate(moving):
            for b in moving[index + 1 :]:
                if a.position == b.move_to and b.position == a.move_to:
                    self.counters["edge_conflicts"] += 1
                    raise RuntimeError("opposing edge movement detected independently of planner")
        if any(not isfinite(r.battery_wh) or r.battery_wh < -1e-9 for r in self.robots.values()):
            raise RuntimeError("invalid physical battery state")
