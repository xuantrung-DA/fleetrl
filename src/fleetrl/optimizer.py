"""Joint task/reposition/wait/charge scheduling with six RL-selected profiles.

CP-SAT selects one next operation per idle robot. Physical port intervals and
station waiting intervals are resource constraints; the action only changes
objective weights, never feasibility. All times in the model are integer ticks.

One correction to the proposal is deliberate: additive task, energy, occupancy,
delay and charge-gain terms are ALL averaged over fleet size. Averaging only
task completion while summing charge gain otherwise makes large fleets prefer
charging almost everybody. Profile multipliers remain exactly those proposed.
The return-to-charger check is conditional on the current calendar, not a
promise that future online decisions cannot change availability.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import ceil, isfinite
from time import perf_counter
from typing import Any

from ortools.sat.python import cp_model

from .config import FleetConfig
from .energy import charge_duration, energy_for, reserve_wh, travel_time
from .types import Cell, Decision, DecisionPlan, Robot, Snapshot, Task

COST_SCALE = 1_000_000


@dataclass(frozen=True)
class ObjectiveProfile:
    """Fixed, auditable cost multipliers controlled by Discrete(6)."""

    name: str
    tardiness: float = 1.0
    gain: float = 1.0
    availability: float = 1.0
    zone: float = 1.0
    energy: float = 0.2


PROFILES = (
    ObjectiveProfile("balanced"),
    ObjectiveProfile("deadline", tardiness=3.0, gain=0.5),
    ObjectiveProfile("store_energy", gain=3.0, availability=0.5),
    ObjectiveProfile("keep_available", gain=0.5, availability=3.0),
    ObjectiveProfile("reduce_empty_travel", energy=1.0),
    ObjectiveProfile("balance_zones", zone=3.0),
)


def validate_profile(value: Any) -> tuple[int, str | None]:
    """Reject NaN, non-integer and out-of-range policy actions safely."""
    try:
        scalar = float(value)
        if isfinite(scalar) and scalar.is_integer() and 0 <= scalar < len(PROFILES):
            return int(scalar), None
    except (ValueError, TypeError, OverflowError):
        pass
    return 0, "invalid_policy_action"


def earliest_slot(
    earliest: int,
    latest: int,
    duration: int,
    occupied: list[tuple[int, int]],
    quantum: int = 1,
) -> int | None:
    """Find a start without clipping interval tails to the start horizon.

    Intervals are half-open. The *start* must be <= latest; its end may be much
    later. ``quantum`` aligns starts to the dispatch grid, including after a
    reservation whose end is not itself on that grid.
    """
    start = int(ceil(earliest / quantum) * quantum)
    for left, right in sorted(occupied):
        if right <= start:
            continue
        if start + duration <= left:
            break
        start = int(ceil(right / quantum) * quantum)
        if start > latest:
            return None
    return start if start <= latest else None


def queue_fits(
    arrival: int,
    start: int,
    existing: list[tuple[int, int]],
    capacity: int,
) -> bool:
    """Check a proposed [arrival,start) against a station queue sweep line."""
    if start <= arrival:
        return True
    if capacity <= 0:
        return False
    events = [(arrival, 1), (start, -1)]
    for left, right in existing:
        if right > arrival and left < start:
            events.extend([(max(left, arrival), 1), (min(right, start), -1)])
    active = 0
    for _, change in sorted(events, key=lambda pair: (pair[0], pair[1])):
        active += change
        if active > capacity:
            return False
    return True


@dataclass
class Candidate:
    """An eligible operation and its physical estimates before CP-SAT selection."""

    decision: Decision
    duration_s: float = 0.0
    energy_wh: float = 0.0
    gain_wh: float = 0.0
    tardiness_s: float = 0.0
    zone: int = 0
    available: bool = True
    earliest_tick: int = 0
    latest_tick: int = 0
    duration_tick: int = 0
    station_id: int | None = None
    selected: Any = None
    start_var: Any = None
    delay_var: Any = None


class FleetOptimizer:
    """Bounded rolling optimizer used identically by fixed-profile B1 and PPO.

    ``decide`` has no access to simulator state or event tapes. It produces
    proposals, which the simulator must validate before committing them.
    """

    def __init__(self, config: FleetConfig):
        self.config = config
        self._distance_key: Any = None
        self._distances: dict[Cell, dict[Cell, int]] = {}
        self.last_rejections: dict[str, int] = {}
        self._urgent_ids: set[int] = set()
        self._robot_soc: dict[int, float] = {}

    def _reject(self, reason: str) -> None:
        self.last_rejections[reason] = self.last_rejections.get(reason, 0) + 1

    def _distance(self, snapshot: Snapshot, start: Cell, goal: Cell) -> float:
        """Cache reverse BFS from fixed destinations; invalidate on map changes."""
        grid = snapshot.warehouse
        key = (grid.width, grid.height, frozenset(grid.walls), snapshot.blocked_cells)
        if key != self._distance_key:
            self._distance_key = key
            self._distances.clear()
        if goal not in self._distances:
            if not grid.is_free(goal, set(snapshot.blocked_cells)):
                return float("inf")
            distances = {goal: 0}
            queue = deque([goal])
            while queue:
                cell = queue.popleft()
                for neighbor in grid.neighbors(cell, snapshot.blocked_cells):
                    if neighbor not in distances:
                        distances[neighbor] = distances[cell] + 1
                        queue.append(neighbor)
            self._distances[goal] = distances
        return float(self._distances[goal].get(start, float("inf")))

    def _ticks(self, seconds: float) -> int:
        return int(ceil(seconds / self.config.tick_s - 1e-9))

    def _calendars(self, snapshot: Snapshot) -> tuple[dict, dict]:
        """Return complete live port/queue calendars, including long tails."""
        ports = {p.id: [] for p in snapshot.ports}
        queues: dict[int, list[tuple[int, int]]] = {}
        station_of = {p.id: p.station_id for p in snapshot.ports}
        now = self._ticks(snapshot.sim_time_s)
        for booking in snapshot.bookings:
            if booking.cancelled_at_s is not None or booking.actual_end_s is not None:
                continue
            if booking.end_s <= snapshot.sim_time_s or booking.port_id not in ports:
                continue
            left, right = max(now, self._ticks(booking.start_s)), self._ticks(booking.end_s)
            ports[booking.port_id].append((left, right))
            arrival = max(now, self._ticks(booking.arrival_s))
            if left > arrival:
                queues.setdefault(station_of[booking.port_id], []).append((arrival, left))
        return ports, queues

    def _queue_capacity(self, snapshot: Snapshot, station_id: int) -> int:
        # A station's two ports normally share the same four physical queue cells.
        return len(
            {cell for p in snapshot.ports if p.station_id == station_id for cell in p.queue_cells}
        )

    def _energy_urgent(self, snapshot: Snapshot, robot: Robot, occupied: dict) -> bool:
        """Reserve enough headroom *before* the full wait guard becomes infeasible.

        Includes the earliest current calendar gap even if it lies beyond the
        online start horizon. That future gap is only a starvation-risk estimate;
        actual admission retains the300s road +300s queue bounds.
        """
        cfg = self.config
        needs = []
        for port in snapshot.ports:
            if robot.kind not in port.compatible_kinds:
                continue
            distance = self._distance(snapshot, robot.position, port.position)
            if not isfinite(distance):
                continue
            duration = travel_time(robot, distance, cfg)
            eta = snapshot.sim_time_s + duration
            energy = energy_for(robot, distance, duration, 0.0, cfg)
            charge_s = charge_duration(
                robot,
                max(0.0, robot.battery_wh - energy),
                max(0.6, min(0.95, robot.soc)),
                port.power_w,
                cfg,
            )
            start = earliest_slot(
                self._ticks(eta),
                self._ticks(eta + 86400),
                self._ticks(charge_s + 2 * cfg.docking_s),
                occupied[port.id],
                self._ticks(cfg.decision_s),
            )
            queue_estimate = (
                cfg.charger_wait_bound_s
                if start is None
                else max(cfg.charger_wait_bound_s, start * cfg.tick_s - eta)
            )
            needed = energy + reserve_wh(robot, cfg)
            needed += (
                cfg.auxiliary_power_w
                * (cfg.route_wait_bound_s + queue_estimate + 2 * cfg.docking_s)
                / 3600
            )
            needs.append(needed)
        if not needs:
            return robot.soc < cfg.charge_threshold
        # 10% capacity is an intervention buffer, not a reduced physical reserve.
        return robot.battery_wh < min(needs) + 0.10 * robot.capacity_wh

    def _task_candidate(
        self, snapshot: Snapshot, robot: Robot, task: Task, occupied: dict
    ) -> Candidate | None:
        cfg, now = self.config, snapshot.sim_time_s
        if task.weight_kg > robot.payload_capacity_kg:
            self._reject("payload")
            return None
        empty = self._distance(snapshot, robot.position, task.pickup)
        loaded = self._distance(snapshot, task.pickup, task.dropoff)
        if not isfinite(empty + loaded):
            self._reject("unreachable_task")
            return None
        empty_t, loaded_t = travel_time(robot, empty, cfg), travel_time(robot, loaded, cfg)
        duration = empty_t + loaded_t + 2 * cfg.service_s
        use = energy_for(robot, empty, empty_t, 0.0, cfg)
        use += energy_for(robot, loaded, loaded_t + 2 * cfg.service_s, task.weight_kg, cfg)
        waiting_use = (
            cfg.auxiliary_power_w * (cfg.route_wait_bound_s + cfg.charger_wait_bound_s) / 3600
        )
        witnesses = []
        for port in snapshot.ports:
            if robot.kind not in port.compatible_kinds:
                continue
            return_distance = self._distance(snapshot, task.dropoff, port.position)
            if not isfinite(return_distance):
                continue
            return_time = travel_time(robot, return_distance, cfg)
            return_use = energy_for(robot, return_distance, return_time, 0.0, cfg)
            required = use + return_use + waiting_use + reserve_wh(robot, cfg)
            if robot.battery_wh + 1e-8 < required:
                continue
            eta = now + duration + return_time
            # A conditional calendar witness. Reserve a meaningful minimum session
            # to60%, or docking-only if already above60; execution rechecks later.
            lower_energy = robot.battery_wh - use - return_use - waiting_use
            target = max(0.60, min(0.95, lower_energy / robot.capacity_wh))
            charge_s = charge_duration(robot, lower_energy, target, port.power_w, cfg)
            occupancy = max(cfg.tick_s, charge_s + 2 * cfg.docking_s)
            slot = earliest_slot(
                self._ticks(eta),
                self._ticks(eta + cfg.charger_wait_bound_s),
                self._ticks(occupancy),
                occupied[port.id],
                self._ticks(cfg.decision_s),
            )
            if slot is not None:
                witnesses.append((required, port.id, eta, slot * cfg.tick_s))
        if not witnesses:
            self._reject("insufficient_energy_or_return_calendar")
            return None
        required, return_port, return_eta, return_start = min(witnesses)
        age = max(0.0, now - task.created_s)
        # Subtract already-incurred lateness: otherwise old impossible deadlines
        # make doing an overdue task look worse than leaving it forever.
        incremental_late = max(0.0, now + duration - task.deadline_s) - max(
            0.0, now - task.deadline_s
        )
        deferral_pressure = min(600.0, age) * 0.5 + max(0.0, 120.0 - (task.deadline_s - now))
        tardiness = task.priority * (incremental_late - min(600.0, deferral_pressure))
        decision = Decision(
            robot.id,
            "task",
            task_id=task.id,
            target_position=task.pickup,
            estimated_energy_wh=required,
            reason="eligible_task",
            metadata={
                "travel_m": empty + loaded,
                "empty_travel_m": empty,
                "expected_completion_s": now + duration,
                "required_with_reserve_wh": required,
                "return_port_id": return_port,
                "return_eta_s": return_eta,
                "return_calendar_start_s": return_start,
            },
        )
        return Candidate(
            decision, duration, use, tardiness_s=tardiness, zone=task.zone, available=True
        )

    def _charge_candidates(
        self,
        snapshot: Snapshot,
        robot: Robot,
        occupied: dict,
        allow_charge: bool,
        targets: tuple[float, ...],
    ) -> list[Candidate]:
        if not allow_charge:
            return []
        cfg, now = self.config, snapshot.sim_time_s
        result = []
        for port in snapshot.ports:
            if robot.kind not in port.compatible_kinds:
                continue
            distance = self._distance(snapshot, robot.position, port.position)
            if not isfinite(distance):
                self._reject("unreachable_charger")
                continue
            travel_s = travel_time(robot, distance, cfg)
            use = energy_for(robot, distance, travel_s, 0.0, cfg)
            extra_wait = cfg.route_wait_bound_s + cfg.charger_wait_bound_s + cfg.docking_s
            lower_energy = robot.battery_wh - use - cfg.auxiliary_power_w * extra_wait / 3600
            if lower_energy < reserve_wh(robot, cfg) - 1e-8:
                self._reject("insufficient_energy_to_charge")
                continue
            arrival_energy = robot.battery_wh - use
            queue_trips = [
                (
                    self._distance(snapshot, robot.position, cell),
                    self._distance(snapshot, cell, port.position),
                )
                for cell in port.queue_cells
            ]
            queue_trips = [(a, b) for a, b in queue_trips if isfinite(a + b)]
            if robot.position == port.position:
                queue_arrival_s, port_arrival_s = now, now
            elif queue_trips:
                # Queue reservations begin at the earliest physical queue ETA,
                # not the later port ETA. The selected free queue bay is assigned
                # at commit; bound entry using every possible reachable bay.
                queue_arrival_s = now + min(travel_time(robot, a, cfg) for a, _ in queue_trips)
                port_arrival_s = now + max(travel_time(robot, a + b, cfg) for a, b in queue_trips)
                distance = max(a + b for a, b in queue_trips)
                travel_s = travel_time(robot, distance, cfg)
                use = energy_for(robot, distance, travel_s, 0.0, cfg)
                lower_energy = robot.battery_wh - use - cfg.auxiliary_power_w * extra_wait / 3600
                arrival_energy = robot.battery_wh - use
                if lower_energy < reserve_wh(robot, cfg) - 1e-8:
                    self._reject("insufficient_energy_via_queue")
                    continue
            else:
                self._reject("no_reachable_queue")
                continue
            eta = self._ticks(queue_arrival_s)
            earliest_port = self._ticks(port_arrival_s)
            earliest = int(
                ceil(earliest_port / self._ticks(cfg.decision_s)) * self._ticks(cfg.decision_s)
            )
            # A modest schedule margin absorbs ingress/egress calendar updates;
            # execution still enforces the full300s physical wait limit.
            schedule_margin_s = min(30.0, 0.10 * cfg.charger_wait_bound_s)
            latest = min(
                self._ticks(now + cfg.charging_horizon_s),
                self._ticks(queue_arrival_s + cfg.charger_wait_bound_s - schedule_margin_s),
            )
            for target in targets:
                # Do not repeatedly charge from79.9→80 due to waiting-energy
                # subtraction. A target must exceed the actual nominal arrival SoC.
                if target * robot.capacity_wh - arrival_energy < 0.05 * robot.capacity_wh - 1e-6:
                    self._reject("charge_gain_below_5pct")
                    continue
                charging_s = charge_duration(robot, lower_energy, target, port.power_w, cfg)
                duration_s = charging_s + 2 * cfg.docking_s
                duration_tick = self._ticks(duration_s)
                if (
                    earliest_slot(
                        earliest,
                        latest,
                        duration_tick,
                        occupied[port.id],
                        self._ticks(cfg.decision_s),
                    )
                    is None
                ):
                    self._reject("charger_wait_limit")
                    continue
                target_energy = target * robot.capacity_wh
                knee = 0.8 * robot.capacity_wh
                useful_gain = max(0.0, min(knee, target_energy) - arrival_energy)
                useful_gain += 0.3 * max(0.0, target_energy - max(knee, arrival_energy))
                decision = Decision(
                    robot.id,
                    "charge",
                    port_id=port.id,
                    target_soc=target,
                    arrival_s=eta * cfg.tick_s,
                    target_position=port.position,
                    estimated_energy_wh=use + cfg.auxiliary_power_w * extra_wait / 3600,
                    reason="eligible_charge",
                    metadata={
                        "travel_m": distance,
                        "lower_bound_arrival_wh": lower_energy,
                        "queue_arrival_lower_bound_s": queue_arrival_s,
                        "port_arrival_upper_bound_s": port_arrival_s,
                        "schedule_wait_margin_s": schedule_margin_s,
                        "reserved_duration_s": duration_tick * cfg.tick_s,
                    },
                )
                result.append(
                    Candidate(
                        decision,
                        travel_s + duration_s,
                        use,
                        gain_wh=useful_gain,
                        zone=snapshot.warehouse.zone_of(port.position),
                        available=False,
                        earliest_tick=earliest,
                        latest_tick=latest,
                        duration_tick=duration_tick,
                        station_id=port.station_id,
                    )
                )
        return result

    def generate_candidates(self, snapshot: Snapshot) -> list[Candidate]:
        """Filter physical eligibility once, independently of the selected profile."""
        cfg = self.config
        self.last_rejections = {}
        if snapshot.sim_time_s - snapshot.observed_at_s > cfg.stale_after_s:
            self._reject("stale_snapshot")
            return []
        occupied, _ = self._calendars(snapshot)
        self._robot_soc = {r.id: r.soc for r in snapshot.robots}
        self._urgent_ids = {
            r.id
            for r in snapshot.robots
            if r.status == "idle"
            and r.task_id is None
            and self._energy_urgent(snapshot, r, occupied)
        }
        waiting = [
            t
            for t in snapshot.tasks
            if t.status in {"waiting", "blocked"} and t.created_s <= snapshot.sim_time_s
        ]
        # Keep urgent/deadline and oldest candidates together under the40 cap.
        priority = sorted(waiting, key=lambda t: (-t.priority, t.deadline_s, t.created_s, t.id))
        oldest = sorted(waiting, key=lambda t: (t.created_s, t.deadline_s, t.id))
        shortlist: dict[int, Task] = {}
        for i in range(max(len(priority), len(oldest))):
            for ordered in (priority, oldest):
                if i < len(ordered) and len(shortlist) < cfg.max_tasks_observed:
                    shortlist[ordered[i].id] = ordered[i]
        result: list[Candidate] = []
        positions = {r.position for r in snapshot.robots}
        for robot in sorted(snapshot.robots, key=lambda r: r.id):
            if robot.status != "idle" or robot.task_id is not None:
                continue
            if snapshot.sim_time_s - robot.observed_at_s > cfg.stale_after_s:
                self._reject("stale_robot")
                continue
            task_candidates = []
            for task in shortlist.values():
                candidate = self._task_candidate(snapshot, robot, task, occupied)
                if candidate is not None:
                    task_candidates.append(candidate)
            # Old/urgent tasks are protected by shortlist order; nearest feasible
            # tasks fill the remaining slots and lower wasted travel.
            task_candidates.sort(
                key=lambda c: (
                    -next(t.priority for t in shortlist.values() if t.id == c.decision.task_id),
                    c.tardiness_s,
                    c.duration_s,
                    c.decision.task_id,
                )
            )
            limit = cfg.task_candidates_per_robot
            if len(task_candidates) > limit:
                # Every robot keeps an urgent/old candidate; rotating the rest
                # prevents20 robots all having the same six tasks in their graph.
                protected = task_candidates[: min(2, limit)]
                remaining_tasks = task_candidates[len(protected) :]
                offset = robot.id % len(remaining_tasks) if remaining_tasks else 0
                rotated = remaining_tasks[offset:] + remaining_tasks[:offset]
                task_candidates = protected + rotated[: limit - len(protected)]
            compatible_work = any(t.weight_kg <= robot.payload_capacity_kg for t in waiting)
            emergency = compatible_work and not task_candidates
            threshold_mode = not cfg.advanced_charging or cfg.controller == "heuristic"
            due_charge = robot.soc < cfg.charge_threshold or emergency
            urgent = robot.id in self._urgent_ids
            if (threshold_mode and due_charge) or urgent:
                # Threshold ablation has genuine locked charge timing, not merely
                # a95% candidate competing against tasks in the learned objective.
                task_candidates = []
            targets = (0.95,) if threshold_mode else cfg.charge_targets
            if urgent and not threshold_mode:
                # Recover enough energy promptly, freeing scarce ports for the
                # other critical robots before elective longer sessions resume.
                targets = (min(cfg.charge_targets),)
            allow_charge = (not threshold_mode) or due_charge
            if self._urgent_ids and not urgent and not threshold_mode:
                allow_charge = False
            charges = self._charge_candidates(snapshot, robot, occupied, allow_charge, targets)
            result.extend(task_candidates)
            result.extend(charges)
            ready = robot.battery_wh > reserve_wh(robot, cfg) + cfg.auxiliary_power_w * 600 / 3600
            result.append(
                Candidate(
                    Decision(
                        robot.id, "wait", target_position=robot.position, reason="hold_available"
                    ),
                    duration_s=cfg.decision_s,
                    energy_wh=cfg.auxiliary_power_w * cfg.decision_s / 3600,
                    available=ready and not urgent and not (threshold_mode and due_charge),
                    zone=snapshot.warehouse.zone_of(robot.position),
                )
            )
            if (threshold_mode and due_charge) or urgent:
                continue
            parking = [
                (self._distance(snapshot, robot.position, cell), cell)
                for cell in snapshot.warehouse.parking
                if cell != robot.position and cell not in positions
            ]
            parking.sort()
            chosen_parking: list[Cell] = []
            for zone in range(4):
                zone_cells = [
                    cell for _, cell in parking if snapshot.warehouse.zone_of(cell) == zone
                ]
                if zone_cells:
                    chosen_parking.append(zone_cells[0])
            for _, cell in parking:
                if len(chosen_parking) >= 6:
                    break
                if cell not in chosen_parking:
                    chosen_parking.append(cell)
            for cell in chosen_parking:
                if cell == robot.position or cell in positions:
                    continue
                distance = self._distance(snapshot, robot.position, cell)
                if not isfinite(distance):
                    continue
                travel_s = travel_time(robot, distance, cfg)
                use = energy_for(robot, distance, travel_s, 0.0, cfg)
                # Reposition still needs energy to reach a compatible charger.
                return_costs = []
                for port in snapshot.ports:
                    if robot.kind in port.compatible_kinds:
                        back = self._distance(snapshot, cell, port.position)
                        if isfinite(back):
                            return_costs.append(
                                energy_for(robot, back, travel_time(robot, back, cfg), 0, cfg)
                            )
                required = use + min(return_costs, default=float("inf")) + reserve_wh(robot, cfg)
                required += (
                    cfg.auxiliary_power_w
                    * (cfg.route_wait_bound_s + cfg.charger_wait_bound_s)
                    / 3600
                )
                if robot.battery_wh < required:
                    continue
                result.append(
                    Candidate(
                        Decision(
                            robot.id,
                            "reposition",
                            target_position=cell,
                            estimated_energy_wh=required,
                            reason="eligible_reposition",
                            metadata={"travel_m": distance},
                        ),
                        travel_s,
                        use,
                        zone=snapshot.warehouse.zone_of(cell),
                        available=ready,
                    )
                )
        # Preserve one hold/robot, task feasibility and ALL three charge targets.
        # The usual20×(6task+12charge+6parking+hold) fits500 exactly.
        if len(result) > cfg.max_candidates:
            holds = [c for c in result if c.decision.kind == "wait"]
            others = [c for c in result if c.decision.kind != "wait"]
            others.sort(
                key=lambda c: (
                    {"task": 0, "charge": 1, "reposition": 2}[c.decision.kind],
                    c.decision.robot_id,
                    c.duration_s,
                )
            )
            result = holds + others[: max(0, cfg.max_candidates - len(holds))]
            self._reject("candidate_cap")
        return result

    def _candidate_cost(self, candidate: Candidate, profile: ObjectiveProfile, n: int) -> float:
        cost = -4.0 if candidate.decision.kind == "task" else 0.0
        cost += profile.tardiness * candidate.tardiness_s / 600
        cost += profile.energy * candidate.energy_wh / 100
        cost += 0.3 * candidate.duration_s / 900
        cost -= profile.gain * candidate.gain_wh / 100
        if candidate.decision.kind == "charge" and candidate.decision.robot_id in self._urgent_ids:
            # A physical intervention priority: profiles cannot trade an imminent
            # energy-starvation event for more attractive elective battery gain.
            cost -= 1000 + 100 * (1 - self._robot_soc[candidate.decision.robot_id])
        return cost / max(1, n)

    def _hold(self, snapshot: Snapshot, profile_id: int, reason: str) -> DecisionPlan:
        return DecisionPlan(
            snapshot.version,
            profile_id,
            [Decision(r.id, "wait", reason=reason) for r in snapshot.robots if r.status == "idle"],
            solver_status="HOLD",
            fallback_reason=reason,
            metadata={"rejections": dict(self.last_rejections)},
        )

    def _heuristic(
        self,
        snapshot: Snapshot,
        candidates: list[Candidate],
        profile_id: int,
        reason: str | None = None,
    ) -> DecisionPlan:
        """B0 and solver fallback: urgent/oldest task, nearest eligible robot."""
        cfg = self.config
        occupied, queues = self._calendars(snapshot)
        robots = {r.id: r for r in snapshot.robots}
        decisions: dict[int, Decision] = {}
        used_tasks: set[int] = set()
        by_robot: dict[int, list[Candidate]] = {}
        for candidate in candidates:
            by_robot.setdefault(candidate.decision.robot_id, []).append(candidate)
        # Low energy first, then forced no-task emergency charging.
        for rid in sorted(by_robot, key=lambda r: (robots[r].soc, r)):
            options = by_robot[rid]
            eligible_task = any(c.decision.kind == "task" for c in options)
            pending = any(
                t.status in {"waiting", "blocked"}
                and t.weight_kg <= robots[rid].payload_capacity_kg
                for t in snapshot.tasks
            )
            if (
                rid not in self._urgent_ids
                and robots[rid].soc >= cfg.charge_threshold
                and (eligible_task or not pending)
            ):
                continue
            available_charges = []
            for candidate in options:
                if candidate.decision.kind != "charge":
                    continue
                if candidate.decision.target_soc != 0.95 and rid not in self._urgent_ids:
                    continue
                start = earliest_slot(
                    candidate.earliest_tick,
                    candidate.latest_tick,
                    candidate.duration_tick,
                    occupied[candidate.decision.port_id],
                    self._ticks(cfg.decision_s),
                )
                if start is not None and queue_fits(
                    self._ticks(candidate.decision.arrival_s),
                    start,
                    queues.get(candidate.station_id, []),
                    self._queue_capacity(snapshot, candidate.station_id),
                ):
                    available_charges.append(
                        (start, candidate.duration_tick, candidate.decision.port_id, candidate)
                    )
            if available_charges:
                start, duration, _, candidate = min(available_charges, key=lambda x: x[:3])
                decision = replace(
                    candidate.decision,
                    start_s=start * cfg.tick_s,
                    end_s=(start + duration) * cfg.tick_s,
                    reason="threshold_or_emergency_charge",
                )
                decisions[rid] = decision
                occupied[decision.port_id].append((start, start + duration))
                queues.setdefault(candidate.station_id, []).append(
                    (self._ticks(decision.arrival_s), start)
                )
        pending = sorted(
            (t for t in snapshot.tasks if t.status in {"waiting", "blocked"}),
            key=lambda t: (-t.priority, t.created_s, t.deadline_s, t.id),
        )
        for task in pending:
            options = [
                c
                for c in candidates
                if c.decision.kind == "task"
                and c.decision.task_id == task.id
                and c.decision.robot_id not in decisions
            ]
            if options:
                best = min(
                    options,
                    key=lambda c: (
                        c.decision.metadata["empty_travel_m"],
                        c.duration_s,
                        c.decision.robot_id,
                    ),
                )
                decisions[best.decision.robot_id] = replace(
                    best.decision, reason="priority_age_nearest_feasible"
                )
                used_tasks.add(task.id)
        for rid in by_robot:
            decisions.setdefault(rid, Decision(rid, "wait", reason="no_safe_heuristic_assignment"))
        return DecisionPlan(
            snapshot.version,
            profile_id,
            list(decisions.values()),
            solver_status="HEURISTIC" if reason is None else "FALLBACK",
            fallback_reason=reason,
            candidate_count=len(candidates),
            metadata={"rejections": dict(self.last_rejections)},
        )

    def decide(
        self, snapshot: Snapshot, profile_id: int = 0, *, weights=None, deadline=None
    ) -> DecisionPlan:
        """Return a feasible proposal or explicit bounded conservative fallback."""
        started = perf_counter()
        cfg = self.config
        deadline = min(deadline or float("inf"), started + cfg.decision_budget_s)
        profile_id, policy_error = validate_profile(profile_id)
        if cfg.controller == "fixed":
            profile_id = cfg.fixed_profile
        if snapshot.sim_time_s - snapshot.observed_at_s > cfg.stale_after_s:
            plan = self._hold(snapshot, profile_id, "stale_snapshot")
            plan.total_ms = (perf_counter() - started) * 1000
            return plan
        candidates = self.generate_candidates(snapshot)
        if not candidates:
            plan = self._hold(snapshot, profile_id, "no_idle_eligible_robot")
            plan.fallback_reason = policy_error
            plan.total_ms = (perf_counter() - started) * 1000
            return plan
        if cfg.controller == "heuristic":
            plan = self._heuristic(snapshot, candidates, profile_id)
            plan.total_ms = (perf_counter() - started) * 1000
            return plan
        if perf_counter() >= deadline:
            plan = self._hold(snapshot, profile_id, "candidate_generation_budget")
            plan.candidate_count = len(candidates)
            plan.total_ms = (perf_counter() - started) * 1000
            return plan
        profile = weights if weights is not None else PROFILES[profile_id]
        if cfg.backend == "milp":
            from .optimization.milp import solve_dispatch

            return solve_dispatch(
                self, snapshot, candidates, profile, profile_id, started, deadline
            )
        model = cp_model.CpModel()
        n = max(1, len(snapshot.robots))
        occupied, queues = self._calendars(snapshot)
        port_intervals: dict[int, list] = {p.id: [] for p in snapshot.ports}
        queue_intervals: dict[int, list] = {p.station_id: [] for p in snapshot.ports}
        robot_vars: dict[int, list] = {}
        task_vars: dict[int, list] = {}
        parking_vars: dict[Cell, list] = {}
        objective_terms = []
        tick_s = cfg.tick_s
        for pid, intervals in occupied.items():
            for i, (left, right) in enumerate(intervals):
                port_intervals[pid].append(
                    model.NewFixedSizeIntervalVar(left, right - left, f"port_fixed_{pid}_{i}")
                )
        for sid, intervals in queues.items():
            for i, (left, right) in enumerate(intervals):
                if right > left:
                    queue_intervals[sid].append(
                        model.NewFixedSizeIntervalVar(left, right - left, f"queue_fixed_{sid}_{i}")
                    )
        for index, candidate in enumerate(candidates):
            decision = candidate.decision
            chosen = model.NewBoolVar(f"choose_{index}_{decision.kind}")
            candidate.selected = chosen
            robot_vars.setdefault(decision.robot_id, []).append(chosen)
            if decision.kind == "task":
                task_vars.setdefault(decision.task_id, []).append(chosen)
            elif decision.kind == "reposition":
                parking_vars.setdefault(decision.target_position, []).append(chosen)
            elif decision.kind == "charge":
                quantum = self._ticks(cfg.decision_s)
                slot = model.NewIntVar(
                    int(ceil(candidate.earliest_tick / quantum)),
                    candidate.latest_tick // quantum,
                    f"start_slot_{index}",
                )
                start = model.NewIntVar(
                    candidate.earliest_tick, candidate.latest_tick, f"start_{index}"
                )
                model.Add(start == slot * quantum)
                end = model.NewIntVar(
                    candidate.earliest_tick + candidate.duration_tick,
                    candidate.latest_tick + candidate.duration_tick,
                    f"end_{index}",
                )
                interval = model.NewOptionalIntervalVar(
                    start, candidate.duration_tick, end, chosen, f"charge_{index}"
                )
                port_intervals[decision.port_id].append(interval)
                arrival = self._ticks(decision.arrival_s)
                wait = model.NewIntVar(0, max(0, candidate.latest_tick - arrival), f"wait_{index}")
                model.Add(wait == start - arrival).OnlyEnforceIf(chosen)
                model.Add(wait == 0).OnlyEnforceIf(chosen.Not())
                queue = model.NewOptionalIntervalVar(arrival, wait, start, chosen, f"queue_{index}")
                queue_intervals[candidate.station_id].append(queue)
                candidate.start_var, candidate.delay_var = start, wait
                # Integer objective precision is kept by scaling all costs to1e-6 precision.
                # Wait cost coefficient uses a separately scaled linear expression.
                wait_coeff = max(1, round(COST_SCALE * 0.3 * tick_s / 900 / n))
                objective_terms.append(wait_coeff * wait)
            coefficient = round(COST_SCALE * self._candidate_cost(candidate, profile, n))
            objective_terms.append(coefficient * chosen)
        for variables in robot_vars.values():
            model.AddExactlyOne(variables)
        for variables in task_vars.values():
            model.AddAtMostOne(variables)
        for variables in parking_vars.values():
            model.AddAtMostOne(variables)
        for intervals in port_intervals.values():
            if intervals:
                model.AddNoOverlap(intervals)
        for station, intervals in queue_intervals.items():
            if intervals:
                model.AddCumulative(
                    intervals, [1] * len(intervals), self._queue_capacity(snapshot, station)
                )
        backlog = [t for t in snapshot.tasks if t.status in {"waiting", "blocked"}]
        rates = list(snapshot.demand_rate_by_zone)
        recent_service = [
            t.completed_at_s - t.assigned_at_s
            for t in snapshot.tasks
            if t.completed_at_s is not None
            and t.assigned_at_s is not None
            and snapshot.sim_time_s - 60 <= t.completed_at_s <= snapshot.sim_time_s
            and t.completed_at_s >= t.assigned_at_s
        ]
        mean_service = sum(recent_service) / len(recent_service) if recent_service else 120.0
        work_count = sum(
            r.port_id is None and (r.task_id is not None or r.status == "reposition")
            for r in snapshot.robots
        )
        required = min(n, max(len(backlog), int(ceil(sum(rates) * mean_service))))
        ready_variables = [c.selected for c in candidates if c.available]
        shortage = model.NewIntVar(0, n, "available_shortage")
        model.Add(shortage >= required - work_count - sum(ready_variables))
        objective_terms.append(round(COST_SCALE * profile.availability / n) * shortage)
        # Use current backlog and trailing arrival rates only, never future demand.
        demand = [sum(t.zone == z for t in backlog) + rates[z] * 60 for z in range(4)]
        total_demand = sum(demand)
        if total_demand > 0:
            target_total = max(
                required, min(n, sum(c.decision.kind == "wait" for c in candidates) + work_count)
            )
            for zone in range(4):
                target = round(100 * target_total * demand[zone] / total_demand)
                current_work = sum(
                    snapshot.warehouse.zone_of(r.position) == zone
                    and r.port_id is None
                    and (r.task_id is not None or r.status == "reposition")
                    for r in snapshot.robots
                )
                count = current_work + sum(
                    c.selected for c in candidates if c.available and c.zone == zone
                )
                imbalance = model.NewIntVar(0, 100 * n, f"zone_deviation_{zone}")
                model.AddAbsEquality(imbalance, 100 * count - target)
                # 100 units/fleet; multiplying common objective by100 would be
                # equivalent; high integer precision preserves target influence.
                objective_terms.append(
                    max(1, round(COST_SCALE * profile.zone / (100 * n))) * imbalance
                )
        model.Minimize(sum(objective_terms))
        # A physically checked greedy incumbent prevents a short timeout from
        # returning an all-HOLD plan before discovering ordinary task service.
        hint_plan = self._heuristic(snapshot, candidates, profile_id)
        hint_decisions = {d.robot_id: d for d in hint_plan.decisions}
        for candidate in candidates:
            decision = candidate.decision
            hint = hint_decisions.get(decision.robot_id)
            selected = hint is not None and (
                decision.kind,
                decision.task_id,
                decision.port_id,
                decision.target_soc,
                decision.target_position,
            ) == (hint.kind, hint.task_id, hint.port_id, hint.target_soc, hint.target_position)
            # Heuristic HOLD does not require an explicit target cell.
            if decision.kind == "wait" and hint is not None and hint.kind == "wait":
                selected = True
            model.AddHint(candidate.selected, int(selected))
            if selected and candidate.start_var is not None:
                model.AddHint(candidate.start_var, self._ticks(hint.start_s))
                model.AddHint(
                    candidate.delay_var, self._ticks(hint.start_s) - self._ticks(hint.arrival_s)
                )
        remaining = deadline - perf_counter()
        if remaining <= 0:
            plan = self._hold(snapshot, profile_id, "model_generation_budget")
            plan.candidate_count = len(candidates)
            plan.total_ms = (perf_counter() - started) * 1000
            return plan
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = min(cfg.solver_time_limit_s, remaining)
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = cfg.seed
        solver.parameters.cp_model_presolve = False
        solver_started = perf_counter()
        try:
            status = solver.Solve(model)
        except Exception as error:
            plan = self._heuristic(
                snapshot, candidates, profile_id, f"solver_exception:{type(error).__name__}"
            )
            plan.solver_ms = (perf_counter() - solver_started) * 1000
            plan.total_ms = (perf_counter() - started) * 1000
            return plan
        solver_ms = (perf_counter() - solver_started) * 1000
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            reason = f"solver_{solver.StatusName(status).lower()}"
            if perf_counter() < deadline:
                plan = self._heuristic(snapshot, candidates, profile_id, reason)
            else:
                plan = self._hold(snapshot, profile_id, reason + "_budget_exhausted")
            plan.metadata["raw_solver_status"] = solver.StatusName(status)
        else:
            decisions = []
            for candidate in candidates:
                if not solver.BooleanValue(candidate.selected):
                    continue
                decision = replace(candidate.decision, metadata=dict(candidate.decision.metadata))
                if decision.kind == "charge":
                    start = solver.Value(candidate.start_var)
                    decision.start_s = start * tick_s
                    decision.end_s = (start + candidate.duration_tick) * tick_s
                decision.metadata["surrogate_cost"] = self._candidate_cost(candidate, profile, n)
                decisions.append(decision)
            plan = DecisionPlan(
                snapshot.version,
                profile_id,
                decisions,
                solver_status=solver.StatusName(status),
                fallback_reason=policy_error,
                objective=solver.ObjectiveValue() / COST_SCALE,
                candidate_count=len(candidates),
                metadata={
                    "profile_name": profile.name,
                    "objective_normalization": "all_additive_terms_per_robot",
                    "rejections": dict(self.last_rejections),
                    "best_bound": solver.BestObjectiveBound() / COST_SCALE,
                    "required_available": required,
                    "mean_service_s": mean_service,
                    "energy_urgent_robot_ids": sorted(self._urgent_ids),
                    "return_calendar": "conditional_on_existing_bookings",
                },
            )
        plan.metadata.update(
            solver_called=True,
            raw_solver_status=solver.StatusName(status),
            nonoptimal=status != cp_model.OPTIMAL,
            timed_out=None,
            termination_reason="optimal" if status == cp_model.OPTIMAL else None,
            backend="cpsat",
        )
        plan.solver_ms = solver_ms
        plan.total_ms = (perf_counter() - started) * 1000
        return plan
