"""Causal EWMA demand forecast and two-operation receding-horizon MILP.

Columns are feasible known-task/charge/parking sequences. Both operations enter
the objective and resource constraints; only the first operation is committed.
The bounded two-operation horizon is explicit, not an oracle or a one-step
optimizer renamed MPC. Arrival forecasts affect terminal readiness, never
create fictional task identities or expose future event-tape entries.
"""

from collections import defaultdict
from dataclasses import asdict, replace
from math import ceil
from time import perf_counter

import numpy as np

from ..optimization.milp import available_solver, occupancy_constraints
from ..optimizer import COST_SCALE, PROFILES, FleetOptimizer, earliest_slot, queue_fits
from ..types import DecisionPlan


class CausalForecast:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.rates = np.zeros(4)
        self.last_time = None

    def observe(self, snapshot):
        # Re-reading the same delayed snapshot must not count as new evidence.
        if self.last_time is None or snapshot.observed_at_s > self.last_time:
            observed = np.asarray(snapshot.demand_rate_by_zone, dtype=float)
            self.rates = (
                observed.copy()
                if self.last_time is None
                else self.alpha * observed + (1 - self.alpha) * self.rates
            )
            self.last_time = snapshot.observed_at_s
        return self.rates.copy()


class MPCController(FleetOptimizer):
    def __init__(self, config):
        super().__init__(config)
        self.reset()

    def reset(self):
        self.forecast = CausalForecast(self.config.forecast_alpha)

    def _scheduled(self, snapshot, candidate):
        c = replace(
            candidate,
            decision=replace(candidate.decision, metadata=dict(candidate.decision.metadata)),
        )
        if c.decision.kind == "charge":
            occupied, queues = self._calendars(snapshot)
            start = earliest_slot(
                c.earliest_tick,
                c.latest_tick,
                c.duration_tick,
                occupied[c.decision.port_id],
                self._ticks(self.config.decision_s),
            )
            if start is None or not queue_fits(
                self._ticks(c.decision.arrival_s),
                start,
                queues.get(c.station_id, []),
                self._queue_capacity(snapshot, c.station_id),
            ):
                return None
            c.decision.start_s = start * self.config.tick_s
            c.decision.end_s = (start + c.duration_tick) * self.config.tick_s
            c.duration_s = c.decision.end_s - snapshot.sim_time_s
        return c

    def _project(self, snapshot, candidate):
        d = candidate.decision
        duration = max(self.config.decision_s, candidate.duration_s)
        future = (
            snapshot.sim_time_s + ceil(duration / self.config.decision_s) * self.config.decision_s
        )
        robots = []
        tasks = []
        for robot in snapshot.robots:
            if robot.id != d.robot_id:
                # Other idle robots are excluded from this robot's route expansion.
                robots.append(
                    replace(robot, status="hold" if robot.status == "idle" else robot.status)
                )
                continue
            position = d.target_position or robot.position
            battery = robot.battery_wh - candidate.energy_wh
            if d.kind == "task":
                position = next(t.dropoff for t in snapshot.tasks if t.id == d.task_id)
            if d.kind == "charge":
                port = next(p for p in snapshot.ports if p.id == d.port_id)
                # Include egress to a free adjacent cell in the horizon projection.
                neighbors = snapshot.warehouse.neighbors(port.position, set(snapshot.blocked_cells))
                if not neighbors:
                    return None
                position = neighbors[0]
                future += self.config.decision_s
                battery = (
                    robot.capacity_wh * d.target_soc
                    - self.config.auxiliary_power_w * self.config.decision_s / 3600
                )
            robots.append(
                replace(
                    robot,
                    position=position,
                    battery_wh=max(0, battery),
                    status="idle",
                    task_id=None,
                    port_id=None,
                    load_kg=0.0,
                    target_position=None,
                    path=[],
                    move_to=None,
                    observed_at_s=future,
                )
            )
        for task in snapshot.tasks:
            tasks.append(
                replace(task, status="completed", completed_at_s=future)
                if d.kind == "task" and task.id == d.task_id
                else task
            )
        return replace(
            snapshot,
            sim_time_s=future,
            observed_at_s=future,
            robots=tuple(robots),
            tasks=tuple(tasks),
        )

    def decide(self, snapshot, profile_id=0, *, weights=None, deadline=None):
        started = perf_counter()
        cfg = self.config
        deadline = min(deadline or float("inf"), started + cfg.decision_budget_s)
        if snapshot.sim_time_s - snapshot.observed_at_s > cfg.stale_after_s:
            return self._hold(snapshot, profile_id, "stale_snapshot")
        forecast_start = perf_counter()
        rates = self.forecast.observe(snapshot)
        forecast_ms = (perf_counter() - forecast_start) * 1000
        firsts = self.generate_candidates(snapshot)
        urgent = set(self._urgent_ids)
        soc = dict(self._robot_soc)
        if not firsts:
            plan = self._hold(snapshot, profile_id, "no_idle_eligible_robot")
            plan.fallback_reason = None
            plan.metadata.update(
                backend="mpc_milp",
                solver_called=False,
                raw_solver_status="NOT_CALLED",
                timed_out=None,
                nonoptimal=False,
                forecast_ms=forecast_ms,
            )
            plan.total_ms = (perf_counter() - started) * 1000
            return plan
        profile = weights or PROFILES[profile_id]
        n = max(1, len(snapshot.robots))
        routes = []
        # Reserve solver time: bounded route expansion is logged, never silently oracle-assisted.
        expansion_deadline = deadline - min(
            cfg.solver_time_limit_s, max(0, deadline - started) * 0.4
        )
        scheduled = [self._scheduled(snapshot, c) for c in firsts]
        scheduled = [c for c in scheduled if c is not None]
        for c in scheduled:
            routes.append((c.decision.robot_id, [c], snapshot.sim_time_s))
        expanded = 0
        for c in sorted(
            scheduled, key=lambda c: (c.decision.kind == "wait", c.decision.robot_id, c.duration_s)
        ):
            if perf_counter() >= expansion_deadline:
                break
            future = self._project(snapshot, c)
            if future is None or future.sim_time_s >= snapshot.sim_time_s + cfg.mpc_horizon_s:
                continue
            seconds = self.generate_candidates(future)
            for second in seconds:
                if (
                    second.decision.robot_id != c.decision.robot_id
                    or second.decision.kind == "wait"
                ):
                    continue
                second = self._scheduled(future, second)
                if second is None:
                    continue
                if future.sim_time_s + second.duration_s > snapshot.sim_time_s + cfg.mpc_horizon_s:
                    continue
                routes.append((c.decision.robot_id, [c, second], future.sim_time_s))
                expanded += 1
                if expanded >= cfg.max_candidates or perf_counter() >= expansion_deadline:
                    break
            if expanded >= cfg.max_candidates:
                break
        self._urgent_ids = urgent
        self._robot_soc = soc
        solver = available_solver()
        by_robot = defaultdict(list)
        by_task = defaultdict(list)
        by_parking = defaultdict(list)
        ports = defaultdict(list)
        queues = defaultdict(list)
        costs = []
        variables = []
        for i, (rid, ops, second_start) in enumerate(routes):
            x = solver.BoolVar(f"route_{i}")
            variables.append(x)
            by_robot[rid].append(x)
            cost = 0.0
            for stage, c in enumerate(ops):
                d = c.decision
                cost += (0.95**stage) * self._candidate_cost(c, profile, n)
                if d.kind == "task":
                    by_task[d.task_id].append(x)
                if d.kind == "reposition":
                    by_parking[d.target_position].append(x)
                if d.kind == "charge":
                    ports[d.port_id].append((x, self._ticks(d.start_s), self._ticks(d.end_s)))
                    queues[c.station_id].append(
                        (x, self._ticks(d.arrival_s), self._ticks(d.start_s))
                    )
                    cost += 0.3 * (d.start_s - d.arrival_s) / 900 / n
            terminal = ops[-1]
            finish = (snapshot.sim_time_s if len(ops) == 1 else second_start) + terminal.duration_s
            readiness = (
                max(0, snapshot.sim_time_s + cfg.mpc_horizon_s - finish) * rates[terminal.zone]
            )
            cost -= profile.availability * min(1.0, readiness) / n
            costs.append(round(cost * COST_SCALE) * x)
        for xs in by_robot.values():
            solver.Add(solver.Sum(xs) == 1)
        for xs in list(by_task.values()) + list(by_parking.values()):
            solver.Add(solver.Sum(xs) <= 1)
        occupied, fixed_queues = self._calendars(snapshot)
        for pid in set(ports) | set(occupied):
            occupancy_constraints(solver, ports[pid], occupied.get(pid, []), 1)
        for sid in set(queues) | set(fixed_queues):
            occupancy_constraints(
                solver, queues[sid], fixed_queues.get(sid, []), self._queue_capacity(snapshot, sid)
            )
        solver.Minimize(solver.Sum(costs))
        remaining = min(cfg.solver_time_limit_s, deadline - perf_counter())
        if remaining <= 0:
            plan = self._hold(snapshot, profile_id, "mpc_model_generation_budget")
            called = False
            raw = "NOT_CALLED"
            elapsed = 0
        else:
            solver.SetTimeLimit(max(1, int(remaining * 1000)))
            solve_start = perf_counter()
            status = solver.Solve()
            elapsed = (perf_counter() - solve_start) * 1000
            called = True
            raw = {
                solver.OPTIMAL: "OPTIMAL",
                solver.FEASIBLE: "FEASIBLE",
                solver.INFEASIBLE: "INFEASIBLE",
                solver.NOT_SOLVED: "NOT_SOLVED",
            }.get(status, str(status))
            if (
                status in (solver.OPTIMAL, solver.FEASIBLE)
                and solver.VerifySolution(1e-6, False)
                and all(
                    abs(x.solution_value() - round(x.solution_value())) < 1e-6 for x in variables
                )
            ):
                chosen = [route for x, route in zip(variables, routes) if x.solution_value() > 0.5]
                plan = DecisionPlan(
                    snapshot.version,
                    profile_id,
                    [replace(ops[0].decision) for _, ops, _ in chosen],
                    solver_status=raw,
                    objective=solver.Objective().Value() / COST_SCALE,
                    candidate_count=len(firsts),
                )
                plan.metadata["selected_sequences"] = [
                    [asdict(c.decision) for c in ops] for _, ops, _ in chosen
                ]
            elif perf_counter() < deadline:
                plan = self._heuristic(snapshot, firsts, profile_id, "mpc_" + raw.lower())
            else:
                plan = self._hold(snapshot, profile_id, "mpc_" + raw.lower() + "_budget_exhausted")
        plan.metadata.update(
            backend="mpc_milp",
            solver_called=called,
            raw_solver_status=raw,
            timed_out=None,
            nonoptimal=called and raw != "OPTIMAL",
            forecast_rates_per_s=rates.tolist(),
            forecast_ms=forecast_ms,
            forecast_observed_at_s=self.forecast.last_time,
            route_count=len(routes),
            two_operation_routes=expanded,
            horizon_s=cfg.mpc_horizon_s,
            max_operations=2,
            committed_operations_per_robot=1,
            forecast_source="EWMA of observed trailing arrival rates",
            route_expansion_bounded=perf_counter() >= expansion_deadline,
        )
        plan.solver_ms = elapsed
        plan.total_ms = (perf_counter() - started) * 1000
        return plan


def controller(config):
    return MPCController(config)
