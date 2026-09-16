"""SCIP integer dispatch; same quantized costs, candidates and full-tail calendars.

Charge start binaries enumerate the CP-SAT five-second lattice. Occupancy is
half-open, including every fixed booking and the complete tail of each start.
"""

from collections import defaultdict
from dataclasses import replace
from math import ceil
from time import perf_counter

from ortools.linear_solver import pywraplp

from ..optimizer import COST_SCALE
from ..types import DecisionPlan


def available_solver():
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP MILP backend unavailable; no substitution is permitted")
    solver.SetNumThreads(1)
    return solver


def occupancy_constraints(solver, alternatives, fixed, capacity):
    """An alternative is (binary, inclusive_start, exclusive_end)."""
    events = defaultdict(list)
    fixed_events = defaultdict(int)
    for variable, a, b in alternatives:
        if b > a:
            events[a].append((variable, 1))
            events[b].append((variable, -1))
    for a, b in fixed:
        if b > a:
            fixed_events[a] += 1
            fixed_events[b] -= 1
    points = sorted(set(events) | set(fixed_events))
    active = {}
    occupied = 0
    # Merge identical active sets to avoid redundant constraints.
    seen = set()
    for point in points[:-1]:
        for variable, delta in events[point]:
            index = variable.index()
            count = active.get(index, (variable, 0))[1] + delta
            if count:
                active[index] = (variable, count)
            else:
                active.pop(index, None)
        occupied += fixed_events[point]
        key = (tuple(sorted((index, count) for index, (_, count) in active.items())), occupied)
        if key in seen:
            continue
        seen.add(key)
        solver.Add(
            solver.Sum(variable * count for variable, count in active.values())
            <= capacity - occupied
        )


def violated_resource_rows(solver, resources):
    """Separate exact resource cuts from an integral incumbent, using full tails.

    Every added row is a valid row of the full time-indexed formulation. If the
    relaxed optimum passes every calendar, it is also the full MILP optimum.
    This avoids constructing millions of coefficients before the first solve.
    """
    incumbent = {
        x.index(): x.solution_value()
        for alternatives, _, _ in resources
        for x, _, _ in alternatives
    }
    rows = []
    for alternatives, fixed, capacity in resources:
        selected = [(x, a, b) for x, a, b in alternatives if incumbent[x.index()] > 0.5 and b > a]
        points = sorted(
            {v for _, a, b in selected for v in (a, b)} | {v for a, b in fixed for v in (a, b)}
        )
        probes = points[:-1] + [(a + b) / 2 for a, b in zip(points, points[1:])]
        best = None
        for point in probes:
            existing = sum(a <= point < b for a, b in fixed)
            if existing + sum(a <= point < b for _, a, b in selected) > capacity:
                active = [x for x, a, b in alternatives if a <= point < b]
                if best is None or len(active) > len(best[0]):
                    best = (active, capacity - existing)
        if best is not None:
            rows.append(best)
    # Reading solution_value after adding a row invalidates the SCIP incumbent.
    # Capture every resource first, then mutate the model.
    for active, capacity in rows:
        solver.Add(solver.Sum(active) <= capacity)
    return len(rows)


def solve_dispatch(owner, snapshot, candidates, profile, profile_id, started, deadline):
    cfg = owner.config
    n = max(1, len(snapshot.robots))
    solver = available_solver()
    selected = []
    charge_starts = {}
    robot = defaultdict(list)
    task = defaultdict(list)
    parking = defaultdict(list)
    ports = defaultdict(list)
    queues = defaultdict(list)
    costs = []
    occupied, fixed_queues = owner._calendars(snapshot)
    quantum = owner._ticks(cfg.decision_s)
    wait_coeff = max(1, round(COST_SCALE * 0.3 * cfg.tick_s / 900 / n))
    for i, c in enumerate(candidates):
        x = solver.BoolVar(f"choose_{i}")
        selected.append(x)
        d = c.decision
        robot[d.robot_id].append(x)
        costs.append(round(COST_SCALE * owner._candidate_cost(c, profile, n)) * x)
        if d.kind == "task":
            task[d.task_id].append(x)
        if d.kind == "reposition":
            parking[d.target_position].append(x)
        if d.kind == "charge":
            starts = []
            arrival = owner._ticks(d.arrival_s)
            for start in range(
                ceil(c.earliest_tick / quantum) * quantum, c.latest_tick + 1, quantum
            ):
                if any(
                    start < b and a < start + c.duration_tick
                    for a, b in occupied.get(d.port_id, [])
                ):
                    continue
                from ..optimizer import queue_fits

                if not queue_fits(
                    arrival,
                    start,
                    fixed_queues.get(c.station_id, []),
                    owner._queue_capacity(snapshot, c.station_id),
                ):
                    continue
                y = solver.BoolVar(f"start_{i}_{start}")
                starts.append((y, start))
                ports[d.port_id].append((y, start, start + c.duration_tick))
                if start > arrival:
                    queues[c.station_id].append((y, arrival, start))
                costs.append(wait_coeff * (start - arrival) * y)
            solver.Add(solver.Sum(y for y, _ in starts) == x)
            charge_starts[i] = starts
        if perf_counter() >= deadline:
            plan = owner._hold(snapshot, profile_id, "milp_model_generation_budget")
            plan.metadata.update(backend="milp", solver_called=False)
            plan.total_ms = (perf_counter() - started) * 1000
            return plan
    for xs in robot.values():
        solver.Add(solver.Sum(xs) == 1)
    for xs in list(task.values()) + list(parking.values()):
        solver.Add(solver.Sum(xs) <= 1)
    resources = [(ports[pid], occupied.get(pid, []), 1) for pid in set(ports) | set(occupied)]
    resources += [
        (queues[sid], fixed_queues.get(sid, []), owner._queue_capacity(snapshot, sid))
        for sid in set(queues) | set(fixed_queues)
    ]
    backlog = [t for t in snapshot.tasks if t.status in {"waiting", "blocked"}]
    rates = list(snapshot.demand_rate_by_zone)
    recent = [
        t.completed_at_s - t.assigned_at_s
        for t in snapshot.tasks
        if t.completed_at_s is not None
        and t.assigned_at_s is not None
        and snapshot.sim_time_s - 60 <= t.completed_at_s <= snapshot.sim_time_s
        and t.completed_at_s >= t.assigned_at_s
    ]
    mean_service = sum(recent) / len(recent) if recent else 120.0
    work = sum(
        r.port_id is None and (r.task_id is not None or r.status == "reposition")
        for r in snapshot.robots
    )
    required = min(n, max(len(backlog), ceil(sum(rates) * mean_service)))
    shortage = solver.IntVar(0, n, "shortage")
    solver.Add(
        shortage
        >= required - work - solver.Sum(x for c, x in zip(candidates, selected) if c.available)
    )
    costs.append(round(COST_SCALE * profile.availability / n) * shortage)
    demand = [sum(t.zone == z for t in backlog) + rates[z] * 60 for z in range(4)]
    if sum(demand) > 0:
        total = max(required, min(n, sum(c.decision.kind == "wait" for c in candidates) + work))
        for z in range(4):
            target = round(100 * total * demand[z] / sum(demand))
            current = sum(
                snapshot.warehouse.zone_of(r.position) == z
                and r.port_id is None
                and (r.task_id is not None or r.status == "reposition")
                for r in snapshot.robots
            )
            count = current + solver.Sum(
                x for c, x in zip(candidates, selected) if c.available and c.zone == z
            )
            dev = solver.IntVar(0, 100 * n, f"zone_{z}")
            solver.Add(dev >= 100 * count - target)
            solver.Add(dev >= target - 100 * count)
            costs.append(max(1, round(COST_SCALE * profile.zone / (100 * n))) * dev)
    solver.Minimize(solver.Sum(costs))
    remaining = min(cfg.solver_time_limit_s, deadline - perf_counter())
    if remaining <= 0:
        plan = owner._hold(snapshot, profile_id, "milp_model_generation_budget")
        plan.metadata.update(backend="milp", solver_called=False)
        plan.total_ms = (perf_counter() - started) * 1000
        return plan
    solve_start = perf_counter()
    solve_deadline = min(deadline, solve_start + cfg.solver_time_limit_s)
    rounds = 0
    resource_valid = False
    raw_statuses = []
    while perf_counter() < solve_deadline:
        solver.SetTimeLimit(max(1, int((solve_deadline - perf_counter()) * 1000)))
        status = solver.Solve()
        rounds += 1
        raw_statuses.append(status)
        if status not in (solver.OPTIMAL, solver.FEASIBLE):
            break
        if not solver.VerifySolution(1e-6, False):
            break
        if not all(
            abs(x.solution_value() - round(x.solution_value())) < 1e-6
            for x in solver.variables()
            if x.integer()
        ):
            break
        if violated_resource_rows(solver, resources) == 0:
            resource_valid = True
            break
    if not rounds:
        status = solver.NOT_SOLVED
    elapsed = (perf_counter() - solve_start) * 1000
    names = {
        solver.OPTIMAL: "OPTIMAL",
        solver.FEASIBLE: "FEASIBLE",
        solver.INFEASIBLE: "INFEASIBLE",
        solver.UNBOUNDED: "UNBOUNDED",
        solver.ABNORMAL: "ABNORMAL",
        solver.NOT_SOLVED: "NOT_SOLVED",
    }
    raw = names.get(status, str(status))
    valid = (
        resource_valid
        and status in (solver.OPTIMAL, solver.FEASIBLE)
        and solver.VerifySolution(1e-6, False)
    )
    if valid:
        valid = all(
            abs(x.solution_value() - round(x.solution_value())) < 1e-6
            for x in solver.variables()
            if x.integer()
        )
    if valid:
        decisions = []
        for i, (c, x) in enumerate(zip(candidates, selected)):
            if x.solution_value() < 0.5:
                continue
            d = replace(c.decision, metadata=dict(c.decision.metadata))
            if d.kind == "charge":
                chosen = [s for y, s in charge_starts[i] if y.solution_value() > 0.5]
                if len(chosen) != 1:
                    raise RuntimeError("MILP returned inconsistent charge start")
                d.start_s = chosen[0] * cfg.tick_s
                d.end_s = (chosen[0] + c.duration_tick) * cfg.tick_s
            decisions.append(d)
        plan = DecisionPlan(
            snapshot.version,
            profile_id,
            decisions,
            solver_status=raw,
            objective=solver.Objective().Value() / COST_SCALE,
            candidate_count=len(candidates),
        )
        plan.metadata["best_bound"] = solver.Objective().BestBound() / COST_SCALE
    elif perf_counter() < deadline:
        plan = owner._heuristic(
            snapshot, candidates, profile_id, f"milp_{raw.lower()}_no_valid_incumbent"
        )
    else:
        plan = owner._hold(
            snapshot, profile_id, f"milp_{raw.lower()}_no_valid_incumbent_budget_exhausted"
        )
    plan.metadata.update(
        backend="milp",
        solver_called=rounds > 0,
        raw_solver_status=raw,
        nonoptimal=not valid or status != solver.OPTIMAL,
        timed_out=None,
        termination_reason="optimal" if valid and status == solver.OPTIMAL else None,
        variable_count=solver.NumVariables(),
        constraint_count=solver.NumConstraints(),
        required_available=required,
        mean_service_s=mean_service,
        rejections=dict(owner.last_rejections),
        resource_separation_rounds=rounds,
        resource_validated=resource_valid,
        solver_invocations=rounds,
        raw_solver_statuses=[names.get(s, str(s)) for s in raw_statuses],
    )
    plan.solver_ms = elapsed
    plan.total_ms = (perf_counter() - started) * 1000
    return plan
