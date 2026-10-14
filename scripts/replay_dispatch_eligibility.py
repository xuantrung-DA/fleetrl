"""Replay recorded plans to separate parking exclusion from charger-calendar rejection."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def replay(artifact: Path) -> dict[str, Any]:
    from fleetrl.config import FleetConfig
    from fleetrl.energy import travel_time
    from fleetrl.optimizer import Candidate, FleetOptimizer
    from fleetrl.simulator import Simulator
    from fleetrl.types import Decision, DecisionPlan, EventTape

    saved = json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
    config = FleetConfig(**saved["config"]).validate()
    tape = EventTape.from_dict(json.loads((artifact / "tape.json").read_text(encoding="utf-8")))
    cycles = json.loads((artifact / "cycles.json").read_text(encoding="utf-8"))
    sim = Simulator(config)
    sim.reset(saved["seed"], tape)
    optimizer = FleetOptimizer(config)
    totals: Counter = Counter()
    trace = []
    slower_parking_choices = []
    for cycle in cycles:
        snapshot = sim.snapshot()
        candidates = optimizer.generate_candidates(snapshot)
        offered = {
            (c.decision.robot_id, c.decision.task_id)
            for c in candidates
            if c.decision.kind == "task"
        }
        occupied, _ = optimizer._calendars(snapshot)
        free_calendars = {port.id: [] for port in snapshot.ports}
        waiting = [t for t in snapshot.tasks if t.status in {"waiting", "blocked"}]
        robots = {r.id: r for r in snapshot.robots}
        tasks = {t.id: t for t in snapshot.tasks}
        used_robots = set()
        for planned in cycle["decisions"]:
            if planned["kind"] == "task" and robots[planned["robot_id"]].status == "reposition":
                task = tasks[planned["task_id"]]
                contenders = [
                    c
                    for c in candidates
                    if c.decision.kind == "task"
                    and c.decision.task_id == task.id
                    and c.decision.robot_id not in used_robots
                ]

                def pickup_time(candidate: Candidate) -> float:
                    r = robots[candidate.decision.robot_id]
                    loaded = optimizer._distance(snapshot, task.pickup, task.dropoff)
                    return (
                        candidate.duration_s - travel_time(r, loaded, config) - 2 * config.service_s
                    )

                selected = next(
                    (c for c in contenders if c.decision.robot_id == planned["robot_id"]), None
                )
                idle = [c for c in contenders if robots[c.decision.robot_id].status == "idle"]
                if selected is not None and idle:
                    best_idle = min(idle, key=pickup_time)
                    if pickup_time(selected) > pickup_time(best_idle) + 1e-8:
                        slower_parking_choices.append(
                            {
                                "time_s": sim.time_s,
                                "task_id": task.id,
                                "selected_robot_id": selected.decision.robot_id,
                                "selected_pickup_s": pickup_time(selected),
                                "idle_robot_id": best_idle.decision.robot_id,
                                "idle_pickup_s": pickup_time(best_idle),
                            }
                        )
            if planned["kind"] != "wait":
                used_robots.add(planned["robot_id"])
        pairs = []
        for robot in snapshot.robots:
            if (
                robot.task_id is not None
                or robot.port_id is not None
                or robot.status not in {"idle", "reposition"}
            ):
                continue
            for task in waiting:
                if task.weight_kg > robot.payload_capacity_kg:
                    continue
                optimizer.last_rejections = {}
                physical = optimizer._task_candidate(snapshot, robot, task, occupied)
                reason = next(iter(optimizer.last_rejections), None)
                if physical is None:
                    without_booking = optimizer._task_candidate(
                        snapshot, robot, task, free_calendars
                    )
                    reason = "return_calendar" if without_booking is not None else reason
                elif (robot.id, task.id) in offered:
                    reason = "offered"
                elif robot.id in optimizer._urgent_ids:
                    reason = "urgent_charge_priority"
                elif robot.status == "reposition":
                    reason = "parking_commit"
                else:
                    reason = "shortlist_or_candidate_cap"
                totals[reason] += 1
                pairs.append(
                    {
                        "robot_id": robot.id,
                        "task_id": task.id,
                        "weight_kg": task.weight_kg,
                        "status": robot.status,
                        "position": robot.position,
                        "goal": robot.target_position,
                        "soc": robot.soc,
                        "reason": reason,
                        "offered": (robot.id, task.id) in offered,
                    }
                )
        trace.append({"time_s": sim.time_s, "pairs": pairs})
        decisions = []
        for value in cycle["decisions"]:
            value = dict(value)
            if value["target_position"] is not None:
                value["target_position"] = tuple(value["target_position"])
            decisions.append(Decision(**value))
        profile = None if config.action_mode == "multiagent" else config.fixed_profile
        committed = sim.submit_plan(
            DecisionPlan(snapshot.version, profile, decisions, solver_status=cycle["solver_status"])
        )
        for key in ("accepted", "rejected"):
            if committed[key] != cycle["commit"][key]:
                raise AssertionError(f"replay commit diverged at {sim.time_s}: {key}")
        sim.advance(config.decision_s)
    for metric in ("completed", "distance_m", "consumed_wh"):
        actual = sim.counters[metric]
        expected = saved["metrics"][metric]
        if abs(actual - expected) > 1e-7:
            raise AssertionError(f"replay final metric diverged: {metric}: {actual} != {expected}")
    result = {
        "verified_replay": True,
        "source_hash": saved["source_hash"],
        "tape_hash": tape.tape_hash,
        "counts": dict(totals),
        "cycles": trace,
        "slower_parking_choices": slower_parking_choices,
    }
    (artifact / "eligibility_replay.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", default=["S4"])
    parser.add_argument("--methods", nargs="+", default=["fixed_cpsat"])
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve() / "src"))
    status = json.loads((args.stage / "status.json").read_text(encoding="utf-8"))
    for name, job in status["jobs"].items():
        if job["status"] != "complete":
            continue
        saved = json.loads((Path(job["artifact"]) / "summary.json").read_text(encoding="utf-8"))
        if saved["scenario"] not in args.scenarios or saved["method"] not in args.methods:
            continue
        result = replay(Path(job["artifact"]))
        print(f"{name}: verified replay; {result['counts']}", flush=True)


if __name__ == "__main__":
    main()
