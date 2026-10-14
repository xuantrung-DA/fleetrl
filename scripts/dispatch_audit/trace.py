"""Record task dispatch, counterfactual eligibility, charging and cycle decisions."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from fleetrl.config import FleetConfig
from fleetrl.env import FleetEnv
from fleetrl.evaluation import choose_action, file_sha256, source_tree_hash, write_json
from fleetrl.learning.policies import InferenceSession
from fleetrl.methods.registry import get_method
from fleetrl.simulator import Simulator
from fleetrl.types import Cell, DecisionPlan, EventTape, Robot, ScenarioEvent, Snapshot


def state_counts(sim: Simulator) -> dict[str, int]:
    robots = list(sim.robots.values())
    return {
        "idle": sum(r.status == "idle" and r.task_id is None for r in robots),
        "parked": sum(r.status == "idle" and r.position in sim.warehouse.parking for r in robots),
        "reposition": sum(r.status == "reposition" and r.task_id is None for r in robots),
        "charging": sum(r.status == "charging" for r in robots),
        "charge_committed": sum(r.port_id is not None for r in robots),
    }


def robot_record(sim: Simulator, robot_id: int) -> dict[str, Any]:
    r = sim.robots[robot_id]
    return {
        "robot_id": r.id,
        "status": r.status,
        "position": r.position,
        "goal": r.target_position,
        "move_to": r.move_to,
        "soc": r.soc,
        "load_kg": r.load_kg,
        "task_id": r.task_id,
        "port_id": r.port_id,
        "parking_changed_at_s": r.metadata.get("parking_changed_at_s"),
    }


class DispatchTraceSimulator(Simulator):
    def reset(self, seed: int | None = None, tape: EventTape | None = None) -> Snapshot:
        self.task_arrivals: dict[int, dict] = {}
        self.assignments: dict[int, list[dict]] = defaultdict(list)
        self.charge_proposals: list[dict] = []
        self.charge_starts: list[dict] = []
        self.charge_samples: Counter = Counter()
        self.parking_changes: list[dict] = []
        return super().reset(seed, tape)

    def _apply_event(self, event: ScenarioEvent) -> None:
        if event.kind == "task":
            task_id = int(event.payload["id"])
            self.task_arrivals[task_id] = {
                "task_id": task_id,
                "created_s": event.time_s,
                "arrival_observed_s": self.time_s,
                "counts_at_arrival": state_counts(self),
                "robots_at_arrival": [robot_record(self, rid) for rid in self.robots],
            }
        super()._apply_event(event)

    def submit_plan(self, plan: DecisionPlan) -> dict:
        before = {rid: robot_record(self, rid) for rid in self.robots}
        result = super().submit_plan(plan)
        for accepted in result["accepted"]:
            rid = accepted["robot_id"]
            decision = next(d for d in plan.decisions if d.robot_id == rid)
            if accepted["kind"] == "task":
                task = self.tasks[accepted["task_id"]]
                self.assignments[task.id].append(
                    {
                        "time_s": self.time_s,
                        "robot_before": before[rid],
                        "distance_to_pickup_m": self._distance(
                            before[rid]["position"], task.pickup
                        ),
                        "counts_at_assignment": state_counts(self),
                        "decision": asdict(decision),
                    }
                )
            elif accepted["kind"] == "charge":
                self.charge_proposals.append(
                    {
                        "time_s": self.time_s,
                        "robot_before": before[rid],
                        "decision": asdict(decision),
                    }
                )
        return result

    def _finish_service(self, robot: Robot) -> None:
        before = robot.status
        super()._finish_service(robot)
        if before == "docking" and robot.status == "charging":
            self.charge_starts.append(
                {
                    "time_s": self.time_s,
                    "robot_id": robot.id,
                    "port_id": robot.port_id,
                    "soc": robot.soc,
                }
            )

    def _set_goal(self, robot: Robot, goal: Cell | None) -> None:
        super()._set_goal(robot, goal)
        if robot.status == "reposition":
            self.parking_changes.append(
                {
                    "time_s": self.time_s,
                    "robot_id": robot.id,
                    "position": robot.position,
                    "goal": robot.target_position,
                }
            )


def counterfactual_eligibility(env: FleetEnv) -> dict[str, Any]:
    snapshot = env._decision_snapshot
    candidates = env.optimizer.generate_candidates(snapshot)
    offered = {c.decision.robot_id for c in candidates if c.decision.kind == "task"}
    waiting = [t for t in snapshot.tasks if t.status in {"waiting", "blocked"}]
    occupied, _ = env.optimizer._calendars(snapshot)
    rejections = dict(env.optimizer.last_rejections)
    blocked = []
    for robot in snapshot.robots:
        if robot.status != "reposition" or robot.task_id is not None or robot.port_id is not None:
            continue
        options = [
            env.optimizer._task_candidate(snapshot, robot, task, occupied) for task in waiting
        ]
        feasible = [c.decision.task_id for c in options if c is not None]
        if feasible and robot.id not in offered:
            blocked.append(
                {
                    "robot_id": robot.id,
                    "feasible_task_ids": feasible,
                    "position": robot.position,
                    "goal": robot.target_position,
                    "move_to": robot.move_to,
                }
            )
    env.optimizer.last_rejections = rejections
    return {
        "excluded_reposition_robots": blocked,
        "offered_task_robots": len(offered),
        "parked_task_offered": sum(
            r.id in offered and r.position in snapshot.warehouse.parking and r.status == "idle"
            for r in snapshot.robots
        ),
        "candidate_rejections": rejections,
    }


def run_episode(
    config: FleetConfig,
    method: str,
    seed: int,
    tape: EventTape,
    output: Path,
    policy: Any,
    checkpoint: Path,
    variant: str,
    solver_control: dict[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("episode output already exists")
    output.mkdir(parents=True)
    config = get_method(method).configure(config)
    inference = None
    if method == "mappo_dispatch":
        metadata = policy._fleetrl_metadata
        config = config.copy(
            observation_mode=metadata["observation_mode"],
            charge_threshold=metadata["env"]["charge_threshold"],
        )
        inference = InferenceSession(policy)
    env = FleetEnv(config)
    sim = DispatchTraceSimulator(config)
    env.sim = sim
    cycles = []
    initial_positions: dict[int, tuple[int, int]] = {}
    no_order_moves = Counter()
    zero_order_stable_ticks = Counter()

    def observe_tick() -> None:
        charging = sum(r.status == "charging" for r in sim.robots.values())
        sim.charge_samples[charging] += 1
        if not sim.tasks:
            for r in sim.robots.values():
                if r.position != initial_positions.get(r.id, r.position):
                    no_order_moves[r.id] += 1
                if r.status == "idle" and r.target_position is None and r.move_to is None:
                    zero_order_stable_ticks[r.id] += 1

    try:
        observation, _ = env.reset(seed=seed, options={"tape": tape})
        initial_positions = {r.id: r.position for r in sim.robots.values()}
        env.set_tick_observer(observe_tick)
        done = False
        while not done:
            counts = state_counts(sim)
            eligibility = counterfactual_eligibility(env)
            action, failure, inference_ms = choose_action(inference, observation, config)
            env.set_inference_context(inference_ms, failure)
            observation, _, done, _, info = env.step(action)
            decision = info["decision"]
            cycles.append(
                {
                    "time_s": decision["sim_time_s"],
                    "counts": counts,
                    **eligibility,
                    "decisions": decision["decisions"],
                    "commit": decision["commit"],
                    "decision_counts": dict(Counter(d["kind"] for d in decision["decisions"])),
                    "solver_status": decision["solver_status"],
                    "fallback_reason": decision["fallback_reason"],
                    "metadata": decision["metadata"],
                }
            )
        tasks = []
        for task in sim.tasks.values():
            history = sim.assignments[task.id]
            first = history[0] if history else None
            tasks.append(
                {
                    **asdict(task),
                    **sim.task_arrivals[task.id],
                    "assignments": history,
                    "first_assignment_s": first["time_s"] if first else None,
                    "assignment_delay_s": first["time_s"] - task.created_s if first else None,
                    "assignment_delay_censored_s": (first["time_s"] if first else sim.time_s)
                    - task.created_s,
                    "first_pickup_distance_m": first["distance_to_pickup_m"] if first else None,
                }
            )
        assigned = [t for t in tasks if t["first_assignment_s"] is not None]
        diagnostics = {
            "mean_assignment_delay_s": mean(t["assignment_delay_s"] for t in assigned)
            if assigned
            else None,
            "mean_assignment_delay_censored_s": mean(
                t["assignment_delay_censored_s"] for t in tasks
            )
            if tasks
            else None,
            "assigned_tasks": len(assigned),
            "never_assigned_tasks": len(tasks) - len(assigned),
            "mean_pickup_distance_m": mean(t["first_pickup_distance_m"] for t in assigned)
            if assigned
            else None,
            "excluded_reposition_robot_cycles": sum(
                len(c["excluded_reposition_robots"]) for c in cycles
            ),
            "excluded_reposition_unique_robots": len(
                {r["robot_id"] for c in cycles for r in c["excluded_reposition_robots"]}
            ),
            "charge_proposals": len(sim.charge_proposals),
            "charge_starts": len(sim.charge_starts),
            "mean_soc_at_charge_start": mean(c["soc"] for c in sim.charge_starts)
            if sim.charge_starts
            else None,
            "charging_concurrency_seconds": {
                n: ticks * config.tick_s for n, ticks in sim.charge_samples.items()
            },
            "parking_goal_assignments": len(sim.parking_changes),
            "decision_counts": dict(
                sum((Counter(c["decision_counts"]) for c in cycles), Counter())
            ),
            "no_order_position_change_ticks": dict(no_order_moves),
            "no_order_stable_s": {
                rid: ticks * config.tick_s for rid, ticks in zero_order_stable_ticks.items()
            },
        }
        result = {
            "variant": variant,
            "scenario": output.name.split("_")[0],
            "method": method,
            "seed": seed,
            "config": config.to_dict(),
            "tape_hash": tape.tape_hash,
            "source_hash": source_tree_hash(),
            "solver_control": solver_control,
            "checkpoint_sha256": file_sha256(checkpoint) if inference else None,
            "metrics": env.metrics(),
            "diagnostics": diagnostics,
        }
        for name, value in {
            "summary": result,
            "tasks": tasks,
            "cycles": cycles,
            "charging": {
                "proposals": sim.charge_proposals,
                "starts": sim.charge_starts,
                "bookings": [asdict(b) for b in sim.booking_history],
                "port_usage": sim.port_usage,
            },
            "parking": sim.parking_changes,
            "final_state": asdict(sim._live_snapshot()),
            "events": sim.events,
        }.items():
            write_json(output / f"{name}.json", value)
        write_json(output / "tape.json", tape.to_dict())
        flat = [
            {
                "task_id": t["id"],
                "created_s": t["created_s"],
                "first_assignment_s": t["first_assignment_s"],
                "completed_at_s": t["completed_at_s"],
                "status": t["status"],
                "weight_kg": t["weight_kg"],
                "assignment_delay_s": t["assignment_delay_s"],
                "assignment_delay_censored_s": t["assignment_delay_censored_s"],
                "first_pickup_distance_m": t["first_pickup_distance_m"],
            }
            for t in tasks
        ]
        if flat:
            with (output / "tasks.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)
        return result
    finally:
        env.close()
