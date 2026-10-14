"""Trace idle parking motion without changing production metrics or controllers."""

from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TextIO

import torch
from ortools.sat.python import cp_model

from fleetrl.config import FleetConfig
from fleetrl.env import FleetEnv
from fleetrl.evaluation import (
    choose_action,
    file_sha256,
    serializable,
    source_tree_hash,
    write_json,
)
from fleetrl.learning.policies import InferenceSession
from fleetrl.methods.registry import get_method
from fleetrl.optimizer import PROFILES
from fleetrl.rl import load_policy
from fleetrl.scenario import scenario_config
from fleetrl.simulator import Simulator
from fleetrl.types import EventTape, Robot


def emit(stream: TextIO, value: Any) -> None:
    stream.write(json.dumps(serializable(value), ensure_ascii=False, allow_nan=False) + "\n")


class TracedSimulator(Simulator):
    """Record goal setters locally; the engine implementation stays untouched."""

    def __init__(self, config: FleetConfig, stream: TextIO):
        self.trace_stream = stream
        self.goal_sources: dict[int, str] = {}
        self.goal_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
        super().__init__(config)

    def _set_goal(self, robot: Robot, goal: tuple[int, int] | None) -> None:
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        source = caller.f_code.co_name if caller is not None else "unknown"
        line = caller.f_lineno if caller is not None else None
        previous = robot.target_position
        super()._set_goal(robot, goal)
        self.goal_sources[robot.id] = source
        record = {
            "tick": self.tick,
            "time_s": self.time_s,
            "robot_id": robot.id,
            "position": robot.position,
            "previous_goal": previous,
            "goal": robot.target_position,
            "status": robot.status,
            "source": source,
            "source_line": line,
            "free": robot.task_id is None and self._booking(robot) is None,
        }
        self.goal_history[robot.id].append(record)
        emit(self.trace_stream, record)
        del frame, caller


def run_case(
    config: FleetConfig,
    method: str,
    seed: int,
    output: Path,
    checkpoint: Path,
    tape_directory: Path | None,
    deterministic_time: float | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    config = get_method(method).configure(config)
    policy = None
    if method == "mappo_dispatch":
        policy = InferenceSession(load_policy(checkpoint, device="cpu"))
        metadata = policy.policy._fleetrl_metadata
        config = config.copy(
            observation_mode=metadata["observation_mode"],
            charge_threshold=metadata["env"]["charge_threshold"],
        )
    env = FleetEnv(config)
    tape = None
    if tape_directory is not None:
        tape = EventTape.from_dict(
            json.loads((tape_directory / output.name / "tape.json").read_text(encoding="utf-8"))
        )
    previous: dict[int, dict[str, Any]] = {}
    counts: dict[int, Counter] = defaultdict(Counter)
    distance_by_source: Counter = Counter()
    with (
        (output / "goals.jsonl").open("w", encoding="utf-8") as goals,
        (output / "ticks.jsonl").open("w", encoding="utf-8") as ticks,
        (output / "decisions.jsonl").open("w", encoding="utf-8") as decisions,
    ):
        sim = TracedSimulator(config, goals)
        env.sim = sim
        observation, _ = env.reset(seed=seed, options={"tape": tape} if tape else None)

        def observe_tick() -> None:
            decision = env.decision_logs[-1] if env.decision_logs else {}
            applied = {
                d["robot_id"]: d
                for d in decision.get("decisions", [])
                if d["robot_id"]
                in {a["robot_id"] for a in decision.get("commit", {}).get("accepted", [])}
            }
            for robot in sim.robots.values():
                free = robot.task_id is None and sim._booking(robot) is None
                source = sim.goal_sources.get(robot.id, "initial_parking")
                row = {
                    "tick": sim.tick,
                    "time_s": sim.time_s,
                    "robot_id": robot.id,
                    "position": robot.position,
                    "status": robot.status,
                    "goal": robot.target_position,
                    "goal_source": source,
                    "move_to": robot.move_to,
                    "path": robot.path,
                    "free": free,
                    "task_id": robot.task_id,
                    "booking": sim._booking(robot),
                    "decision": applied.get(robot.id),
                    "profile": decision.get("profile_id"),
                    "battery_wh": robot.battery_wh,
                }
                old = previous.get(robot.id)
                if old is not None and old["free"] and free:
                    if old["position"] != robot.position:
                        counts[robot.id]["free_position_changes"] += 1
                        cause = old["goal_source"]
                        status = old["status"]
                        counts[robot.id][f"distance_m:{status}:{cause}"] += 1
                        distance_by_source[f"{status}:{cause}"] += 1
                    if old["goal"] != robot.target_position:
                        counts[robot.id]["free_goal_state_changes"] += 1
                    if robot.status == "idle" and (robot.path or robot.move_to):
                        counts[robot.id]["idle_with_route_ticks"] += 1
                emit(ticks, row)
                previous[robot.id] = row

        env.set_tick_observer(observe_tick)
        observe_tick()
        done = False
        while not done:
            snapshot = env._decision_snapshot
            candidates = env.optimizer.generate_candidates(snapshot)
            action, failure, elapsed = choose_action(policy, observation, config)
            env.set_inference_context(elapsed, failure)
            observation, _, done, _, info = env.step(action)
            emit(
                decisions,
                {
                    **info["decision"],
                    "demand_rates": snapshot.demand_rate_by_zone,
                    "candidates": [
                        {
                            "decision": c.decision,
                            "zone": c.zone,
                            "available": c.available,
                            "cost": env.optimizer._candidate_cost(c, PROFILES[0], config.n_robots),
                        }
                        for c in candidates
                    ],
                },
            )
        metrics = env.metrics()
        robots = {}
        for rid in sim.robots:
            history = [r for r in sim.goal_history[rid] if r["free"] and r["goal"] is not None]
            cycles = []
            for a, b, c in zip(history, history[1:], history[2:]):
                if a["goal"] == c["goal"] and a["goal"] != b["goal"]:
                    cycles.append(
                        {
                            "a": a["goal"],
                            "b": b["goal"],
                            "start_s": a["time_s"],
                            "end_s": c["time_s"],
                        }
                    )
            robots[rid] = {
                **dict(counts[rid]),
                "free_goal_assignments": len(history),
                "goal_intervals_s": [
                    b["time_s"] - a["time_s"] for a, b in zip(history, history[1:])
                ],
                "cycles": cycles,
            }
        result = {
            "method": method,
            "seed": seed,
            "config": config.to_dict(),
            "tape_hash": sim.tape.tape_hash,
            "source_hash": source_tree_hash(),
            "checkpoint_sha256": file_sha256(checkpoint) if policy else None,
            "solver_deterministic_time_limit": deterministic_time,
            "metrics": metrics,
            "robots": robots,
            "free_distance_by_source_m": dict(distance_by_source),
            "cycles": sum(len(r["cycles"]) for r in robots.values()),
            "yield_events": [
                e for e in sim.events if "yield" in e["kind"] or "deadlock" in e["kind"]
            ],
        }
        write_json(output / "tape.json", sim.tape.to_dict())
        write_json(output / "config.json", config.to_dict())
        write_json(output / "events.json", sim.events)
        write_json(output / "final_state.json", sim._live_snapshot())
        write_json(output / "summary.json", result)
        env.close()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", choices=("diagnostic", "comparison"), default="diagnostic")
    parser.add_argument("--tapes-from", type=Path)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--solver-deterministic-time", type=float)
    parser.add_argument(
        "--methods", nargs="+", choices=("heuristic", "fixed_cpsat", "mappo_dispatch")
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.solver_deterministic_time is not None:
        if not 0 < args.solver_deterministic_time <= 1:
            parser.error("deterministic solver work must be in (0,1]")
        original_solve = cp_model.CpSolver.Solve

        def deterministic_solve(
            solver: cp_model.CpSolver,
            model: cp_model.CpModel,
            solution_callback: cp_model.CpSolverSolutionCallback | None = None,
        ) -> int:
            solver.parameters.max_deterministic_time = args.solver_deterministic_time
            return original_solve(solver, model, solution_callback)

        cp_model.CpSolver.Solve = deterministic_solve
    if args.output.exists():
        parser.error("output already exists; preserve prior evidence")
    args.output.mkdir(parents=True)
    results = []
    base = FleetConfig(horizon_s=600, log_decisions=True)
    if args.solver_deterministic_time is not None:
        # Relax only the diagnostic wall guard so fixed solver work can finish.
        base = base.copy(decision_budget_s=5, solver_time_limit_s=4)
    cases = (
        {
            "zero": base.copy(
                n_robots=10,
                initial_tasks=0,
                demand_per_hour=0,
                initial_soc_min=0.99,
                initial_soc_max=0.99,
                disturbances=False,
            ),
            "sparse": base.copy(
                n_robots=10,
                initial_tasks=0,
                demand_per_hour=12,
                initial_soc_min=0.99,
                initial_soc_max=0.99,
                disturbances=False,
            ),
        }
        if args.suite == "diagnostic"
        else {name: scenario_config(name, base).copy(horizon_s=1200) for name in ("S1", "S2", "S4")}
    )
    if args.cases:
        if set(args.cases) - cases.keys():
            parser.error("unknown case for selected suite")
        cases = {name: config for name, config in cases.items() if name in args.cases}
    seeds = args.seeds or ((1000,) if args.suite == "diagnostic" else (1000, 1001))
    if any(seed < 1000 or seed >= 2000 for seed in seeds):
        parser.error("use validation seeds 1000..1999")
    for case, config in cases.items():
        for seed in seeds:
            for method in args.methods or ("heuristic", "fixed_cpsat", "mappo_dispatch"):
                name = f"{case}_{method}_{seed}"
                result = run_case(
                    config,
                    method,
                    seed,
                    args.output / name,
                    args.checkpoint,
                    args.tapes_from,
                    args.solver_deterministic_time,
                )
                results.append(result)
                write_json(args.output / "summary.json", results)
                print(
                    f"{name}: cycles={result['cycles']} "
                    f"free_distance={sum(result['free_distance_by_source_m'].values())}m "
                    f"pending={result['metrics']['pending']} "
                    f"safety={result['metrics']['safety_incidents']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
