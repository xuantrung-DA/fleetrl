"""Freeze source, run paired multi-seed diagnostics and single-change ablations."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dispatch_audit.snapshots import VARIANTS, prepare, source_hash


def write_status(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def acquire_worker_lock(stage: Path) -> None:
    """Prevent two resumable supervisors from running the same stage twice."""
    stage.mkdir(parents=True, exist_ok=True)
    lock = stage / "worker.lock"
    owner_token = f"{os.getpid()}:{time.monotonic_ns()}"
    while True:
        try:
            with lock.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "owner_token": owner_token}, stream)
            break
        except FileExistsError:
            # Process visibility can differ between Windows execution sandboxes.
            # The owning worker releases this lock; PID lookup is not a lease.
            time.sleep(1)

    def release() -> None:
        if (
            lock.exists()
            and json.loads(lock.read_text(encoding="utf-8")).get("owner_token") == owner_token
        ):
            lock.unlink()

    atexit.register(release)


def worker(args: argparse.Namespace) -> None:
    acquire_worker_lock(args.output / args.variant)
    package = args.source.resolve() / "src/fleetrl"
    sys.path.insert(0, str(package.parent))
    import torch
    from dispatch_audit.trace import run_episode
    from ortools.sat.python import cp_model

    from fleetrl.config import FleetConfig
    from fleetrl.maps import make_map
    from fleetrl.rl import load_policy
    from fleetrl.scenario import generate_tape, scenario_config
    from fleetrl.types import EventTape

    torch.set_num_threads(1)
    if args.control == "deterministic":
        original_solve = cp_model.CpSolver.Solve

        def solve(
            solver: cp_model.CpSolver,
            model: cp_model.CpModel,
            callback: cp_model.CpSolverSolutionCallback | None = None,
        ) -> int:
            solver.parameters.max_deterministic_time = args.deterministic_work
            return original_solve(solver, model, callback)

        cp_model.CpSolver.Solve = solve
    base = FleetConfig(horizon_s=args.horizon, log_decisions=True)
    if args.control == "deterministic":
        base = base.copy(solver_time_limit_s=4, decision_budget_s=5)
    control = {
        "kind": args.control,
        "deterministic_work": args.deterministic_work if args.control == "deterministic" else None,
        "solver_wall_s": base.solver_time_limit_s,
        "decision_wall_s": base.decision_budget_s,
    }
    policy = (
        load_policy(args.checkpoint, device="cpu") if "mappo_dispatch" in args.methods else None
    )
    stage = args.output / args.variant
    stage.mkdir(parents=True, exist_ok=True)
    status_file = stage / "status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {"jobs": {}}
    frozen_hash = source_hash(package)
    status.update(source_hash=frozen_hash, variant=args.variant, control=control)
    for scenario in args.scenarios:
        config = scenario_config(scenario, base)
        for seed in args.seeds:
            tape_path = args.output / "tapes" / f"{scenario}_{seed}.json"
            if tape_path.exists():
                tape = EventTape.from_dict(json.loads(tape_path.read_text(encoding="utf-8")))
            else:
                tape = generate_tape(config, make_map(config), seed)
                write_status(tape_path, tape.to_dict())
            for method in args.methods:
                name = f"{scenario}_{method}_{seed}"
                job = status["jobs"].get(name)
                if job is not None and job["status"] == "complete":
                    if job["source_hash"] != frozen_hash:
                        raise ValueError("refuse to resume with a changed source")
                    continue
                case_output = stage / name
                attempt = 1
                while case_output.exists():
                    attempt += 1
                    case_output = stage / f"{name}_attempt_{attempt:03d}"
                status["jobs"][name] = {
                    "status": "running",
                    "artifact": str(case_output),
                    "source_hash": frozen_hash,
                    "attempt": attempt,
                }
                write_status(status_file, status)
                try:
                    result = run_episode(
                        config,
                        method,
                        seed,
                        tape,
                        case_output,
                        policy,
                        args.checkpoint,
                        args.variant,
                        control,
                    )
                    status["jobs"][name].update(
                        status="complete",
                        metrics=result["metrics"],
                        diagnostics=result["diagnostics"],
                        tape_hash=result["tape_hash"],
                    )
                    write_status(status_file, status)
                    print(
                        f"{args.variant}/{name}: pending={result['metrics']['pending']} "
                        f"completion={result['metrics']['completion_rate']:.3f} "
                        f"safety={result['metrics']['safety_incidents']} "
                        f"excluded={result['diagnostics']['excluded_reposition_robot_cycles']}",
                        flush=True,
                    )
                except Exception:
                    status["jobs"][name].update(status="failed", error=traceback.format_exc())
                    write_status(status_file, status)
                    raise
    status["complete"] = True
    write_status(status_file, status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--variant", default="parking_fix")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument(
        "--scenarios", nargs="+", choices=("S2", "S4", "S6"), default=["S2", "S4", "S6"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1005)))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("heuristic", "fixed_cpsat", "mappo_dispatch"),
        default=["heuristic", "fixed_cpsat", "mappo_dispatch"],
    )
    parser.add_argument("--horizon", type=float, default=1200)
    parser.add_argument("--control", choices=("deterministic", "default"), default="deterministic")
    parser.add_argument("--deterministic-work", type=float, default=0.03)
    parser.add_argument("--parallel", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    args.output, args.checkpoint = args.output.resolve(), args.checkpoint.resolve()
    if any(seed < 1000 or seed >= 2000 for seed in args.seeds):
        parser.error("validation seeds must be in 1000..1999")
    if not 0 < args.deterministic_work <= 1:
        parser.error("deterministic work must be in (0,1]")
    if args.prepare:
        manifest = prepare(Path(__file__).resolve().parents[1], args.output / "sources")
        print(json.dumps(manifest, indent=2))
    if args.matrix:
        manifest = json.loads((args.output / "sources/manifest.json").read_text())

        def launch(variant: str) -> None:
            if variant not in manifest:
                parser.error(f"missing source variant {variant}")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(args.output),
                "--checkpoint",
                str(args.checkpoint),
                "--source",
                str(Path(manifest[variant]["package"]).parents[1]),
                "--variant",
                variant,
                "--control",
                args.control,
                "--deterministic-work",
                str(args.deterministic_work),
                "--horizon",
                str(args.horizon),
                "--scenarios",
                *args.scenarios,
                "--seeds",
                *map(str, args.seeds),
                "--methods",
                *args.methods,
            ]
            with (args.output / f"{variant}.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
                if process.stdout is None:
                    raise RuntimeError("worker stdout unavailable")
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
                if code:
                    raise RuntimeError(f"diagnostic worker {variant} failed with exit {code}")

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            list(executor.map(launch, args.variants))
    elif args.source:
        worker(args)
    elif not args.prepare:
        parser.error("choose --prepare, --matrix, or --source")


if __name__ == "__main__":
    main()
