"""Portable command line. No training runs at import; safe under Windows spawn."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from .config import FleetConfig, TrainConfig, load_config, save_config


def _json(value):
    from .evaluation import serializable

    print(json.dumps(serializable(value), indent=2, ensure_ascii=False, allow_nan=False))


def _csv_int(value):
    return [int(s.strip()) for s in value.split(",") if s.strip()]


def _csv_str(value):
    return [s.strip() for s in value.split(",") if s.strip()]


def doctor() -> dict:
    """Import real dependencies, check torch device, no model downloads."""
    import torch

    # Import these packages to check that their runtime dependencies load too.
    for package in ("gymnasium", "stable_baselines3", "ortools", "numpy", "pygame"):
        importlib.import_module(package)
    from .methods.registry import METHODS
    from .optimization.milp import available_solver

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "milp_backend": available_solver().SolverVersion(),
        "methods": list(METHODS),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "fleetrl",
                "torch",
                "gymnasium",
                "stable-baselines3",
                "sb3-contrib",
                "ortools",
                "numpy",
                "pygame",
                "pytest",
            )
        },
        "cuda_available": torch.cuda.is_available(),
        "recommended_device": "cpu",
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_smoke(output: str | Path, train_steps: int = 128) -> dict:
    """Run real model checks, 10/15/20 rollout, PPO update/save/load/resume; raise on failure."""
    import numpy as np
    from stable_baselines3.common.env_checker import check_env

    from .env import FleetEnv
    from .evaluation import benchmark, run_episode, write_json
    from .rl import ensure_finite_parameters, evaluate_checkpoint, load_policy, train

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("smoke output must be new/empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"status": "running", "runtime": doctor(), "checks": []}
    write_json(output / "smoke_report.json", report)

    def mark(name, detail=None):
        report["checks"].append({"name": name, "passed": True, "detail": detail})
        print(f"[PASS] {name}", flush=True)

    try:
        base = FleetConfig(n_robots=5, horizon_s=120, disturbances=False, solver_time_limit_s=0.03)
        env = FleetEnv(base)
        check_env(env, warn=True)
        env.close()
        mark("SB3 Gymnasium environment contract")
        obs_a = FleetEnv(base)
        obs_b = FleetEnv(base)
        a, _ = obs_a.reset(seed=42)
        b, _ = obs_b.reset(seed=42)
        assert all(np.array_equal(a[k], b[k]) for k in a)
        assert obs_a.sim.tape.tape_hash == obs_b.sim.tape.tape_hash
        mark("identical reset, independent event tape and finite masked observations")
        obs_a.close()
        obs_b.close()
        for n in (10, 15, 20):
            cfg = base.copy(n_robots=n, horizon_s=120, demand_per_hour=n * 6)
            result = run_episode(cfg, "fixed", 2000, output_dir=output / f"fleet_{n}")
            m = result["metrics"]
            assert (
                m["collisions"] == 0 and m["edge_conflicts"] == 0 and m["energy_emergencies"] == 0
            )
            assert m["arrived"] == m["completed"] + m["pending"]
            mark(f"{n} robots:120 simulated seconds without conflict/task loss", m)
        tc = TrainConfig(
            total_timesteps=train_steps,
            n_envs=1,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            eval_freq=max(64, train_steps),
            checkpoint_freq=max(64, train_steps),
            eval_seeds=(1000,),
            curriculum=False,
            tensorboard=False,
            seed=11,
        )
        manifest = train(base, tc, output / "training")
        model = load_policy(manifest["final_model"])
        ensure_finite_parameters(model)
        assert manifest["actual_additional_timesteps"] >= train_steps
        assert getattr(model, "_n_updates", 0) > 0
        mark(
            "PPO optimizer update + finite parameters + checkpoint load",
            {
                "steps": model.num_timesteps,
                "parameters": sum(p.numel() for p in model.policy.parameters()),
            },
        )
        r = evaluate_checkpoint(manifest["final_model"], base, [1001], output / "checkpoint_eval")
        mark("held-out inference from reloaded checkpoint", r)
        resume = train(
            base,
            replace(tc, total_timesteps=64),
            output / "resumed",
            resume=manifest["final_model"],
        )
        assert resume["actual_total_timesteps"] > manifest["actual_total_timesteps"]
        mark(
            "resume adds optimizer updates in new output directory",
            {"steps": resume["actual_total_timesteps"]},
        )
        paired = benchmark(
            base.copy(horizon_s=30), ["heuristic", "fixed"], [2000], ["S1"], output / "paired"
        )
        assert len({e["tape_hash"] for e in paired["episodes"]}) == 1
        mark("paired benchmark, CSV/JSON, manifest and replay output")
        report.update(
            status="passed",
            wall_time_s=time.perf_counter() - started,
            interpretation="Smoke validates execution, not convergence, generalization or 5% improvement.",
        )
    except BaseException as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            wall_time_s=time.perf_counter() - started,
        )
        write_json(output / "smoke_report.json", report)
        raise
    write_json(output / "smoke_report.json", report)
    return report


def tune_baselines(
    config: FleetConfig,
    seeds: list[int],
    profiles: list[int],
    thresholds: list[float],
    output: str | Path,
) -> dict:
    """Validation-only fixed-profile/threshold sweep, never trains or tunes on test."""
    from .evaluation import run_episode, write_json
    from .maps import make_map
    from .rl import validation_rank
    from .scenario import generate_tape

    if not seeds or any(s < 1000 or s >= 2000 for s in seeds):
        raise ValueError("tune uses validation seeds1000..1999 only")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("tune output must be new/empty")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    ranked = []
    if not profiles or not thresholds:
        raise ValueError("profile and threshold grids cannot be empty")
    tapes = {seed: generate_tape(config, make_map(config), seed) for seed in seeds}
    for profile in profiles:
        vals = []
        for seed in seeds:
            result = run_episode(
                config.copy(fixed_profile=profile), "fixed", seed, tape=tapes[seed]
            )
            vals.append(result["metrics"])
            records.append({"profile": profile, **result})
        ranked.append((validation_rank(vals), profile))
        print(f"profile {profile}:rank={ranked[-1][0]}", flush=True)
    # validation_rank is a maximization tuple (-safety, completed, -lateness,-pending).
    best_profile = max(ranked, key=lambda x: x[0])[1]
    threshold_ranks = []
    for threshold in thresholds:
        vals = []
        for seed in seeds:
            result = run_episode(
                config.copy(charge_threshold=threshold), "heuristic", seed, tape=tapes[seed]
            )
            vals.append(result["metrics"])
            records.append({"threshold": threshold, **result})
        threshold_ranks.append((validation_rank(vals), threshold))
    best_threshold = max(threshold_ranks, key=lambda x: x[0])[1]
    chosen = config.copy(fixed_profile=best_profile, charge_threshold=best_threshold)
    save_config(output / "selected.yaml", chosen)
    result = {
        "validation_seeds": seeds,
        "best_fixed_profile": best_profile,
        "best_threshold": best_threshold,
        "records": records,
    }
    write_json(output / "tuning.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fleetrl", description="Train and evaluate constrained warehouse fleet PPO guidance"
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check installed packages/device")
    sub.add_parser("methods", help="List all fourteen registered methods")
    ready = sub.add_parser(
        "ready-check", help="Real update/save/load/resume/eval for all 14 methods"
    )
    ready.add_argument("--output", required=True)
    ready.add_argument("--steps", type=int, default=32)
    ready.add_argument("--methods")
    study = sub.add_parser("study", help="Resumable controlled experiments")
    study.add_argument("phase", choices=["plan", "search", "tune", "train", "test", "report"])
    study.add_argument("--config", default="configs/study.yaml")
    study.add_argument("--output", default="runs/study")
    study.add_argument("--limit", type=int)
    smoke = sub.add_parser("smoke", help="Real engine+PPO save/load/resume smoke")
    smoke.add_argument("--output", default="runs/smoke")
    smoke.add_argument("--train-steps", type=int, default=128)

    def common(q):
        q.add_argument("--config", default=None)
        q.add_argument("--robots", type=int)
        q.add_argument("--horizon", type=float)

    trainp = sub.add_parser("train")
    common(trainp)
    trainp.add_argument("--output", required=True)
    trainp.add_argument("--steps", type=int)
    trainp.add_argument("--seed", type=int)
    trainp.add_argument("--n-envs", type=int)
    trainp.add_argument("--encoder", choices=["entities", "mlp"])
    trainp.add_argument("--device", choices=["cpu", "cuda", "auto"])
    trainp.add_argument("--resume")
    trainp.add_argument("--no-curriculum", action="store_true")
    trainp.add_argument("--no-tensorboard", action="store_true")
    from .methods.registry import METHODS as variants

    method_choices = list(variants) + ["B0", "B1", "H", "A", "fixed", "hybrid", "ablation"]
    trainp.add_argument("--method", choices=method_choices)
    trainp.add_argument(
        "--ablation-charging", action="store_true", help="Retrain with threshold charging"
    )
    ev = sub.add_parser("eval")
    common(ev)
    ev.add_argument("--method", choices=method_choices, default="fixed")
    ev.add_argument("--checkpoint")
    ev.add_argument("--seed", type=int, default=2000)
    ev.add_argument("--scenario")
    ev.add_argument("--tape")
    ev.add_argument("--output", required=True)
    ev.add_argument(
        "--frozen-backend-control",
        action="store_true",
        help="Explicit frozen PPO CP-SAT/MILP transfer, outside the 14 core training runs",
    )
    bench = sub.add_parser("benchmark")
    common(bench)
    bench.add_argument("--methods", default="heuristic,fixed")
    bench.add_argument("--seeds", default="2000,2001,2002,2003,2004")
    bench.add_argument("--scenarios", default="S1,S2,S3,S4,S5,S6,S7,S8,S9")
    bench.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Repeat method=path for independent trained seeds",
    )
    bench.add_argument("--output", required=True)
    gui = sub.add_parser("gui")
    common(gui)
    gui.add_argument("--method", choices=method_choices, default="fixed")
    gui.add_argument("--checkpoint")
    gui.add_argument("--seed", type=int, default=2000)
    gui.add_argument("--output")
    gui.add_argument("--max-frames", type=int)
    gui.add_argument("--screenshot")
    replay = sub.add_parser("replay")
    replay.add_argument("path")
    replay.add_argument("--max-frames", type=int)
    replay.add_argument("--screenshot")
    demo3d = sub.add_parser("demo3d", help="Run the local live Three.js warehouse demo")
    demo3d.add_argument("--study", default="runs/study14")
    demo3d.add_argument("--scenario", choices=["S2", "S4", "S7", "S9"], default="S4")
    demo3d.add_argument("--seed", type=int, default=2000)
    demo3d.add_argument("--port", type=int, default=8765)
    demo3d.add_argument("--output", default="runs/demo3d")
    profile = sub.add_parser("profile")
    common(profile)
    profile.add_argument("--decisions", type=int, default=1000)
    profile.add_argument("--output", required=True)
    tune = sub.add_parser("tune")
    common(tune)
    tune.add_argument("--seeds", default="1000,1001,1002")
    tune.add_argument("--profiles", default="0,1,2,3,4,5")
    tune.add_argument("--thresholds", default="0.2,0.3,0.4")
    tune.add_argument("--output", required=True)
    scenario = sub.add_parser("scenario")
    common(scenario)
    scenario.add_argument("--name", default="S4")
    scenario.add_argument("--seed", type=int, default=2000)
    scenario.add_argument("--output", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            _json(doctor())
            return 0
        if args.command == "methods":
            from .methods.registry import METHODS

            _json([asdict(spec) for spec in METHODS.values()])
            return 0
        if args.command == "ready-check":
            from .experiments.acceptance import ready_check

            report = ready_check(
                args.output, args.steps, _csv_str(args.methods) if args.methods else None
            )
            _json(
                {
                    "status": report["status"],
                    "methods": len(report["methods"]),
                    "report": str(Path(args.output) / "ready_report.json"),
                }
            )
            return 0 if report["status"] == "passed" else 1
        if args.command == "study":
            from .experiments.runner import run_study

            result = run_study(args.config, args.output, args.phase, args.limit)
            _json(result)
            return (
                1
                if result.get("failed_jobs", 0)
                or (result.get("status") == "partial" and args.limit is None)
                else 0
            )
        if args.command == "smoke":
            _json(run_smoke(args.output, args.train_steps))
            return 0
        if args.command == "replay":
            from .ui import run_ui

            _json(
                run_ui(
                    replay_path=args.path,
                    max_frames=args.max_frames,
                    screenshot_path=args.screenshot,
                )
            )
            return 0
        if args.command == "demo3d":
            from .demo3d.server import run_server

            run_server(args.study, args.scenario, args.seed, args.port, args.output)
            return 0
        cfg, tc = load_config(args.config)
        updates = {}
        if args.robots is not None:
            updates.update(n_robots=args.robots, demand_per_hour=args.robots * 6)
        if args.horizon is not None:
            updates["horizon_s"] = args.horizon
        cfg = cfg.copy(**updates)
        if args.command == "train":
            from .methods.registry import get_method
            from .rl import train

            spec = get_method(
                args.method
                or cfg.method
                or ("ppo_threshold_cpsat" if args.ablation_charging else "ppo_cpsat")
            )
            cfg = cfg if cfg.method == spec.name else spec.configure(cfg)
            tc = replace(tc, algorithm=spec.algorithm or "ppo")
            t_updates = {
                k: getattr(args, a)
                for a, k in [
                    ("steps", "total_timesteps"),
                    ("seed", "seed"),
                    ("n_envs", "n_envs"),
                    ("encoder", "encoder"),
                    ("device", "device"),
                ]
                if getattr(args, a) is not None
            }
            if args.no_curriculum:
                t_updates["curriculum"] = False
            if args.no_tensorboard:
                t_updates["tensorboard"] = False
            tc = replace(tc, **t_updates).validate()
            _json(train(cfg, tc, args.output, args.resume))
        elif args.command == "eval":
            from .evaluation import run_episode
            from .scenario import load_tape, scenario_config

            if args.scenario:
                cfg = scenario_config(args.scenario, cfg)
            policy = None
            if args.checkpoint:
                from .rl import load_policy

                policy = load_policy(args.checkpoint)
            _json(
                run_episode(
                    cfg,
                    args.method,
                    args.seed,
                    policy,
                    args.output,
                    load_tape(args.tape) if args.tape else None,
                    frozen_backend_control=args.frozen_backend_control,
                )
            )
        elif args.command == "benchmark":
            from .evaluation import benchmark

            checkpoints = {}
            for item in args.checkpoint:
                method, path = item.split("=", 1)
                checkpoints.setdefault(method, []).append(path)
            result = benchmark(
                cfg,
                _csv_str(args.methods),
                _csv_int(args.seeds),
                _csv_str(args.scenarios),
                args.output,
                checkpoints,
            )
            _json(
                {
                    "episodes": len(result["episodes"]),
                    "manifest": result["manifest"],
                    "summary": result["summary"],
                }
            )
        elif args.command == "gui":
            from .ui import run_ui

            policy = None
            if args.checkpoint:
                from .rl import load_policy

                policy = load_policy(args.checkpoint)
            _json(
                run_ui(
                    cfg,
                    args.method,
                    args.seed,
                    policy,
                    output_dir=args.output,
                    max_frames=args.max_frames,
                    screenshot_path=args.screenshot,
                )
            )
        elif args.command == "profile":
            from .env import FleetEnv
            from .evaluation import write_json

            if args.decisions < 1:
                raise ValueError("decisions must be positive")
            env = FleetEnv(cfg.copy(controller="fixed"))
            env.reset(seed=1)
            begin = time.perf_counter()
            lat = []
            for i in range(args.decisions):
                _, _, done, _, info = env.step(cfg.fixed_profile)
                lat.append(info["decision"]["total_ms"])
                if done and i + 1 < args.decisions:
                    env.reset(seed=1 + i)
            import numpy as np

            elapsed = time.perf_counter() - begin
            result = {
                "decisions": args.decisions,
                "elapsed_s": elapsed,
                "decisions_per_second": args.decisions / elapsed,
                "mean_online_ms": float(np.mean(lat)),
                "p95_online_ms": float(np.percentile(lat, 95)),
                "runtime": doctor(),
            }
            write_json(args.output, result)
            _json(result)
            env.close()
        elif args.command == "tune":
            _json(
                tune_baselines(
                    cfg,
                    _csv_int(args.seeds),
                    _csv_int(args.profiles),
                    [float(s) for s in _csv_str(args.thresholds)],
                    args.output,
                )
            )
        elif args.command == "scenario":
            from .maps import make_map, save_map
            from .scenario import generate_tape, save_tape, scenario_config

            directory = Path(args.output)
            directory.mkdir(parents=True, exist_ok=True)
            cfg = scenario_config(args.name, cfg)
            warehouse = make_map(cfg)
            tape = generate_tape(cfg, warehouse, args.seed)
            save_map(directory / "map.json", warehouse)
            save_tape(directory / "tape.json", tape)
            save_config(directory / "config.yaml", cfg)
            _json({"output": str(directory), "tape_hash": tape.tape_hash})
        return 0
    except KeyboardInterrupt:
        print(
            "Interrupted. Check run manifest/interrupted checkpoint before resuming.",
            file=sys.stderr,
        )
        return 130
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
