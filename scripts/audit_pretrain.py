"""Longer paired execution checks; deliberately not a convergence benchmark."""

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("audit output must be new/empty")
    out.mkdir(parents=True, exist_ok=True)
    import torch

    torch.set_num_threads(1)
    from fleetrl.config import FleetConfig, TrainConfig
    from fleetrl.evaluation import run_episode, source_tree_hash, write_json
    from fleetrl.learning.training import load_checkpoint, train_method
    from fleetrl.maps import make_map
    from fleetrl.methods.registry import METHODS, get_method
    from fleetrl.scenario import generate_tape

    started = time.perf_counter()
    report = {"status": "running", "source_hash": source_tree_hash(), "episodes": [], "checks": []}
    cases = {
        "nominal10": FleetConfig(
            n_robots=10, horizon_s=600, demand_per_hour=60, disturbances=False
        ),
        "stress20": FleetConfig(
            n_robots=20,
            horizon_s=600,
            demand_per_hour=120,
            map_name="B",
            initial_soc_max=0.45,
            heavy_fraction=0.5,
            disturbances=True,
            burst_start_s=150,
            burst_end_s=450,
        ),
    }
    tapes = {
        (case, seed): generate_tape(cfg, make_map(cfg), seed)
        for case, cfg in cases.items()
        for seed in (2000, 2001)
    }
    for name, spec in METHODS.items():
        policy = (
            load_checkpoint(Path(args.checkpoints) / name / "train/final_model.zip")
            if spec.learnable
            else None
        )
        for case, cfg in cases.items():
            for seed in (2000, 2001):
                row = {"method": name, "case": case, "seed": seed}
                result = None
                try:
                    result = run_episode(
                        cfg, name, seed, policy, out / name / case / str(seed), tapes[case, seed]
                    )
                    metrics = result["metrics"]
                    assert result["status"] == "complete"
                    assert metrics["safety_incidents"] == 0, metrics["invariant_incidents"]
                    assert metrics["arrived"] == metrics["completed"] + metrics["pending"]
                    assert abs(metrics["battery_balance_error_wh"]) < 1e-5
                    assert metrics["policy_failures"] == 0
                    assert all(
                        math.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))
                    )
                    assert result["tape_hash"] == tapes[case, seed].tape_hash
                    row.update(status="passed", result=result)
                    print(
                        f"[PASS] {name}/{case}/{seed}: completed={metrics['completed']}/{metrics['arrived']}, "
                        f"fallback={metrics['fallback_rate']:.3f}, p95={metrics['decision_p95_ms']:.1f}ms",
                        flush=True,
                    )
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    if result is not None:
                        row["diagnostic_result"] = result
                    print(f"[FAIL] {name}/{case}/{seed}: {row['error']}", flush=True)
                report["episodes"].append(row)
                write_json(out / "audit_report.json", report)

    # Real off-policy curriculum with multiple stages, global epsilon scheduling,
    # and continuation after the planned budget. Separate from paired evaluation.
    cfg = get_method("dqn_cpsat").configure(
        FleetConfig(n_robots=10, horizon_s=10, initial_tasks=2, disturbances=False)
    )
    tc = TrainConfig(
        algorithm="dqn",
        total_timesteps=64,
        n_envs=1,
        n_steps=4,
        batch_size=4,
        learning_starts=4,
        buffer_size=128,
        train_freq=4,
        eval_freq=16,
        checkpoint_freq=16,
        eval_seeds=(1000,),
        tensorboard=False,
        curriculum=True,
        exploration_fraction=1,
    )
    trained = train_method(cfg, tc, out / "curriculum")
    assert [s["name"] for s in trained["stages"]] == ["warmup", "intermediate", "target"]
    policy = load_checkpoint(trained["final_model"])
    assert abs(policy.exploration_rate - tc.exploration_final_eps) < 1e-8
    resumed = train_method(
        cfg, replace(tc, total_timesteps=16), out / "curriculum_resume", trained["final_model"]
    )
    assert resumed["actual_additional_timesteps"] == 16
    assert all(s["name"] == "target" for s in resumed["stages"])
    assert load_checkpoint(resumed["final_model"]).exploration_rate == tc.exploration_final_eps
    report["checks"].append(
        {
            "name": "DQN actual curriculum and exploration-preserving resume",
            "status": "passed",
            "training": trained,
            "resumed": resumed,
        }
    )
    report.update(
        status="passed" if all(r["status"] == "passed" for r in report["episodes"]) else "failed",
        wall_s=time.perf_counter() - started,
        interpretation="56 paired execution episodes after 128-step smoke training; not evidence of convergence or superiority.",
    )
    write_json(out / "audit_report.json", report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
