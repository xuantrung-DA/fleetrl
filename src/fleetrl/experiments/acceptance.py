"""Real execution acceptance, including learning/save/load/resume for every learner."""

import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..config import FleetConfig, TrainConfig
from ..env import FleetEnv
from ..evaluation import run_episode, runtime_manifest, source_tree_hash, write_json
from ..learning.training import finite_parameters, load_checkpoint, train_method
from ..methods.registry import METHODS


def ready_check(output, steps=32, methods=None):
    import torch
    from stable_baselines3.common.env_checker import check_env

    torch.set_num_threads(1)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("acceptance output must be new/empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {
        "status": "running",
        "source_hash": source_tree_hash(),
        "runtime": runtime_manifest(),
        "methods": [],
    }
    base = FleetConfig(
        n_robots=5,
        horizon_s=30,
        initial_tasks=3,
        disturbances=False,
        service_s=2.0,
        solver_time_limit_s=0.05,
    )
    tc = TrainConfig(
        total_timesteps=max(32, steps),
        n_envs=1,
        n_steps=16,
        batch_size=8,
        n_epochs=2,
        eval_freq=max(32, steps),
        checkpoint_freq=max(32, steps),
        eval_seeds=(1000,),
        curriculum=False,
        tensorboard=False,
        buffer_size=256,
        learning_starts=4,
        train_freq=4,
        gradient_steps=1,
    )
    for name, spec in METHODS.items():
        if methods and name not in methods:
            continue
        row = {"method": name, "variant_id": spec.variant_id, "status": "running"}
        try:
            cfg = spec.configure(base)
            opts = replace(tc, algorithm=spec.algorithm or "ppo")
            env = FleetEnv(cfg)
            check_env(env, warn=False)
            env.close()
            policy = None
            if spec.learnable:
                trained = train_method(cfg, opts, output / name / "train")
                policy = load_checkpoint(trained["final_model"], expected_method=name)
                finite_parameters(policy)
                assert trained["initial_parameter_sha256"] != trained["final_parameter_sha256"], (
                    "parameters did not change"
                )
                assert trained["additional_training_updates"] > 0
                env = FleetEnv(cfg)
                obs, _ = env.reset(seed=2000)
                before = policy.predict(obs, deterministic=True)[0]
                again = load_checkpoint(trained["final_model"], expected_method=name)
                after = again.predict(obs, deterministic=True)[0]
                np.testing.assert_allclose(before, after, atol=1e-7)
                env.close()
                resumed = train_method(
                    cfg,
                    replace(opts, total_timesteps=16),
                    output / name / "resume",
                    trained["final_model"],
                )
                assert resumed["actual_total_timesteps"] > trained["actual_total_timesteps"]
                assert resumed["training_updates"] > trained["training_updates"]
                row.update(
                    training_steps=trained["actual_total_timesteps"],
                    updates=trained["training_updates"],
                    resumed_steps=resumed["actual_total_timesteps"],
                    checkpoint=trained["final_model"],
                )
            result = run_episode(cfg, name, 2000, policy, output / name / "eval")
            metrics = result["metrics"]
            assert metrics["arrived"] == metrics["completed"] + metrics["pending"]
            assert metrics["safety_incidents"] == 0, metrics["invariant_incidents"]
            assert abs(metrics["battery_balance_error_wh"]) < 1e-5
            row.update(status="passed", metrics=metrics)
            print(
                f"[PASS] {name}: "
                + ("update/save/load/resume/eval" if spec.learnable else "controller/eval"),
                flush=True,
            )
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            import traceback

            row["traceback"] = traceback.format_exc()
            print(f"[FAIL] {name}: {row['error']}", flush=True)
        report["methods"].append(row)
        write_json(output / "ready_report.json", report)
    report.update(
        status="passed" if all(r["status"] == "passed" for r in report["methods"]) else "failed",
        wall_s=time.perf_counter() - started,
        interpretation="Execution readiness only; full training and held-out comparison remain study jobs.",
    )
    write_json(output / "ready_report.json", report)
    return report
