"""Additional real integrations beyond the per-method ready-check.

Run after ready-check, passing its output directory. Executed as a script to
exercise Windows spawn, TensorBoard, controls, replay and longer fleet episodes.
"""

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    import torch

    torch.set_num_threads(1)
    from fleetrl.config import FleetConfig, TrainConfig
    from fleetrl.evaluation import load_replay, run_episode, source_tree_hash, write_json
    from fleetrl.learning.training import load_checkpoint, train_method
    from fleetrl.methods.registry import get_method
    from fleetrl.ui import run_ui

    base = FleetConfig(n_robots=5, horizon_s=30, initial_tasks=3, disturbances=False)
    tc = TrainConfig(
        total_timesteps=32,
        n_envs=2,
        n_steps=8,
        batch_size=8,
        n_epochs=2,
        eval_freq=32,
        checkpoint_freq=32,
        eval_seeds=(1000,),
        curriculum=False,
        tensorboard=True,
    )
    checks = []
    started = time.perf_counter()

    def mark(name, detail):
        checks.append({"name": name, "passed": True, "detail": detail})
        print("[PASS] " + name, flush=True)

    result = train_method(
        get_method("recurrent_ppo_cpsat").configure(base),
        replace(tc, algorithm="recurrent_ppo"),
        out / "spawn_lstm",
    )
    assert result["additional_training_updates"] > 0
    assert list((out / "spawn_lstm" / "logs").glob("events.out.tfevents.*"))
    mark(
        "2 Windows spawn workers, recurrent rollout, TensorBoard", result["actual_total_timesteps"]
    )
    for name, observation, action in [
        ("ppo_cpsat", "compact", "discrete"),
        ("dqn_cpsat", "compact", "discrete"),
        ("ppo_cpsat", "entities", "continuous"),
    ]:
        cfg = (
            get_method(name).configure(base).copy(observation_mode=observation, action_mode=action)
        )
        opts = replace(
            tc,
            n_envs=1,
            encoder="mlp" if observation == "compact" else "entities",
            algorithm=get_method(name).algorithm,
            buffer_size=256,
            learning_starts=4,
            train_freq=4,
            tensorboard=False,
        )
        control = train_method(cfg, opts, out / f"{name}_{observation}_{action}")
        assert control["additional_training_updates"] > 0
        mark(f"control {name}/{observation}/{action}", control["actual_total_timesteps"])
    for n in (10, 15, 20):
        metrics = run_episode(
            FleetConfig(n_robots=n, horizon_s=600, demand_per_hour=n * 6, disturbances=True),
            "fixed_cpsat",
            2003,
            output_dir=out / f"fleet_{n}",
        )["metrics"]
        assert metrics["safety_incidents"] == 0, metrics
        assert metrics["arrived"] == metrics["completed"] + metrics["pending"]
        assert abs(metrics["battery_balance_error_wh"]) < 1e-5
        mark(f"{n} robots, 600 simulated seconds, disturbances and invariant audit", metrics)
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    policy = load_checkpoint(Path(args.checkpoints) / "recurrent_ppo_cpsat/train/final_model.zip")
    run_ui(
        base,
        method="recurrent_ppo_cpsat",
        policy=policy,
        max_frames=2,
        screenshot_path=out / "live.png",
    )
    replay = Path(args.checkpoints) / "recurrent_ppo_cpsat/eval"
    header, frames = load_replay(replay)
    assert frames
    run_ui(replay_path=replay, max_frames=2, screenshot_path=out / "replay.png")
    mark("headless live/replay screenshots and artifact checksum", {"frames": len(frames)})
    result = {
        "status": "passed",
        "source_hash": source_tree_hash(),
        "wall_s": time.perf_counter() - started,
        "checks": checks,
    }
    write_json(out / "integration_report.json", result)


if __name__ == "__main__":
    main()
