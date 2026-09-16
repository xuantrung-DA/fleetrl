"""Exercise tuning/search/train/test/report, partial jobs and frozen selections."""

import argparse
import json
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("audit output must be new/empty")
    out.mkdir(parents=True, exist_ok=True)
    import torch
    import yaml

    torch.set_num_threads(1)
    from fleetrl.config import load_config, save_config
    from fleetrl.evaluation import source_tree_hash, write_json
    from fleetrl.experiments.runner import run_study

    methods = ["fixed_cpsat", "dqn_cpsat", "ppo_threshold_cpsat"]
    configs = out / "method_configs"
    configs.mkdir()
    for name in methods:
        cfg, tc = load_config(Path("configs/methods") / (name + ".yaml"))
        tc = replace(
            tc,
            n_envs=1,
            n_steps=8,
            batch_size=4,
            n_epochs=2,
            buffer_size=64,
            learning_starts=4,
            train_freq=4,
            eval_freq=16,
            checkpoint_freq=16,
            curriculum=False,
            tensorboard=False,
        )
        save_config(configs / (name + ".yaml"), cfg.copy(n_robots=5, initial_tasks=2), tc)
    study = {
        "schema_version": 2,
        "study": {
            "methods": methods,
            "training_seeds": [11],
            "validation_seeds": [1000],
            "tape_seeds": [2000, 2001],
            "total_timesteps": 32,
            "horizon_s": 10,
            "scenarios": ["S1"],
            "matched_ood": True,
            "bootstrap_samples": 50,
            "search_steps": 16,
            "search_trials": [0.0003],
            "search_q_alphas": [0.1],
            "search_thresholds": [0.3],
            "config_dir": str(configs),
        },
    }
    path = out / "study.yaml"
    path.write_text(yaml.safe_dump(study), encoding="utf-8")
    report = {"source_hash": source_tree_hash(), "phases": []}
    for phase, limit in [
        ("tune", None),
        ("search", None),
        ("train", 1),
        ("train", None),
        ("test", 1),
        ("test", None),
        ("report", None),
        ("train", 0),
        ("test", 0),
    ]:
        result = run_study(path, out / "study", phase, limit)
        assert result.get("failed_jobs", 0) == 0, result
        assert result["status"] == ("partial" if limit == 1 else "complete"), result
        report["phases"].append({"phase": phase, "limit": limit, "result": result})
    assert report["phases"][-1]["result"]["complete_episodes"] == 18
    assert all(
        json.loads(p.read_text())["attempt"] == 1 for p in (out / "study/jobs").glob("*/job.json")
    )
    report["status"] = "passed"
    write_json(out / "study_audit_report.json", report)
    print(
        "PASS: all study phases, 2 trained policies, 18 episodes, resume without rerunning completed jobs",
        flush=True,
    )


if __name__ == "__main__":
    main()
