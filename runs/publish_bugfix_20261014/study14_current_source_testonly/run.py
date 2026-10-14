"""Evaluate old MAPPO/DQN checkpoints on the current source without training."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src"))

from fleetrl.config import FleetConfig
from fleetrl.evaluation import file_sha256, run_episode, source_tree_hash, write_json
from fleetrl.learning.training import load_checkpoint
from fleetrl.maps import make_map
from fleetrl.metrics import paired_summary
from fleetrl.scenario import generate_tape, scenario_config
from fleetrl.types import EventTape

EXPECTED_SOURCE = "8bb96d90d4c7ac92595fdb4805c33ce5ffb958736237611a42f5b59948319c95"
METHODS = ("heuristic", "fixed_cpsat", "mappo_dispatch", "dqn_cpsat")
SEEDS = (3005, 3006, 3007, 3008, 3009)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO:
        raise ValueError("Run from the repository root")
    if source_tree_hash() != EXPECTED_SOURCE:
        raise ValueError("Source changed; prepare a separate evaluation plan")
    checkpoints = {}
    for method in METHODS[2:]:
        for seed in (11, 12, 13):
            lock = REPO / "runs/study14" / f"frozen_checkpoint_{method}_{seed}.json"
            item = json.loads(lock.read_text(encoding="utf-8"))
            path = Path(item["path"])
            if file_sha256(path) != item["sha256"]:
                raise ValueError(f"Checkpoint changed: {path}")
            meta_path = Path(str(path) + ".json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["method"] != method:
                raise ValueError(f"Checkpoint method mismatch: {path}")
            checkpoints[method, seed] = item
    print("CHECK OK: current source, six Study14 checkpoints; 360 test-only cases.", flush=True)
    if args.check:
        return
    if (ROOT / "STOP_REQUIRED.json").exists():
        raise RuntimeError("A failed gate needs review; preserve STOP_REQUIRED.json")
    import torch
    torch.set_num_threads(1)
    lock = ROOT / "running.lock"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"started": time.time()}, handle)
    try:
        execute(checkpoints)
    finally:
        lock.unlink(missing_ok=True)


def execute(checkpoints: dict) -> None:
    selected = json.loads((REPO / "runs/study14/selected_baselines.json").read_text(encoding="utf-8"))["selected"]
    identity = dict(source_hash=EXPECTED_SOURCE, checkpoints={f"{m}_{s}": v for (m, s), v in checkpoints.items()},
                    selected_baselines=selected, scenarios=[f"S{i}" for i in range(1, 10)],
                    tape_seeds=SEEDS, horizon_s=3600, n_robots=15,
                    checkpoint_label="checkpoint train trên source cũ", training_or_tuning=False,
                    interpretation="Observed tapes reused for source transfer assessment, not a fresh holdout")
    identity_path = ROOT / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != json.loads(json.dumps(identity)):
        raise ValueError("Run identity changed")
    write_json(identity_path, identity)
    policies = {}
    records = []
    started = time.perf_counter()
    for scenario in identity["scenarios"]:
        base = scenario_config(scenario, FleetConfig(n_robots=15, horizon_s=3600))
        for seed in SEEDS:
            tape = generate_tape(base, make_map(base), seed)
            for method in METHODS:
                replicas = (11, 12, 13) if method in METHODS[2:] else (None,)
                for training_seed in replicas:
                    key = f"{scenario}_{seed}_{method}_{training_seed}"
                    folder = ROOT / "episodes" / key
                    result_path = folder / "result.json"
                    if result_path.exists():
                        record = json.loads(result_path.read_text(encoding="utf-8"))
                        if record["tape_hash"] != tape.tape_hash or record["source_hash"] != EXPECTED_SOURCE:
                            raise ValueError(f"Completed artifact identity mismatch: {key}")
                    else:
                        if folder.exists():
                            raise RuntimeError(f"Preserve and review incomplete episode: {folder}")
                        if time.perf_counter() - started > 5.5 * 3600:
                            print("TIME_LIMIT: paused before six hours; run the same command to resume.", flush=True)
                            return
                        if source_tree_hash() != EXPECTED_SOURCE:
                            raise ValueError("Source changed while evaluating")
                        cfg = base.copy(**selected.get(method, {}))
                        policy = None
                        checkpoint = None
                        if training_seed is not None:
                            pair = method, training_seed
                            if pair not in policies:
                                policies[pair] = load_checkpoint(checkpoints[pair]["path"], expected_method=method, device="cpu")
                            policy = policies[pair]
                            checkpoint = checkpoints[pair]["path"]
                        print(f"[{len(records)+1}/360] START {key}", flush=True)
                        record = run_episode(cfg, method, seed, policy, folder, EventTape.from_dict(tape.to_dict()))
                        record.update(scenario=scenario, training_seed=training_seed,
                                      checkpoint=checkpoint, source_hash=EXPECTED_SOURCE,
                                      checkpoint_label="checkpoint train trên source cũ" if checkpoint else "controller không học")
                        write_json(result_path, record)
                    records.append(record)
                    metrics = record["metrics"]
                    print(f"[{len(records)}/360] completed={metrics['completed']} pending={metrics['pending']} safety={metrics['safety_incidents']} lost={metrics['lost_tasks']}", flush=True)
                    write_json(ROOT / "status.json", dict(completed=len(records), planned=360, last_case=key, complete=False))
                    failures = []
                    if metrics["safety_incidents"]: failures.append("safety")
                    if metrics["lost_tasks"]: failures.append("lost_tasks")
                    if metrics["completion_rate"] < .5: failures.append("completion_below_50_percent")
                    if metrics["online_decision_p95_ms"] > 250: failures.append("p95_over_250_ms")
                    timeout_rate = metrics.get("solver_timeout_rate")
                    if timeout_rate is not None and timeout_rate >= .05:
                        failures.append("solver_timeout_ge_5_percent")
                    if failures:
                        write_json(ROOT / "STOP_REQUIRED.json", dict(case=key, failures=failures, artifact=str(folder)))
                        print(f"STOP_REQUIRED: {failures}; artifacts preserved, no source changes.", flush=True)
                        return
    fields = ["scenario", "method", "seed", "training_seed", "checkpoint_label", "checkpoint"]
    metric_names = sorted({name for r in records for name in r["metrics"] if isinstance(r["metrics"][name], (int, float)) or r["metrics"][name] is None})
    with (ROOT / "episodes.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + metric_names)
        writer.writeheader()
        for r in records:
            writer.writerow({**{k: r.get(k) for k in fields}, **{k: r["metrics"].get(k) for k in metric_names}})
    write_json(ROOT / "summary.json", paired_summary(records, reference="fixed_cpsat"))
    write_json(ROOT / "status.json", dict(completed=360, planned=360, complete=True))
    print("COMPLETE 360/360. Results: episodes.csv, summary.json. No training performed.", flush=True)


if __name__ == "__main__":
    main()
