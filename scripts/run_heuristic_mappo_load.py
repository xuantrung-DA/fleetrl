"""Run a paired S4 threshold search and held-out demand sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fleetrl.config import FleetConfig  # noqa: E402
from fleetrl.evaluation import file_sha256, run_episode, source_tree_hash  # noqa: E402
from fleetrl.learning.training import load_checkpoint  # noqa: E402
from fleetrl.maps import make_map  # noqa: E402
from fleetrl.scenario import generate_tape, scenario_config  # noqa: E402
from fleetrl.types import EventTape  # noqa: E402

OUTPUT = ROOT / "runs" / "heuristic_mappo_load_20261015"
SOURCE_HASH = "0dfa553da4ee08622977f67fb60b54ece07588056f98dcfbb1fcaca4882a187f"
TRAINING_SEEDS = (11, 12, 13)
TUNE_SEEDS = (1100, 1101, 1102, 1103, 1104)
TEST_SEEDS = (4100, 4101, 4102, 4103, 4104)
THRESHOLDS = (0.2, 0.3, 0.4)
FACTORS = (0.5, 1.0, 1.5, 2.0, 3.0)


def checkpoint_paths() -> dict[int, Path]:
    paths = {}
    for seed in TRAINING_SEEDS:
        lock = ROOT / "runs" / "study14" / f"frozen_checkpoint_mappo_dispatch_{seed}.json"
        saved = json.loads(lock.read_text(encoding="utf-8"))
        path = Path(saved["path"])
        if file_sha256(path) != saved["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
        paths[seed] = path
    return paths


def identity(paths: dict[int, Path]) -> dict:
    return {
        "source_hash": SOURCE_HASH,
        "checkpoint_sha256": {str(seed): file_sha256(path) for seed, path in paths.items()},
        "scenario": "S4",
        "tune_seeds": TUNE_SEEDS,
        "test_seeds": TEST_SEEDS,
        "thresholds": THRESHOLDS,
        "demand_factors": FACTORS,
        "horizon_s": 3600,
        "selection_rule": "maximize mean completion_rate, then minimize mean total_lateness_s",
        "source_note": "MAPPO checkpoints trained on earlier source; this tests transfer only",
    }


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def selected_threshold() -> float:
    from statistics import mean

    scores = []
    for threshold in THRESHOLDS:
        rows = []
        for seed in TUNE_SEEDS:
            path = OUTPUT / "tune" / f"heuristic_{threshold:.1f}_{seed}.json"
            if not path.exists():
                raise ValueError(f"Tune grid is incomplete: {path}")
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        scores.append(
            (
                -mean(row["metrics"]["completion_rate"] for row in rows),
                mean(row["metrics"]["total_lateness_s"] for row in rows),
                threshold,
            )
        )
    return min(scores)[2]


def cases(phase: str) -> list[tuple[str, float, int, float | None, int | None]]:
    if phase == "tune":
        return [
            ("S4", 1.0, seed, threshold, None) for seed in TUNE_SEEDS for threshold in THRESHOLDS
        ] + [
            ("S4", 1.0, seed, None, training_seed)
            for seed in TUNE_SEEDS
            for training_seed in TRAINING_SEEDS
        ]
    threshold = selected_threshold()
    if phase == "sweep":
        return [
            ("S4", factor, seed, charge_threshold, training_seed)
            for factor in FACTORS
            for seed in TEST_SEEDS
            for charge_threshold, training_seed in [(threshold, None)]
            + [(None, item) for item in TRAINING_SEEDS]
        ]
    if phase == "ood":
        return [
            ("OOD_A", 1.0, seed, charge_threshold, training_seed)
            for seed in TEST_SEEDS
            for charge_threshold, training_seed in [(threshold, None)]
            + [(None, item) for item in TRAINING_SEEDS]
        ]
    raise ValueError(f"Unknown phase: {phase}")


def run_phase(phase: str, paths: dict[int, Path]) -> None:
    import torch

    torch.set_num_threads(1)
    policies = {}
    planned = cases(phase)
    for index, (scenario, factor, seed, threshold, training_seed) in enumerate(planned, 1):
        if source_tree_hash() != SOURCE_HASH:
            raise ValueError("Source changed during evaluation")
        label = (
            f"heuristic_{threshold:.1f}_{seed}"
            if training_seed is None
            else f"mappo_{training_seed}_{seed}"
        )
        folder = OUTPUT / phase
        if phase == "sweep":
            folder = folder / f"factor_{factor:.1f}"
        path = folder / f"{label}.json"
        if scenario == "S4":
            base = scenario_config("S4", FleetConfig(horizon_s=3600))
            config = base.copy(demand_per_hour=base.demand_per_hour * factor)
        else:
            config = FleetConfig(horizon_s=3600).copy(
                n_robots=20,
                demand_per_hour=120,
                map_name="A",
                disturbances=False,
                burst_multiplier=1,
            )
        tape = generate_tape(config, make_map(config), seed)
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (
                saved["tape_hash"] != tape.tape_hash
                or saved["source_hash"] != SOURCE_HASH
                or saved["status"] != "complete"
            ):
                raise ValueError(f"Saved case identity/status mismatch: {path}")
            print(f"[{index}/{len(planned)}] reuse {path.name}", flush=True)
            continue
        method = "heuristic" if training_seed is None else "mappo_dispatch"
        if training_seed is None:
            config = config.copy(charge_threshold=threshold)
            policy = None
        else:
            if training_seed not in policies:
                policies[training_seed] = load_checkpoint(
                    paths[training_seed], expected_method="mappo_dispatch", device="cpu"
                )
            policy = policies[training_seed]
        print(f"[{index}/{len(planned)}] start {phase} {label} factor={factor}", flush=True)
        record = run_episode(
            config, method, seed, policy, None, EventTape.from_dict(tape.to_dict())
        )
        if record["status"] != "complete":
            raise RuntimeError(f"Incomplete episode: {phase}/{label}")
        record.update(
            scenario=scenario,
            demand_factor=factor,
            charge_threshold=threshold,
            training_seed=training_seed,
            checkpoint_sha256=file_sha256(paths[training_seed]) if training_seed else None,
            source_hash=SOURCE_HASH,
        )
        save_json(path, record)
        print(
            f"[{index}/{len(planned)}] done completion={record['metrics']['completion_rate']:.3f} "
            f"lateness={record['metrics']['total_lateness_s']:.1f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "tune", "sweep", "ood"))
    args = parser.parse_args()
    if source_tree_hash() != SOURCE_HASH:
        raise ValueError("Source hash differs from frozen experiment definition")
    paths = checkpoint_paths()
    current = identity(paths)
    manifest = OUTPUT / "identity.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != json.loads(json.dumps(current)):
            raise ValueError("Experiment identity changed")
    else:
        save_json(manifest, current)
    print(f"CHECK OK: source and 3 MAPPO checkpoints; phase={args.phase}", flush=True)
    if args.phase != "check":
        run_phase(args.phase, paths)


if __name__ == "__main__":
    main()
