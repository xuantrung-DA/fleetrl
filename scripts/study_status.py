"""Display progress for the full 14-method FleetRL study."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fleetrl.experiments.protocol import load_study  # noqa: E402
from fleetrl.methods.registry import get_method  # noqa: E402

STUDY = load_study(ROOT / "configs" / "study.yaml")
OUTPUT = ROOT / "runs" / "study14"
JOBS = OUTPUT / "jobs"
TUNE_TRIALS = {"heuristic": 3, "fixed_cpsat": 6, "forecast_mpc_milp": 6}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def job_state(key: str) -> dict[str, Any]:
    return read_json(JOBS / key / "job.json") or {}


def logged_steps(output: Path, state: dict[str, Any]) -> int:
    if state.get("status") == "complete":
        return int(state.get("result", {}).get("actual_total_timesteps", 0))
    steps = 0
    progress = output / "logs" / "progress.csv"
    if progress.is_file():
        try:
            with progress.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    value = row.get("time/total_timesteps")
                    if value:
                        steps = max(steps, int(float(value)))
        except (OSError, UnicodeError, ValueError):
            pass
    for name in ("latest_model.zip.json", "interrupted_model.zip.json", "final_model.zip.json"):
        metadata = read_json(output / name)
        if metadata:
            steps = max(steps, int(metadata.get("num_timesteps", 0)))
    return steps


def train_steps(state: dict[str, Any]) -> int:
    output = state.get("output")
    if not output:
        return 0
    return logged_steps(Path(str(output)), state)


def related_jobs(name: str, prefix: str) -> list[tuple[str, dict[str, Any]]]:
    if prefix == "train":
        keys = [f"train_{name}_{seed}" for seed in STUDY.training_seeds]
    elif prefix == "search":
        keys = [f"search_{name}_{trial}" for trial in range(len(STUDY.search_trials))]
    elif prefix == "tune":
        keys = [f"tune_{name}_{trial}" for trial in range(TUNE_TRIALS[name])]
    else:
        keys = []
    return [(key, job_state(key)) for key in keys]


def method_row(name: str) -> tuple[str, str, str, str, str, str]:
    spec = get_method(name)
    prep_kind = "search" if spec.learnable else "tune"
    prep_jobs = related_jobs(name, prep_kind)
    train_jobs = related_jobs(name, "train") if spec.learnable else []
    prep_done = sum(state.get("status") == "complete" for _, state in prep_jobs)
    train_done = sum(state.get("status") == "complete" for _, state in train_jobs)
    replicas = len(STUDY.training_seeds) if spec.learnable else 1
    test_expected = replicas * len(STUDY.scenarios_with_controls()) * len(STUDY.tape_seeds)
    test_states = [
        read_json(path) or {} for path in JOBS.glob(f"test_{name}_*/job.json")
    ]
    test_done = sum(state.get("status") == "complete" for state in test_states)
    preparation_states = [state for _, state in prep_jobs + train_jobs]

    active = next(
        ((key, state) for key, state in prep_jobs + train_jobs if state.get("status") == "running"),
        None,
    )
    if active is None:
        active = next(
            (
                (path.parent.name, state)
                for path in JOBS.glob(f"test_{name}_*/job.json")
                if (state := read_json(path)) and state.get("status") == "running"
            ),
            None,
        )
    detail = "-"
    if active:
        key, state = active
        if key.startswith(("train_", "search_")):
            steps = train_steps(state)
            target = STUDY.total_timesteps if key.startswith("train_") else STUDY.search_steps
            detail = f"{key.split('_')[-1]}: {steps:,}/{target:,}"
        elif key.startswith("tune_"):
            detail = f"trial {int(key.split('_')[-1]) + 1}/{len(prep_jobs)}"
        else:
            detail = key.removeprefix(f"test_{name}_")[-24:]
        status = "TRAIN" if key.startswith("train_") else prep_kind.upper()
        if key.startswith("test_"):
            status = "TEST"
    elif any(
        state.get("status") in {"failed", "interrupted"}
        for state in preparation_states + test_states
    ):
        status = "FAILED"
    elif test_done == test_expected:
        status = "DONE"
    elif train_done == len(train_jobs) and spec.learnable:
        status = "TRAINED"
    elif prep_done == len(prep_jobs):
        status = "READY"
    else:
        status = "WAIT"
    return (
        spec.variant_id,
        name,
        f"{prep_done}/{len(prep_jobs)}",
        f"{train_done}/{len(train_jobs)}" if spec.learnable else "-",
        f"{test_done}/{test_expected}",
        f"{status} {detail}" if detail != "-" else status,
    )


def show_status() -> None:
    rows = [method_row(name) for name in STUDY.methods]
    learners = [name for name in STUDY.methods if get_method(name).learnable]
    train_states = [
        job_state(f"train_{name}_{seed}")
        for name in learners
        for seed in STUDY.training_seeds
    ]
    train_done = sum(state.get("status") == "complete" for state in train_states)
    logged_total = sum(min(STUDY.total_timesteps, train_steps(state)) for state in train_states)
    plan = read_json(OUTPUT / "plan.json")
    expected_steps = len(train_states) * STUDY.total_timesteps
    print(f"\nFleetRL full study | {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(
        f"Main training: {train_done}/{len(train_states)} jobs | "
        f"logged {logged_total:,}/{expected_steps:,} steps"
    )
    if plan:
        print(
            f"Test episodes: {sum(int(row[4].split('/')[0]) for row in rows)}"
            f"/{plan['core_test_episodes'] + plan['matched_control_episodes']}"
        )
    print(
        f"{'ID':<4} {'Method':<23} {'Prep':>6} {'Train':>6} "
        f"{'Test':>8}  Status / latest logged step"
    )
    print("-" * 95)
    for variant, name, prep, train, test, status in rows:
        print(f"{variant:<4} {name:<23} {prep:>6} {train:>6} {test:>8}  {status}")
    summary = read_json(OUTPUT / "summary.json")
    if summary:
        print(f"Study report: {summary.get('status', 'unknown')}")
    print("Steps may lag until the next CSV update or checkpoint.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be positive")
    while True:
        show_status()
        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
