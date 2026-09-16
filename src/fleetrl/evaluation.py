"""Paired, tape-controlled evaluation, auditable run artifacts and replay logs.

Benchmark units are scenario/tape seeds and independent policy checkpoints,
never simulator ticks. Missing and right-censored values remain visible.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import sqlite3
import subprocess
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import FleetConfig
from .methods.registry import METHODS as REGISTRY
from .methods.registry import get_method
from .metrics import _finite as _finite
from .metrics import _metric_keys, paired_summary
from .metrics import _paired_ci as _paired_ci
from .types import EventTape, Snapshot

METHODS = {
    "B0": "heuristic",
    "B1": "fixed",
    "H": "hybrid",
    "A": "ablation",
    "heuristic": "heuristic",
    "fixed": "fixed",
    "hybrid": "hybrid",
    "ablation": "ablation",
}

METHODS.update({name: name for name in REGISTRY})
METHODS.update({spec.variant_id: spec.name for spec in REGISTRY.values()})
SCHEMA_VERSION = 2


def serializable(value: Any) -> Any:
    """Convert dataclasses/NumPy/sets to strict JSON; non-finite metrics are null."""
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [serializable(v) for v in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, value: Any) -> None:
    """Write strict UTF-8 JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(serializable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    from .artifacts import atomic_replace

    atomic_replace(temporary, path)


def method_config(config: FleetConfig, method: str) -> FleetConfig:
    """Apply the proposal's baseline/full/charging-ablation definitions."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose {', '.join(METHODS)}")
    spec = get_method(method)
    result = spec.configure(config)
    # Explicit representation/action controls retain their requested schema.
    if config.method == spec.name:
        result = result.copy(
            action_mode=config.action_mode, observation_mode=config.observation_mode
        )
    return result


def choose_action(
    policy: Any, observation: Any, config: FleetConfig
) -> tuple[int, str | None, float]:
    """Deterministic policy inference with explicit conservative failure records."""
    if config.controller != "hybrid":
        return config.fixed_profile, None, 0.0
    started = time.perf_counter()
    try:
        prediction = policy.predict(observation, deterministic=True)
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        scalar = np.asarray(action)
        if not np.isfinite(scalar).all():
            raise ValueError("nonfinite policy action")
        if config.action_mode == "continuous":
            if scalar.shape != (5,) or (np.abs(scalar) > 1).any():
                raise ValueError("invalid continuous action")
        elif config.action_mode == "multiagent":
            if (
                scalar.shape != (20,)
                or not np.equal(scalar, np.floor(scalar)).all()
                or (scalar < 0).any()
                or (scalar >= 64).any()
            ):
                raise ValueError("invalid multi-agent action")
        elif (
            scalar.size != 1
            or float(scalar.item()) != int(scalar.item())
            or not 0 <= int(scalar.item()) < 6
        ):
            raise ValueError("policy returned invalid action")
        elapsed_ms = (time.perf_counter() - started) * 1000
        if elapsed_ms > config.decision_budget_s * 1000:
            return config.fixed_profile, "policy_inference_budget_exceeded", elapsed_ms
        return int(scalar.item()) if config.action_mode == "discrete" else scalar, None, elapsed_ms
    except Exception as exc:
        return (
            config.fixed_profile,
            f"policy_error:{type(exc).__name__}:{str(exc)[:160]}",
            (time.perf_counter() - started) * 1000,
        )


def account_policy_latency(
    env: Any, info: dict, inference_ms: float, failure: str | None = None
) -> None:
    """Include policy inference in online decision latency and fallback statistics.

    Call exactly once after env.step(); terminal metrics must then be recomputed.
    """
    records = getattr(env, "decision_logs", [])
    record = records[-1] if records else info.get("decision", {})
    already = record.get("latency_accounted", False) and record.get("policy_inference_ms", 0) > 0
    if not already:
        record["policy_inference_ms"] = float(inference_ms)
        record["total_ms"] = float(record.get("total_ms", 0)) + float(inference_ms)
    record["budget_exceeded"] = record["total_ms"] > env.config.decision_budget_s * 1000
    if failure and failure not in (record.get("fallback_reason") or ""):
        previous = record.get("fallback_reason")
        record["fallback_reason"] = f"{failure};{previous}" if previous else failure
    info["decision"] = record


def file_sha256(path: str | Path) -> str:
    """Hash a file incrementally, including large checkpoints."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_hash() -> str:
    """Identify unpacked Python sources even when no git metadata is present."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            serializable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def snapshot_record(
    snapshot: Snapshot, metrics: dict | None = None, events: list | None = None
) -> dict:
    """Serialize a replay frame without exposing future tape events to a policy."""
    return {
        "type": "frame",
        "time_s": snapshot.sim_time_s,
        "state_hash": snapshot.state_hash,
        "robots": serializable(snapshot.robots),
        "tasks": serializable(snapshot.tasks),
        "ports": serializable(snapshot.ports),
        "bookings": serializable(snapshot.bookings),
        "blocked_cells": serializable(snapshot.blocked_cells),
        "metrics": serializable(metrics or {}),
        "events": serializable(events or []),
    }


def runtime_manifest() -> dict:
    """Record versions and source revision if the unpacked tree is a git checkout."""
    packages = {}
    for name in (
        "numpy",
        "gymnasium",
        "torch",
        "stable-baselines3",
        "sb3-contrib",
        "ortools",
        "pygame",
        "psutil",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        commit = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit,
        "source_tree_sha256": source_tree_hash(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


class RunRecorder:
    """Append decisions and compressed snapshots; index complete runs in SQLite."""

    def __init__(
        self,
        output_dir: str | Path,
        config: FleetConfig,
        tape: EventTape,
        method: str,
        seed: int,
        warehouse: Any,
        checkpoint: str | None = None,
    ):
        self.path = Path(output_dir)
        if self.path.exists() and any(self.path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing run: {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        self.replay = gzip.open(self.path / "replay.jsonl.gz", "wt", encoding="utf-8")
        self.decisions = (self.path / "decisions.jsonl").open("w", encoding="utf-8")
        self.closed = False
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "method": method,
            "seed": seed,
            "tape_hash": tape.tape_hash,
            "initial_tape_hash": tape.tape_hash,
            "config": config.to_dict(),
            "checkpoint": checkpoint,
            "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
            "config_hash": object_hash(config.to_dict()),
            "map_hash": object_hash(warehouse),
            "runtime": runtime_manifest(),
            "artifacts": {
                "tape": "tape.json",
                "replay": "replay.jsonl.gz",
                "decisions": "decisions.jsonl",
                "metrics": "metrics.json",
            },
        }
        write_json(self.path / "tape.json", tape.to_dict())
        self._line(
            self.replay,
            {
                "type": "header",
                "schema_version": SCHEMA_VERSION,
                "mode": "REPLAY",
                "map": serializable(warehouse),
                "manifest": self.manifest,
            },
        )

    @staticmethod
    def _line(handle, value: Any) -> None:
        handle.write(json.dumps(serializable(value), separators=(",", ":"), allow_nan=False) + "\n")

    def record(self, frame: dict, decision: dict | None = None) -> None:
        if self.closed:
            raise RuntimeError("Recorder is closed")
        if decision is not None:
            self._line(self.decisions, decision)
            frame = {**frame, "decision": decision}
        self._line(self.replay, frame)

    def close(self, metrics: dict | None = None, tape: EventTape | None = None) -> None:
        if self.closed:
            return
        self.replay.close()
        self.decisions.close()
        self.closed = True
        if tape is not None:
            write_json(self.path / "tape.json", tape.to_dict())
            self.manifest["tape_hash"] = tape.tape_hash
        self.manifest["status"] = (
            "interrupted"
            if metrics is None
            else metrics.get("status")
            if metrics.get("status") in {"failed", "partial", "interrupted"}
            else "partial"
            if metrics.get("episode_complete") is False
            else "complete"
        )
        if metrics is not None:
            write_json(self.path / "metrics.json", metrics)
        for name in ("tape.json", "replay.jsonl.gz", "decisions.jsonl", "metrics.json"):
            path = self.path / name
            if path.exists():
                self.manifest.setdefault("sha256", {})[name] = file_sha256(path)
        write_json(self.path / "manifest.json", self.manifest)
        with sqlite3.connect(self.path.parent / "runs.sqlite3") as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS runs (path TEXT PRIMARY KEY, method TEXT, seed INTEGER, tape_hash TEXT, status TEXT, metrics_json TEXT)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)",
                (
                    str(self.path.resolve()),
                    self.manifest["method"],
                    self.manifest["seed"],
                    self.manifest["tape_hash"],
                    self.manifest["status"],
                    json.dumps(serializable(metrics)),
                ),
            )


def load_replay(path: str | Path) -> tuple[dict, list[dict]]:
    """Read a recorded session; reject unknown schema and decreasing timestamps."""
    path = Path(path)
    if path.is_dir():
        path = path / "replay.jsonl.gz"
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        header = json.loads(next(handle))
        if header.get("type") != "header" or header.get("schema_version") not in {
            1,
            SCHEMA_VERSION,
        }:
            raise ValueError("Unsupported replay header/schema")
        frames = [json.loads(line) for line in handle if line.strip()]
    if any(f.get("type") != "frame" for f in frames):
        raise ValueError("Unexpected replay record")
    if any(b["time_s"] < a["time_s"] for a, b in zip(frames, frames[1:])):
        raise ValueError("Replay timestamps must be monotonic")
    if not frames:
        raise ValueError("Replay contains no frames")
    manifest_path = path.parent / "manifest.json"
    if manifest_path.exists():
        finalized = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = finalized.get("sha256", {}).get(path.name)
        if expected and file_sha256(path) != expected:
            raise ValueError("Replay content does not match its manifest checksum")
        header["manifest"] = finalized
    return header, frames


def run_episode(
    config: FleetConfig,
    method: str = "fixed",
    seed: int = 2000,
    policy: Any = None,
    output_dir: str | Path | None = None,
    tape: EventTape | None = None,
    *,
    frozen_backend_control: bool = False,
) -> dict:
    """Evaluate one complete episode, optionally writing its evidence bundle."""
    from .env import FleetEnv

    run_config = method_config(config, method)
    metadata = getattr(policy, "_fleetrl_metadata", {})
    if metadata:
        if metadata["method"] != get_method(method).name:
            allowed = frozen_backend_control and {metadata["method"], get_method(method).name} == {
                "ppo_cpsat",
                "ppo_milp",
            }
            if not allowed:
                raise ValueError(
                    "checkpoint method mismatch; use explicit frozen-policy backend control"
                )
        run_config = run_config.copy(
            observation_mode=metadata["observation_mode"],
            action_mode=metadata["action_mode"],
            charge_threshold=metadata["env"]["charge_threshold"],
        )
    if run_config.controller == "hybrid" and policy is None:
        raise ValueError("Hybrid/ablation evaluation requires a trained policy")
    env = FleetEnv(run_config)
    if policy is not None:
        from .learning.policies import InferenceSession

        policy = InferenceSession(policy)
    recorder = None
    complete = False
    started = time.perf_counter()
    total_reward = 0.0
    steps = failures = 0
    policy_times = []
    online_times = []
    try:
        observation, _ = env.reset(seed=seed, options={"tape": tape} if tape is not None else None)
        if output_dir is not None:
            recorder = RunRecorder(
                output_dir,
                run_config,
                env.sim.tape,
                METHODS[method],
                seed,
                env.sim.warehouse,
                getattr(policy, "_fleetrl_checkpoint_path", None),
            )
            recorder.record(snapshot_record(env.sim.snapshot(), env.metrics()))
        previous_event = getattr(env.sim, "_event_serial", len(env.sim.events))
        done = False
        while not done:
            action, failure, inference_ms = choose_action(policy, observation, run_config)
            policy_times.append(inference_ms)
            failures += int(failure is not None)
            if hasattr(env, "set_inference_context"):
                env.set_inference_context(inference_ms, failure)
            observation, reward, terminated, truncated, info = env.step(action)
            account_policy_latency(env, info, inference_ms, failure)
            online_times.append(float(info.get("decision", {}).get("total_ms", 0)))
            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)
            if recorder is not None:
                decision = {
                    "time_s": env.sim.time_s,
                    "step": steps,
                    "profile": action,
                    "policy_fallback": failure,
                    "inference_ms": inference_ms,
                    "reward": reward,
                    "reward_components": info.get("reward_components", {}),
                    "plan": info.get("decision", {}),
                }
                new_events = (
                    [event for event in env.sim.events if event.get("event_id", 0) > previous_event]
                    if hasattr(env.sim, "_event_serial")
                    else env.sim.events[previous_event:]
                )
                recorder.record(
                    snapshot_record(env.sim.snapshot(), env.metrics(), new_events), decision
                )
            previous_event = getattr(env.sim, "_event_serial", len(env.sim.events))
        from .metrics import episode_metrics

        metrics = episode_metrics(
            env.metrics(),
            total_reward,
            steps,
            failures,
            policy_times,
            online_times,
            run_config.decision_budget_s * 1000,
            time.perf_counter() - started,
        )
        result = {
            "method": METHODS[method],
            "variant_id": get_method(method).variant_id,
            "status": "complete" if terminated and not truncated else "partial",
            "seed": seed,
            "tape_hash": env.sim.tape.tape_hash,
            "map_name": run_config.map_name,
            "n_robots": run_config.n_robots,
            "horizon_s": run_config.horizon_s,
            "metrics": serializable(metrics),
        }
        if frozen_backend_control:
            result.update(
                comparison_role="frozen_backend_control",
                training_method=metadata.get("method", "legacy_ppo"),
                evaluation_backend=run_config.backend,
            )
        complete = True
        if recorder is not None:
            recorder.close(result, env.sim.tape)
        return result
    finally:
        if recorder is not None and not complete:
            recorder.close({"status": "partial", "metrics": env.metrics()})
        env.close()


def _checkpoint_seed(path: Path) -> int | str:
    sidecar = Path(str(path) + ".json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))["training_seed"]
    for file in (
        path.parent / "manifest.json",
        path.parent / "training_manifest.json",
        path.parent / "run_manifest.json",
    ):
        if file.exists():
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                for candidate in (
                    data.get("training", {}),
                    data.get("train_config", {}),
                    data.get("train", {}),
                    data,
                ):
                    if isinstance(candidate, dict) and "seed" in candidate:
                        return int(candidate["seed"])
            except (ValueError, TypeError, OSError):
                pass
    # Identity remains distinct; never invent a training seed.
    return str(path.resolve())


def benchmark(
    config: FleetConfig,
    methods: Sequence[str],
    seeds: Iterable[int],
    scenarios: Sequence[str],
    output_dir: str | Path,
    checkpoints: Mapping[str, str | Path | Sequence[str | Path]] | None = None,
) -> dict:
    """Run methods on the exact same immutable tape and seed within each case.

    For full proposal evaluation supply 3 H checkpoints, 3 independently trained
    A checkpoints, scenarios S1..S9, and seeds 2000..2004 (360 episodes).
    """
    from .maps import make_map
    from .scenario import generate_tape, scenario_config

    methods = [METHODS.get(method, method) for method in methods]
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Methods must be nonempty and unique")
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Tape seeds must be nonempty and unique")
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Scenarios must be nonempty and unique")
    for method in methods:
        method_config(config, method)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing benchmark: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = {}
    checkpoints = {METHODS.get(k, k): v for k, v in (checkpoints or {}).items()}
    for method in methods:
        if get_method(method).learnable:
            paths = checkpoints.get(method)
            if paths is None:
                raise ValueError(
                    f"Provide checkpoint(s) for {method}; no untrained policy comparisons"
                )
            paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
            if not paths:
                raise ValueError(f"No checkpoints for {method}")
            from .rl import load_policy

            policies[method] = [
                (str(path), _checkpoint_seed(Path(path)), load_policy(path, device="cpu"))
                for path in paths
            ]
            identities = [entry[1] for entry in policies[method]]
            if len(identities) != len(set(identities)):
                raise ValueError(f"Duplicate training seed/checkpoint identity for {method}")
            hashes = [file_sha256(entry[0]) for entry in policies[method]]
            if len(hashes) != len(set(hashes)):
                raise ValueError("duplicate checkpoint contents are not independent replicas")
        else:
            policies[method] = [(None, None, None)]
    records = []
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config": config.to_dict(),
        "methods": methods,
        "seeds": seeds,
        "scenarios": list(scenarios),
        "checkpoints": serializable(checkpoints),
        "runtime": runtime_manifest(),
        "config_hash": object_hash(config.to_dict()),
        "checkpoint_sha256": {
            method: [
                {
                    "path": path,
                    "sha256": file_sha256(getattr(policy, "_fleetrl_checkpoint_path", path)),
                }
                for path, _, policy in group
                if path is not None
            ]
            for method, group in policies.items()
        },
        "status": "running",
        "tapes": [],
    }
    write_json(output_dir / "manifest.json", manifest)
    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for scenario in scenarios:
            case = scenario_config(scenario, config)
            for seed in seeds:
                tape = generate_tape(case, make_map(case), seed)
                original_hash = tape.tape_hash
                manifest["tapes"].append(
                    {"scenario": scenario, "seed": seed, "hash": original_hash}
                )
                for method in methods:
                    for index, (checkpoint, training_seed, policy) in enumerate(policies[method]):
                        run_path = output_dir / str(scenario) / f"seed_{seed}" / f"{method}_{index}"
                        # Reconstruct so a defective controller cannot contaminate another run.
                        result = run_episode(
                            case,
                            method,
                            seed,
                            policy,
                            run_path,
                            EventTape.from_dict(tape.to_dict()),
                        )
                        if result["tape_hash"] != original_hash:
                            raise RuntimeError(
                                f"Tape mutated during benchmark: {scenario}/{seed}/{method}"
                            )
                        result.update(
                            {
                                "scenario": scenario,
                                "checkpoint": checkpoint,
                                "training_seed": training_seed,
                                "run_dir": str(run_path),
                            }
                        )
                        records.append(result)
                        stream.write(json.dumps(serializable(result), allow_nan=False) + "\n")
                        stream.flush()
    keys = _metric_keys(records)
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["scenario", "method", "seed", "training_seed", "tape_hash", "checkpoint", *keys]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{key: record.get(key) for key in fields[:6]},
                    **{key: record["metrics"].get(key) for key in keys},
                }
            )
    summary = paired_summary(records)
    write_json(output_dir / "summary.json", summary)
    plot_benchmark(summary, output_dir / "benchmark.png")
    manifest["status"] = "complete"
    manifest["episode_count"] = len(records)
    write_json(output_dir / "manifest.json", manifest)
    return {"episodes": records, "summary": summary, "manifest": str(output_dir / "manifest.json")}


def plot_benchmark(summary: dict, path: str | Path) -> None:
    """Plot mean episode KPIs by scenario; no hidden cross-scenario averaging."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = summary["groups"]
    scenarios = sorted({g["scenario"] for g in groups})
    methods = sorted({g["method"] for g in groups})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        ("throughput_per_hour", "Completed tasks / simulated hour"),
        ("total_lateness_s", "Total lateness incl. pending (s)"),
        ("energy_per_completed_wh", "Consumed Wh / completed task"),
        ("decision_p95_ms", "Online decision p95 incl. inference (ms)"),
    ]
    lookup = {(g["scenario"], g["method"]): g["statistics"] for g in groups}
    x = np.arange(len(scenarios))
    for ax, (metric, title) in zip(axes.flat, panels):
        for method in methods:
            entries = [lookup.get((scenario, method), {}).get(metric, {}) for scenario in scenarios]
            mean = np.array(
                [
                    entry.get("mean") if entry.get("mean") is not None else np.nan
                    for entry in entries
                ]
            )
            std = np.array(
                [entry.get("std") if entry.get("std") is not None else 0.0 for entry in entries]
            )
            ax.plot(x, mean, marker="o", label=method)
            ax.fill_between(x, mean - std, mean + std, alpha=0.12)
        ax.set_xticks(x, scenarios)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "FleetRL benchmark | mean ± episode SD; consult paired CIs in summary.json", fontsize=12
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
