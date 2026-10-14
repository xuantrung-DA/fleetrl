"""Aggregate complete paired episodes without discarding failed or missing jobs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

METRICS = (
    "pending",
    "completion_rate",
    "throughput_per_hour",
    "total_lateness_s",
    "consumed_wh",
    "empty_distance_ratio",
    "safety_incidents",
    "lost_tasks",
)
DIAGNOSTICS = (
    "mean_assignment_delay_s",
    "mean_assignment_delay_censored_s",
    "mean_pickup_distance_m",
    "excluded_reposition_robot_cycles",
    "charge_starts",
    "mean_soc_at_charge_start",
    "parking_goal_assignments",
)


def load_results(root: Path) -> list[dict[str, Any]]:
    results = []
    for stage in root.iterdir():
        status = stage / "status.json"
        if not stage.is_dir() or not status.exists():
            continue
        data = json.loads(status.read_text(encoding="utf-8"))
        for job in data["jobs"].values():
            if job["status"] == "complete":
                recorded = Path(job["artifact"])
                first = stage / recorded.name.split("_attempt_")[0]
                chosen = first if (first / "summary.json").exists() else recorded
                result = json.loads((chosen / "summary.json").read_text(encoding="utf-8"))
                result["artifact"] = str(chosen)
                results.append(result)
    return results


def aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(result["variant"], result["scenario"], result["method"])].append(result)
    rows = []
    for (variant, scenario, method), group in sorted(groups.items()):
        if len({r["seed"] for r in group}) != len(group):
            raise ValueError("duplicate seeds in aggregate")
        row: dict[str, Any] = {
            "variant": variant,
            "scenario": scenario,
            "method": method,
            "n": len(group),
            "seeds": ",".join(str(r["seed"]) for r in group),
        }
        for section, keys in (("metrics", METRICS), ("diagnostics", DIAGNOSTICS)):
            for key in keys:
                values = [r[section][key] for r in group if r[section][key] is not None]
                row[f"{key}_mean"] = mean(values) if values else None
                row[f"{key}_sd"] = stdev(values) if len(values) > 1 else None
                row[f"{key}_n"] = len(values)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(results: list[dict[str, Any]], reference: str) -> list[dict[str, Any]]:
    old = {(r["scenario"], r["method"], r["seed"]): r for r in results if r["variant"] == reference}
    rows = []
    for result in results:
        if result["variant"] == reference:
            continue
        key = result["scenario"], result["method"], result["seed"]
        baseline = old.get(key)
        if baseline is None:
            continue
        if (
            result["tape_hash"] != baseline["tape_hash"]
            or result["checkpoint_sha256"] != baseline["checkpoint_sha256"]
        ):
            raise ValueError(f"unpaired result {key}")
        if result["solver_control"] != baseline["solver_control"]:
            raise ValueError(f"unpaired solver control {key}")
        row = {
            "reference": reference,
            "variant": result["variant"],
            "scenario": key[0],
            "method": key[1],
            "seed": key[2],
            "tape_hash": result["tape_hash"],
            "reference_source_hash": baseline["source_hash"],
            "source_hash": result["source_hash"],
        }
        for section, keys in (("metrics", METRICS), ("diagnostics", DIAGNOSTICS)):
            for metric in keys:
                a, b = baseline[section][metric], result[section][metric]
                row[f"{metric}_before"], row[f"{metric}_after"] = a, b
                row[f"{metric}_delta"] = b - a if a is not None and b is not None else None
        rows.append(row)
    return rows


def build_summary(root: Path) -> dict[str, Any]:
    results = load_results(root)
    rows = aggregate(results)
    write_csv(root / "aggregate.csv", rows)
    write_csv(root / "paired_vs_baseline.csv", paired_rows(results, "baseline"))
    write_csv(root / "ablation_vs_parking_fix.csv", paired_rows(results, "parking_fix"))
    summary = {"complete_episodes": len(results), "groups": rows}
    (root / "aggregate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
