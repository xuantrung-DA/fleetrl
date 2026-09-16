"""Artifact formatting and plots; all numerical reductions live in metrics.py."""

import csv
import json
from pathlib import Path

from ..metrics import METRIC_GROUPS, comparison_rows


def write_dictionaries(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    keys = {
        "M01": (
            "throughput_per_hour",
            "completed / (sim_time_s/3600)",
            "maximize",
            "simulated hours",
            "episode",
        ),
        "M02": (
            "pending, completion_rate",
            "completed/arrived; pending=arrived-completed",
            "pending minimize",
            "arrived tasks",
            "episode",
        ),
        "M03": (
            "total_lateness_s",
            "sum(max(0, completion_or_T-deadline))",
            "minimize",
            "all arrived tasks",
            "episode",
        ),
        "M04": ("deadline_violation_rate", "late_due / due", "minimize", "deadline<=T", "episode"),
        "M05": (
            "mean_wait_s",
            "sum(pickup_start_or_T-created)/arrived",
            "minimize",
            "arrived tasks",
            "episode",
        ),
        "M06": (
            "consumed_wh, energy_per_completed_wh",
            "consumed/completed; initial+charged-consumed-final",
            "minimize, balance=0",
            "completed tasks",
            "episode",
        ),
        "M07": (
            "empty_distance_ratio",
            "empty_distance/total_distance",
            "minimize",
            "traversed metres",
            "episode",
        ),
        "M08": (
            "mean_charge_wait_s",
            "mean(start_or_cancel_or_T-requested)",
            "minimize",
            "accepted charge requests",
            "episode",
        ),
        "M09": (
            "charger_utilization, port_occupied_utilization",
            "active_seconds / (ports*T)",
            "context dependent",
            "port-seconds",
            "episode",
        ),
        "M10": (
            "safety_incidents",
            "unique state audit incidents + physical safety counters",
            "zero",
            "all committed states",
            "episode",
        ),
        "M11": (
            "deadlock_recovery_rate, deadlock_recovery_s",
            "confirmed_resolved / detected; resolved-detected",
            "maximize rate, minimize delay",
            "detected incidents",
            "episode",
        ),
        "M12": (
            "decision_p95_ms, latency_components_ms",
            "percentiles of end-to-end measured decisions",
            "minimize",
            "dispatch decisions",
            "episode",
        ),
        "M13": (
            "solver_nonoptimal_rate, solver_timeout_rate, fallback_rate",
            "nonoptimal/calls; confirmed timeouts/calls; fallback/dispatch",
            "minimize",
            "solver calls or dispatch explicitly",
            "episode",
        ),
        "M14": (
            "resources",
            "actual steps/updates, measured wall, max simultaneous process-tree RSS",
            "context dependent",
            "training run",
            "training manifest",
        ),
        "M15": (
            "statistics, paired_vs_reference",
            "sample SD, crossed tape x training-seed bootstrap CI95",
            "context dependent",
            "complete expected run grid",
            "study summary",
        ),
        "M16": (
            "generalization",
            "OOD-control; (OOD-control)/abs(control)",
            "context dependent",
            "matched workload controls",
            "study summary",
        ),
    }
    rows = []
    for identity, (name, unit, description) in METRIC_GROUPS.items():
        key, formula, direction, denominator, scope = keys[identity]
        rows.append(
            dict(
                id=identity,
                name=name,
                keys=key,
                unit=unit,
                formula=formula,
                direction=direction,
                denominator=denominator,
                scope=scope,
                description=description,
                schema_version=2,
                missing="null; no zero substitution",
                censoring="pending at T where applicable; failed runs excluded from KPI means but retained in coverage",
            )
        )
    with (output / "metrics_dictionary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Metrics dictionary (schema 2)",
        "",
        "| ID | Name / keys | Unit | Formula | Denominator | Scope |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['name']}: {r['keys']} | {r['unit']} | {r['formula']} | {r['denominator']} | {r['scope']} |"
        )
    lines += [
        "",
        "Null means unavailable or an undefined denominator. Pending tasks/requests are right-censored at T.",
        "FEASIBLE/UNKNOWN do not establish a timeout. The solver wrapper leaves an unreported termination reason null.",
        "Episode output includes M01–M13. M14 comes from training manifests; M15–M16 require the study grid.",
    ]
    (output / "metrics_dictionary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(output, records, training):
    output = Path(output)
    rows = comparison_rows(records, training)
    with (output / "algorithm_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Algorithm comparison",
        "",
        "Aggregates below are descriptive. Use per-scenario paired CIs for conclusions; action/representation/backend changes are distinct comparisons.",
        "",
        "| ID | Method | Comparison | Complete / expected | Throughput/h | Lateness (s) | Safety | Training steps | Peak RAM (bytes) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        values = [
            r[k]
            for k in (
                "throughput_per_hour",
                "total_lateness_s",
                "safety_incidents",
                "training_steps",
                "peak_ram_bytes",
            )
        ]
        lines.append(
            f"| {r['variant_id']} | {r['method']} | {r['comparison']} | {r['complete_runs']}/{r['expected_runs']} | "
            + " | ".join("—" if v is None else f"{v:.3f}" for v in values)
            + " |"
        )
    (output / "algorithm_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def study_plots(root, summary, training):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for record in training:
        path = Path(record["final_model"]).parent / "validation.json"
        if not path.exists():
            continue
        values = json.loads(path.read_text(encoding="utf-8"))
        label = f"{record['method']}/{record['training_seed']}"
        axes[0].plot([v["steps"] for v in values], [v["rank"][1] for v in values], label=label)
        axes[1].plot(
            [v.get("elapsed_wall_s", 0) for v in values],
            [v["rank"][1] for v in values],
            label=label,
        )
    axes[0].set_xlabel("Fleet training steps")
    axes[1].set_xlabel("Elapsed wall seconds (including validation)")
    for ax in axes:
        ax.set_ylabel("Validation completed tasks")
        ax.grid(alpha=0.2)
    if training:
        axes[0].legend(fontsize=6)
    fig.savefig(root / "learning_curves.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for group in summary["groups"]:
        stats = group["statistics"]

        def get(key):
            return stats.get(key, {}).get("mean")

        throughput = get("throughput_per_hour")
        lateness = get("total_lateness_s")
        latency = get("decision_p95_ms")
        if throughput is None:
            continue
        if lateness is not None:
            axes[0].scatter(lateness, throughput, s=18, label=group["method"])
        if latency is not None:
            axes[1].scatter(latency, throughput, s=18)
    axes[0].set_xlabel("Lateness incl. pending (s)")
    axes[1].set_xlabel("Decision p95 (ms)")
    for ax in axes:
        ax.set_ylabel("Completed tasks/hour")
        ax.grid(alpha=0.2)
    fig.savefig(root / "quality_tradeoffs.png", dpi=140)
    plt.close(fig)
