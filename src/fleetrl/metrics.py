"""Censored-at-horizon fleet KPIs; unfinished and impossible tasks remain counted."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

METRICS_SCHEMA_VERSION = 2
METRIC_GROUPS = {
    "M01": ("throughput", "tasks/hour", "Completed tasks per simulated hour"),
    "M02": (
        "completion_backlog",
        "tasks, fraction",
        "All arrived tasks retained, including blocked work",
    ),
    "M03": (
        "lateness",
        "seconds",
        "Unweighted tardiness; pending tasks censored at observation time",
    ),
    "M04": (
        "deadline_violations",
        "tasks, fraction",
        "Denominator is tasks whose deadline has elapsed",
    ),
    "M05": ("task_wait", "seconds", "Creation to pickup start; pending pickups censored"),
    "M06": ("energy", "Wh, Wh/task", "Consumed/grid/battery balance and breakdown by robot kind"),
    "M07": ("travel", "metres, fraction", "Actually traversed loaded and empty edges"),
    "M08": ("charge_wait", "seconds", "Accepted request to actual start/cancel/time limit"),
    "M09": (
        "charger_utilization",
        "seconds, fraction",
        "Actual charging and physical port-cell occupancy",
    ),
    "M10": (
        "safety",
        "incidents",
        "Independent state audits; rejected proposals reported separately",
    ),
    "M11": (
        "deadlock_recovery",
        "incidents, seconds",
        "First progress and confirmed recovery, unresolved censored",
    ),
    "M12": (
        "online_latency",
        "milliseconds",
        "Preparation/inference/solver/commit; advance reported separately",
    ),
    "M13": (
        "solver_fallback",
        "calls, fraction",
        "Raw solver status, nonoptimal solves, confirmed timeout and fallback",
    ),
    "M14": (
        "training_resources",
        "steps, updates, seconds, bytes",
        "Actual fleet samples/updates, process-tree RAM and wall time",
    ),
    "M15": (
        "statistical_stability",
        "mean, SD, CI95",
        "Crossed tape/training-seed bootstrap with expected-grid coverage",
    ),
    "M16": (
        "generalization",
        "KPI change, fraction",
        "Matched OOD controls; compound shift reported separately",
    ),
}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def distribution(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "p50": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "max": max(values) if values else None,
    }


def fleet_metrics(
    sim, decision_logs: list[dict] | None = None, initial_battery_wh: float | None = None
) -> dict:
    now = float(sim.time_s)
    tasks = [t for t in sim.tasks.values() if t.created_s <= now]
    completed = [t for t in tasks if t.completed_at_s is not None]
    pending = [t for t in tasks if t.completed_at_s is None]
    due = [t for t in tasks if t.deadline_s <= now]
    late = [t for t in due if t.completed_at_s is None or t.completed_at_s > t.deadline_s]
    lateness = sum(
        max(0, (t.completed_at_s if t.completed_at_s is not None else now) - t.deadline_s)
        for t in tasks
    )
    wait = sum(
        max(0, (t.picked_at_s if t.picked_at_s is not None else now) - t.created_s) for t in tasks
    )
    system = sum(
        max(0, (t.completed_at_s if t.completed_at_s is not None else now) - t.created_s)
        for t in tasks
    )
    history = getattr(sim, "booking_history", [])
    waits = []
    for b in history:
        end = (
            b.actual_start_s
            if b.actual_start_s is not None
            else (b.cancelled_at_s if b.cancelled_at_s is not None else now)
        )
        waits.append(max(0, end - b.requested_at_s))
    logs = decision_logs or []
    lat = [float(x.get("total_ms", 0)) for x in logs]
    counters = sim.counters
    res = {k: float(v) for k, v in counters.items()}
    res.update(
        sim_time_s=now,
        arrived=len(tasks),
        completed=len(completed),
        pending=len(pending),
        throughput_per_hour=ratio(len(completed), now / 3600),
        completion_rate=len(completed) / len(tasks) if tasks else None,
        total_lateness_s=float(lateness),
        deadline_due=len(due),
        deadline_violated=len(late),
        deadline_violation_rate=len(late) / len(due) if due else None,
        pending_not_yet_due=sum(t.deadline_s > now for t in pending),
        mean_wait_s=ratio(wait, len(tasks)),
        wait_censored=sum(t.picked_at_s is None for t in tasks),
        mean_system_time_s=ratio(system, len(tasks)),
        blocked_tasks=sum(bool(t.blocked_reason) for t in pending),
        energy_per_completed_wh=counters.get("consumed_wh", 0) / len(completed)
        if completed
        else None,
        initial_battery_wh=initial_battery_wh,
        final_battery_wh=sum(r.battery_wh for r in sim.robots.values()),
        mean_final_soc=float(np.mean([r.soc for r in sim.robots.values()])),
        mean_charge_wait_s=float(np.mean(waits)) if waits else None,
        charging_requests=len(history),
        charge_wait_censored=sum(
            b.actual_start_s is None and b.cancelled_at_s is None for b in history
        ),
        charge_cancelled=sum(b.cancelled_at_s is not None for b in history),
        decision_count=len(logs),
        decision_p50_ms=float(np.percentile(lat, 50)) if lat else None,
        decision_p95_ms=float(np.percentile(lat, 95)) if lat else None,
        decision_max_ms=max(lat) if lat else None,
        fallback_rate=ratio(sum(bool(x.get("fallback_reason")) for x in logs), len(logs)),
    )
    for mass in [5, 15, 40, 80]:
        group = [t for t in tasks if t.weight_kg == mass]
        res[f"weight_{mass}_arrived"] = len(group)
        res[f"weight_{mass}_completed"] = sum(t.completed_at_s is not None for t in group)
    for priority in [1, 3]:
        group = [t for t in tasks if t.priority == priority]
        res[f"priority_{priority}_arrived"] = len(group)
        res[f"priority_{priority}_completed"] = sum(t.completed_at_s is not None for t in group)
    res.update(_operational_metrics(sim, logs, initial_battery_wh))
    return res


def _operational_metrics(sim, logs, initial_battery_wh):
    now = sim.time_s
    c = sim.counters
    robots = list(sim.robots.values())
    tasks = list(sim.tasks.values())
    calls = [
        x
        for x in logs
        if x.get("metadata", {}).get(
            "solver_called",
            x.get("solver_status") in {"OPTIMAL", "FEASIBLE", "UNKNOWN", "INFEASIBLE"},
        )
    ]
    statuses = Counter(
        status
        for x in calls
        for status in x.get("metadata", {}).get(
            "raw_solver_statuses",
            [x.get("metadata", {}).get("raw_solver_status", x.get("solver_status"))],
        )
    )
    solver_invocations = sum(statuses.values())
    known_timeout = [x for x in calls if x.get("metadata", {}).get("timed_out") is not None]
    confirmed = sum(x.get("metadata", {}).get("timed_out") is True for x in calls)
    incidents = getattr(sim, "deadlock_incidents", [])
    resolved = [i for i in incidents if i["resolved_s"] is not None]
    invariant = getattr(sim, "invariant_incidents", [])
    expected = {
        int(e.payload["id"])
        for e in sim.tape.events
        if e.kind == "task" and e.time_s <= now and "id" in e.payload
    }
    lost = len(expected - set(sim.tasks))
    audit = Counter(i["kind"] for i in invariant)
    usage = getattr(sim, "port_usage", {})
    port_stats = {
        str(pid): {
            **v,
            "charging_utilization": ratio(v["charging_s"], now),
            "occupied_utilization": ratio(v["occupied_s"], now),
        }
        for pid, v in usage.items()
    }
    total_battery = sum(r.battery_wh for r in robots)
    energy_error = (
        None
        if initial_battery_wh is None
        else initial_battery_wh
        + c.get("charged_battery_wh", 0)
        - c.get("consumed_wh", 0)
        - total_battery
    )
    result = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "metric_groups": list(METRIC_GROUPS),
        "pending_rate": ratio(sum(t.completed_at_s is None for t in tasks), len(tasks)),
        "empty_distance_ratio": ratio(c.get("empty_distance_m", 0), c.get("distance_m", 0)),
        "battery_balance_error_wh": energy_error,
        "port_usage": port_stats,
        "charging_time_s": sum(v["charging_s"] for v in usage.values()),
        "port_occupied_time_s": sum(v["occupied_s"] for v in usage.values()),
        "charger_utilization": ratio(
            sum(v["charging_s"] for v in usage.values()), len(usage) * now
        ),
        "port_occupied_utilization": ratio(
            sum(v["occupied_s"] for v in usage.values()), len(usage) * now
        ),
        "invariant_incidents": dict(audit),
        "lost_tasks": lost,
        "payload_violations": audit["payload"],
        "duplicate_assignments": audit["duplicate_assignment"],
        "task_ownership_violations": audit["task_ownership"],
        "safety_incidents": sum(audit.values())
        + lost
        + sum(
            c.get(k, 0)
            for k in ("collisions", "edge_conflicts", "reserve_violations", "energy_emergencies")
        ),
        "deadlock_incident_count": len(incidents),
        "deadlock_confirmed_resolved": len(resolved),
        "deadlock_pending": len(incidents) - len(resolved),
        "deadlock_recovery_rate": ratio(len(resolved), len(incidents)),
        "deadlock_first_progress_s": distribution(
            [
                i["first_progress_s"] - i["detected_s"]
                for i in incidents
                if i["first_progress_s"] is not None
            ]
        ),
        "deadlock_recovery_s": distribution([i["resolved_s"] - i["detected_s"] for i in resolved]),
        "deadlock_duration_censored_s": distribution(
            [
                (i["resolved_s"] if i["resolved_s"] is not None else now) - i["detected_s"]
                for i in incidents
            ]
        ),
        "deadlock_incidents": incidents,
        "solver_calls": solver_invocations,
        "solver_dispatches": len(calls),
        "solver_status_counts": dict(statuses),
        "solver_nonoptimal_rate": ratio(
            sum(v for k, v in statuses.items() if k != "OPTIMAL"), solver_invocations
        ),
        "solver_confirmed_timeouts": confirmed,
        "solver_timeout_unknown_calls": len(calls) - len(known_timeout),
        "solver_timeout_rate": ratio(confirmed, len(calls))
        if len(known_timeout) == len(calls)
        else None,
        "solver_known_timeout_rate": ratio(confirmed, len(known_timeout)),
        "fallback_reasons": dict(
            Counter(x.get("fallback_reason") for x in logs if x.get("fallback_reason"))
        ),
        "proposal_rejection_reasons": dict(
            Counter(r["reason"] for x in logs for r in x.get("commit", {}).get("rejected", []))
        ),
        "proposal_rejection_rate": ratio(
            sum(len(x.get("commit", {}).get("rejected", [])) for x in logs),
            sum(len(x.get("decisions", [])) for x in logs),
        ),
        "arbitration_rejection_count": sum(
            len(x.get("metadata", {}).get("arbitration_rejections", [])) for x in logs
        ),
        "online_budget_exceeded_rate": ratio(
            sum(bool(x.get("budget_exceeded")) for x in logs), len(logs)
        ),
        "latency_components_ms": {
            key: distribution([x.get(key) for x in logs])
            for key in (
                "preparation_ms",
                "policy_inference_ms",
                "solver_ms",
                "commit_ms",
                "total_ms",
                "advance_ms",
            )
        },
        "training_resources": None,
        "statistical_stability": None,
        "generalization": None,
    }
    for field in ("weight_kg", "priority"):
        for value in [5, 15, 40, 80] if field == "weight_kg" else [1, 3]:
            group = [t for t in tasks if getattr(t, field) == value]
            due = [t for t in group if t.deadline_s <= now]
            prefix = f"weight_{value}" if field == "weight_kg" else f"priority_{value}"
            result[prefix + "_completion_rate"] = ratio(
                sum(t.completed_at_s is not None for t in group), len(group)
            )
            result[prefix + "_lateness_s"] = sum(
                max(0, (t.completed_at_s if t.completed_at_s is not None else now) - t.deadline_s)
                for t in group
            )
            result[prefix + "_deadline_violation_rate"] = ratio(
                sum(t.completed_at_s is None or t.completed_at_s > t.deadline_s for t in due),
                len(due),
            )
    return result


def training_metrics(model, wall_s, timings, samples, initial_steps=0, initial_updates=0):
    import torch

    steps = int(getattr(model, "num_timesteps", 0))
    updates = int(getattr(model, "_n_updates", 0))
    result = {
        "wall_s": wall_s,
        **timings,
        "actual_total_steps": steps,
        "additional_fleet_steps": steps - initial_steps,
        "training_updates": updates,
        "additional_updates": updates - initial_updates,
        "updates_per_fleet_step": ratio(updates - initial_updates, steps - initial_steps),
        "steps_per_wall_second": ratio(steps - initial_steps, wall_s),
        "peak_process_tree_rss_bytes": max((s["tree_rss_bytes"] for s in samples), default=None),
        "ram_sample_count": len(samples),
        "ram_sampling_interval_s": 0.2,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else None,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved()
        if torch.cuda.is_available()
        else None,
        "parameter_count": sum(p.numel() for p in model.policy.parameters())
        if model is not None and hasattr(model, "policy")
        else None,
    }
    if model is not None and hasattr(model, "q"):
        result.update(
            visited_states=len(model.q),
            state_visits=sum(model.visits.values()),
            unseen_prediction_rate=ratio(model.unseen_predictions, model.predictions),
        )
    return result


def _metric_keys(records: Sequence[dict]) -> list[str]:
    return sorted(
        {
            key
            for record in records
            for key, value in record.get("metrics", {}).items()
            if value is None or isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _paired_ci(
    candidate: Sequence[dict],
    reference: Sequence[dict],
    metric: str,
    samples: int,
    rng: np.random.Generator,
    expected_tapes=None,
    expected_replicates=None,
) -> dict:
    baseline = defaultdict(list)
    candidate_cells = defaultdict(lambda: defaultdict(list))
    for record in reference:
        if record.get("status", "complete") == "complete" and _finite(
            record["metrics"].get(metric)
        ):
            baseline[(record["seed"], record["tape_hash"])].append(float(record["metrics"][metric]))
    for record in candidate:
        key = (record["seed"], record["tape_hash"])
        if (
            record.get("status", "complete") == "complete"
            and key in baseline
            and _finite(record["metrics"].get(metric))
        ):
            identity = str(record.get("training_seed", record.get("checkpoint", "deterministic")))
            candidate_cells[key][identity].append(float(record["metrics"][metric]))
    keys = sorted(candidate_cells)
    identities = sorted({identity for values in candidate_cells.values() for identity in values})
    if not keys:
        return {"paired_tapes": 0, "training_replicates": 0, "mean_difference": None, "ci95": None}
    # Only a complete crossed tape/checkpoint grid is inferentially comparable.
    # Report missing cells instead of averaging a different mix of policies/tapes.
    complete = all(set(candidate_cells[key]) == set(identities) for key in keys)
    if expected_tapes is not None:
        complete = complete and {k[0] for k in keys} == set(expected_tapes)
    if expected_replicates is not None:
        complete = complete and set(identities) == {str(v) for v in expected_replicates}
    complete = complete and all(
        len(values) == 1 for cells in candidate_cells.values() for values in cells.values()
    )
    # B1 is deterministic: duplicated reference cells must not silently gain weight.
    complete = complete and all(len(baseline[key]) == 1 for key in keys)
    per_tape = [
        np.mean([np.mean(values) for values in candidate_cells[key].values()])
        - np.mean(baseline[key])
        for key in keys
    ]
    result = {
        "paired_tapes": len(keys),
        "training_replicates": len(identities),
        "mean_difference": float(np.mean(per_tape)),
        "ci95": None,
        "matched_seeds": [key[0] for key in keys],
        "complete_crossed_grid": complete,
        "bootstrap_unit": "matched event tapes and independent training checkpoints",
    }
    if len(keys) < 2 or not complete or samples < 1:
        result["ci_note"] = "CI withheld: need >=2 paired tapes and a complete checkpoint/tape grid"
        return result
    delta = np.array(
        [
            [
                np.mean(candidate_cells[key][identity]) - np.mean(baseline[key])
                for identity in identities
            ]
            for key in keys
        ]
    )
    rows = rng.integers(0, len(keys), (samples, len(keys)))
    cols = rng.integers(0, len(identities), (samples, len(identities)))
    draws = delta[rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))
    result["ci95"] = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
    if len(identities) < 3:
        result["training_seed_note"] = (
            "Fewer than 3 training replicas; policy-seed uncertainty is weakly measured"
        )
    return result


def paired_summary(
    records: Sequence[dict],
    reference: str = "fixed",
    bootstrap_samples: int = 2000,
    seed: int = 12345,
    expected_grid=None,
) -> dict:
    """Summarize each scenario/method and paired differences against B1.

    Null denominators (e.g. zero completions) never become zeros. Metrics with
    censored tasks/charge requests are retained exactly as computed by the env.
    """
    from .evaluation import METHODS

    reference = METHODS.get(reference, reference)
    grouped = defaultdict(list)
    for record in records:
        grouped[(record.get("scenario", "reference"), record["method"])].append(record)
    rng = np.random.default_rng(seed)
    summaries = []
    for (scenario, method), group in sorted(grouped.items()):
        stats = {}
        for key in _metric_keys(group):
            values = [
                float(r["metrics"][key])
                for r in group
                if r.get("status", "complete") == "complete" and _finite(r["metrics"].get(key))
            ]
            stats[key] = {
                "n": len(values),
                "missing": len(group) - len(values),
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        baseline = grouped.get((scenario, reference), [])
        expected = (expected_grid or {}).get(method, {})
        paired = (
            {}
            if method == reference
            else {
                key: _paired_ci(
                    group,
                    baseline,
                    key,
                    bootstrap_samples,
                    rng,
                    expected.get("tape_seeds"),
                    expected.get("training_seeds"),
                )
                for key in _metric_keys(group)
            }
        )
        summaries.append(
            {
                "scenario": scenario,
                "method": method,
                "episodes": len(group),
                "complete_episodes": sum(r.get("status", "complete") == "complete" for r in group),
                "failed_or_partial": sum(r.get("status", "complete") != "complete" for r in group),
                "unique_tapes": len({(r["seed"], r["tape_hash"]) for r in group}),
                "statistics": stats,
                "paired_vs_reference": paired,
            }
        )
    return {
        "reference": reference,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "interpretation": "Smoke runs verify execution only. These intervals do not establish RL superiority; inspect every scenario, censored counts and safety metrics.",
        "groups": summaries,
    }


def generalization_summary(records, pairs):
    """Only predeclared matched-control scenario pairs permit an OOD delta."""
    result = []
    for pair in pairs:
        control = pair["control"]
        ood = pair["ood"]
        metrics = pair.get(
            "metrics", ["throughput_per_hour", "total_lateness_s", "pending", "consumed_wh"]
        )
        for method in sorted({r["method"] for r in records}):
            left = {
                (r["seed"], str(r.get("training_seed", "deterministic"))): r
                for r in records
                if r["method"] == method
                and r.get("scenario") == control
                and r.get("status", "complete") == "complete"
            }
            right = {
                (r["seed"], str(r.get("training_seed", "deterministic"))): r
                for r in records
                if r["method"] == method
                and r.get("scenario") == ood
                and r.get("status", "complete") == "complete"
            }
            matched = set(left) & set(right)
            # Tape hashes differ with map metadata; workload hashes must agree.
            matched = {
                k
                for k in matched
                if left[k].get("workload_hash") is not None
                and left[k].get("workload_hash") == right[k].get("workload_hash")
            }
            for metric in metrics:
                deltas = []
                relative = []
                for key in matched:
                    a = left[key]["metrics"].get(metric)
                    b = right[key]["metrics"].get(metric)
                    if _finite(a) and _finite(b):
                        deltas.append(b - a)
                        if a != 0:
                            relative.append((b - a) / abs(a))
                result.append(
                    {
                        "method": method,
                        "control": control,
                        "ood": ood,
                        "metric": metric,
                        "matched_runs": len(matched),
                        "difference": distribution(deltas),
                        "relative_change": distribution(relative),
                        "interpretation": pair.get(
                            "interpretation", "matched workload; isolated layout shift"
                        ),
                    }
                )
    return result


def episode_metrics(
    metrics, total_reward, steps, failures, policy_times, online_times, budget_ms, wall_s
):
    """Runner measurements are reduced here with all other KPI formulas."""
    return {
        **metrics,
        "episode_reward": total_reward,
        "decision_steps": steps,
        "policy_failures": failures,
        "policy_inference_mean_ms": distribution(policy_times)["mean"],
        "policy_inference_p95_ms": distribution(policy_times)["p95"],
        "online_decision_p95_ms": distribution(online_times)["p95"],
        "online_decision_mean_ms": distribution(online_times)["mean"],
        "online_budget_exceeded_rate": ratio(
            sum(t > budget_ms for t in online_times), len(online_times)
        ),
        "wall_time_s": wall_s,
    }


def comparison_rows(records, training):
    from .methods.registry import METHODS

    rows = []
    for name, spec in METHODS.items():
        group = [r for r in records if r["method"] == name]
        complete = [r for r in group if r.get("status") == "complete"]
        learned = [r for r in training if r["method"] == name]
        row = {
            "variant_id": spec.variant_id,
            "method": name,
            "comparison": spec.comparison,
            "expected_runs": len(group),
            "complete_runs": len(complete),
            "status": "complete"
            if group and len(group) == len(complete)
            else "partial"
            if group
            else "not_requested",
        }
        for key in (
            "throughput_per_hour",
            "total_lateness_s",
            "pending",
            "safety_incidents",
            "decision_p95_ms",
            "consumed_wh",
        ):
            row[key] = distribution([r["metrics"].get(key) for r in complete])["mean"]
        row["training_steps"] = (
            sum(r["actual_total_timesteps"] for r in learned) if learned else None
        )
        row["training_wall_s"] = sum(r["resources"]["wall_s"] for r in learned) if learned else None
        row["peak_ram_bytes"] = max(
            (r["resources"]["peak_process_tree_rss_bytes"] for r in learned), default=None
        )
        rows.append(row)
    return rows
