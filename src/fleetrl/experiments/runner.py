"""Sequential, resumable study jobs with explicit failures and expected coverage."""

import csv
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from ..config import load_config
from ..evaluation import file_sha256, plot_benchmark, run_episode, source_tree_hash, write_json
from ..methods.registry import get_method
from ..metrics import _metric_keys, generalization_summary, paired_summary
from .protocol import load_study


def method_config(study, name):
    cfg, tc = load_config(Path(study.config_dir) / (name + ".yaml"))
    cfg = get_method(name).configure(cfg).copy(horizon_s=study.horizon_s)
    tc = replace(
        tc,
        total_timesteps=study.total_timesteps,
        eval_seeds=tuple(study.validation_seeds),
        train_seed_min=study.train_seed_min,
        train_seed_max=study.train_seed_max,
    )
    return cfg, tc


def study_plan(study):
    learned = [m for m in study.methods if get_method(m).learnable]
    deterministic = len(study.methods) - len(learned)
    models = len(learned) * len(study.training_seeds)
    return {
        "study": study.to_dict(),
        "protocol_hash": study.digest,
        "training_jobs": 0 if study.pretrained_study else models,
        "requested_training_steps": 0 if study.pretrained_study else models * study.total_timesteps,
        "core_test_episodes": (models + deterministic)
        * len(study.scenarios)
        * len(study.tape_seeds),
        "matched_control_episodes": (models + deterministic) * 2 * len(study.tape_seeds)
        if study.matched_ood
        else 0,
        "search_jobs": len(learned) * len(study.search_trials),
        "search_steps_per_method": len(study.search_trials) * study.search_steps,
        "expected_grid": study.expected_grid(),
        "note": "Execution readiness is separate from convergence or algorithm superiority. Budgets count fleet decisions.",
    }


def _read(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def _locked_root(output, study):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "protocol_hash": study.digest,
        "source_hash": source_tree_hash(),
        "method_config_hashes": {
            name: file_sha256(Path(study.config_dir) / (name + ".yaml")) for name in study.methods
        },
    }
    previous = _read(output / "identity.json")
    if previous and previous != identity:
        migration = _read(output / "source_migration.json", {})
        same_protocol = (
            previous.get("protocol_hash") == identity["protocol_hash"]
            and previous.get("method_config_hashes") == identity["method_config_hashes"]
        )
        authorized_source = (
            migration.get("from_source_hash") == previous.get("source_hash")
            and migration.get("to_source_hash") == identity["source_hash"]
        )
        if not (same_protocol and authorized_source):
            raise ValueError("source/config/protocol changed; use a new study output directory")
    else:
        write_json(output / "identity.json", identity)
    write_json(output / "plan.json", study_plan(study))
    return output


def _verified_job(job, key):
    artifacts = job.get("artifact_hashes", {})
    if not artifacts or not all(
        Path(p).is_file() and file_sha256(p) == digest for p, digest in artifacts.items()
    ):
        raise ValueError(f"completed job {key} has missing or modified artifacts")
    return job["result"]


def _selection(root, source, kind, freeze=False):
    """Freeze both selected parameters and the explicit choice to use defaults."""
    path = Path(source) / f"selected_{kind}.json"
    value = _read(path)
    lock = Path(root) / f"frozen_{kind}.json"
    previous = _read(lock)
    identity = {
        "source": str(path.resolve()),
        "sha256": file_sha256(path) if path.exists() else None,
    }
    if previous is not None and previous != identity:
        raise ValueError(f"frozen {kind} selection changed; use a new study output directory")
    if value and value.get("complete") is False:
        raise ValueError(f"{kind} selection is incomplete; finish tuning/search first")
    if freeze:
        write_json(lock, identity)
    return (value or {}).get("selected", {})


def _job(root, key, execute, limit_state):
    folder = root / "jobs" / key
    state_path = folder / "job.json"
    previous = _read(state_path, {})
    if previous.get("status") == "complete":
        return _verified_job(previous, key)
    if limit_state["remaining"] == 0:
        return None
    if limit_state["remaining"] is not None:
        limit_state["remaining"] -= 1
    attempt = previous.get("attempt", 0) + 1
    run = folder / f"attempt_{attempt:03d}"
    state = {"job": key, "status": "running", "attempt": attempt, "output": str(run)}
    write_json(state_path, state)
    started = time.perf_counter()
    try:
        result, artifacts = execute(run, previous)
        if result.get("status", "complete") != "complete":
            raise ValueError("job returned an incomplete result")
        state.update(
            status="complete",
            result=result,
            artifact_hashes={str(Path(p).resolve()): file_sha256(p) for p in artifacts},
        )
    except KeyboardInterrupt:
        state.update(status="interrupted")
        raise
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(f"[FAILED] {key}: {state['error']}", flush=True)
        return None
    finally:
        state["wall_s"] = time.perf_counter() - started
        write_json(state_path, state)
    print(f"[COMPLETE] {key}", flush=True)
    return result


def _training_artifacts(result):
    files = []
    for key in ("final_model", "best_model"):
        p = Path(result[key])
        files.extend([p, Path(str(p) + ".json")])
        metadata = _read(str(p) + ".json")
        if metadata.get("replay_file"):
            files.append(p.parent / metadata["replay_file"])
    return files + [
        Path(result["final_model"]).parent / name for name in ("manifest.json", "validation.json")
    ]


def _resume_candidate(directory):
    from ..learning.training import checkpoint_metadata

    candidates = []
    for name in ("final_model.zip", "interrupted_model.zip", "latest_model.zip"):
        path = Path(directory) / name
        if not path.exists():
            continue
        try:
            meta = checkpoint_metadata(path)
            if not meta:
                continue
            if meta.get("replay_file"):
                replay = path.parent / meta["replay_file"]
                if replay.parent != path.parent or file_sha256(replay) != meta["replay_sha256"]:
                    continue
            candidates.append((meta["num_timesteps"], meta["training_updates"], path))
        except (OSError, ValueError, KeyError):
            continue
    return max(candidates, key=lambda x: x[:2])[2] if candidates else None


def run_study(path, output, phase="plan", limit=None):
    study = load_study(path)
    if phase == "plan":
        return study_plan(study)
    root = _locked_root(output, study)
    original_source_hash = _read(root / "identity.json")["source_hash"]
    resume_source_hash = (
        original_source_hash if original_source_hash != source_tree_hash() else None
    )
    limit_state = {"remaining": limit}
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    source = Path(study.pretrained_study) if study.pretrained_study else root
    for kind in ("learning", "baselines"):
        if (root / f"frozen_{kind}.json").exists():
            _selection(root, source, kind)
    if phase == "search":
        if (root / "frozen_learning.json").exists():
            raise ValueError("learning selection is frozen after training/testing starts")
        from ..learning.training import train_method

        selected = {}
        for name in study.methods:
            if not get_method(name).learnable:
                continue
            cfg, tc = method_config(study, name)
            results = []
            for trial, lr in enumerate(study.search_trials):
                train_params = (
                    {"q_alpha": study.search_q_alphas[trial]}
                    if get_method(name).algorithm == "qlearning"
                    else {"learning_rate": lr}
                )
                env_params = (
                    {"charge_threshold": study.search_thresholds[trial]}
                    if name == "ppo_threshold_cpsat"
                    else {}
                )

                def execute(run, previous, train_params=train_params, env_params=env_params):
                    opts = replace(
                        tc,
                        total_timesteps=study.search_steps,
                        seed=study.training_seeds[0],
                        **train_params,
                    )
                    result = train_method(cfg.copy(**env_params), opts, run)
                    return result, _training_artifacts(result)

                result = _job(root, f"search_{name}_{trial}", execute, limit_state)
                if result:
                    vals = _read(Path(result["final_model"]).parent / "validation.json")
                    best = max(vals, key=lambda v: tuple(v["rank"]))
                    results.append((tuple(best["rank"]), train_params, env_params, trial))
            if len(results) == len(study.search_trials):
                rank, train_params, env_params, trial = max(results, key=lambda x: (x[0], -x[3]))
                selected[name] = {
                    "train": train_params,
                    "env": env_params,
                    "trial": trial,
                    "rank": rank,
                }
        write_json(
            root / "selected_learning.json",
            {
                "protocol_hash": study.digest,
                "selected": selected,
                "complete": len(selected) == sum(get_method(n).learnable for n in study.methods),
            },
        )
        return {
            "phase": phase,
            "selected": selected,
            "status": "complete"
            if len(selected) == sum(get_method(n).learnable for n in study.methods)
            else "partial",
            "failed_jobs": _failures(root, "search_"),
        }
    if phase == "tune":
        if (root / "frozen_baselines.json").exists():
            raise ValueError("baseline selection is frozen after testing starts")
        return tune_deterministic(root, study, limit_state)
    if phase == "train":
        if study.pretrained_study:
            raise ValueError("confirmatory study uses frozen checkpoints; training is disabled")
        from ..learning.training import train_method

        selected = _selection(root, root, "learning", freeze=True)
        results = []
        for name in study.methods:
            if not get_method(name).learnable:
                continue
            cfg, tc = method_config(study, name)
            if name in selected:
                tc = replace(tc, **selected[name]["train"])
                cfg = cfg.copy(**selected[name]["env"])
            for seed in study.training_seeds:

                def execute(run, previous, seed=seed):
                    resume = None
                    if previous.get("output"):
                        resume = _resume_candidate(previous["output"])
                    opts = replace(tc, seed=seed)
                    result = train_method(
                        cfg,
                        opts,
                        run,
                        resume,
                        target_total_timesteps=tc.total_timesteps,
                        resume_source_hash=resume_source_hash,
                    )
                    result["selection"] = (
                        "validation_search" if name in selected else "predeclared_default"
                    )
                    return result, _training_artifacts(result)

                result = _job(root, f"train_{name}_{seed}", execute, limit_state)
                if result:
                    results.append(result)
        plan = study_plan(study)
        return {
            "phase": phase,
            "completed_training_jobs": len(results),
            "plan": plan,
            "status": "complete" if len(results) == plan["training_jobs"] else "partial",
            "failed_jobs": _failures(root, "train_"),
        }
    if phase == "test":
        return test_study(root, study, limit_state, resume_source_hash)
    if phase == "report":
        return report_study(root, study)
    raise ValueError("unknown study phase")


def tune_deterministic(root, study, limit_state):
    from ..rl import validation_rank

    selected = {}
    grids = {
        "heuristic": [{"charge_threshold": x} for x in (0.2, 0.3, 0.4)],
        "fixed_cpsat": [{"fixed_profile": i} for i in range(6)],
        "forecast_mpc_milp": [
            {"forecast_alpha": a, "mpc_horizon_s": h}
            for a in (0.1, 0.25, 0.5)
            for h in (150.0, 300.0)
        ],
    }
    for name in study.methods:
        if name not in grids:
            continue
        cfg, _ = method_config(study, name)
        ranked = []
        for trial, params in enumerate(grids[name]):

            def execute(run, previous, params=params):
                results = [
                    run_episode(cfg.copy(**params), name, seed, output_dir=run / str(seed))
                    for seed in study.validation_seeds
                ]
                if any(r["status"] != "complete" for r in results):
                    raise RuntimeError("tuning requires complete episodes")
                result = {
                    "parameters": params,
                    "rank": validation_rank([r["metrics"] for r in results]),
                    "records": results,
                }
                write_json(run / "tuning.json", result)
                return result, [run / "tuning.json"]

            result = _job(root, f"tune_{name}_{trial}", execute, limit_state)
            if result:
                ranked.append(result)
        if len(ranked) == len(grids[name]):
            selected[name] = max(ranked, key=lambda r: tuple(r["rank"]))["parameters"]
    write_json(
        root / "selected_baselines.json",
        {
            "protocol_hash": study.digest,
            "selected": selected,
            "complete": len(selected) == sum(name in grids for name in study.methods),
        },
    )
    return {
        "phase": "tune",
        "selected": selected,
        "status": "complete"
        if len(selected) == sum(name in grids for name in study.methods)
        else "partial",
        "failed_jobs": _failures(root, "tune_"),
    }


def _failures(root, prefix):
    return sum(
        _read(p).get("status") == "failed" for p in (root / "jobs").glob(prefix + "*/job.json")
    )


def _case(study, scenario):
    from ..config import FleetConfig
    from ..scenario import scenario_config

    base = FleetConfig(horizon_s=study.horizon_s)
    if scenario in {"OOD_A", "OOD_B"}:
        return base.copy(
            n_robots=20,
            demand_per_hour=120,
            map_name=scenario[-1],
            disturbances=False,
            burst_multiplier=1,
        )
    return scenario_config(scenario, base)


def _tape(study, scenario, seed):
    from ..maps import make_map
    from ..scenario import generate_tape

    cfg = _case(study, scenario)
    if scenario in {"OOD_A", "OOD_B"}:
        # Same coordinates, task timing, deadlines, masses and priorities in both layouts.
        source = cfg.copy(map_name="A")
        tape = generate_tape(source, make_map(source), seed)
    else:
        tape = generate_tape(cfg, make_map(cfg), seed)
    workload = hashlib.sha256(
        json.dumps(
            [{"time": e.time_s, "kind": e.kind, "payload": e.payload} for e in tape.events],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return cfg, tape, workload


def test_study(root, study, limit_state, resume_source_hash=None):
    from ..learning.training import load_checkpoint

    allowed_sources = {source_tree_hash()}
    if resume_source_hash:
        allowed_sources.add(resume_source_hash)
    training_root = Path(study.pretrained_study) if study.pretrained_study else root
    tuned = _selection(root, training_root, "baselines", freeze=True)
    _selection(root, training_root, "learning", freeze=True)
    for name in study.methods:
        spec = get_method(name)
        replicas = study.training_seeds if spec.learnable else ["deterministic"]
        for training_seed in replicas:
            policy = None
            checkpoint = None
            if spec.learnable:
                job = _read(
                    training_root / "jobs" / f"train_{name}_{training_seed}" / "job.json", {}
                )
                if job.get("status") != "complete":
                    print(f"[PENDING] missing training job {name}/{training_seed}", flush=True)
                    continue
                _verified_job(job, f"train_{name}_{training_seed}")
                checkpoint = job["result"]["best_model"]
                policy = load_checkpoint(checkpoint, expected_method=name)
                if policy._fleetrl_metadata["source_hash"] not in allowed_sources:
                    raise ValueError("training checkpoint source differs from test source")
                identity = {
                    "path": str(Path(checkpoint).resolve()),
                    "sha256": file_sha256(checkpoint),
                }
                lock = root / f"frozen_checkpoint_{name}_{training_seed}.json"
                old = _read(lock)
                if old is not None and old != identity:
                    raise ValueError(
                        "frozen training checkpoint changed; use a new study output directory"
                    )
                write_json(lock, identity)
            for scenario in study.scenarios_with_controls():
                for seed in study.tape_seeds:

                    def execute(run, previous):
                        cfg, tape, workload = _tape(study, scenario, seed)
                        cfg = cfg.copy(**tuned.get(name, {}))
                        if policy is not None and hasattr(policy, "_fleetrl_metadata"):
                            cfg = cfg.copy(
                                charge_threshold=policy._fleetrl_metadata["env"]["charge_threshold"]
                            )
                        result = run_episode(cfg, name, seed, policy, run, tape)
                        result.update(
                            scenario=scenario,
                            training_seed=training_seed,
                            checkpoint=checkpoint,
                            checkpoint_source_hash=(
                                policy._fleetrl_metadata["source_hash"] if policy else None
                            ),
                            workload_hash=workload,
                        )
                        write_json(run / "result.json", result)
                        return result, [run / "result.json", run / "manifest.json"]

                    _job(
                        root, f"test_{name}_{training_seed}_{scenario}_{seed}", execute, limit_state
                    )
    return report_study(root, study)


def report_study(root, study):
    records = []
    missing = []
    failures = []
    for name in study.methods:
        for replica in study.training_seeds if get_method(name).learnable else ["deterministic"]:
            for scenario in study.scenarios_with_controls():
                for seed in study.tape_seeds:
                    key = f"test_{name}_{replica}_{scenario}_{seed}"
                    job = _read(root / "jobs" / key / "job.json", {})
                    if job.get("status") == "complete":
                        records.append(_verified_job(job, key))
                    else:
                        record = {
                            "method": name,
                            "training_seed": replica,
                            "scenario": scenario,
                            "seed": seed,
                            "tape_hash": "missing",
                            "status": job.get("status", "missing"),
                            "metrics": {},
                            "error": job.get("error"),
                        }
                        records.append(record)
                        missing.append(key)
                        if job.get("status") == "failed":
                            failures.append(record)
    summary = paired_summary(
        records,
        reference="fixed_cpsat",
        bootstrap_samples=study.bootstrap_samples,
        expected_grid=study.expected_grid(),
    )
    summary.update(
        status="complete" if not missing else "partial",
        expected_episodes=len(records),
        complete_episodes=len(records) - len(missing),
        missing_jobs=missing,
        failed_jobs=failures,
    )
    migration = _read(root / "source_migration.json")
    if migration:
        summary["source_migration"] = migration
    summary["generalization"] = (
        generalization_summary(records, [{"control": "OOD_A", "ood": "OOD_B"}])
        if study.matched_ood
        else []
    )
    write_json(root / "summary.json", summary)
    with (root / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, allow_nan=False) + "\n")
    keys = _metric_keys(records)
    with (root / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        columns = ["method", "scenario", "training_seed", "seed", "status"] + keys
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {**{k: r.get(k) for k in columns[:5]}, **{k: r["metrics"].get(k) for k in keys}}
            )
    lines = [
        "# FleetRL study report",
        "",
        f"Status: {summary['status']}. Complete {summary['complete_episodes']}/{len(records)} expected episodes.",
        "",
        "| Method | Scenario | Complete / expected | Throughput/h | Pending | Lateness (s) | Safety incidents |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if migration:
        lines[4:4] = [
            "Source migration: this study resumed after a simulator fix. "
            f"Original hash `{migration['from_source_hash']}`; "
            f"patched hash `{migration['to_source_hash']}`. "
            "Training checkpoints may come from either source; see source_migration.json.",
            "",
        ]
    for group in summary["groups"]:
        stats = group["statistics"]
        values = [
            stats.get(k, {}).get("mean")
            for k in ("throughput_per_hour", "pending", "total_lateness_s", "safety_incidents")
        ]
        lines.append(
            f"| {group['method']} | {group['scenario']} | {group['complete_episodes']}/{group['episodes']} | "
            + " | ".join("—" if v is None else f"{v:.3f}" for v in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "Paired CIs and all 16 metric groups are in summary.json. Missing/failed runs remain in denominators.",
            "S9 is a compound distribution shift. OOD_A/OOD_B isolate layout with the same task tape.",
            "Execution checks do not establish convergence, superiority, or a 5% improvement.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    from .reporting import study_plots, write_comparison, write_dictionaries

    training = []
    training_root = Path(study.pretrained_study) if study.pretrained_study else root
    for file in (training_root / "jobs").glob("train_*/job.json"):
        job = _read(file)
        if job.get("status") == "complete":
            training.append(_verified_job(job, file.parent.name))
    write_dictionaries(root)
    write_comparison(root, records, training)
    if summary["complete_episodes"]:
        plot_benchmark(summary, root / "benchmark.png")
        study_plots(root, summary, training)
    return {
        "status": summary["status"],
        "complete_episodes": summary["complete_episodes"],
        "expected_episodes": len(records),
        "failed_jobs": len(failures),
        "report": str(root / "report.md"),
    }
