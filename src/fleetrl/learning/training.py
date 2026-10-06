"""Shared lifecycle for eleven actual learners; method identity travels with artifacts."""

import copy
import hashlib
import json
import time
import uuid
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from torch import nn

from ..config import FleetConfig, TrainConfig, save_config
from ..evaluation import file_sha256, run_episode, runtime_manifest, source_tree_hash, write_json
from ..methods.registry import get_method

CHECKPOINT_SCHEMA = 2
OFF_POLICY = {"dqn", "qrdqn", "sac", "td3"}
CHECKPOINT_COMMENT = b"FleetRL checkpoint v2\n"


def _payload_hash(path):
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            digest.update(item.filename.encode("utf-8"))
            with archive.open(item) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def encoder_kwargs(tc):
    from ..rl import MaskedEntityEncoder

    result = {
        "activation_fn": nn.Tanh,
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
        "optimizer_kwargs": {"eps": 1e-5},
    }
    if tc.encoder == "entities":
        result.update(
            features_extractor_class=MaskedEntityEncoder,
            features_extractor_kwargs={"features_dim": 256, "entity_dim": 64},
        )
    return result


def finite_parameters(model):
    if hasattr(model, "q"):
        if any(not np.isfinite(v).all() for v in model.q.values()):
            raise FloatingPointError("nonfinite Q table")
        return
    for name, p in model.policy.named_parameters():
        if not torch.isfinite(p).all():
            raise FloatingPointError(f"nonfinite parameter {name}")
    optimizers = [
        getattr(model, "optimizer", None),
        getattr(model.policy, "optimizer", None),
        getattr(model, "ent_coef_optimizer", None),
    ]
    optimizers += [
        getattr(getattr(model, name, None), "optimizer", None) for name in ("actor", "critic")
    ]
    for optimizer in optimizers:
        if optimizer is not None:
            for state in optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value) and not torch.isfinite(value).all():
                        raise FloatingPointError("nonfinite optimizer state")


def checkpoint(path, model, cfg, tc, lineage=None):
    from ..artifacts import atomic_replace

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    finite_parameters(model)
    temporary = path.with_name(path.stem + ".pending.zip")
    previous = checkpoint_metadata(path) if path.exists() else None
    model.save(str(temporary))
    replay = None
    if tc.algorithm in OFF_POLICY:
        # A new generation cannot overwrite the replay referenced by the live ZIP.
        replay = path.with_name(path.stem + "." + uuid.uuid4().hex + ".replay.pkl")
        replay_temporary = replay.with_name(replay.name + ".tmp")
        model.save_replay_buffer(str(replay_temporary))
        atomic_replace(replay_temporary, replay)
    spec = get_method(cfg.method)
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA,
        "method": spec.name,
        "variant_id": spec.variant_id,
        "algorithm": tc.algorithm,
        "backend": cfg.backend,
        "action_mode": cfg.action_mode,
        "observation_mode": cfg.observation_mode,
        "observation_schema": "fleet-entities-v1"
        if cfg.observation_mode == "entities"
        else "compact-bins-v1",
        "metrics_schema_version": 2,
        "action_schema": {
            "discrete": "profiles-6-v1",
            "continuous": "weights-5-v1",
            "multiagent": "robot-candidates-64-v1",
        }[cfg.action_mode],
        "encoder": tc.encoder,
        "advanced_charging": cfg.advanced_charging,
        "training_seed": tc.seed,
        "num_timesteps": int(model.num_timesteps),
        "training_updates": int(model._n_updates),
        "env": asdict(cfg),
        "train": asdict(tc),
        "source_hash": source_tree_hash(),
        "lineage": lineage,
        "replay_file": replay.name if replay else None,
        "replay_sha256": file_sha256(replay) if replay else None,
        "training_plan": getattr(model, "_fleetrl_training_plan", None),
        "payload_sha256": _payload_hash(temporary),
        "validation_rank": getattr(model, "_fleetrl_validation_rank", None),
        "exploration_schedule_steps": getattr(getattr(model, "tc", None), "total_timesteps", None),
        "resume_scope": "policy/optimizer/counters and replay or tabular exploration; starts fresh simulator episodes",
    }
    # ZIP comment is read by SB3, torch and our tabular loader without affecting
    # their payloads. Metadata + replay reference commit atomically with the model.
    comment = CHECKPOINT_COMMENT + json.dumps(metadata, allow_nan=False).encode("utf-8")
    if len(comment) > 65535:
        raise ValueError("checkpoint metadata exceeds ZIP comment capacity")
    with zipfile.ZipFile(temporary, "a") as archive:
        archive.comment = comment
    atomic_replace(temporary, path)
    metadata["checkpoint_sha256"] = file_sha256(path)
    write_json(str(path) + ".json", metadata)
    if (
        previous
        and previous.get("replay_file")
        and previous["replay_file"] != metadata["replay_file"]
    ):
        old = path.parent / previous["replay_file"]
        if old.parent == path.parent and old.name.startswith(path.stem + "."):
            try:
                old.unlink(missing_ok=True)
            except PermissionError:
                pass  # Harmless stale generation; never break a committed save.
    return str(path.resolve())


def checkpoint_metadata(path):
    """Embedded metadata is authoritative; sidecars also support old v2 files."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        comment = archive.comment
    if comment.startswith(CHECKPOINT_COMMENT):
        metadata = json.loads(comment[len(CHECKPOINT_COMMENT) :])
        if _payload_hash(path) != metadata["payload_sha256"]:
            raise ValueError("checkpoint payload digest mismatch")
        metadata["checkpoint_sha256"] = file_sha256(path)
        return metadata
    sidecar = Path(str(path) + ".json")
    if not sidecar.exists():
        return None
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if file_sha256(path) != metadata["checkpoint_sha256"]:
        raise ValueError("checkpoint digest mismatch")
    return metadata


def load_checkpoint(path, device="cpu", env=None, resume=False, expected_method=None):
    path = Path(path)
    metadata = checkpoint_metadata(path)
    if metadata is None:
        from stable_baselines3 import PPO

        if resume:
            raise ValueError("resume requires FleetRL checkpoint metadata")
        with zipfile.ZipFile(path) as archive:
            if "data" not in archive.namelist() or "clip_range" not in json.loads(
                archive.read("data")
            ):
                raise ValueError("missing FleetRL metadata; unidentified legacy algorithm")
        if expected_method and get_method(expected_method).name not in {
            "ppo_cpsat",
            "ppo_threshold_cpsat",
        }:
            raise ValueError("legacy PPO checkpoint cannot identify this method")
        model = PPO.load(path, device=device, env=env)
    else:
        if metadata.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint schema mismatch")
        spec = get_method(metadata["method"])
        if expected_method and spec.name != get_method(expected_method).name:
            raise ValueError("checkpoint method mismatch")
        kwargs = {"device": device}
        if env is not None and spec.algorithm not in {"qlearning", "mappo"}:
            kwargs["env"] = env
        model = spec.module().load(str(path), **kwargs)
        if resume and spec.algorithm in OFF_POLICY:
            replay = path.parent / metadata["replay_file"]
            if replay.parent != path.parent:
                raise ValueError("replay must be in the checkpoint directory")
            if file_sha256(replay) != metadata["replay_sha256"]:
                raise ValueError("replay digest mismatch")
            model.load_replay_buffer(str(replay), truncate_last_traj=True)
        model._fleetrl_metadata = metadata
        model._fleetrl_training_plan = metadata.get("training_plan")
        model._fleetrl_validation_rank = metadata.get("validation_rank")
    finite_parameters(model)
    model._fleetrl_checkpoint_path = str(path.resolve())
    return model


def replay_bytes(env, tc):
    # DictReplayBuffer holds both observations plus action/reward/done/timeout.
    obs = sum(
        int(np.prod(s.shape)) * np.dtype(s.dtype).itemsize
        for s in env.observation_space.spaces.values()
    )
    return int(tc.buffer_size * (2 * obs + np.prod(env.action_space.shape or (1,)) * 8 + 12))


def training_stages(cfg, tc, metadata=None):
    """Keep absolute curriculum boundaries and exploration horizon across resume."""
    from ..rl import curriculum_stages

    unit = (
        1
        if tc.algorithm == "qlearning"
        else (tc.train_freq * tc.n_envs if tc.algorithm in OFF_POLICY else tc.n_steps * tc.n_envs)
    )

    def rounded(n):
        return ((int(n) + unit - 1) // unit) * unit

    initial = int((metadata or {}).get("num_timesteps", 0))
    original = (metadata or {}).get("training_plan")
    if original:
        plan = copy.deepcopy(original)
    else:
        plan = {"stages": [], "schedule_steps": initial + tc.total_timesteps}
        cursor = initial
        for name, env_cfg, budget in (
            curriculum_stages(cfg, tc) if not metadata else [("target", cfg, tc.total_timesteps)]
        ):
            end = cursor + rounded(budget)
            plan["stages"].append(
                {"name": name, "env": asdict(env_cfg), "start": cursor, "end": end}
            )
            cursor = end
        # Warmup rounding must not lengthen the global exploration horizon.
        # Allocate the remainder to the final stage, with at least one rollout.
        plan["stages"][-1]["end"] = max(
            plan["stages"][-1]["start"] + unit, initial + rounded(tc.total_timesteps)
        )
        plan["schedule_steps"] = plan["stages"][-1]["end"]
    requested_end = initial + tc.total_timesteps
    tail = plan["stages"][-1]["end"]
    if requested_end > tail:
        plan["stages"].append(
            {
                "name": "target",
                "env": asdict(cfg),
                "start": tail,
                "end": tail + rounded(requested_end - tail),
            }
        )
        # Extending a finished run preserves its original exploration schedule.
    stages = []
    for index, stage in enumerate(plan["stages"]):
        start = max(initial, stage["start"])
        end = min(requested_end, stage["end"])
        if end > start:
            stages.append(
                (index, stage["name"], FleetConfig(**stage["env"]).validate(), end - start)
            )
    return plan, stages


def train_method(
    cfg: FleetConfig,
    tc: TrainConfig,
    output: str | Path,
    resume: str | Path | None = None,
    *,
    target_total_timesteps: int | None = None,
    resume_source_hash: str | None = None,
) -> dict:
    from ..env import FleetEnv
    from ..experiments.resources import ResourceSampler
    from ..metrics import training_metrics
    from ..rl import (
        SeededTrainingEnv,
        curriculum_variants,
        make_training_env,
        validation_rank,
    )

    spec = get_method(
        cfg.method or ("ppo_cpsat" if cfg.advanced_charging else "ppo_threshold_cpsat")
    )
    if not spec.learnable:
        raise ValueError(f"{spec.name} has no learned policy; use study tune/eval")
    cfg = cfg.copy(
        method=spec.name,
        backend=spec.backend,
        controller=spec.controller,
        advanced_charging=spec.advanced_charging,
    )
    if cfg.observation_mode == "compact":
        tc = replace(tc, encoder="mlp")
    tc = replace(tc, algorithm=spec.algorithm).validate()
    output = Path(output)
    if not tc.eval_seeds or len(tc.eval_seeds) != len(set(tc.eval_seeds)):
        raise ValueError("validation seeds must be nonempty and unique")
    if min(tc.eval_freq, tc.checkpoint_freq, tc.torch_threads) < 1:
        raise ValueError("frequencies and torch_threads must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("training output must be new/empty; resume into a new directory")
    output.mkdir(parents=True, exist_ok=True)
    save_config(output / "config.yaml", cfg, tc)
    torch.set_num_threads(tc.torch_threads)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sampler = ResourceSampler().start()
    env = None
    model = None
    timings = {"validation_s": 0.0, "checkpoint_s": 0.0, "learning_s": 0.0}
    evals = []
    best = None
    best_step = None
    manifest = {
        "schema_version": 2,
        "status": "running",
        "method": spec.name,
        "variant_id": spec.variant_id,
        "training_seed": tc.seed,
        "requested_timesteps": tc.total_timesteps,
        "runtime": runtime_manifest(),
        "source_hash": source_tree_hash(),
        "validation_seeds": list(tc.eval_seeds),
        "resume": str(resume) if resume else None,
    }
    write_json(output / "manifest.json", manifest)
    initial_steps = 0
    initial_updates = 0
    next_eval = 0
    next_save = 0
    last_eval = None
    can_checkpoint = False

    def save(name):
        begin = time.perf_counter()
        path = checkpoint(output / name, model, cfg, tc, lineage=str(resume) if resume else None)
        timings["checkpoint_s"] += time.perf_counter() - begin
        return path

    def validate():
        nonlocal best, best_step, last_eval
        identity = (model.num_timesteps, model._n_updates)
        if identity == last_eval:
            return
        begin = time.perf_counter()
        episodes = [run_episode(cfg, spec.name, int(seed), model) for seed in tc.eval_seeds]
        if any(e["status"] != "complete" for e in episodes):
            raise RuntimeError("validation requires complete episodes")
        records = [episode["metrics"] for episode in episodes]
        rank = validation_rank(records)
        model._fleetrl_validation_rank = rank
        evals.append(
            {
                "steps": model.num_timesteps,
                "updates": model._n_updates,
                "parameter_sha256": parameter_digest(model),
                "rank": rank,
                "records": records,
                "elapsed_wall_s": time.perf_counter() - started,
            }
        )
        timings["validation_s"] += time.perf_counter() - begin
        last_eval = identity
        if best is None or rank >= best:
            best = rank
            best_step = model.num_timesteps
            save("best_model.zip")
        write_json(output / "validation.json", evals)

    def progress(current, info=None):
        nonlocal next_eval, next_save
        finite_parameters(current)
        if current.num_timesteps >= next_eval:
            validate()
            next_eval = current.num_timesteps + tc.eval_freq
        if current.num_timesteps >= next_save:
            save("latest_model.zip")
            next_save = current.num_timesteps + tc.checkpoint_freq

    class Callback(BaseCallback):
        def _on_training_start(self):
            # Off-policy collectors use this for epsilon/progress. The learn loop
            # retains its local stage stop count, while exploration stays global.
            self.model._total_timesteps = plan["schedule_steps"]

        def _on_step(self):
            return True

        def _on_rollout_start(self):
            progress(self.model)

        def _on_training_end(self):
            progress(self.model)

    try:
        prior = checkpoint_metadata(resume) if resume else None
        remaining = (
            tc.total_timesteps
            if target_total_timesteps is None
            else max(0, target_total_timesteps - int((prior or {}).get("num_timesteps", 0)))
        )
        plan, stages = training_stages(cfg, replace(tc, total_timesteps=max(1, remaining)), prior)
        if remaining == 0:
            stages = [(len(plan["stages"]) - 1, "finalize", cfg, 0)]
        manifest.update(
            requested_timesteps=remaining, target_total_timesteps=target_total_timesteps
        )
        manifest["training_plan"] = plan
        manifest["stages"] = []
        for stage_index, name, stage_cfg, budget in stages:
            if tc.algorithm in {"qlearning", "mappo"}:
                env = SeededTrainingEnv(
                    FleetEnv(stage_cfg),
                    tc.seed + stage_index * 100003,
                    tc.train_seed_min,
                    tc.train_seed_max,
                    variant_configs=curriculum_variants(stage_cfg, tc) if name == "target" else [],
                )
            else:
                env = make_training_env(
                    stage_cfg,
                    tc,
                    output / f"workers_{stage_index}",
                    stage_index,
                    mix_curriculum=name == "target",
                )
            if tc.algorithm in OFF_POLICY:
                estimate = replay_bytes(env, tc)
                manifest["estimated_replay_bytes"] = estimate
                if estimate > tc.max_replay_memory_mb * 1024**2:
                    raise ValueError("replay exceeds max_replay_memory_mb; reduce buffer_size")
            if model is None:
                model = (
                    load_checkpoint(resume, tc.device, env, True, spec.name)
                    if resume
                    else spec.module().build(env, tc)
                )
                if resume:
                    meta = getattr(model, "_fleetrl_metadata", {})
                    if meta and meta["training_seed"] != tc.seed:
                        raise ValueError("resume must preserve training seed and lineage")
                    allowed = {
                        "total_timesteps",
                        "eval_freq",
                        "checkpoint_freq",
                        "tensorboard",
                        "torch_threads",
                        "device",
                    }
                    for key, value in asdict(tc).items():
                        if (
                            meta
                            and key not in allowed
                            and json.dumps(meta["train"].get(key), sort_keys=True)
                            != json.dumps(value, sort_keys=True)
                        ):
                            raise ValueError(f"resume configuration differs: train.{key}")
                    for key, value in [
                        ("action_mode", cfg.action_mode),
                        ("observation_mode", cfg.observation_mode),
                        ("encoder", tc.encoder),
                        ("advanced_charging", cfg.advanced_charging),
                    ]:
                        if key in meta and meta[key] != value:
                            raise ValueError(f"incompatible resume {key}")
                    if meta and meta.get("source_hash") != source_tree_hash():
                        if not resume_source_hash or meta.get("source_hash") != resume_source_hash:
                            raise ValueError("resume source differs; use a new training run")
                        manifest["source_migration"] = {
                            "from_source_hash": resume_source_hash,
                            "to_source_hash": source_tree_hash(),
                        }
                    for key, value in asdict(cfg).items():
                        if meta and json.dumps(meta["env"].get(key), sort_keys=True) != json.dumps(
                            value, sort_keys=True
                        ):
                            raise ValueError(f"resume configuration differs: env.{key}")
                initial_steps = model.num_timesteps
                initial_updates = model._n_updates
                can_checkpoint = True
                model._fleetrl_training_plan = plan
                manifest["initial_parameter_sha256"] = parameter_digest(model)
                next_eval = initial_steps + tc.eval_freq
                next_save = initial_steps + tc.checkpoint_freq
                if hasattr(model, "set_logger"):
                    formats = ["csv"] + (["tensorboard"] if tc.tensorboard else [])
                    model.set_logger(configure(str(output / "logs"), formats))
                if resume:
                    # Retain the best validated policy across interruptions, even
                    # when the latest policy is worse. Loading it must not change
                    # the RNG used by the continuing learner.
                    parent_best = Path(resume).parent / "best_model.zip"
                    if parent_best.exists():
                        best_meta = checkpoint_metadata(parent_best)
                        rank = (best_meta or {}).get("validation_rank")
                        if (
                            rank is not None
                            and best_meta["method"] == spec.name
                            and best_meta["num_timesteps"] <= initial_steps
                        ):
                            import random

                            rng = (
                                random.getstate(),
                                np.random.get_state(),
                                torch.get_rng_state(),
                                torch.cuda.get_rng_state_all()
                                if torch.cuda.is_available()
                                else None,
                            )
                            try:
                                inherited = load_checkpoint(
                                    parent_best, tc.device, resume=True, expected_method=spec.name
                                )
                                checkpoint(
                                    output / "best_model.zip",
                                    inherited,
                                    cfg,
                                    tc,
                                    lineage=str(parent_best),
                                )
                                best = tuple(rank)
                                best_step = best_meta["num_timesteps"]
                            finally:
                                random.setstate(rng[0])
                                np.random.set_state(rng[1])
                                torch.set_rng_state(rng[2])
                                if rng[3] is not None:
                                    torch.cuda.set_rng_state_all(rng[3])
                validate()
            elif hasattr(model, "set_env"):
                model.set_env(env)
            begin = time.perf_counter()
            before = model.num_timesteps
            overhead_before = timings["validation_s"] + timings["checkpoint_s"]
            if budget:
                if hasattr(model, "learn_custom"):
                    model.learn_custom(env, budget, progress)
                else:
                    model.learn(
                        total_timesteps=budget,
                        callback=Callback(),
                        reset_num_timesteps=False,
                        progress_bar=False,
                    )
            timings["learning_s"] += (
                time.perf_counter()
                - begin
                - (timings["validation_s"] + timings["checkpoint_s"] - overhead_before)
            )
            manifest["stages"].append(
                {
                    "name": name,
                    "robots": stage_cfg.n_robots,
                    "requested_steps": budget,
                    "actual_steps": model.num_timesteps - before,
                }
            )
            validate()
            env.close()
            env = None
        finite_parameters(model)
        validate()
        final = save("final_model.zip")
        if model._n_updates <= initial_updates and remaining > 0:
            raise RuntimeError("training completed without an optimizer update")
        if model._n_updates <= 0:
            raise RuntimeError("checkpoint has no optimizer updates")
        manifest.update(
            status="complete",
            final_model=final,
            best_model=str((output / "best_model.zip").resolve()),
            best_step=best_step,
            actual_total_timesteps=model.num_timesteps,
            actual_additional_timesteps=model.num_timesteps - initial_steps,
            training_updates=model._n_updates,
            additional_training_updates=model._n_updates - initial_updates,
            final_parameter_sha256=parameter_digest(model),
        )
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        if model is not None and can_checkpoint:
            try:
                manifest["interrupted_model"] = save("interrupted_model.zip")
            except Exception as save_error:
                manifest["checkpoint_error"] = str(save_error)
        raise
    finally:
        if env is not None:
            env.close()
        samples = sampler.stop()
        wall = time.perf_counter() - started
        manifest["resources"] = training_metrics(
            model, wall, timings, samples, initial_steps, initial_updates
        )
        write_json(output / "resource_samples.json", samples)
        write_json(output / "manifest.json", manifest)
    return manifest


def parameter_digest(model):
    digest = hashlib.sha256()
    if hasattr(model, "q"):
        for k in sorted(model.q):
            digest.update(repr(k).encode())
            digest.update(model.q[k].tobytes())
    else:
        for p in model.policy.parameters():
            digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
