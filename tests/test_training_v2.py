"""Regression coverage for checkpoint consistency and post-update validation."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from fleetrl.config import FleetConfig, TrainConfig
from fleetrl.learning.training import (
    checkpoint,
    checkpoint_metadata,
    load_checkpoint,
    parameter_digest,
    train_method,
    training_stages,
)
from fleetrl.methods.registry import get_method


@pytest.fixture(params=["ppo_cpsat", "dqn_cpsat", "mappo_dispatch"])
def trained(request, tmp_path):
    spec = get_method(request.param)
    cfg = spec.configure(
        FleetConfig(n_robots=2, horizon_s=10, initial_tasks=2, disturbances=False, service_s=2)
    )
    tc = TrainConfig(
        algorithm=spec.algorithm,
        total_timesteps=16,
        n_envs=1,
        n_steps=8,
        batch_size=4,
        n_epochs=2,
        learning_starts=2,
        buffer_size=32,
        train_freq=4,
        eval_freq=16,
        checkpoint_freq=16,
        eval_seeds=(1000,),
        curriculum=False,
        tensorboard=False,
    )
    result = train_method(cfg, tc, tmp_path / "train")
    return cfg, tc, result


def test_final_validation_and_periodic_checkpoint_include_last_update(trained):
    cfg, tc, result = trained
    directory = Path(result["final_model"]).parent
    final = load_checkpoint(result["final_model"])
    latest = load_checkpoint(directory / "latest_model.zip", resume=True)
    evaluations = json.loads((directory / "validation.json").read_text())
    assert evaluations[-1]["steps"] == final.num_timesteps
    assert evaluations[-1]["updates"] == final._n_updates
    assert evaluations[-1]["parameter_sha256"] == parameter_digest(final)
    assert parameter_digest(latest) == parameter_digest(final)
    if tc.algorithm == "dqn":
        assert latest.replay_buffer.size() == final.num_timesteps


def test_embedded_metadata_survives_missing_or_stale_sidecar(trained, monkeypatch):
    import fleetrl.learning.training as training

    cfg, tc, result = trained
    path = Path(result["final_model"])
    model = load_checkpoint(path, resume=True)
    old_steps = model.num_timesteps
    model.num_timesteps += 4
    original = training.write_json

    def fail_sidecar(destination, data):
        if str(destination) == str(path) + ".json":
            raise OSError("simulated interruption after ZIP commit")
        return original(destination, data)

    monkeypatch.setattr(training, "write_json", fail_sidecar)
    with pytest.raises(OSError):
        checkpoint(path, model, cfg, tc)
    restored = load_checkpoint(path, resume=True)
    assert restored.num_timesteps == old_steps + 4
    assert restored._fleetrl_metadata["num_timesteps"] == old_steps + 4
    Path(str(path) + ".json").unlink()
    assert load_checkpoint(path, expected_method=cfg.method).num_timesteps == old_steps + 4


def test_failed_zip_commit_preserves_model_and_replay(trained, monkeypatch):
    import fleetrl.artifacts as artifacts

    cfg, tc, result = trained
    path = Path(result["final_model"])
    old = checkpoint_metadata(path)
    model = load_checkpoint(path, resume=True)
    model.num_timesteps += 4
    original = artifacts.atomic_replace

    def fail_commit(source, target):
        if Path(target) == path:
            raise OSError("simulated interruption before ZIP commit")
        return original(source, target)

    monkeypatch.setattr(artifacts, "atomic_replace", fail_commit)
    with pytest.raises(OSError):
        checkpoint(path, model, cfg, tc)
    restored = load_checkpoint(path, resume=True)
    assert restored.num_timesteps == old["num_timesteps"]
    assert restored._fleetrl_metadata["replay_sha256"] == old["replay_sha256"]


def test_resume_rejects_changed_reward(trained, tmp_path):
    cfg, tc, result = trained
    with pytest.raises(ValueError, match="env.reward_backlog"):
        train_method(cfg.copy(reward_backlog=0.7), tc, tmp_path / "changed", result["final_model"])


def test_resume_accepts_recorded_source_migration(trained, tmp_path, monkeypatch):
    import fleetrl.learning.training as training

    cfg, tc, result = trained
    original_hash = checkpoint_metadata(result["final_model"])["source_hash"]
    monkeypatch.setattr(training, "source_tree_hash", lambda: "patched")
    with pytest.raises(ValueError, match="resume source differs"):
        train_method(
            cfg,
            tc,
            tmp_path / "blocked",
            result["final_model"],
            target_total_timesteps=16,
        )

    resumed = train_method(
        cfg,
        tc,
        tmp_path / "migrated",
        result["final_model"],
        target_total_timesteps=16,
        resume_source_hash=original_hash,
    )
    manifest = json.loads((tmp_path / "migrated" / "manifest.json").read_text())
    assert resumed["status"] == "complete"
    assert resumed["actual_additional_timesteps"] == 0
    assert manifest["source_migration"] == {
        "from_source_hash": original_hash,
        "to_source_hash": "patched",
    }


def test_completed_budget_can_be_finalized_without_extra_updates(trained, tmp_path):
    cfg, tc, result = trained
    finalized = train_method(
        cfg, tc, tmp_path / "finalize", result["final_model"], target_total_timesteps=16
    )
    assert finalized["status"] == "complete"
    assert finalized["actual_additional_timesteps"] == 0
    assert finalized["additional_training_updates"] == 0
    assert finalized["final_parameter_sha256"] == result["final_parameter_sha256"]


def test_checkpoint_rejects_modified_payload(trained):
    import zipfile

    cfg, tc, result = trained
    path = Path(result["final_model"])
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("unexpected-payload", b"changed")
    with pytest.raises(ValueError, match="payload digest mismatch"):
        load_checkpoint(path)


def test_resume_preserves_better_parent_checkpoint(trained, tmp_path, monkeypatch):
    cfg, tc, result = trained
    parent = Path(result["best_model"])
    best = load_checkpoint(parent, resume=True)
    best._fleetrl_validation_rank = (0.0, 1000000.0, 0.0, 0.0)
    checkpoint(parent, best, cfg, tc)
    resumed = train_method(cfg, tc, tmp_path / "resume", result["final_model"])
    inherited = load_checkpoint(resumed["best_model"])
    assert inherited.num_timesteps == best.num_timesteps
    assert parameter_digest(inherited) == parameter_digest(best)


def test_resume_keeps_curriculum_phase_and_original_exploration_horizon():
    cfg = FleetConfig(n_robots=15)
    tc = TrainConfig(algorithm="dqn", total_timesteps=1000, n_envs=1, n_steps=8, train_freq=4)
    plan, stages = training_stages(cfg, tc)
    metadata = {"num_timesteps": 40, "training_plan": plan}
    resumed_plan, resumed = training_stages(cfg, replace(tc, total_timesteps=960), metadata)
    assert resumed_plan == plan
    assert resumed[0][1] == "warmup" and resumed[0][2].n_robots == 5
    assert resumed[-1][1] == "target"
    assert resumed_plan["schedule_steps"] == plan["schedule_steps"]
    extended, work = training_stages(
        cfg,
        replace(tc, total_timesteps=100),
        {"num_timesteps": plan["stages"][-1]["end"], "training_plan": plan},
    )
    assert all(stage[1] == "target" for stage in work)
    assert extended["schedule_steps"] == plan["schedule_steps"]


def test_curriculum_rounding_does_not_extend_exploration_past_actual_budget():
    cfg = FleetConfig(n_robots=10)
    tc = TrainConfig(algorithm="dqn", total_timesteps=64, n_envs=1, n_steps=4, train_freq=4)
    plan, stages = training_stages(cfg, tc)
    assert sum(stage[3] for stage in stages) == 64
    assert plan["schedule_steps"] == 64
    assert plan["stages"][-1]["end"] == 64
