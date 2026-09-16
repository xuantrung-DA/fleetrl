import json
from pathlib import Path

import pytest

from fleetrl.experiments.protocol import Study
from fleetrl.experiments.runner import run_study, study_plan


def test_expected_study_counts_and_seed_split():
    plan = study_plan(Study().validate())
    assert plan["training_jobs"] == 33
    assert plan["requested_training_steps"] == 9900000
    assert plan["core_test_episodes"] == 1620
    assert plan["matched_control_episodes"] == 360
    with pytest.raises(ValueError):
        Study(tape_seeds=(10,)).validate()


def test_atomic_replace_retries_transient_windows_lock(tmp_path, monkeypatch):
    import fleetrl.artifacts as artifacts
    from fleetrl.artifacts import atomic_replace

    source = tmp_path / "pending"
    target = tmp_path / "final"
    source.write_text("new")
    target.write_text("old")
    original = Path.replace
    attempts = []

    def locked(path, destination):
        attempts.append(1)
        if len(attempts) < 8:
            raise PermissionError("transient sharing lock")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(artifacts.time, "sleep", lambda _: None)
    atomic_replace(source, target)
    assert target.read_text() == "new" and len(attempts) == 8


def test_resume_checks_artifacts_and_keeps_missing_cells(tmp_path):
    path = tmp_path / "study.yaml"
    path.write_text(
        "study:\n  methods: [heuristic]\n  tape_seeds: [2000, 2001]\n  validation_seeds: [1000]\n  training_seeds: [11]\n  scenarios: [S1]\n  horizon_s: 5\n  matched_ood: false\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    first = run_study(path, output, "test", limit=1)
    assert (
        first["status"] == "partial"
        and first["complete_episodes"] == 1
        and first["expected_episodes"] == 2
    )
    second = run_study(path, output, "test")
    assert second["status"] == "complete" and second["complete_episodes"] == 2
    state = next((output / "jobs").glob("*/job.json"))
    assert json.loads(state.read_text())["attempt"] == 1
    result = Path(json.loads(state.read_text())["output"]) / "result.json"
    result.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="modified artifacts"):
        run_study(path, output, "test")
    with pytest.raises(ValueError, match="modified artifacts"):
        run_study(path, output, "report")


def test_test_phase_freezes_baseline_defaults_and_blocks_late_tuning(tmp_path):
    path = tmp_path / "study.yaml"
    path.write_text(
        "study:\n  methods: [heuristic]\n  scenarios: [S1]\n  tape_seeds: [2000]\n  horizon_s: 5\n  matched_ood: false\n"
    )
    output = tmp_path / "study"
    run_study(path, output, "test")
    with pytest.raises(ValueError, match="frozen"):
        run_study(path, output, "tune")
    (output / "selected_baselines.json").write_text(
        json.dumps({"selected": {"heuristic": {"charge_threshold": 0.4}}})
    )
    with pytest.raises(ValueError, match="selection changed"):
        run_study(path, output, "report")


def test_partial_result_cannot_be_marked_complete(tmp_path):
    from fleetrl.experiments.runner import _job

    def execute(run, previous):
        return {"status": "partial", "metrics": {"safety_incidents": 0}}, []

    assert _job(tmp_path, "partial", execute, {"remaining": 1}) is None
    job = json.loads((tmp_path / "jobs/partial/job.json").read_text())
    assert job["status"] == "failed" and "incomplete result" in job["error"]


def test_cli_returns_failure_for_unlimited_incomplete_study(monkeypatch):
    import fleetrl.experiments.runner as runner
    from fleetrl.cli import main

    monkeypatch.setattr(
        runner, "run_study", lambda *a, **k: {"status": "partial", "failed_jobs": 0}
    )
    assert main(["study", "test"]) == 1
    assert main(["study", "test", "--limit", "1"]) == 0
