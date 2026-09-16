"""Evaluation integrity tests: matched tapes, censoring, policy fallback, replay."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from fleetrl.config import FleetConfig
from fleetrl.evaluation import (
    RunRecorder,
    choose_action,
    load_replay,
    method_config,
    paired_summary,
    run_episode,
    serializable,
    snapshot_record,
)
from fleetrl.types import EventTape, GridMap, Robot, Snapshot


def small_snapshot(time_s=0.0):
    robot = Robot(0, "L", (1, 1), 100.0, 50.0, 1.0, 10.0, 0.02)
    warehouse = GridMap(8, 6, {(3, 3)}, ((1, 2),), ((6, 4),), ((1, 1),), ())
    return Snapshot(time_s, time_s, int(time_s), (robot,), (), (), (), warehouse)


def test_method_definitions_match_proposal():
    config = FleetConfig()
    assert method_config(config, "B0").controller == "heuristic"
    assert not method_config(config, "B0").advanced_charging
    assert method_config(config, "B1").advanced_charging
    assert method_config(config, "H").controller == "hybrid"
    assert not method_config(config, "A").advanced_charging
    with pytest.raises(ValueError):
        method_config(config, "random")


@pytest.mark.parametrize("action", [np.nan, np.inf, -1, 6, 0.5, [0, 1]])
def test_policy_invalid_action_falls_back_and_records_reason(action):
    policy = SimpleNamespace(predict=lambda *a, **k: (action, None))
    result, reason, elapsed = choose_action(policy, {}, FleetConfig(fixed_profile=2))
    assert result == 2 and reason.startswith("policy_error:") and elapsed >= 0


def test_policy_exception_is_logged_and_baseline_skips_policy():
    def fail(*args, **kwargs):
        raise RuntimeError("injected inference fault")

    policy = SimpleNamespace(predict=fail)
    action, reason, _ = choose_action(policy, {}, FleetConfig())
    assert action == 0 and "injected inference fault" in reason
    assert choose_action(policy, {}, FleetConfig(controller="fixed")) == (0, None, 0.0)


def test_record_replay_matches_hashes_and_sqlite_index(tmp_path):
    snap = small_snapshot()
    tape = EventTape(4, [])
    path = tmp_path / "run"
    recorder = RunRecorder(path, FleetConfig(), tape, "fixed", 4, snap.warehouse)
    recorder.record(snapshot_record(snap, {"completed": 0}))
    recorder.record(snapshot_record(small_snapshot(5), {"completed": 1}), {"profile": 2})
    recorder.close({"metrics": {"completed": 1}}, tape)
    recorder.close()  # idempotent cleanup
    header, frames = load_replay(path)
    assert header["mode"] == "REPLAY"
    assert frames[0]["state_hash"] == snap.state_hash
    assert frames[1]["decision"]["profile"] == 2
    assert frames[1]["metrics"]["completed"] == 1
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    for filename, expected in manifest["sha256"].items():
        assert hashlib.sha256((path / filename).read_bytes()).hexdigest() == expected
    with sqlite3.connect(tmp_path / "runs.sqlite3") as conn:
        assert conn.execute("SELECT method, seed, tape_hash, status FROM runs").fetchone() == (
            "fixed",
            4,
            tape.tape_hash,
            "complete",
        )


def records_for_ci():
    rows = []
    for seed in range(3):
        rows.append(
            {
                "method": "fixed",
                "scenario": "S1",
                "seed": seed,
                "tape_hash": str(seed),
                "metrics": {
                    "throughput_per_hour": 10 + seed,
                    "energy_per_completed_wh": None,
                    "pending": 5,
                },
            }
        )
        for training_seed in (11, 22, 33):
            rows.append(
                {
                    "method": "hybrid",
                    "scenario": "S1",
                    "seed": seed,
                    "tape_hash": str(seed),
                    "training_seed": training_seed,
                    "metrics": {
                        "throughput_per_hour": 12 + seed,
                        "energy_per_completed_wh": None,
                        "pending": 4,
                    },
                }
            )
    return rows


def test_paired_bootstrap_uses_matched_tape_and_training_replicates():
    report = paired_summary(records_for_ci(), bootstrap_samples=100)
    group = next(g for g in report["groups"] if g["method"] == "hybrid")
    paired = group["paired_vs_reference"]["throughput_per_hour"]
    assert paired["paired_tapes"] == 3
    assert paired["training_replicates"] == 3
    assert paired["mean_difference"] == 2
    assert paired["ci95"] == [2.0, 2.0]
    assert group["statistics"]["energy_per_completed_wh"] == {
        "n": 0,
        "missing": 9,
        "mean": None,
        "std": None,
    }
    assert group["statistics"]["pending"]["mean"] == 4


def test_unmatched_hash_and_tiny_smoke_cannot_claim_ci():
    records = records_for_ci()
    for record in records:
        if record["method"] == "hybrid":
            record["tape_hash"] = "DIFFERENT"
    group = next(g for g in paired_summary(records)["groups"] if g["method"] == "hybrid")
    assert group["paired_vs_reference"]["throughput_per_hour"]["paired_tapes"] == 0
    one_tape = [r for r in records_for_ci() if r["seed"] == 0]
    group = next(g for g in paired_summary(one_tape)["groups"] if g["method"] == "hybrid")
    assert group["paired_vs_reference"]["throughput_per_hour"]["ci95"] is None


def test_missing_crossed_checkpoint_cell_withholds_ci():
    records = records_for_ci()
    records.pop()
    group = next(g for g in paired_summary(records)["groups"] if g["method"] == "hybrid")
    pair = group["paired_vs_reference"]["throughput_per_hour"]
    assert not pair["complete_crossed_grid"]
    assert pair["ci95"] is None


def test_serialization_uses_json_null_for_nonfinite():
    value = serializable({"bad": float("nan"), "array": np.array([1, np.inf]), "set": {(1, 2)}})
    assert value == {"bad": None, "array": [1.0, None], "set": [[1, 2]]}
    json.dumps(value, allow_nan=False)


def test_run_episode_logs_inference_failure_and_preserves_tape(monkeypatch, tmp_path):
    class FakeEnv:
        def __init__(self, config):
            self.config = config
            self.sim = SimpleNamespace(
                time_s=0, tape=None, warehouse=small_snapshot().warehouse, events=[]
            )
            self.sim.snapshot = lambda: small_snapshot(self.sim.time_s)
            self.steps = 0

        def reset(self, seed, options=None):
            self.sim.tape = options["tape"] if options else EventTape(seed, [])
            return {}, {}

        def snapshot(self):
            return small_snapshot(self.sim.time_s)

        def metrics(self):
            return {
                "completed": self.steps,
                "pending": 5 - self.steps,
                "energy_per_completed_wh": None if not self.steps else 1.2,
            }

        def step(self, action):
            self.steps += 1
            self.sim.time_s += 5
            return (
                {},
                1.0,
                False,
                self.steps == 2,
                {"metrics": self.metrics(), "decision": {"solver_status": "OPTIMAL"}},
            )

        def close(self):
            pass

    module = ModuleType("fleetrl.env")
    module.FleetEnv = FakeEnv
    monkeypatch.setitem(sys.modules, "fleetrl.env", module)
    tape = EventTape(123, [])
    policy = SimpleNamespace(predict=lambda *a, **k: (np.nan, None))
    result = run_episode(FleetConfig(), "hybrid", 123, policy, tmp_path / "run", tape)
    assert result["tape_hash"] == tape.tape_hash
    assert result["metrics"]["policy_failures"] == 2
    assert result["metrics"]["episode_reward"] == 2.0
    assert result["metrics"]["pending"] == 3
    assert len(load_replay(tmp_path / "run")[1]) == 3


def test_dashboard_headless_render_and_robot_selection(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from fleetrl.ui import Dashboard

    snapshot = small_snapshot()
    dashboard = Dashboard(serializable(snapshot.warehouse))
    try:
        pixels = dashboard.draw(snapshot_record(snapshot), mode="REPLAY")
        assert pixels.shape == (dashboard.height, dashboard.width, 3)
        assert pixels.dtype == np.uint8 and pixels.std() > 10
        assert dashboard.select_at(dashboard._rect((1, 1)).center, snapshot_record(snapshot)) == 0
        dashboard.save(tmp_path / "ui.png")
        assert (tmp_path / "ui.png").stat().st_size > 1000
    finally:
        dashboard.close()


def test_evidence_refuses_overwrite(tmp_path):
    snap = small_snapshot()
    path = tmp_path / "run"
    recorder = RunRecorder(path, FleetConfig(), EventTape(1, []), "fixed", 1, snap.warehouse)
    recorder.close()
    with pytest.raises(FileExistsError):
        RunRecorder(path, FleetConfig(), EventTape(1, []), "fixed", 1, snap.warehouse)


def test_inference_latency_and_failure_enter_env_metrics():
    from fleetrl.evaluation import account_policy_latency

    record = {"total_ms": 7.0, "fallback_reason": "solver_timeout"}
    env = SimpleNamespace(config=FleetConfig(), decision_logs=[record])
    info = {"decision": record}
    account_policy_latency(env, info, 251.0, "policy_error")
    assert record["total_ms"] == 258.0 and record["budget_exceeded"]
    assert record["fallback_reason"] == "policy_error;solver_timeout"
    assert info["decision"] is record


def test_ui_injections_record_identical_tape_for_all_methods(tmp_path):
    from fleetrl.env import FleetEnv
    from fleetrl.ui import inject_demo_event

    config = FleetConfig(
        n_robots=5, horizon_s=10, initial_tasks=1, demand_per_hour=0, disturbances=False
    )
    env = FleetEnv(config)
    env.reset(seed=2000)
    original_events = len(env.sim.tape.events)
    inject_demo_event(env, "tasks")
    inject_demo_event(env, "pause", 0)
    inject_demo_event(env, "block", cell=(1, 1))
    tape = EventTape.from_dict(env.sim.tape.to_dict())
    assert len(tape.events) == original_events + 7
    assert [e.kind for e in tape.events].count("unblock") == 0  # duration belongs to block
    a = run_episode(config, "fixed", 2000, tape=tape)
    b = run_episode(config, "heuristic", 2000, tape=EventTape.from_dict(tape.to_dict()))
    assert a["tape_hash"] == b["tape_hash"] == tape.tape_hash
    assert a["metrics"]["arrived"] == b["metrics"]["arrived"] == 6
    env.close()


def test_live_ui_pause_start_and_fast_playback(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from fleetrl.ui import Dashboard, run_ui

    original = Dashboard.draw
    timestamps = []

    def draw_and_start(self, frame, *args, **kwargs):
        timestamps.append(frame["time_s"])
        pixels = original(self, frame, *args, **kwargs)
        if len(timestamps) == 2:
            self.pg.event.post(self.pg.event.Event(self.pg.KEYDOWN, key=self.pg.K_SPACE))
            for _ in range(7):
                self.pg.event.post(self.pg.event.Event(self.pg.KEYDOWN, key=self.pg.K_PLUS))
        return pixels

    monkeypatch.setattr(Dashboard, "draw", draw_and_start)
    config = FleetConfig(
        n_robots=5, horizon_s=10, initial_tasks=1, demand_per_hour=0, disturbances=False
    )
    result = run_ui(
        config, max_frames=8, output_dir=tmp_path / "ui", screenshot_path=tmp_path / "ui.png"
    )
    assert timestamps[:2] == [0, 0]
    assert result["time_s"] == 10 and result["episode_complete"]
    assert (tmp_path / "ui.png").exists()
    header, frames = load_replay(tmp_path / "ui" / "reset_000")
    assert frames[-1]["time_s"] == 10
    assert header["manifest"]["status"] == "complete"
