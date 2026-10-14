"""Saved playback must remain bounded, persistent and separate from live simulation."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from fleetrl.config import FleetConfig
from fleetrl.demo3d.recordings import RecordingLibrary
from fleetrl.demo3d.session import DemoSession, MethodAvailability, discover_methods


def write_recording(root: Path, status: str | None = "complete") -> Path:
    folder = root / "saved01"
    folder.mkdir()
    header = {
        "type": "header",
        "schema": "fleetrl-viewer/1",
        "map": {},
        "session": {"method": "heuristic", "scenario": "S4", "seed": 1000},
    }
    lines = [header, *({"type": "frame", "t": i * 0.5} for i in range(1000))]
    (folder / "replay.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    if status:
        (folder / "recording.json").write_text(json.dumps({"status": status}))
    return folder


def test_saved_recording_survives_restart_and_seeks_bounded_windows(tmp_path: Path) -> None:
    write_recording(tmp_path)
    library = RecordingLibrary(tmp_path)
    assert library.list()[0]["status"] == "complete"
    page = library.window("saved01", 400.2, 20)
    assert len(page["frames"]) == 20
    assert page["frames"][0]["t"] == 400
    assert page["end_s"] == 499.5
    assert page["total_frames"] == 1000
    assert page["has_more"]
    assert RecordingLibrary(tmp_path).window("saved01", 499.5)["has_more"] is False


def test_active_recording_is_not_playable(tmp_path: Path) -> None:
    write_recording(tmp_path, "running")
    library = RecordingLibrary(tmp_path)
    assert library.list() == []
    with pytest.raises(ValueError, match="chưa kết thúc"):
        library.window("saved01")


def test_bundled_gzip_is_verified_cached_and_playable_after_restart(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    folder = write_recording(samples)
    raw = (folder / "replay.jsonl").read_bytes()
    compressed = gzip.compress(raw, mtime=0)
    (folder / "replay.jsonl.gz").write_bytes(compressed)
    (folder / "replay.jsonl").unlink()
    provenance = {
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "gzip_sha256": hashlib.sha256(compressed).hexdigest(),
    }
    (folder / "provenance.json").write_text(json.dumps(provenance))
    output = tmp_path / "output"
    output.mkdir()
    library = RecordingLibrary(output, samples)
    assert library.list()[0]["bytes"] == len(compressed)
    assert not (output / "_replay_cache").exists()
    assert library.window("saved01", 400)["frames"][0]["t"] == 400
    assert len(list((output / "_replay_cache").glob("*.jsonl"))) == 1
    assert RecordingLibrary(output, samples).window("saved01", 499.5)["has_more"] is False
    provenance["gzip_sha256"] = "0" * 64
    (folder / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="Checksum"):
        RecordingLibrary(output, samples).window("saved01")
    provenance["gzip_sha256"] = hashlib.sha256(compressed).hexdigest()
    (folder / "provenance.json").write_text(json.dumps(provenance))
    next((output / "_replay_cache").glob("*.jsonl")).write_bytes(b"broken")
    with pytest.raises(ValueError, match="Cache replay hỏng"):
        RecordingLibrary(output, samples).window("saved01")


def test_relative_model_lock_is_anchored_to_its_folder(tmp_path: Path, monkeypatch) -> None:
    study = tmp_path / "models"
    checkpoint = study / "dqn_cpsat/seed11/best_model.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    (study / "frozen_checkpoint_dqn_cpsat_11.json").write_text(
        json.dumps(
            {
                "path": "dqn_cpsat/seed11/best_model.zip",
                "sha256": hashlib.sha256(b"model").hexdigest(),
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "fleetrl.demo3d.session.checkpoint_metadata",
        lambda _: {
            "method": "dqn_cpsat",
            "observation_mode": "entities",
            "action_mode": "discrete",
            "env": {"charge_threshold": 0.2},
        },
    )
    methods = discover_methods(study)
    assert methods["dqn_cpsat"].checkpoint == str(checkpoint.resolve())
    assert methods["heuristic"].available


@pytest.mark.parametrize("recording_id", ["../saved01", "C:/file", "", "saved01/extra"])
def test_recording_cannot_escape_output_root(tmp_path: Path, recording_id: str) -> None:
    write_recording(tmp_path)
    with pytest.raises(ValueError):
        RecordingLibrary(tmp_path).window(recording_id)


def test_legacy_recording_is_labelled_and_bad_limits_rejected(tmp_path: Path) -> None:
    write_recording(tmp_path, None)
    library = RecordingLibrary(tmp_path)
    assert library.list()[0]["status"] == "legacy"
    for limit in [0, 1, 241]:
        with pytest.raises(ValueError):
            library.window("saved01", limit=limit)
    with pytest.raises(ValueError):
        library.window("saved01", start=float("nan"))


def test_record_only_finishes_short_episode_without_wall_clock_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fleetrl.demo3d.session.load_config",
        lambda _: (FleetConfig(horizon_s=5, disturbances=False), {}),
    )
    waits = []
    monkeypatch.setattr("fleetrl.demo3d.session.time.sleep", waits.append)
    session = DemoSession(
        "heuristic",
        "S2",
        2000,
        tmp_path,
        tmp_path,
        MethodAvailability("heuristic", True, None),
        record_only=True,
    )
    session._run()
    assert session.status()["status"] == "complete", session.status()["error"]
    assert not waits
    assert json.loads((session.output_dir / "recording.json").read_text())["status"] == "complete"
    page = RecordingLibrary(tmp_path).window(session.id)
    assert [f["t"] for f in page["frames"]] == [i * 0.5 for i in range(11)]
    session.record_only = False
    session._wait_for_tick(0.1)
    assert waits and waits[-1] > 0
