import hashlib
import json

import pytest

from fleetrl.config import FleetConfig
from fleetrl.demo3d.contract import frame_to_viewer, map_to_viewer
from fleetrl.demo3d.session import discover_methods
from fleetrl.env import FleetEnv
from fleetrl.evaluation import method_config
from fleetrl.scenario import scenario_config


def test_live_contract_uses_every_physical_tick() -> None:
    config = method_config(
        scenario_config("S2", FleetConfig(horizon_s=5.0, disturbances=False)), "heuristic"
    )
    env = FleetEnv(config)
    env.reset(seed=2000)
    frames = []
    env.set_tick_observer(
        lambda: frames.append(frame_to_viewer(env.sim.snapshot(), env.metrics(), env.config))
    )

    _, _, terminated, truncated, _ = env.step(0)

    assert terminated is True
    assert truncated is False
    assert [frame["t"] for frame in frames] == pytest.approx(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    )
    assert all(len(frame["robots"]) == 15 for frame in frames)
    assert all(0 <= robot["soc"] <= 1 for robot in frames[-1]["robots"])
    assert all(frame["observed_at"] <= frame["t"] for frame in frames)
    env.close()


def test_viewer_map_preserves_map_b_layout() -> None:
    config = scenario_config("S9", FleetConfig(horizon_s=5.0))
    env = FleetEnv(config)
    env.reset(seed=2000)

    result = map_to_viewer(env.sim.warehouse)

    assert result["name"] == "B"
    assert result["width"] == 40
    assert result["height"] == 30
    assert len(result["ports"]) == 4
    assert result["ports"][0]["cell"][1] == 14
    env.close()


def test_checkpoint_discovery_rejects_digest_mismatch(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "best_model.zip"
    checkpoint.write_bytes(b"checkpoint")
    lock = tmp_path / "frozen_checkpoint_dqn_cpsat_11.json"
    lock.write_text(json.dumps({"path": str(checkpoint), "sha256": "0" * 64}), encoding="utf-8")

    methods = discover_methods(tmp_path)

    assert methods["heuristic"].available is True
    assert methods["dqn_cpsat"].available is False
    assert methods["dqn_cpsat"].reason == "Checksum checkpoint không khớp"

    lock.write_text(
        json.dumps(
            {
                "path": str(checkpoint),
                "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "fleetrl.demo3d.session.checkpoint_metadata",
        lambda _: {
            "method": "dqn_cpsat",
            "observation_mode": "vector",
            "action_mode": "joint",
            "env": {"charge_threshold": 0.2},
        },
    )
    assert discover_methods(tmp_path)["dqn_cpsat"].available is True


def test_checkpoint_discovery_rejects_wrong_method_metadata(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "best_model.zip"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "frozen_checkpoint_mappo_dispatch_11.json").write_text(
        json.dumps(
            {
                "path": str(checkpoint),
                "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "fleetrl.demo3d.session.checkpoint_metadata",
        lambda _: {
            "method": "dqn_cpsat",
            "observation_mode": "vector",
            "action_mode": "joint",
            "env": {"charge_threshold": 0.2},
        },
    )

    availability = discover_methods(tmp_path)["mappo_dispatch"]

    assert availability.available is False
    assert availability.reason == "Metadata checkpoint không khớp phương pháp"
