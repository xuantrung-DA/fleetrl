"""Short clean-package demo checks; never train or execute a study."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import threading
from pathlib import Path
from urllib.request import urlopen

import fleetrl
from fleetrl.demo3d import session as sessions
from fleetrl.demo3d.recordings import RecordingLibrary
from fleetrl.demo3d.server import DemoServer
from fleetrl.evaluation import source_tree_hash


def main() -> None:
    root = Path.cwd().resolve()
    if root not in Path(fleetrl.__file__).resolve().parents:
        raise ValueError("Package imports source outside the clean directory")
    for name in ("numpy", "torch", "gymnasium", "stable_baselines3", "ortools"):
        specification = importlib.util.find_spec(name)
        if specification is None or root not in Path(specification.origin).resolve().parents:
            raise ValueError(f"Dependency imported outside the clean venv: {name}")
    output = root / "runs/portability_smoke"
    manager = sessions.SessionManager(root / "demo3d/models", output, "S4", 1001)
    if not all(method.available for method in manager.methods.values()):
        raise ValueError(str(manager.methods))
    if not sessions.discover_methods(root / "absent-models")["heuristic"].available:
        raise ValueError("Heuristic must not require a model")
    load_config = sessions.load_config

    def short_config(path: Path) -> tuple:
        config, extra = load_config(path)
        return config.copy(horizon_s=5, disturbances=False), extra

    results = []
    sessions.load_config = short_config
    try:
        for method in ("heuristic", "mappo_dispatch", "dqn_cpsat"):
            demo = sessions.DemoSession(
                method,
                "S4",
                1001,
                manager.study,
                output,
                manager.methods[method],
                record_only=True,
            )
            demo._run()
            status = demo.status()
            if status["status"] != "complete":
                raise ValueError(str(status))
            page = RecordingLibrary(output).window(demo.id)
            if len(page["frames"]) != 11 or page["end_s"] != 5:
                raise ValueError("Short recording lost physical ticks")
            decisions = demo._env.decision_logs
            reasons = [decision.get("fallback_reason") for decision in decisions]
            if any("policy_error:" in (reason or "") for reason in reasons):
                raise ValueError(str(reasons))
            results.append(
                {
                    "method": method,
                    "status": "complete",
                    "duration_s": 5,
                    "frames": 11,
                    "fallback_reasons": reasons,
                }
            )
        live = sessions.DemoSession(
            "heuristic",
            "S4",
            1001,
            manager.study,
            output,
            manager.methods["heuristic"],
        )
        live._run()
        if live.status()["status"] != "complete" or live.record_only:
            raise ValueError("Heuristic LIVE failed")
    finally:
        sessions.load_config = load_config

    web = root / "demo3d/web/dist"
    api_results = []
    for restart in range(2):
        server = DemoServer(("127.0.0.1", 0), manager, web)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(base, timeout=60) as response:
                if b"<html" not in response.read():
                    raise ValueError("Built frontend not served")
            with urlopen(base + "/api/recordings", timeout=60) as response:
                listing = json.load(response)
            sample = next(row for row in listing if row["id"] == "mappo_s4_1001_full")
            if sample["status"] != "complete":
                raise ValueError("Sample completion metadata missing")
            for start in (0, 1800, 3599):
                url = base + f"/api/recording-window?id=mappo_s4_1001_full&start={start}"
                with urlopen(url, timeout=120) as response:
                    page = json.load(response)
                if page["total_frames"] != 7201 or page["end_s"] != 3600:
                    raise ValueError("Full sample truncated")
                if len(page["frames"]) > 120 or page["frames"][0]["t"] != start:
                    raise ValueError("Window seek not bounded or incorrect")
                api_results.append(
                    {
                        "restart": restart,
                        "start_s": start,
                        "frames": len(page["frames"]),
                        "end_s": page["end_s"],
                    }
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    evidence = {
        "source_hash": source_tree_hash(),
        "clean_import": "src/fleetrl/__init__.py",
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "gymnasium", "stable-baselines3", "ortools", "pytest")
        },
        "models": {key: value.available for key, value in manager.methods.items()},
        "record_only_smoke": results,
        "heuristic_live_duration_s": 5,
        "http_checks": api_results,
        "limits": "No training, study rerun or FPS benchmark; full replay is historical",
    }
    (root / "runs/portability.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
