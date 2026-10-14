"""Freeze baseline/current packages and change one mechanism per diagnostic copy."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

VARIANTS = (
    "baseline",
    "parking_fix",
    "no_cost_correction",
    "no_idle_return",
    "no_residence",
    "no_benefit_guard",
    "no_goal_reservation",
    "no_recovery_stability",
)


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def function_text(source: str, name: str) -> str:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "".join(source.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])
    raise ValueError(f"missing function {name}")


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"ambiguous ablation fragment: {old[:80]!r}")
    return source.replace(old, new, 1)


def prepare(repo: Path, output: Path) -> dict[str, dict[str, str]]:
    if output.exists():
        raise ValueError("source snapshot directory already exists")
    files = sorted((repo / "src/fleetrl").rglob("*.py"))
    tracked = (
        subprocess.check_output(["git", "ls-files", "src/fleetrl"], cwd=repo).decode().splitlines()
    )
    baseline = {
        str(Path(name).relative_to("src/fleetrl")).replace("\\", "/"): subprocess.check_output(
            ["git", "show", f"HEAD:{name}"], cwd=repo
        )
        for name in tracked
        if name.endswith(".py")
    }
    current = {
        path.relative_to(repo / "src/fleetrl").as_posix(): path.read_bytes() for path in files
    }
    manifest = {}
    for variant in VARIANTS:
        contents = dict(baseline if variant == "baseline" else current)
        optimizer = contents["optimizer.py"].decode()
        simulator = contents["simulator.py"].decode()
        if variant == "no_cost_correction":
            optimizer = replace_once(
                optimizer,
                function_text(optimizer, "_candidate_cost"),
                function_text(baseline["optimizer.py"].decode(), "_candidate_cost"),
            )
        elif variant == "no_idle_return":
            simulator = replace_once(
                simulator,
                '            elif (\n                decision.kind == "wait"\n'
                "                and robot.position not in self.warehouse.parking\n"
                "                and self._booking(robot) is None\n            ):\n"
                '                self._park_idle_blocker(robot, robot.id, reason="idle_parking")\n',
                "",
            )
        elif variant == "no_residence":
            config = contents["config.py"].decode()
            contents["config.py"] = replace_once(
                config,
                "parking_min_residence_s: float = 60.0",
                "parking_min_residence_s: float = 0.0",
            ).encode()
        elif variant == "no_benefit_guard":
            parking = contents["parking.py"].decode()
            start = parking.index("    demand = [")
            contents["parking.py"] = (parking[:start] + "    return None\n").encode()
        elif variant == "no_goal_reservation":
            optimizer = replace_once(
                optimizer,
                "        positions = {r.position for r in snapshot.robots} | {\n"
                "            r.target_position for r in snapshot.robots if r.target_position is not None\n"
                "        }",
                "        positions = {r.position for r in snapshot.robots}",
            )
        elif variant == "no_recovery_stability":
            simulator = replace_once(
                simulator,
                '                robot.metadata["recovery_goal"] = (\n'
                '                    (bay, "reposition")\n'
                '                    if robot.status == "reposition" and robot.task_id is None\n'
                "                    else (robot.target_position, robot.status)\n                )",
                '                robot.metadata["recovery_goal"] = (robot.target_position, robot.status)',
            )
        contents["optimizer.py"], contents["simulator.py"] = optimizer.encode(), simulator.encode()
        root = output / variant / "src/fleetrl"
        for name, value in contents.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
        manifest[variant] = {"package": str(root), "source_hash": source_hash(root)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
