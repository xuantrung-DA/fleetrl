"""Remove unreferenced latest checkpoints from completed study jobs."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def log(root: Path, message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with (root / "cleanup.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def latest_bundle(job_path: Path) -> list[Path]:
    job = read_json(job_path)
    if not job or job.get("status") != "complete":
        return []
    output = job.get("output")
    if not isinstance(output, str):
        return []
    run = Path(output).resolve()
    if run.parent != job_path.parent.resolve() or not re.fullmatch(r"attempt_\d+", run.name):
        return []

    sidecar = run / "latest_model.zip.json"
    metadata = read_json(sidecar)
    if not metadata:
        return []
    candidates = [run / "latest_model.zip", sidecar]
    replay_name = metadata.get("replay_file")
    if replay_name:
        if not isinstance(replay_name, str) or not re.fullmatch(
            r"latest_model\.[0-9a-f]{32}\.replay\.pkl", replay_name
        ):
            return []
        candidates.append(run / replay_name)

    protected = {Path(path).resolve() for path in job.get("artifact_hashes", {})}
    for name in ("best_model.zip.json", "final_model.zip.json"):
        saved = read_json(run / name)
        if saved and saved.get("replay_file"):
            protected.add((run / saved["replay_file"]).resolve())
    if any(path.resolve().parent != run or path.resolve() in protected for path in candidates):
        return []
    return candidates


def clean_once(root: Path) -> tuple[int, int]:
    removed = 0
    bytes_freed = 0
    for prefix in ("search", "train"):
        for job_path in (root / "jobs").glob(f"{prefix}_*/job.json"):
            try:
                candidates = latest_bundle(job_path)
                if not candidates:
                    continue
                for path in candidates:
                    if path.is_file():
                        size = path.stat().st_size
                        path.unlink()
                        removed += 1
                        bytes_freed += size
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                log(root, f"Skipped {job_path.parent.name}: {exc}")
    return removed, bytes_freed


def parent_alive(parent: psutil.Process | None, started: float | None) -> bool:
    if parent is None:
        return False
    try:
        return parent.is_running() and parent.create_time() == started
    except psutil.NoSuchProcess:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/study14"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be positive")
    root = args.output.resolve()
    if not (root / "jobs").is_dir():
        parser.error(f"Study jobs directory does not exist: {root / 'jobs'}")
    parent = None
    if args.parent_pid is not None:
        try:
            parent = psutil.Process(args.parent_pid)
            parent_started = parent.create_time()
        except psutil.NoSuchProcess:
            parent = None
            parent_started = None

    while True:
        removed, bytes_freed = clean_once(root)
        if removed:
            log(root, f"Removed {removed} temporary checkpoint files; freed {bytes_freed / 2**30:.2f} GiB")
        if not args.watch:
            return 0
        if args.parent_pid is not None and not parent_alive(parent, parent_started):
            return 0
        summary = read_json(root / "summary.json")
        if summary and summary.get("status") == "complete":
            return 0
        print(
            f"[CLEANUP] Active at {datetime.now():%H:%M:%S}; "
            f"checking completed jobs again in {args.interval}s.",
            flush=True,
        )
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
