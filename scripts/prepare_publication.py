"""Export an explicit publication allowlist without changing original run artifacts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "runs/publish_bugfix_20261014"
GROUPS = {
    "idle_parking_20261013": "REPORT.md DIAGNOSIS_BEFORE.md comparison_default.csv "
    "comparison_controlled.csv acceptance.json walltime_variation.json",
    "dispatch_regression_20261013": "REPORT.md DIAGNOSIS_DISPATCH.md "
    "DIAGNOSIS_CHARGE_HANDOFF.md protocol.json aggregate.csv paired_vs_baseline.csv "
    "ablation_vs_parking_fix.csv acceptance.json completion_audit.json "
    "default_validation/aggregate.csv default_validation/paired_vs_baseline.csv",
    "charging_access_20261013": "CHARGING_FIX.md QUALITY_FINAL.json SOURCE_REVIEW.json",
    "paused_candidate_20261013": "REPORT.md FIX_VERIFICATION.json QUALITY_FINAL.json",
    "stale_fallback_20261013": "REPORT.md FIX_VERIFICATION.json QUALITY_FINAL.json",
    "endpoint_wait_fix_20261013": "FIX_REPORT.md BEFORE_AFTER_CI95.json "
    "before_after_S4_validation.csv QUALITY_FINAL.json SOURCE_REVIEW.json "
    "VALIDATION_SUMMARY.json pytest_full.xml controlled_validation1000/plan.json "
    "controlled_validation1000/result.json controlled_validation1000/map.json",
    "endpoint_wait_validation_20261013": "controlled_validation1000/plan.json "
    "controlled_validation1000/result.json controlled_validation1000/map.json",
    "readiness_nonlearning_fresh_20261013": "FINAL_REPORT.md nonlearning_aggregate.csv "
    "nonlearning_metrics.csv nonlearning_paired_CI95.csv COMPLETION_AUDIT.json "
    "PHASE_COMPLETE.json traffic_light.csv STOP_REQUIRED_first_completion.json "
    "S6_CALENDAR_WITNESS.json",
    "study14_current_source_testonly": "README.md identity.json status.json episodes.csv "
    "summary.json run.py",
    "demo3d_recorded_validation": "REPORT.md pytest_final.xml api_evidence.json",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def sanitize_text(value: str) -> str:
    def clean_path(match: re.Match[str]) -> str:
        path = match.group().replace("\\", "/")
        if path.lower().startswith(ROOT.as_posix().lower() + "/"):
            return path[len(ROOT.as_posix()) + 1 :]
        if re.fullmatch(r"C:/Users/[^/]+/\.conda/envs/[^/]+/python.exe", path, re.IGNORECASE):
            return "python"
        if re.match(r"C:/Users/[^/]+/AppData/Local/Temp/", path, re.IGNORECASE):
            return "<temporary-path>"
        return path

    value = re.sub(r"[A-Za-z]:[/\\][^\r\n<>`\"']+", clean_path, value)
    value = re.sub(r'hostname="[^"]*"', 'hostname="anonymous-host"', value)
    return value


def sanitize(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: "anonymous-host" if key == "hostname" else sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return sanitize_text(value) if isinstance(value, str) else value


def payload_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            digest.update(entry.filename.encode())
            digest.update(archive.read(entry))
    return digest.hexdigest()


def export_models() -> None:
    evidence = []
    for method, attempt in (("mappo_dispatch", "002"), ("dqn_cpsat", "001")):
        source = ROOT / f"runs/study14/jobs/train_{method}_11/attempt_{attempt}/best_model.zip"
        target = ROOT / f"demo3d/models/{method}/seed11/best_model.zip"
        target.parent.mkdir(parents=True, exist_ok=True)
        original_sha = sha(source)
        lock = ROOT / f"runs/study14/frozen_checkpoint_{method}_11.json"
        if json.loads(lock.read_text(encoding="utf-8"))["sha256"] != original_sha:
            raise ValueError("Original checkpoint does not match frozen lock")
        original_payload = payload_sha(source)
        shutil.copyfile(source, target)
        prefix = b"FleetRL checkpoint v2\n"
        with zipfile.ZipFile(target, "a") as archive:
            if not archive.comment.startswith(prefix):
                raise ValueError(f"Missing checkpoint metadata: {source}")
            metadata = json.loads(archive.comment[len(prefix) :])
            clean = sanitize(metadata)
            if clean != metadata:
                archive.comment = prefix + json.dumps(clean, ensure_ascii=False).encode()
        if payload_sha(target) != original_payload:
            raise ValueError("Export changed model payload")
        exported_sha = sha(target)
        sidecar = sanitize(json.loads(source.with_suffix(".zip.json").read_text(encoding="utf-8")))
        sidecar["checkpoint_sha256"] = exported_sha
        write_json(target.with_suffix(".zip.json"), sidecar)
        write_json(
            ROOT / f"demo3d/models/frozen_checkpoint_{method}_11.json",
            {
                "path": f"{method}/seed11/best_model.zip",
                "sha256": exported_sha,
            },
        )
        evidence.append(
            {
                "method": method,
                "train_seed": 11,
                "source": source.relative_to(ROOT).as_posix(),
                "export": target.relative_to(ROOT).as_posix(),
                "original_sha256": original_sha,
                "export_sha256": exported_sha,
                "payload_sha256": original_payload,
                "payload_unchanged": True,
                "normalization": "Repository-relative lineage paths; sidecar and lock export hash",
            }
        )
        if sha(source) != original_sha:
            raise ValueError("Original checkpoint changed")
    write_json(ROOT / "demo3d/models/provenance.json", evidence)


def export_replay() -> None:
    source = ROOT / "runs/demo3d/748f38c95f4e/replay.jsonl"
    folder = ROOT / "demo3d/samples/mappo_s4_1001_full"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "replay.jsonl.gz"
    raw_hash = sha(source)
    with source.open("rb") as stream, target.open("wb") as output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9
        ) as zipped:
            shutil.copyfileobj(stream, zipped, length=1024 * 1024)
    status = json.loads(source.with_name("recording.json").read_text(encoding="utf-8"))
    if status["status"] != "complete" or status["time_s"] != 3600:
        raise ValueError("Expected completed full replay")
    write_json(folder / "recording.json", status)
    with source.open("rb") as stream:
        header = json.loads(stream.readline())
        frames = 0
        last = -1.0
        for line in stream:
            frame = json.loads(line)
            if frame["type"] != "frame" or frame["t"] <= last:
                raise ValueError("Invalid replay frame")
            frames += 1
            last = frame["t"]
    if frames != 7201 or last != 3600:
        raise ValueError("Expected all 7201 frames of the 3600-second replay")
    with gzip.open(target, "rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != raw_hash:
            raise ValueError("Replay compression altered bytes")
    write_json(
        folder / "provenance.json",
        {
            "source": source.relative_to(ROOT).as_posix(),
            "source_session_id": status["id"],
            "method": status["method"],
            "scenario": status["scenario"],
            "seed": status["seed"],
            "status": status["status"],
            "modified": status["modified"],
            "duration_s": last,
            "frames": frames,
            "source_hash": header["source_hash"],
            "raw_bytes": source.stat().st_size,
            "gzip_bytes": target.stat().st_size,
            "raw_sha256": raw_hash,
            "gzip_sha256": sha(target),
            "normalization": "None; gzip contains every original byte, including historical header",
        },
    )
    if sha(source) != raw_hash:
        raise ValueError("Original replay changed")


def export_reports() -> None:
    manifest = []
    names = [f"{group}/{name}" for group, files in GROUPS.items() for name in files.split()]
    for name in names:
        source = ROOT / "runs" / name
        target = EXPORT / name
        original_sha = sha(source)
        text = source.read_text(encoding="utf-8")
        if source.suffix == ".json":
            result = json.dumps(sanitize(json.loads(text)), ensure_ascii=False, indent=2) + "\n"
        elif source.suffix == ".csv":
            output = io.StringIO(newline="")
            writer = csv.writer(output, lineterminator="\n")
            writer.writerows(
                [[sanitize_text(cell) for cell in row] for row in csv.reader(io.StringIO(text))]
            )
            result = output.getvalue()
        else:
            result = sanitize_text(text)
            if source.suffix == ".md":

                def relative_link(match: re.Match[str]) -> str:
                    ref = match.group(1)
                    relative = ref.removeprefix("runs/")
                    destination = EXPORT / relative if relative in names else ROOT / ref
                    return (
                        "](<"
                        + os.path.relpath(destination, target.parent).replace("\\", "/")
                        + ">)"
                    )

                result = re.sub(r"\]\(<(runs/[^>]+)>\)", relative_link, result)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result, encoding="utf-8", newline="\n")
        manifest.append(
            {
                "source": source.relative_to(ROOT).as_posix(),
                "export": target.relative_to(ROOT).as_posix(),
                "original_sha256": original_sha,
                "export_sha256": sha(target),
                "bytes": target.stat().st_size,
                "normalization": "LF; repository-relative personal paths; anonymous hostname; "
                "JSON/CSV serialization; exported Markdown links where available",
            }
        )
        if sha(source) != original_sha:
            raise ValueError("Original report changed")
    write_json(EXPORT / "manifest.json", manifest)


def main() -> None:
    export_models()
    export_replay()
    export_reports()
    print("Exported 2 models, 1 full gzip replay and selected reports; originals unchanged")


if __name__ == "__main__":
    main()
