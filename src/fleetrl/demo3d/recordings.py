"""Read saved viewer recordings in bounded windows without running a simulator."""

from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import math
import re
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any


class RecordingLibrary:
    def __init__(self, root: Path, samples: Path | None = None) -> None:
        self.root = root.resolve()
        self.roots = [self.root]
        if samples is not None and samples.resolve() != self.root:
            self.roots.append(samples.resolve())
        self._expanded: dict[Path, tuple[int, int, Path]] = {}
        self._indexes: OrderedDict[str, tuple[dict, list[float], list[int]]] = OrderedDict()
        self._lock = threading.Lock()

    def _source(self, recording_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", recording_id):
            raise ValueError("ID bản ghi không hợp lệ")
        for root in self.roots:
            for filename in ("replay.jsonl", "replay.jsonl.gz"):
                path = (root / recording_id / filename).resolve()
                if root in path.parents and path.is_file():
                    return path
        raise ValueError("Không tìm thấy bản ghi")

    def _path(self, recording_id: str) -> Path:
        source = self._source(recording_id)
        if source.suffix != ".gz":
            return source
        stat = source.stat()
        previous = self._expanded.get(source)
        if previous and previous[:2] == (stat.st_size, stat.st_mtime_ns):
            return previous[2]
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        provenance_path = source.parent / "provenance.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.is_file()
            else {}
        )
        if provenance.get("gzip_sha256", digest) != digest:
            raise ValueError("Checksum bản ghi gzip không khớp")
        cache = (self.root / "_replay_cache").resolve()
        if self.root not in cache.parents:
            raise ValueError("Thư mục cache nằm ngoài output")
        cache.mkdir(parents=True, exist_ok=True)
        target = (cache / f"{digest}.jsonl").resolve()
        if cache not in target.parents:
            raise ValueError("File cache nằm ngoài thư mục cache")
        if not target.is_file():
            temporary = target.with_suffix(".tmp")
            try:
                with gzip.open(source, "rb") as compressed, temporary.open("wb") as expanded:
                    shutil.copyfileobj(compressed, expanded, length=1024 * 1024)
                with temporary.open("rb") as stream:
                    raw_digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if provenance.get("raw_sha256", raw_digest) != raw_digest:
                    raise ValueError("Checksum dữ liệu giải nén không khớp")
                temporary.replace(target)
            except EOFError as error:
                raise ValueError("Bản ghi gzip bị cắt ngắn") from error
            finally:
                temporary.unlink(missing_ok=True)
        elif provenance.get("raw_sha256"):
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != provenance["raw_sha256"]:
                    raise ValueError("Cache replay hỏng; dừng server và xóa cache để tạo lại")
        self._expanded[source] = stat.st_size, stat.st_mtime_ns, target
        return target

    def list(self) -> list[dict[str, Any]]:
        result = []
        folders = [
            folder
            for root in self.roots
            if root.is_dir()
            for folder in sorted(root.iterdir(), reverse=True)
        ]
        seen = set()
        for folder in folders:
            status_path = folder / "recording.json"
            if not folder.is_dir() or folder.name in seen:
                continue
            try:
                path = self._source(folder.name)
                opener = gzip.open if path.suffix == ".gz" else Path.open
                with opener(path, "rb") as stream:
                    header = json.loads(stream.readline())
                if header.get("schema") != "fleetrl-viewer/1":
                    continue
                status = (
                    json.loads(status_path.read_text(encoding="utf-8"))
                    if status_path.is_file()
                    else None
                )
                if status is not None and status.get("status") not in {"complete", "stopped"}:
                    continue
                session = header.get("session", {})
                result.append(
                    {
                        "id": folder.name,
                        "method": session.get("method", "unknown"),
                        "scenario": session.get("scenario", "unknown"),
                        "seed": session.get("seed"),
                        "status": status["status"] if status else "legacy",
                        "bytes": path.stat().st_size,
                        "source_hash": header.get("source_hash"),
                    }
                )
                seen.add(folder.name)
            except (OSError, ValueError, KeyError, EOFError):
                continue
        return result

    def _index(self, recording_id: str) -> tuple[dict, list[float], list[int]]:
        with self._lock:
            if recording_id in self._indexes:
                self._indexes.move_to_end(recording_id)
                return self._indexes[recording_id]
            path = self._path(recording_id)
            status_path = self._source(recording_id).parent / "recording.json"
            if status_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("status") not in {"complete", "stopped"}:
                    raise ValueError("Bản ghi chưa kết thúc; hãy chờ chạy xong")
            times, offsets = [], []
            with path.open("rb") as stream:
                header = json.loads(stream.readline())
                if header.get("schema") != "fleetrl-viewer/1":
                    raise ValueError("Định dạng replay không được hỗ trợ")
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    frame = json.loads(line)
                    stamp = float(frame["t"])
                    if frame.get("type") != "frame" or not math.isfinite(stamp):
                        raise ValueError("Frame replay không hợp lệ")
                    if times and stamp <= times[-1]:
                        raise ValueError("Thời gian replay không tăng")
                    times.append(stamp)
                    offsets.append(offset)
            if not times:
                raise ValueError("Replay không có frame")
            value = header, times, offsets
            self._indexes[recording_id] = value
            if len(self._indexes) > 2:
                self._indexes.popitem(last=False)
            return value

    def window(self, recording_id: str, start: float = 0, limit: int = 120) -> dict:
        if not math.isfinite(start) or not 2 <= limit <= 240:
            raise ValueError("Đoạn replay không hợp lệ (2–240 frame)")
        header, times, offsets = self._index(recording_id)
        index = max(0, bisect.bisect_right(times, start) - 1)
        stop = min(len(times), index + limit)
        frames = []
        with self._path(recording_id).open("rb") as stream:
            stream.seek(offsets[index])
            for _ in range(index, stop):
                frames.append(json.loads(stream.readline()))
        return {
            "header": header,
            "frames": frames,
            "start_s": times[0],
            "end_s": times[-1],
            "total_frames": len(times),
            "has_more": stop < len(times),
        }
