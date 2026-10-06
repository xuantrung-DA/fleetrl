"""FleetRL -> viewer adapter. Không tính planner, energy, reward hay KPI.

CLI: python export_fleetrl_replay.py raw.jsonl -o replay.json --map map.json
API: snapshot_to_frame(snapshot, kpi=metrics_record, decision=decision_plan)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any


def plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [plain(v) for v in value]
    return value


def required(d: dict, *names: str):
    for name in names:
        if name in d:
            return d[name]
    raise ValueError(f"Thiếu trường bắt buộc {names}")


def map_to_viewer(warehouse: Any) -> dict:
    m = plain(warehouse)
    queues = m.get("queue_cells", {})
    ports = []
    for p in m["ports"]:
        station = required(p, "station", "station_id")
        if "queue_cells" in p:
            queues[str(station)] = p["queue_cells"]
        ports.append({"id": p["id"], "station": station,
                      "cell": required(p, "cell", "position"), "power_w": p["power_w"],
                      "compatible": required(p, "compatible", "compatible_kinds")})
    return {**m, "ports": ports, "queue_cells": queues, "blocked_edges": m.get("blocked_edges", [])}


def snapshot_to_frame(snapshot: Any, *, kpi=None, decision=None, extras=None) -> dict:
    """Snapshot dataclass hoặc dict. Mọi extras phải là dữ liệu đã biết tại t.

    KPI nhận nguyên bản từ metrics.py; thiếu thì {}. Không lấy final episode metrics.
    Snapshot.ports là static: occupancy chỉ suy từ position thực + status charging.
    Không suy thời gian sạc thực từ booking.start_s/end_s (đó chỉ là kế hoạch).
    """
    state_hash = getattr(snapshot, "state_hash", None)
    source = plain(snapshot)
    t = required(source, "t", "sim_time_s")
    robots = []
    for r in source["robots"]:
        capacity = r.get("capacity_wh")
        soc = r.get("soc")
        if soc is None:
            if not capacity or capacity <= 0:
                raise ValueError("Robot cần soc hoặc battery_wh/capacity_wh hợp lệ")
            soc = r["battery_wh"] / capacity  # Chỉ đổi đơn vị, không mô phỏng pin.
        robot = {**r, "type": required(r, "type", "kind"),
                 "cell": required(r, "cell", "position"), "soc": soc,
                 "status": r["status"], "load_kg": r["load_kg"],
                 "route": r.get("route", r.get("path", [])), "task_id": r.get("task_id")}
        if r.get("move_to") is not None:
            duration = r.get("edge_duration_s")
            if duration is None:
                speed = r.get("speed_mps")
                if not speed or speed <= 0:
                    raise ValueError("move_to cần edge_duration_s hoặc speed_mps")
                duration = 1.0 / speed  # 1 m / vận tốc cấu hình; không tạo cạnh mới.
            robot["edge_duration_s"] = duration
            robot["move_remaining_s"] = r["move_remaining_s"]
        robots.append(robot)
    tasks = []
    for task in source["tasks"]:
        created = required(task, "created", "created_s")
        if created > t:
            raise ValueError("Snapshot chứa task tương lai; không xuất event tape")
        tasks.append({**task, "created": created, "deadline": required(task, "deadline", "deadline_s"),
                      "kg": required(task, "kg", "weight_kg"),
                      "state": required(task, "state", "status")})
    ports = []
    for p in source["ports"]:
        if "occupied_by" in p:
            ports.append(p)
            continue
        position = required(p, "cell", "position")
        occupants = [r for r in robots if r["cell"] == position]
        if len(occupants) > 1:
            raise ValueError("Nhiều robot cùng cổng")
        bookings = []
        for b in source.get("bookings", []):
            if b["port_id"] != p["id"]:
                continue
            known = b.get("requested_at_s", t)
            if known > t:
                raise ValueError("Booking chưa được quan sát")
            bookings.append({**b, "start": required(b, "start", "start_s"),
                             "end": required(b, "end", "end_s"), "known_at": known})
        occupant = occupants[0] if occupants else None
        queue = [r["id"] for r in robots if r["cell"] in p.get("queue_cells", [])
                 and r["status"] == "waiting_charge"]
        ports.append({"id": p["id"], "occupied_by": occupant["id"] if occupant else None,
                      "charging_robot": occupant["id"] if occupant and occupant["status"] == "charging" else None,
                      "bookings": bookings, "queue_robot_ids": queue})
    decision = plain(decision if decision is not None else source.get("decision"))
    if decision:
        decision = {**decision, "profile": decision.get("profile", decision.get("profile_id"))}
    frame = {"t": t, "observed_at": required(source, "observed_at", "observed_at_s"),
             "map_version": source.get("map_version", 0),
             "state_hash": state_hash or source.get("state_hash"), "robots": robots,
             "tasks": tasks, "ports": ports, "blocked_cells": source.get("blocked_cells", []),
             "blocked_edges": source.get("blocked_edges", []),
             "kpi": plain(kpi if kpi is not None else source.get("kpi", {})), "decision": decision}
    for key in ("reservations", "heatmap", "corridors"):
        if key in (extras or source):
            frame[key] = plain((extras or source)[key])
    return frame


def read_records(path: Path):
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL lỗi dòng {line_number}: {exc}") from exc
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--map", type=Path)
    args = parser.parse_args()
    try:
        data = read_records(args.input)
        records = data if isinstance(data, list) else data.get("frames", [data])
        first = records[0].get("snapshot", records[0])
        warehouse = read_records(args.map) if args.map else (
            data.get("map") if isinstance(data, dict) else None)
        warehouse = warehouse or first.get("warehouse")
        if warehouse is None:
            raise ValueError("Thiếu warehouse/map; truyền --map (save_map của FleetRL)")
        frames = []
        for i, record in enumerate(records):
            try:
                f = snapshot_to_frame(record.get("snapshot", record), kpi=record.get("kpi"),
                                      decision=record.get("decision"), extras=record)
                if frames and f["t"] <= frames[-1]["t"]:
                    raise ValueError("Thời gian không tăng nghiêm ngặt")
                frames.append(f)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Record {i}: {exc}. Cần snapshot đầy đủ; không tự đoán format snapshot_record.") from exc
        result = {"schema": "fleetrl-viewer/1", "mock": data.get("mock", False) if isinstance(data, dict) else False,
                  "map": map_to_viewer(warehouse), "frames": frames}
        args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"Đã xuất {len(frames)} frame -> {args.output}")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Lỗi chuyển replay: {exc}\n")


if __name__ == "__main__":
    main()
