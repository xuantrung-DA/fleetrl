"""Translate authoritative FleetRL state into the read-only 3D viewer contract."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..energy import travel_time
from ..evaluation import serializable
from ..types import GridMap, Snapshot


def map_to_viewer(warehouse: GridMap) -> dict[str, Any]:
    value = serializable(warehouse)
    queue_cells: dict[str, list[list[int]]] = {}
    ports = []
    for port in value["ports"]:
        station = int(port["station_id"])
        queue_cells[str(station)] = port["queue_cells"]
        ports.append(
            {
                "id": port["id"],
                "station": station,
                "cell": port["position"],
                "power_w": port["power_w"],
                "compatible": port["compatible_kinds"],
            }
        )
    return {
        "name": value["name"],
        "width": value["width"],
        "height": value["height"],
        "walls": value["walls"],
        "pickups": value["pickups"],
        "dropoffs": value["dropoffs"],
        "parking": value["parking"],
        "ports": ports,
        "queue_cells": queue_cells,
        "blocked_edges": [],
    }


def _decision_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    plan = record.get("plan", record)
    return {
        "profile": record.get("profile"),
        "solver_status": plan.get("solver_status", "—"),
        "fallback_reason": plan.get("fallback_reason") or record.get("policy_fallback"),
        "total_ms": plan.get("total_ms", record.get("inference_ms", 0.0)),
        "time_s": record.get("time_s"),
    }


def frame_to_viewer(
    snapshot: Snapshot,
    metrics: dict[str, Any],
    config: Any,
    decision: dict[str, Any] | None = None,
    *,
    modified: bool = False,
) -> dict[str, Any]:
    """Build one frame without calculating planner, energy, reward, or KPI values."""
    robots = []
    for source in snapshot.robots:
        robot = asdict(source)
        converted = {
            **serializable(robot),
            "type": source.kind,
            "cell": serializable(source.position),
            "soc": source.soc,
            "route": serializable(source.path),
        }
        if source.move_to is not None:
            converted["edge_duration_s"] = travel_time(source, 1.0, config)
        robots.append(converted)

    tasks = [
        {
            **serializable(asdict(task)),
            "kg": task.weight_kg,
            "created": task.created_s,
            "deadline": task.deadline_s,
            "state": task.status,
        }
        for task in snapshot.tasks
        if task.created_s <= snapshot.observed_at_s + 1e-9
    ]
    ports = []
    for port in snapshot.ports:
        occupants = [robot for robot in snapshot.robots if robot.position == port.position]
        occupant = occupants[0] if occupants else None
        bookings = [
            {
                **serializable(asdict(booking)),
                "start": booking.start_s,
                "end": booking.end_s,
                "known_at": booking.requested_at_s,
            }
            for booking in snapshot.bookings
            if booking.port_id == port.id
            and booking.requested_at_s <= snapshot.observed_at_s + 1e-9
        ]
        ports.append(
            {
                "id": port.id,
                "occupied_by": occupant.id if occupant else None,
                "charging_robot": (
                    occupant.id if occupant is not None and occupant.status == "charging" else None
                ),
                "queue_robot_ids": [
                    robot.id
                    for robot in snapshot.robots
                    if robot.position in port.queue_cells and robot.status == "waiting_charge"
                ],
                "bookings": bookings,
            }
        )

    kpi = serializable(metrics)
    kpi["fallback_count"] = sum(kpi.get("fallback_reasons", {}).values())
    return {
        "type": "frame",
        "t": snapshot.sim_time_s,
        "observed_at": snapshot.observed_at_s,
        "map_version": snapshot.map_version,
        "state_hash": snapshot.state_hash,
        "robots": robots,
        "tasks": tasks,
        "ports": ports,
        "blocked_cells": serializable(snapshot.blocked_cells),
        "blocked_edges": [],
        "kpi": kpi,
        "decision": _decision_record(decision),
        "modified": modified,
    }
