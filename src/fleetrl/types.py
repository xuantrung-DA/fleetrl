"""Shared engine/optimizer contracts. Time is seconds, energy Wh, space 1 m cells."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Khai báo vị trí (x,y) trên grid
Cell = tuple[int, int]


@dataclass
class Robot:
    # Thông tin cơ bản của robot
    id: int
    kind: str
    position: Cell
    capacity_wh: float
    battery_wh: float  # Current battery energy in Watt-hours
    speed_mps: float
    payload_capacity_kg: float  # Maximum payload capacity in kilograms
    movement_wh_m: float  # Energy consumption per meter of movement
    # Trạng thái hiện tại của robot
    status: str = "idle"
    task_id: int | None = None
    load_kg: float = 0.0
    port_id: int | None = None  # ID of the charging port if the robot is charging
    target_soc: float | None = (
        None  # Target state of charge (SOC) for charging, as a fraction of capacity
    )
    paused_until_s: float = 0.0  # Time until which the robot is paused, in seconds
    observed_at_s: float = 0.0
    target_position: Cell | None = None
    path: list[Cell] = field(default_factory=list)
    move_to: Cell | None = None
    move_remaining_s: float = 0.0
    service_remaining_s: float = 0.0
    blocked_since_s: float | None = None  # Time when the robot became blocked, in seconds
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def soc(self) -> float:  # Chuyển pin Wh thành SOC 0 - 1
        return self.battery_wh / self.capacity_wh


@dataclass
class Task:
    id: int
    pickup: Cell  # Điểm lấy (x,y)
    dropoff: Cell  # Điểm thả (x,y)
    weight_kg: float
    created_s: float
    deadline_s: float
    priority: int = 1
    zone: int = 0
    status: str = "waiting"
    robot_id: int | None = None
    assigned_at_s: float | None = None
    picked_at_s: float | None = None
    completed_at_s: float | None = None
    blocked_reason: str | None = None


@dataclass
class Port:
    id: int
    station_id: int
    position: Cell
    power_w: float = 600.0
    compatible_kinds: tuple[str, ...] = ("L", "M", "H")  # Nhận dạng robot loại nào
    queue_cells: tuple[Cell, ...] = ()


@dataclass
class ChargeBooking:
    robot_id: int
    port_id: int
    arrival_s: float
    start_s: float
    end_s: float
    target_soc: float
    requested_at_s: float = 0.0
    actual_start_s: float | None = None
    actual_end_s: float | None = None
    cancelled_at_s: float | None = None


@dataclass
class GridMap:
    width: int
    height: int
    walls: set[Cell]  # Các ô bị chặn (x,y)
    pickups: tuple[Cell, ...]
    dropoffs: tuple[Cell, ...]
    parking: tuple[Cell, ...]
    ports: tuple[Port, ...]
    name: str = "A"

    def is_free(self, cell: Cell, blocked: set[Cell] | None = None) -> bool:
        # Kiểm tra xem ô có nằm trong lưới, không phải tường và không bị chặn
        return (
            0 <= cell[0] < self.width
            and 0 <= cell[1] < self.height
            and cell not in self.walls
            and (blocked is None or cell not in blocked)
        )

    def neighbors(self, cell: Cell, blocked: set[Cell] | None = None) -> list[Cell]:
        # Lấy các ô lân cận (trái, phải, trên, dưới) của ô hiện tại mà không bị chặn
        # Không cho đi chéo
        x, y = cell
        return [
            p for p in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)) if self.is_free(p, blocked)
        ]

    def zone_of(self, cell: Cell) -> int:
        # Xác định khu vực của ô dựa trên vị trí của nó trong lưới
        # Khu vực được xác định bằng cách chia lưới thành 4 phần
        return int(cell[0] >= self.width / 2) + 2 * int(cell[1] >= self.height / 2)


@dataclass(frozen=True)
class Snapshot:  # Trạng thái tại 1 thời điểm gửi cho controller
    sim_time_s: float
    observed_at_s: float
    version: int
    robots: tuple[Robot, ...]
    tasks: tuple[Task, ...]
    ports: tuple[Port, ...]
    bookings: tuple[ChargeBooking, ...]
    warehouse: GridMap
    blocked_cells: frozenset[Cell] = frozenset()
    demand_rate_by_zone: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    map_version: int = 0

    @property
    def state_hash(self) -> str:
        # Stable JSON representation, deliberately excludes future event tape.
        body = {
            "t": self.sim_time_s,
            "v": self.version,
            "robots": [asdict(r) for r in self.robots],
            "tasks": [asdict(t) for t in self.tasks],
            "bookings": [asdict(b) for b in self.bookings],
            "blocked": sorted(self.blocked_cells),
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:20]


@dataclass
class Decision:
    # Thông tin quyết định cho robot
    robot_id: int
    kind: str  # task | charge | reposition | wait
    task_id: int | None = None
    port_id: int | None = None
    target_soc: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    arrival_s: float | None = None
    target_position: Cell | None = None
    estimated_energy_wh: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionPlan:
    # Kế hoạch quyết định cho tất cả robot tại một thời điểm
    snapshot_version: int
    profile_id: int | None
    decisions: list[Decision]
    solver_status: str = "UNSOLVED"
    solver_ms: float = 0.0
    total_ms: float = 0.0
    fallback_reason: str | None = None
    objective: float | None = None
    candidate_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioEvent:
    time_s: float
    kind: str  # task | block | unblock | pause
    payload: dict[str, Any]


@dataclass
class EventTape:
    seed: int
    events: list[ScenarioEvent]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EventTape":
        return cls(
            int(value["seed"]),
            [ScenarioEvent(**e) for e in value["events"]],
            value.get("metadata", {}),
        )

    @property
    def tape_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()
