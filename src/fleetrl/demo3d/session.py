"""Threaded live-demo sessions and replay recording."""

from __future__ import annotations

import copy
import hashlib
import json
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import load_config
from ..evaluation import choose_action, method_config
from ..learning.policies import InferenceSession
from ..learning.training import checkpoint_metadata
from ..rl import load_policy
from ..scenario import scenario_config
from ..types import ScenarioEvent, Task
from .contract import frame_to_viewer, map_to_viewer

SUPPORTED_METHODS = ("heuristic", "mappo_dispatch", "dqn_cpsat")
SUPPORTED_SCENARIOS = ("S2", "S4", "S7", "S9")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class MethodAvailability:
    name: str
    available: bool
    reason: str | None = None
    checkpoint: str | None = None


def discover_methods(study: str | Path) -> dict[str, MethodAvailability]:
    root = Path(study).resolve()
    result = {
        "heuristic": MethodAvailability("heuristic", True),
    }
    for method in ("mappo_dispatch", "dqn_cpsat"):
        locks = sorted(root.glob(f"frozen_checkpoint_{method}_*.json"))
        availability = MethodAvailability(method, False, "Không có checkpoint đã khóa trong study")
        for lock in locks:
            try:
                value = json.loads(lock.read_text(encoding="utf-8"))
                checkpoint = Path(value["path"]).resolve()
                if not checkpoint.is_file():
                    continue
                if file_sha256(checkpoint) != value["sha256"]:
                    availability = MethodAvailability(
                        method, False, "Checksum checkpoint không khớp"
                    )
                    continue
                metadata = checkpoint_metadata(checkpoint)
                if metadata is None or metadata.get("method") != method:
                    availability = MethodAvailability(
                        method, False, "Metadata checkpoint không khớp phương pháp"
                    )
                    continue
                if not all(
                    (
                        metadata.get("observation_mode"),
                        metadata.get("action_mode"),
                        isinstance(metadata.get("env"), dict),
                        "charge_threshold" in metadata.get("env", {}),
                    )
                ):
                    availability = MethodAvailability(
                        method, False, "Metadata checkpoint thiếu cấu hình suy luận"
                    )
                    continue
                availability = MethodAvailability(method, True, checkpoint=str(checkpoint))
                break
            except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
                availability = MethodAvailability(method, False, f"Checkpoint lock lỗi: {error}")
        result[method] = availability
    return result


class SessionStopped(Exception):
    """Internal cooperative cancellation signal."""


class DemoSession:
    def __init__(
        self,
        method: str,
        scenario: str,
        seed: int,
        study: Path,
        output_root: Path,
        availability: MethodAvailability,
    ) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Phương pháp không hỗ trợ: {method}")
        if scenario not in SUPPORTED_SCENARIOS:
            raise ValueError(f"Kịch bản không hỗ trợ: {scenario}")
        if not availability.available:
            raise ValueError(availability.reason or "Phương pháp chưa sẵn sàng")
        self.id = uuid.uuid4().hex[:12]
        self.method = method
        self.scenario = scenario
        self.seed = int(seed)
        self.study = study
        self.output_dir = output_root / self.id
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.replay_path = self.output_dir / "replay.jsonl"
        self._availability = availability
        self._condition = threading.Condition()
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._commands: list[tuple[str, dict[str, Any]]] = []
        self._paused = False
        self._stopping = False
        self._speed = 1.0
        self._modified = False
        self._status = "starting"
        self._error: str | None = None
        self._last_frame: dict[str, Any] | None = None
        self._last_decision_time: float | None = None
        self._env: Any = None
        self._thread = threading.Thread(target=self._run, name=f"demo3d-{self.id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        channel: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=64)
        with self._condition:
            self._subscribers.append(channel)
            channel.put_nowait({"event": "status", "data": self.status()})
            map_data = self.map_data()
            if map_data is not None:
                channel.put_nowait(
                    {"event": "ready", "data": {"map": map_data, "status": self.status()}}
                )
            if self._last_frame is not None:
                channel.put_nowait({"event": "frame", "data": self._last_frame})
        return channel

    def unsubscribe(self, channel: queue.Queue[dict[str, Any]]) -> None:
        with self._condition:
            if channel in self._subscribers:
                self._subscribers.remove(channel)

    def _publish(self, event: str, data: dict[str, Any]) -> None:
        item = {"event": event, "data": data}
        with self._condition:
            for channel in tuple(self._subscribers):
                try:
                    channel.put_nowait(item)
                except queue.Full:
                    try:
                        channel.get_nowait()
                        channel.put_nowait(item)
                    except queue.Empty:
                        pass

    def status(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "method": self.method,
            "scenario": self.scenario,
            "seed": self.seed,
            "status": self._status,
            "paused": self._paused,
            "speed": self._speed,
            "modified": self._modified,
            "error": self._error,
            "replay": f"/api/replay?session={self.id}",
            "time_s": self._last_frame["t"] if self._last_frame else 0.0,
        }

    def map_data(self) -> dict[str, Any] | None:
        return map_to_viewer(self._env.sim.warehouse) if self._env is not None else None

    def control(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._condition:
            if action == "pause":
                self._paused = True
            elif action == "resume":
                self._paused = False
                self._condition.notify_all()
            elif action == "speed":
                speed = float(payload.get("speed", 1.0))
                if speed not in {0.25, 0.5, 1.0, 2.0, 4.0, 8.0}:
                    raise ValueError("Tốc độ phải là 0.25, 0.5, 1, 2, 4 hoặc 8")
                self._speed = speed
            elif action == "stop":
                self._stopping = True
                self._paused = False
                self._condition.notify_all()
            elif action == "urgent_tasks":
                if self._env is None:
                    raise ValueError("Phiên chưa sẵn sàng")
                self._commands.append((action, {}))
                self._modified = True
            elif action in {"block", "unblock"}:
                cell = self._validate_cell_event(action, payload.get("cell"))
                self._commands.append((action, {"cell": cell}))
                self._modified = True
            else:
                raise ValueError(f"Điều khiển không hỗ trợ: {action}")
        status = self.status()
        self._publish("status", status)
        return status

    def stop(self, wait: bool = True) -> None:
        self.control("stop", {})
        if wait and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _inject_urgent_tasks(self) -> None:
        if self._env is None:
            raise ValueError("Phiên chưa sẵn sàng")
        now = self._env.sim.time_s
        tape_ids = [event.payload.get("id", -1) for event in self._env.sim.tape.events]
        first_id = max([*self._env.sim.tasks.keys(), *tape_ids, -1]) + 1
        warehouse = self._env.sim.warehouse
        for index in range(5):
            pickup = warehouse.pickups[index % len(warehouse.pickups)]
            task = Task(
                first_id + index,
                pickup,
                warehouse.dropoffs[index % len(warehouse.dropoffs)],
                5.0,
                now,
                now + 300.0,
                priority=3,
                zone=warehouse.zone_of(pickup),
            )
            self._env.sim.inject_event(ScenarioEvent(now, "task", asdict(task)))

    def _validate_cell_event(self, action: str, cell_value: Any) -> tuple[int, int]:
        if self._env is None:
            raise ValueError("Phiên chưa sẵn sàng")
        if not isinstance(cell_value, list) or len(cell_value) != 2:
            raise ValueError("Cần chọn một ô [x, y]")
        cell = (int(cell_value[0]), int(cell_value[1]))
        if not self._env.sim.warehouse.is_free(cell):
            raise ValueError("Ô được chọn nằm ngoài map hoặc trong kệ")
        occupied = {robot.position for robot in self._env.sim.robots.values()} | {
            robot.move_to for robot in self._env.sim.robots.values() if robot.move_to is not None
        }
        if action == "block" and cell in occupied:
            raise ValueError("Không thể chặn ô robot đang đứng hoặc đang đi tới")
        if action == "block" and cell in self._env.sim.blocked_cells:
            raise ValueError("Ô này đã bị chặn")
        if action == "unblock" and cell not in self._env.sim.blocked_cells:
            raise ValueError("Ô này hiện không bị chặn")
        return cell

    def _apply_commands(self) -> None:
        with self._condition:
            commands = self._commands
            self._commands = []
        for action, payload in commands:
            if action == "urgent_tasks":
                self._inject_urgent_tasks()
            else:
                self._env.sim.inject_event(
                    ScenarioEvent(self._env.sim.time_s, action, {"cell": payload["cell"]})
                )

    def _wait_for_tick(self, tick_s: float) -> None:
        started = time.perf_counter()
        with self._condition:
            while self._paused and not self._stopping:
                self._condition.wait(timeout=1.0)
            if self._stopping:
                raise SessionStopped
            delay = tick_s / self._speed - (time.perf_counter() - started)
        if delay > 0:
            time.sleep(delay)

    def _emit_tick(self, replay) -> None:
        self._apply_commands()
        decision = self._env.decision_logs[-1] if self._env.decision_logs else None
        decision_time = decision.get("sim_time_s") if decision else None
        include_decision = decision if decision_time != self._last_decision_time else None
        if include_decision is not None:
            self._last_decision_time = decision_time
        frame = frame_to_viewer(
            self._env.sim.snapshot(),
            self._env.metrics(),
            self._env.config,
            include_decision,
            modified=self._modified,
        )
        self._last_frame = frame
        replay.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n")
        replay.flush()
        self._publish("frame", frame)
        self._wait_for_tick(self._env.config.tick_s)

    def _run(self) -> None:
        try:
            base, _ = load_config(Path("configs/methods") / f"{self.method}.yaml")
            config = method_config(scenario_config(self.scenario, base), self.method).copy(
                log_decisions=True
            )
            policy = None
            if self._availability.checkpoint:
                policy = load_policy(self._availability.checkpoint)
                metadata = getattr(policy, "_fleetrl_metadata", {})
                if metadata.get("method") != self.method:
                    raise ValueError("Metadata checkpoint không khớp phương pháp")
                config = config.copy(
                    observation_mode=metadata["observation_mode"],
                    action_mode=metadata["action_mode"],
                    charge_threshold=metadata["env"]["charge_threshold"],
                )
                policy = InferenceSession(policy)

            from ..env import FleetEnv

            self._env = FleetEnv(config)
            observation, _ = self._env.reset(seed=self.seed)
            replay_header = {
                "type": "header",
                "schema": "fleetrl-viewer/1",
                "map": map_to_viewer(self._env.sim.warehouse),
                "session": self.status(),
            }
            with self.replay_path.open("w", encoding="utf-8", newline="\n") as replay:
                replay.write(json.dumps(replay_header, ensure_ascii=False) + "\n")
                initial = frame_to_viewer(
                    self._env.sim.snapshot(), self._env.metrics(), config, modified=False
                )
                self._last_frame = initial
                replay.write(json.dumps(initial, ensure_ascii=False, allow_nan=False) + "\n")
                replay.flush()
                self._status = "running"
                self._publish("ready", {"map": replay_header["map"], "status": self.status()})
                self._publish("frame", initial)
                self._env.set_tick_observer(lambda: self._emit_tick(replay))
                done = False
                while not done:
                    action, failure, inference_ms = choose_action(policy, observation, config)
                    self._env.set_inference_context(inference_ms, failure)
                    observation, _, terminated, truncated, _ = self._env.step(action)
                    done = bool(terminated or truncated)
                self._status = "complete"
        except SessionStopped:
            self._status = "stopped"
        except Exception as error:
            self._status = "failed"
            self._error = f"{type(error).__name__}: {error}"
            self._publish("error", {"message": self._error})
        finally:
            if self._env is not None:
                self._env.close()
            self._publish("status", self.status())


class SessionManager:
    def __init__(self, study: str | Path, output: str | Path, scenario: str, seed: int) -> None:
        self.study = Path(study).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.default_scenario = scenario
        self.default_seed = seed
        self.methods = discover_methods(self.study)
        self.current: DemoSession | None = None
        self.sessions: dict[str, DemoSession] = {}
        self._lock = threading.Lock()

    def bootstrap(self) -> dict[str, Any]:
        return {
            "methods": {name: asdict(value) for name, value in self.methods.items()},
            "scenarios": list(SUPPORTED_SCENARIOS),
            "default_scenario": self.default_scenario,
            "default_seed": self.default_seed,
            "session": self.current.status() if self.current else None,
        }

    def create(self, method: str, scenario: str, seed: int) -> DemoSession:
        with self._lock:
            if self.current is not None:
                self.current.stop(wait=True)
            availability = self.methods.get(method)
            if availability is None:
                raise ValueError(f"Phương pháp không hỗ trợ: {method}")
            session = DemoSession(
                method,
                scenario,
                seed,
                self.study,
                self.output,
                copy.deepcopy(availability),
            )
            self.current = session
            self.sessions[session.id] = session
            session.start()
            return session

    def get(self, session_id: str | None = None) -> DemoSession:
        session = self.sessions.get(session_id) if session_id else self.current
        if session is None:
            raise ValueError("Chưa có phiên demo")
        return session
