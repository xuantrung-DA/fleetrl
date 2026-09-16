"""Pygame operator dashboard and honest playback of recorded simulator frames.

The UI uses the same FleetEnv as training. Playback never evaluates a policy.
Keyboard: Space run/pause, R reset, +/- speed, T demand, B block, P pause robot.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import FleetConfig
from .evaluation import (
    RunRecorder,
    account_policy_latency,
    choose_action,
    load_replay,
    method_config,
    serializable,
    snapshot_record,
)
from .types import ScenarioEvent, Task


class Dashboard:
    """Draw isolated serialized frames; renderer is testable without a simulator."""

    def __init__(self, warehouse: dict, title: str = "FleetRL", selected_robot: int | None = 0):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self.pg = pygame
        pygame.init()
        self.warehouse = warehouse
        self.cell = max(12, min(20, int(800 / warehouse["width"])))
        self.map_width = warehouse["width"] * self.cell
        self.map_height = warehouse["height"] * self.cell
        self.sidebar = 355
        self.height = max(self.map_height + 160, 760)
        self.width = self.map_width + self.sidebar + 32
        self.surface = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("dejavusans", 14)
        self.small = pygame.font.SysFont("dejavusans", 12)
        self.bold = pygame.font.SysFont("dejavusans", 16, bold=True)
        self.selected_robot = selected_robot
        self.origin = (16, 82)
        self.buttons = {}
        for index, label in enumerate(("RUN / PAUSE", "RESET", "SLOWER", "FASTER")):
            self.buttons[label] = pygame.Rect(16 + index * 135, 39, 125, 29)
        self.palette = {"L": (93, 190, 249), "M": (128, 219, 180), "H": (249, 192, 105)}

    def _text(
        self, text: Any, x: int, y: int, color=(213, 226, 239), small=False, bold=False
    ) -> None:
        font = self.bold if bold else self.small if small else self.font
        # Side-panel entries remain inside the window even on unusually long IDs/reasons.
        limit = max(5, int((self.width - x - 12) / (7 if not small else 6.1)))
        line = str(text)
        if len(line) > limit:
            line = line[: limit - 3] + "..."
        self.surface.blit(font.render(line, True, color), (x, y))

    def _rect(self, cell, inset=0):
        return self.pg.Rect(
            self.origin[0] + cell[0] * self.cell + inset,
            self.origin[1] + cell[1] * self.cell + inset,
            self.cell - inset * 2,
            self.cell - inset * 2,
        )

    def select_at(self, position: tuple[int, int], frame: dict) -> int | None:
        """Select a robot by a map click; return the selected robot ID."""
        for robot in frame.get("robots", []):
            if self._rect(robot["position"]).collidepoint(position):
                self.selected_robot = int(robot["id"])
                return self.selected_robot
        return None

    def draw(self, frame: dict, mode="LIVE", paused=True, speed=1.0, message="") -> np.ndarray:
        """Draw and return RGB pixels shaped H,W,3 for screenshot verification."""
        pg = self.pg
        self.surface.fill((15, 23, 35))
        mode_color = (249, 192, 105) if mode == "REPLAY" else (128, 219, 180)
        self._text(
            f"FleetRL  |  {mode}  |  {'PAUSED' if paused else 'RUNNING'}  |  {speed:g}x",
            16,
            11,
            mode_color,
            bold=True,
        )
        for label, rect in self.buttons.items():
            pg.draw.rect(self.surface, (42, 57, 77), rect, border_radius=5)
            self._text(label, rect.x + 10, rect.y + 6, small=True)
        ox, oy = self.origin
        pg.draw.rect(self.surface, (26, 38, 52), (ox, oy, self.map_width, self.map_height))
        for x in range(self.warehouse["width"] + 1):
            pg.draw.line(
                self.surface,
                (32, 45, 60),
                (ox + x * self.cell, oy),
                (ox + x * self.cell, oy + self.map_height),
            )
        for y in range(self.warehouse["height"] + 1):
            pg.draw.line(
                self.surface,
                (32, 45, 60),
                (ox, oy + y * self.cell),
                (ox + self.map_width, oy + y * self.cell),
            )
        for cell in self.warehouse.get("walls", []):
            pg.draw.rect(self.surface, (62, 76, 91), self._rect(cell, 1))
        for key, color in (
            ("pickups", (35, 93, 112)),
            ("dropoffs", (92, 70, 126)),
            ("parking", (34, 64, 73)),
        ):
            for cell in self.warehouse.get(key, []):
                pg.draw.rect(self.surface, color, self._rect(cell, 2))
        for cell in frame.get("blocked_cells", []):
            rect = self._rect(cell, 1)
            pg.draw.rect(self.surface, (164, 65, 65), rect)
            pg.draw.line(self.surface, (255, 220, 215), rect.topleft, rect.bottomright, 2)
        for port in frame.get("ports", self.warehouse.get("ports", [])):
            rect = self._rect(port["position"], 1)
            pg.draw.rect(self.surface, (218, 175, 60), rect, 2)
            self._text("C", rect.x + 4, rect.y + 1, (247, 208, 112), small=True)
        selected = next(
            (r for r in frame.get("robots", []) if r["id"] == self.selected_robot), None
        )
        if selected:
            positions = [selected["position"], *selected.get("path", [])]
            centers = [self._rect(cell).center for cell in positions]
            if len(centers) >= 2:
                pg.draw.lines(self.surface, (123, 153, 183), False, centers, 2)
        for robot in frame.get("robots", []):
            center = self._rect(robot["position"]).center
            color = self.palette.get(robot["kind"], (170, 181, 194))
            pg.draw.circle(self.surface, color, center, max(5, self.cell // 2 - 2))
            if robot["id"] == self.selected_robot:
                pg.draw.circle(self.surface, (255, 255, 255), center, self.cell // 2, 2)
            label = self.small.render(str(robot["id"]), True, (10, 24, 36))
            self.surface.blit(label, label.get_rect(center=center))
            fraction = max(0, min(1, robot["battery_wh"] / robot["capacity_wh"]))
            pg.draw.line(
                self.surface,
                (101, 214, 151) if fraction > 0.25 else (245, 123, 105),
                (center[0] - 7, center[1] + 8),
                (center[0] - 7 + round(14 * fraction), center[1] + 8),
                2,
            )
        sx = self.map_width + 34
        y = 82
        self._text(f"SIMULATION  {frame.get('time_s', 0):.1f} s", sx, y, bold=True)
        y += 29
        metrics = frame.get("metrics", {})
        for key, label in (
            ("completed", "Completed"),
            ("pending", "Pending / censored"),
            ("throughput_per_hour", "Throughput / hour"),
            ("total_lateness_s", "Total lateness (s)"),
            ("consumed_wh", "Consumed energy (Wh)"),
            ("mean_charge_wait_s", "Charge wait (s)"),
            ("collisions", "Collisions"),
            ("energy_emergencies", "Energy emergencies"),
        ):
            value = metrics.get(key)
            value = (
                "NA"
                if value is None
                else f"{value:.2f}"
                if isinstance(value, float)
                else str(value)
            )
            self._text(f"{label}: {value}", sx, y)
            y += 22
        y += 11
        self._text("SELECTED ROBOT", sx, y, bold=True)
        y += 27
        if selected:
            entries = [
                f"#{selected['id']}  type {selected['kind']}  {selected['status']}",
                f"Battery: {selected['battery_wh']:.2f} / {selected['capacity_wh']:.0f} Wh",
                f"SoC: {100 * selected['battery_wh'] / selected['capacity_wh']:.1f}%",
                f"Load: {selected.get('load_kg', 0):g} / {selected['payload_capacity_kg']:g} kg",
                f"Task: {selected.get('task_id')}  Port: {selected.get('port_id')}",
                f"Position: {selected['position']}  Goal: {selected.get('target_position')}",
            ]
            booking = next(
                (
                    b
                    for b in frame.get("bookings", [])
                    if b["robot_id"] == selected["id"] and b.get("cancelled_at_s") is None
                ),
                None,
            )
            if booking:
                entries.append(
                    f"Charge C{booking['port_id']} @ {booking['start_s']:.0f}-{booking['end_s']:.0f}s -> {100 * booking['target_soc']:.0f}%"
                )
            for entry in entries:
                self._text(entry, sx, y, small=True)
                y += 20
        else:
            self._text("Click a robot on the map", sx, y, small=True)
            y += 22
        y += 10
        self._text("CHARGER PORTS", sx, y, bold=True)
        y += 25
        for port in frame.get("ports", []):
            occupying = [
                r["id"]
                for r in frame.get("robots", [])
                if r.get("port_id") == port["id"] and r.get("status") == "charging"
            ]
            bookings = [
                b
                for b in frame.get("bookings", [])
                if b["port_id"] == port["id"] and b.get("cancelled_at_s") is None
            ]
            self._text(
                f"C{port['id']} {port['power_w']:g}W  active={occupying}  booked={len(bookings)}",
                sx,
                y,
                small=True,
            )
            y += 21
        decision = frame.get("decision", {})
        plan = decision.get("plan", {})
        y += 13
        self._text("LAST DISPATCH", sx, y, bold=True)
        y += 24
        self._text(
            f"Profile: {decision.get('profile', '-')}  Solver: {plan.get('solver_status', '-')}",
            sx,
            y,
            small=True,
        )
        y += 21
        self._text(
            f"Fallback: {decision.get('policy_fallback') or plan.get('fallback_reason') or 'none'}",
            sx,
            y,
            small=True,
        )
        if selected:
            candidates = [
                d for d in plan.get("decisions", []) if d.get("robot_id") == selected["id"]
            ]
            if candidates:
                y += 21
                self._text(f"Reason: {candidates[-1].get('reason', '-')}", sx, y, small=True)
        fy = self.map_height + self.origin[1] + 12
        self._text(
            "Space pause | R reset | +/- speed | Click robot for details", 16, fy, small=True
        )
        self._text(
            "T add 5 tasks | B temporary block at cursor | P pause selected robot",
            16,
            fy + 20,
            small=True,
        )
        if mode == "REPLAY":
            self._text(
                "REPLAY: recorded snapshots and decisions; policy inference is inactive.",
                16,
                fy + 40,
                (249, 192, 105),
                small=True,
            )
        else:
            self._text(
                message or "Blue / green / amber robots: light / medium / heavy. C: charging port.",
                16,
                fy + 40,
                small=True,
            )
        pg.display.flip()
        return np.transpose(pg.surfarray.array3d(self.surface), (1, 0, 2)).copy()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pg.image.save(self.surface, str(path))

    def close(self) -> None:
        self.pg.quit()


def inject_demo_event(
    env: Any, kind: str, selected_robot: int | None = None, cell: tuple[int, int] | None = None
) -> str:
    """Inject repeatable disturbances through the simulator's event-tape API."""
    now = env.sim.time_s
    if kind == "tasks":
        tape_ids = [e.payload.get("id", -1) for e in env.sim.tape.events if e.kind == "task"]
        first_id = max([*env.sim.tasks.keys(), *tape_ids, -1]) + 1
        warehouse = env.sim.warehouse
        for index in range(5):
            task = Task(
                first_id + index,
                warehouse.pickups[index % len(warehouse.pickups)],
                warehouse.dropoffs[index % len(warehouse.dropoffs)],
                5.0,
                now,
                now + 300.0,
                priority=3,
                zone=warehouse.zone_of(warehouse.pickups[index % len(warehouse.pickups)]),
            )
            env.sim.inject_event(ScenarioEvent(now, "task", serializable(task)))
        return "Injected 5 urgent tasks; recorded in event tape"
    if kind == "pause":
        if selected_robot not in env.sim.robots:
            return "Select a robot first"
        env.sim.inject_event(
            ScenarioEvent(now, "pause", {"robot_id": selected_robot, "duration_s": 60.0})
        )
        return f"Robot {selected_robot} paused for 60 simulated seconds"
    if kind == "block":
        if cell is None or not env.sim.warehouse.is_free(cell):
            return "Point at a free map cell before pressing B"
        env.sim.inject_event(ScenarioEvent(now, "block", {"cell": list(cell), "duration_s": 300.0}))
        return f"Blocked cell {cell} for 300 simulated seconds; recorded in tape"
    raise ValueError(f"Unknown demo event {kind!r}")


def run_ui(
    config: FleetConfig | None = None,
    method: str = "fixed",
    seed: int = 2000,
    policy: Any = None,
    replay_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_frames: int | None = None,
    screenshot_path: str | Path | None = None,
) -> dict:
    """Run live/replay UI. ``max_frames`` permits finite headless smoke checks.

    UI speed controls wall-clock playback only. Reset restores the original seed
    and episode tape. Each recording session has a separate reset_### directory.
    """
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    env = recorder = None
    history = []
    session = 0
    mode = "REPLAY" if replay_path is not None else "LIVE"
    if mode == "REPLAY":
        header, history = load_replay(replay_path)
        warehouse = header["map"]
        frame = history[0]
    else:
        from .env import FleetEnv

        config = method_config(config or FleetConfig(), method).copy(log_decisions=True)
        metadata = getattr(policy, "_fleetrl_metadata", {})
        if metadata:
            from .methods.registry import get_method

            if metadata["method"] != get_method(method).name:
                raise ValueError("checkpoint method mismatch")
            config = config.copy(
                action_mode=metadata["action_mode"],
                observation_mode=metadata["observation_mode"],
                charge_threshold=metadata["env"]["charge_threshold"],
            )
        if config.controller == "hybrid" and policy is None:
            raise ValueError("Live hybrid/ablation UI requires a policy checkpoint")
        env = FleetEnv(config)
        if policy is not None:
            from .learning.policies import InferenceSession

            policy = InferenceSession(policy)
        observation, _ = env.reset(seed=seed)
        warehouse = serializable(env.sim.warehouse)
        frame = snapshot_record(env.sim.snapshot(), env.metrics())
        if output_dir is not None:
            recorder = RunRecorder(
                Path(output_dir) / "reset_000",
                config,
                env.sim.tape,
                method,
                seed,
                env.sim.warehouse,
                getattr(policy, "_fleetrl_checkpoint_path", None),
            )
            recorder.record(frame)
    dashboard = Dashboard(warehouse)
    pg = dashboard.pg
    clock = pg.time.Clock()
    paused = True
    speed = 1.0
    index = frames = 0
    accumulator = 0.0
    running = True
    episode_done = False
    message = "Press Space to start. A reset starts a fresh recording."
    last_event = len(env.sim.events) if env is not None else 0
    last_wall = time.perf_counter()
    try:
        while running and (max_frames is None or frames < max_frames):
            now_wall = time.perf_counter()
            dt_wall = min(now_wall - last_wall, 0.25)
            last_wall = now_wall
            for event in pg.event.get():
                command = None
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    command = {
                        pg.K_SPACE: "RUN / PAUSE",
                        pg.K_r: "RESET",
                        pg.K_EQUALS: "FASTER",
                        pg.K_PLUS: "FASTER",
                        pg.K_KP_PLUS: "FASTER",
                        pg.K_MINUS: "SLOWER",
                        pg.K_KP_MINUS: "SLOWER",
                        pg.K_t: "tasks",
                        pg.K_b: "block",
                        pg.K_p: "pause",
                    }.get(event.key)
                    if event.key == pg.K_ESCAPE:
                        running = False
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    command = next(
                        (
                            name
                            for name, rect in dashboard.buttons.items()
                            if rect.collidepoint(event.pos)
                        ),
                        None,
                    )
                    if command is None:
                        dashboard.select_at(event.pos, frame)
                if command == "RUN / PAUSE":
                    paused = not paused
                elif command == "FASTER":
                    speed = min(128.0, speed * 2)
                elif command == "SLOWER":
                    speed = max(0.25, speed / 2)
                elif command == "RESET":
                    paused, accumulator, index, episode_done = True, 0.0, 0, False
                    if mode == "REPLAY":
                        frame = history[0]
                    else:
                        if recorder:
                            recorder.close(
                                {"metrics": env.metrics(), "episode_complete": False}, env.sim.tape
                            )
                        observation, _ = env.reset(seed=seed)
                        if policy is not None:
                            policy.reset()
                        frame = snapshot_record(env.sim.snapshot(), env.metrics())
                        last_event = len(env.sim.events)
                        session += 1
                        if output_dir is not None:
                            recorder = RunRecorder(
                                Path(output_dir) / f"reset_{session:03d}",
                                config,
                                env.sim.tape,
                                method,
                                seed,
                                env.sim.warehouse,
                                getattr(policy, "_fleetrl_checkpoint_path", None),
                            )
                            recorder.record(frame)
                    message = "Reset to initial seed; paused"
                elif command in {"tasks", "block", "pause"} and mode == "LIVE":
                    mx, my = pg.mouse.get_pos()
                    cell = (
                        (mx - dashboard.origin[0]) // dashboard.cell,
                        (my - dashboard.origin[1]) // dashboard.cell,
                    )
                    message = inject_demo_event(env, command, dashboard.selected_robot, cell)
                    observation = env._observe()
                    frame = snapshot_record(
                        env.sim.snapshot(), env.metrics(), env.sim.events[last_event:]
                    )
                    last_event = len(env.sim.events)
                    if recorder:
                        recorder.record(frame)
            if not paused and not episode_done:
                accumulator += dt_wall * speed
                if mode == "REPLAY":
                    if index + 1 >= len(history):
                        episode_done, paused = True, True
                    else:
                        interval = max(
                            0.01, history[index + 1]["time_s"] - history[index]["time_s"]
                        )
                        if accumulator >= interval:
                            accumulator -= interval
                            index += 1
                            frame = history[index]
                elif accumulator >= config.decision_s:
                    accumulator -= config.decision_s
                    action, failure, inference_ms = choose_action(policy, observation, config)
                    env.set_inference_context(inference_ms, failure)
                    observation, reward, terminated, truncated, info = env.step(action)
                    account_policy_latency(env, info, inference_ms, failure)
                    frame = snapshot_record(
                        env.sim.snapshot(), env.metrics(), env.sim.events[last_event:]
                    )
                    last_event = len(env.sim.events)
                    decision = {
                        "profile": action,
                        "policy_fallback": failure,
                        "inference_ms": inference_ms,
                        "time_s": env.sim.time_s,
                        "reward": reward,
                        "reward_components": info.get("reward_components", {}),
                        "plan": info.get("decision", {}),
                    }
                    frame["decision"] = decision
                    if recorder:
                        recorder.record(frame, decision)
                    episode_done = bool(terminated or truncated)
                    if episode_done:
                        paused = True
                        message = "Episode complete. Press R to reset."
            dashboard.draw(frame, mode, paused, speed, message)
            frames += 1
            clock.tick(30)
        if screenshot_path is not None:
            dashboard.save(screenshot_path)
        return {
            "mode": mode,
            "frames_rendered": frames,
            "time_s": frame["time_s"],
            "episode_complete": episode_done,
            "selected_robot": dashboard.selected_robot,
            "metrics": frame.get("metrics", {}),
            "screenshot": str(screenshot_path) if screenshot_path else None,
        }
    finally:
        if recorder:
            recorder.close(
                {"metrics": env.metrics(), "episode_complete": episode_done}, env.sim.tape
            )
        if env is not None:
            env.close()
        dashboard.close()


def render_frame(snapshot, metrics: dict | None = None) -> np.ndarray:
    """Render a single live snapshot as RGB, including on headless machines."""
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    dashboard = Dashboard(serializable(snapshot.warehouse))
    try:
        return dashboard.draw(snapshot_record(snapshot, metrics), mode="LIVE", paused=True)
    finally:
        dashboard.close()
