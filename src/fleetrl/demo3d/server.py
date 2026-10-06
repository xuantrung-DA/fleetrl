"""Loopback-only HTTP/SSE server for the FleetRL 3D demo."""

from __future__ import annotations

import json
import mimetypes
import queue
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .session import SessionManager


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, manager: SessionManager, web_root: Path):
        self.manager = manager
        self.web_root = web_root
        super().__init__(address, DemoHandler)


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def log_message(self, format: str, *args) -> None:
        print(f"demo3d: {format % args}")

    def _json(self, value, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, error: Exception, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._json({"error": f"{type(error).__name__}: {error}"}, status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024:
            raise ValueError("Request body quá lớn")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            body = self._body()
            if path == "/api/sessions":
                session = self.server.manager.create(
                    str(body.get("method", "heuristic")),
                    str(body.get("scenario", self.server.manager.default_scenario)).upper(),
                    int(body.get("seed", self.server.manager.default_seed)),
                )
                self._json(session.status(), HTTPStatus.CREATED)
                return
            if path == "/api/control":
                session = self.server.manager.get(body.get("session"))
                self._json(session.control(str(body.get("action")), body))
                return
            self._error(ValueError("Endpoint không tồn tại"), HTTPStatus.NOT_FOUND)
        except (ValueError, OSError, json.JSONDecodeError) as error:
            self._error(error)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/bootstrap":
                self._json(self.server.manager.bootstrap())
                return
            if parsed.path == "/api/events":
                self._events(parse_qs(parsed.query).get("session", [None])[0])
                return
            if parsed.path == "/api/replay":
                self._replay(parse_qs(parsed.query).get("session", [None])[0])
                return
            self._static(parsed.path)
        except (ValueError, OSError) as error:
            self._error(error, HTTPStatus.NOT_FOUND)

    def _events(self, session_id: str | None) -> None:
        session = self.server.manager.get(session_id)
        channel = session.subscribe()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                try:
                    item = channel.get(timeout=15)
                    data = json.dumps(item["data"], ensure_ascii=False, allow_nan=False)
                    payload = f"event: {item['event']}\ndata: {data}\n\n".encode("utf-8")
                except queue.Empty:
                    payload = b": keepalive\n\n"
                self.wfile.write(payload)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            session.unsubscribe(channel)

    def _replay(self, session_id: str | None) -> None:
        session = self.server.manager.get(session_id)
        if not session.replay_path.is_file():
            raise ValueError("Replay chưa sẵn sàng")
        body = session.replay_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{session.id}.jsonl"')
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (self.server.web_root / relative).resolve()
        if self.server.web_root not in target.parents and target != self.server.web_root:
            raise ValueError("Đường dẫn không hợp lệ")
        if not target.is_file():
            target = self.server.web_root / "index.html"
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def run_server(study: str, scenario: str, seed: int, port: int, output: str) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("port must be 1..65535")
    web_root = _repository_root() / "demo3d" / "web" / "dist"
    if not (web_root / "index.html").is_file():
        raise FileNotFoundError(
            "Thiếu demo3d/web/dist. Chạy `npm install` rồi `npm run build` trong demo3d/web."
        )
    manager = SessionManager(study, output, scenario, seed)
    server = DemoServer(("127.0.0.1", port), manager, web_root)
    print(f"FleetRL 3D demo: http://127.0.0.1:{port}")
    print("Chỉ lắng nghe trên máy cục bộ. Nhấn Ctrl+C để dừng.")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if manager.current is not None:
            manager.current.stop(wait=True)
        server.server_close()
