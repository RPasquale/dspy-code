from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Dict, Any, List
from pathlib import Path

from .db.factory import get_storage
from .streaming.streaming_config import DEFAULT_CONFIG_PATH, load_config, StreamConfig
from .streaming.streaming_runtime import autodiscover_logs
from .agents.knowledge import summarize_code_graph


def _container_names(workspace: Path) -> List[str]:
    # Prefer configured containers, else autodiscover
    if DEFAULT_CONFIG_PATH.exists():
        try:
            cfg: StreamConfig = load_config(DEFAULT_CONFIG_PATH)
            return [getattr(ct, 'container') for ct in cfg.containers]
        except Exception:
            pass
    try:
        discs = autodiscover_logs(workspace)
        return [d.container for d in discs]
    except Exception:
        return []


class _Handler(BaseHTTPRequestHandler):
    storage = None
    workspace = Path.cwd()

    def _json(self, code: int, obj: Dict[str, Any]):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args):
        # Silence default logging
        return

    def do_GET(self):  # noqa: N802
        st = self.storage
        if self.path == '/health':
            return self._json(200, {"status": "ok"})
        if self.path == '/deploy':
            if st is None:
                return self._json(200, {"status": "unknown", "reason": "no storage"})
            try:
                status = st.get("deploy:last:lightweight:status")
                ts = st.get("deploy:last:lightweight:ts")
                compose_hash = st.get("deploy:last:lightweight:compose_hash")
                image = st.get("deploy:last:lightweight:image")
                return self._json(200, {"status": status, "ts": ts, "compose_hash": compose_hash, "image": image})
            except Exception:
                return self._json(200, {"status": "unknown"})
        if self.path.startswith('/containers'):
            names = _container_names(self.workspace)
            result = {}
            if st is not None:
                for c in names:
                    pref = f"last:{c}:"
                    try:
                        result[c] = {
                            "summary": st.get(pref + "summary"),
                            "key_points": st.get(pref + "key_points"),
                            "plan": st.get(pref + "plan"),
                            "ts": st.get(pref + "ts"),
                        }
                    except Exception:
                        result[c] = {"error": "unavailable"}
            return self._json(200, {"containers": result, "names": names})
        if self.path == '/code':
            st = self.storage
            if st is None:
                return self._json(200, {"status": "no-storage"})
            try:
                graph = st.get("code:graph")
                summary = st.get("code:summary")
                return self._json(200, {"graph": graph, "summary": summary})
            except Exception:
                return self._json(200, {"status": "unavailable"})
        return self._json(404, {"error": "not found"})


def start_status_server(host: str, port: int, workspace: Path) -> threading.Thread:
    handler = _Handler
    handler.storage = get_storage()
    handler.workspace = workspace
    httpd = HTTPServer((host, port), handler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return th
