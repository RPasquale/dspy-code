from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Dict, Any, List
from pathlib import Path

from .db.factory import get_storage
from .streaming.streaming_config import DEFAULT_CONFIG_PATH, load_config, StreamConfig
from .streaming.streaming_runtime import autodiscover_logs
from .agents.knowledge import summarize_code_graph
from .llm import get_circuit_breaker_status


def _container_names(workspace: Path) -> List[str]:
    # Prefer configured containers, else autodiscover
    names: List[str] = []
    if DEFAULT_CONFIG_PATH.exists():
        try:
            cfg: StreamConfig = load_config(DEFAULT_CONFIG_PATH)
            names.extend([getattr(ct, 'container') for ct in cfg.containers])
        except Exception:
            pass
    env_containers = [c.strip() for c in os.getenv('DSPY_DOCKER_CONTAINERS', '').split(',') if c.strip()]
    names.extend(env_containers)
    try:
        discs = autodiscover_logs(workspace)
        names.extend([d.container for d in discs])
    except Exception:
        pass
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


class _Handler(BaseHTTPRequestHandler):
    storage = None
    workspace = Path.cwd()
    bus = None

    def _json(self, code: int, obj: Dict[str, Any]):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _dlq_metrics(self) -> Dict[str, Any]:
        try:
            path = self.workspace / '.dspy_reports' / 'dlq.jsonl'
            if not path.exists():
                return {"total": 0, "by_topic": {}, "last_ts": None}
            total = 0
            by_topic: Dict[str, int] = {}
            last_ts = None
            with path.open('r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        rec = json.loads(line)
                        t = rec.get('topic', 'unknown')
                        by_topic[t] = by_topic.get(t, 0) + 1
                        last_ts = rec.get('ts', last_ts)
                    except Exception:
                        pass
            return {"total": total, "by_topic": by_topic, "last_ts": last_ts}
        except Exception:
            return {"total": 0, "by_topic": {}, "last_ts": None}

    def log_message(self, format: str, *args):
        # Silence default logging
        return

    def do_GET(self):  # noqa: N802
        st = self.storage
        if self.path == '/health':
            cb = get_circuit_breaker_status()
            return self._json(200, {"status": "ok", "lm_circuit": cb})
        if self.path == '/metrics':
            try:
                cb = get_circuit_breaker_status()
            except Exception:
                cb = {"open": False}
            result = {"lm_circuit": cb}
            # Hardware metrics snapshot
            try:
                hw = (self.workspace / '.dspy_hw.json')
                if hw.exists():
                    import json as _json
                    result['hw'] = _json.loads(hw.read_text())
            except Exception:
                pass
            # Include last RL summary if present
            try:
                rl_state = (self.workspace / '.dspy_rl_state.json')
                if rl_state.exists():
                    import json as _json
                    result['rl'] = _json.loads(rl_state.read_text())
            except Exception:
                pass
            # Include streaming RL metrics snapshot
            try:
                srl = (self.workspace / '.dspy_stream_rl.json')
                if srl.exists():
                    import json as _json
                    result['stream_rl'] = _json.loads(srl.read_text())
            except Exception:
                pass
            # Include online bandit snapshot
            try:
                band = (self.workspace / '.dspy_online_bandit.json')
                if band.exists():
                    import json as _json
                    result['online_bandit'] = _json.loads(band.read_text())
            except Exception:
                pass
            # Include DLQ metrics
            try:
                result['dlq'] = self._dlq_metrics()
            except Exception:
                result['dlq'] = {"total": 0}
            # Include bus queue metrics when available
            try:
                if getattr(self, 'bus', None) is not None and hasattr(self.bus, 'metrics'):
                    bm = self.bus.metrics()  # type: ignore[attr-defined]
                    if isinstance(bm, dict):
                        result['bus'] = bm
            except Exception:
                pass
            return self._json(200, result)
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
        if self.path == '/auto':
            status_path = self.workspace / '.dspy_auto_status.json'
            if status_path.exists():
                try:
                    data = json.loads(status_path.read_text())
                    return self._json(200, data)
                except Exception:
                    return self._json(200, {"status": "unavailable"})
            return self._json(200, {"status": "offline"})
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


def start_status_server(host: str, port: int, workspace: Path, bus: Optional[object] = None) -> threading.Thread:
    handler = _Handler
    handler.storage = get_storage()
    handler.workspace = workspace
    handler.bus = bus
    httpd = HTTPServer((host, port), handler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return th
