import json
import threading
import time
from urllib import request
from pathlib import Path
import importlib, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _post_json(url: str, obj: dict, timeout: float = 3.0):
    data = json.dumps(obj).encode('utf-8')
    req = request.Request(url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"}, method='POST')
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        try:
            return resp.status, json.loads(body)
        except Exception:
            return resp.status, body


def _sse_lines(url: str, timeout: float = 4.0, max_lines: int = 2):
    req = request.Request(url, headers={"Accept": "text/event-stream"})
    with request.urlopen(req, timeout=timeout) as resp:
        buf = []
        start = time.time()
        while len(buf) < max_lines and time.time() - start < timeout:
            line = resp.readline().decode("utf-8")
            if not line:
                break
            line = line.strip()
            if line.startswith("data: "):
                buf.append(line[len("data: "):])
        return buf


def test_events_sse_stream(tmp_path, monkeypatch):
    # Route event bus files to tmp
    log_dir = tmp_path / 'logs'
    monkeypatch.setenv('EVENTBUS_LOG_DIR', str(log_dir))

    srv = importlib.import_module('enhanced_dashboard_server')

    class ThreadedHTTPServer(srv.socketserver.ThreadingMixIn, srv.socketserver.TCPServer):
        daemon_threads = True
        allow_reuse_address = True

    httpd = ThreadedHTTPServer(("127.0.0.1", 0), srv.EnhancedDashboardHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"

    try:
        # Post a few events
        for i in range(2):
            _post_json(base + "/api/events", {"topic": "ui.action", "event": {"name": f"unit_test_{i}"}})
        # Connect to SSE stream
        lines = _sse_lines(base + "/api/events/stream?topic=ui.action&limit=5", timeout=5.0, max_lines=1)
        assert isinstance(lines, list) and len(lines) >= 1
        payload = json.loads(lines[-1])
        assert payload.get('topic') == 'ui.action'
        assert ('items' in payload) or ('delta' in payload)
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass

