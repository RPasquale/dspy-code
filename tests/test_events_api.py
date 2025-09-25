import json
import threading
import time
from urllib import request
from pathlib import Path
import importlib, sys, os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _post_json(url: str, obj: dict, timeout: float = 3.0):
    data = json.dumps(obj).encode('utf-8')
    req = request.Request(url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"}, method='POST')
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, (json.loads(body) if body else {})


def _get(url: str, timeout: float = 3.0):
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def test_events_post_and_tail(tmp_path, monkeypatch):
    # Ensure server writes logs to temp dir
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
        # Post an event
        code, res = _post_json(base + "/api/events", {"topic": "ui.action", "event": {"name": "unit_test"}})
        assert code == 200 and res.get('ok') is True

        # Tail the events
        j = _get(base + "/api/events/tail?topic=ui.action&limit=5")
        assert j.get('topic') == 'ui.action'
        assert isinstance(j.get('items'), list)
        # Latest item should contain our name field
        assert any((it.get('event') or {}).get('name') == 'unit_test' for it in j.get('items') or [])
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass

