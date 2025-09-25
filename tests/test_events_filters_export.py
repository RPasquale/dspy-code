import json
import threading
from urllib import request
from pathlib import Path
import importlib, sys, time

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


def _get(url: str, timeout: float = 3.0):
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        try:
            return json.loads(body)
        except Exception:
            return body


def test_events_filters_and_export(tmp_path, monkeypatch):
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
        # Post diverse events
        _post_json(base + "/api/events", {"topic": "ui.action", "event": {"name": "unit_test", "action": "approve", "user": "alice"}})
        _post_json(base + "/api/events", {"topic": "ui.action", "event": {"name": "unit_test2", "action": "reject", "user": "bob"}})
        _post_json(base + "/api/events", {"topic": "spark.app", "event": {"event": "submitted", "name": "jobX"}})

        # Tail with regex filter
        j = _get(base + "/api/events/tail?topic=ui.action&limit=10&q=approve")
        assert isinstance(j, dict) and any((it.get('event') or {}).get('action') == 'approve' for it in j.get('items'))

        # Tail with key/value filter (dot path)
        j2 = _get(base + "/api/events/tail?topic=ui.action&limit=10&key=event.user&value=alice")
        assert any((it.get('event') or {}).get('user') == 'alice' for it in j2.get('items'))

        # Export (NDJSON)
        req = request.Request(base + "/api/events/export?topics=ui.action,spark.app&limit=5", headers={"Accept":"application/x-ndjson"})
        with request.urlopen(req, timeout=3.0) as resp:
            body = resp.read().decode('utf-8').strip().splitlines()
            assert len(body) >= 1
            ln = json.loads(body[0])
            assert 'topic' in ln and 'record' in ln
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass

