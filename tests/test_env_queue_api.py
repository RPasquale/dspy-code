import json
import threading
from urllib import request
from pathlib import Path
import importlib


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


def test_env_queue_submit_and_stats(tmp_path, monkeypatch):
    # ensure queue dir under repo logs
    qdir = Path('logs') / 'env_queue'
    (qdir / 'pending').mkdir(parents=True, exist_ok=True)
    (qdir / 'done').mkdir(parents=True, exist_ok=True)

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
        # Submit new env task
        code, res = _post_json(base + "/api/env-queue/submit", {"id": "unit-task-1", "class": "cpu_short", "payload": "demo"})
        assert code == 200 and res.get('ok') is True

        # Stats should see pending >= 1
        j = _get(base + "/api/env-queue/stats")
        assert 'pending' in j and isinstance(j['pending'], int)
        assert any((it.get('id') == 'unit-task-1') for it in (j.get('items') or []))
    finally:
        try: httpd.shutdown()
        except Exception: pass
        try: httpd.server_close()
        except Exception: pass

