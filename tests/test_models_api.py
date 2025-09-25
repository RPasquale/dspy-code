import json
import threading
from urllib import request
from pathlib import Path
import importlib, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _get(url: str, timeout: float = 3.0):
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def test_models_api_basic():
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
        code, body = _get(base + "/api/models")
        assert code == 200
        assert isinstance(body, dict)
        assert 'code_log' in body and 'grpo' in body
        assert isinstance(body['code_log'], dict)
        assert isinstance(body['grpo'], dict)
    finally:
        try: httpd.shutdown()
        except Exception: pass
        try: httpd.server_close()
        except Exception: pass

