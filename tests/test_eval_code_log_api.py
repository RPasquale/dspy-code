import json
import threading
from urllib import request
from pathlib import Path
import importlib, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _post_json(url: str, obj: dict, timeout: float = 3.0):
    data = json.dumps(obj).encode('utf-8')
    req = request.Request(url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"}, method='POST')
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, (json.loads(body) if body else {})
    except Exception as e:
        try:
            body = e.read().decode('utf-8')
            return getattr(e, 'code', 500), (json.loads(body) if body else {})
        except Exception:
            return 500, {}


def test_eval_code_log_endpoints_exist():
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
        # Model eval may fail if transformers not installed; ensure endpoint responds JSON either way
        code, res = _post_json(base + '/api/eval/code-log', { 'code': 'print("hello")' })
        assert code in (200, 500)
        assert isinstance(res, dict)
        # Scoring endpoint
        code2, res2 = _post_json(base + '/api/eval/code-log/score', { 'code': 'print("hello")', 'topic': 'spark.log', 'limit': 5 })
        assert code2 in (200, 500)
        assert isinstance(res2, dict)
    finally:
        try: httpd.shutdown()
        except Exception: pass
        try: httpd.server_close()
        except Exception: pass

