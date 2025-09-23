import json
import socket
import threading
import time
from urllib import request, error

import pytest


def _get(url: str, timeout: float = 3.0):
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        try:
            return json.loads(body)
        except Exception:
            return body


def _post_json(url: str, obj: dict, timeout: float = 3.0):
    data = json.dumps(obj).encode('utf-8')
    req = request.Request(url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"}, method='POST')
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(body)
            except Exception:
                return resp.status, body
    except error.HTTPError as e:
        # Return status and body for assertions
        try:
            body = e.read().decode('utf-8')
            return e.code, (json.loads(body) if body else {})
        except Exception:
            return e.code, {}


def _sse_lines(url: str, timeout: float = 5.0, max_lines: int = 5):
    # crude SSE consumer for tests
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


@pytest.mark.integration
def test_end_to_end_server_trace_and_endpoints():
    # Start server in-thread on ephemeral port
import importlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
        # Basic endpoints
        j = _get(base + "/api/status")
        assert isinstance(j, dict)

        _ = _get(base + "/api/system/resources")
        _ = _get(base + "/api/kafka/settings")
        _ = _get(base + "/api/kafka/configs?topics=agent.results")
        _ = _get(base + "/api/grpo/status")

        # Set guard to an extremely high disk requirement to force block
        code, body = _post_json(base + "/api/system/guard", {"min_free_gb": 10_000_000, "min_ram_gb": 0, "min_vram_mb": 0})
        assert code == 200 and isinstance(body, dict)
        # Try to start GRPO, expect storage error (507)
        code, body = _post_json(base + "/api/grpo/start", {"dataset_path": "docs/samples/grpo_example.jsonl", "max_steps": 1})
        assert code == 507
        # Stop endpoint should be OK even when not running
        code, body = _post_json(base + "/api/grpo/stop", {})
        assert code == 200 and isinstance(body, dict)

        # SSE trace should receive recent paths
        lines = _sse_lines(base + "/api/debug/trace/stream", timeout=4.0, max_lines=2)
        # Not every environment will have immediate lines; allow empty but no exception
        assert isinstance(lines, list)
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass
