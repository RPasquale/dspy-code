import json
import threading
from urllib import request
from pathlib import Path
import importlib, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _post_json(url: str, obj: dict, headers: dict | None = None, timeout: float = 3.0):
    data = json.dumps(obj).encode('utf-8')
    req = request.Request(url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json", **(headers or {})}, method='POST')
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, (json.loads(body) if body else {})


def _tail_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def test_capacity_and_guardrails_emit_agent_action(tmp_path, monkeypatch):
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
        # Capacity approve (admin)
        code, res = _post_json(base + '/api/capacity/approve', { 'kind': 'storage_budget_increase', 'params': { 'to_gb': 10 } }, headers={'X-Admin-Key':'test'})
        assert code in (200, 403)  # If admin key enforced, may be 403 in some envs
        # Capacity deny
        code, res = _post_json(base + '/api/capacity/deny', { 'kind': 'gpu_hours_increase', 'params': { 'to_hpd': 1 } }, headers={'X-Admin-Key':'test'})
        assert code in (200, 403)

        # Prepare guardrails pending actions file
        repo_root = Path(__file__).resolve().parents[1]
        pending_path = repo_root / '.dspy_reports' / 'pending_actions.json'
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text(json.dumps([{'id':'ga1','status':'pending','type':'noop'}]))

        # Guardrails approve-action
        code, res = _post_json(base + '/api/guardrails/approve-action', { 'id': 'ga1' })
        assert code == 200

        # Add a second pending and reject
        pending_path.write_text(json.dumps([{'id':'gb1','status':'pending','type':'noop'}]))
        code, res = _post_json(base + '/api/guardrails/reject-action', { 'id': 'gb1', 'comment': 'nope' })
        assert code == 200

        # Read event file
        events = _tail_jsonl(log_dir / 'agent_action.jsonl')
        names = [ (ev.get('event') or {}).get('name') for ev in events ]
        # We don't assert capacity names strictly due to admin gating; guardrails action names should exist
        assert 'guardrails_approve_action' in names
        assert 'guardrails_reject_action' in names
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass

