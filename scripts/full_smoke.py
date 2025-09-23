from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run(cmd: list[str], timeout: float = 60.0) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill(); out, err = p.communicate()
    return p.returncode, out, err


def test_cli_help() -> dict:
    code, out, err = run([sys.executable, "-m", "dspy_agent.cli", "--help"], timeout=20)
    return {"ok": code == 0, "code": code, "stdout_len": len(out), "stderr_len": len(err)}


def test_rl_cli() -> dict:
    env = os.environ.copy()
    env["RL_ACTIONS"] = "build"
    cmd = [sys.executable, "-m", "dspy_agent.cli", "rl", "train", "--steps", "40", "--n-envs", "1", "--no-puffer", "--policy", "epsilon-greedy", "--epsilon", "0.1", "--build-cmd", "python -m compileall -q .", "--timeout-sec", "60"]
    code, out, err = run(cmd, timeout=90)
    state = (REPO_ROOT / ".dspy_rl_state.json")
    rl_state = {}
    if state.exists():
        try: rl_state = json.loads(state.read_text())
        except Exception: rl_state = {}
    return {"ok": code == 0, "code": code, "has_state": state.exists(), "state_keys": list(rl_state.keys())[:5]}


def test_stream_stack_and_status() -> dict:
    from dspy_agent.streaming.streaming_runtime import start_local_stack
    from dspy_agent.status_http import start_status_server
    ws = REPO_ROOT
    threads, bus = start_local_stack(ws, None, storage=None, kafka=None)
    th = start_status_server("127.0.0.1", 8765, ws, bus)
    threads.append(th)
    # Send synthetic events
    log_dir = ws / "logs" / "backend"; log_dir.mkdir(parents=True, exist_ok=True)
    test_log = log_dir / "full_smoke_backend.log"
    test_log.write_text("INFO boot\nERROR fail x\n")
    # Give the stack a moment
    time.sleep(1.2)
    # Touch log again
    with test_log.open("a") as f: f.write("ERROR again\n")
    time.sleep(1.2)
    # Wait for server to bind
    time.sleep(0.5)
    # Fetch status
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8765/status", timeout=2.0) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            status = json.loads(raw)
    except Exception:
        status = {}
    srl = None
    if (ws / ".dspy_stream_rl.json").exists():
        try: srl = json.loads((ws / ".dspy_stream_rl.json").read_text())
        except Exception: srl = None
    for t in threads:
        try:
            stop = getattr(t, "stop", None)
            if callable(stop):
                stop()
        except Exception:
            pass
    return {"status_ok": bool(status), "topics": list((status.get("bus", {}).get("topics") or {}).keys()) if isinstance(status, dict) else [], "stream_rl": srl}


def test_enhanced_dashboard_api() -> dict:
    # Start enhanced server briefly and hit a simple endpoint
    import threading
    import http.client
    from enhanced_dashboard_server import EnhancedDashboardHandler
    import socketserver

    class SilentTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    # Bind to an ephemeral port
    with SilentTCPServer(("127.0.0.1", 0), EnhancedDashboardHandler) as httpd:
        port = httpd.server_address[1]
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        ok = False
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request("GET", "/api/status")
            res = conn.getresponse()
            ok = (res.status == 200)
        except Exception:
            ok = False
        finally:
            try: httpd.shutdown()
            except Exception: pass
        return {"ok": ok, "port": port}


def main() -> None:
    results = {
        "cli_help": test_cli_help(),
        "rl_cli": test_rl_cli(),
        "stream_stack": test_stream_stack_and_status(),
        "dashboard_api": test_enhanced_dashboard_api(),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
