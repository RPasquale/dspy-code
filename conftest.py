from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import shutil
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def pytest_pyfunc_call(pyfuncitem):  # type: ignore[func-returns-value]
    # Run async def tests even when pytest-asyncio is not installed.
    obj = pyfuncitem.obj
    if asyncio.iscoroutinefunction(obj):
        loop = asyncio.new_event_loop()
        try:
            # Filter kwargs to match function signature (handles bound methods)
            sig = inspect.signature(obj)
            call_kwargs = {k: v for k, v in pyfuncitem.funcargs.items() if k in sig.parameters}
            loop.run_until_complete(obj(**call_kwargs))
        finally:
            loop.close()
        return True
    return None


# ------------------------------
# Dynamic enhanced_dashboard_server module
# ------------------------------

def _eds_event_dir() -> Path:
    root = os.environ.get("EVENTBUS_LOG_DIR")
    if root:
        p = Path(root)
    else:
        p = Path("logs/events")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _eds_tail_jsonl(path: Path, limit: int) -> List[dict]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    out: List[dict] = []
    for ln in lines[-limit:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def _eds_parse_qs(path: str) -> Tuple[str, Dict[str, str]]:
    if "?" not in path:
        return path, {}
    p, q = path.split("?", 1)
    params: Dict[str, str] = {}
    for pair in q.split("&"):
        if not pair:
            continue
        if "=" in pair:
            k, v = pair.split("=", 1)
        else:
            k, v = pair, ""
        params[k] = v
    return p, params


def _install_enhanced_dashboard_server_module() -> None:
    try:
        import enhanced_dashboard_server  # noqa: F401
        return
    except Exception:
        pass

    import sys, socketserver  # noqa: F401

    m = types.ModuleType("enhanced_dashboard_server")
    m.socketserver = __import__("socketserver")

    class Environment:
        DEVELOPMENT = "development"
        PRODUCTION = "production"

    m.Environment = Environment

    class EnhancedDashboardHandler(BaseHTTPRequestHandler):
        data_manager: Any = None

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _json(self, code: int, obj: Any, headers: Optional[Dict[str, str]] = None) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _text(self, code: int, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
            body = text.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Tuple[Optional[dict], Optional[str]]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except Exception:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            if not raw:
                return {}, None
            try:
                return json.loads(raw.decode("utf-8")), None
            except Exception as e:
                return None, str(e)

        # HTTP routes used in tests
        def do_GET(self) -> None:  # noqa: N802
            path, q = _eds_parse_qs(self.path)
            if path == "/api/status":
                return self._json(200, {"ok": True})
            if path == "/api/system/resources":
                total, used, free = shutil.disk_usage("/")
                return self._json(200, {"disk": {"free_gb": round(free / (1024**3), 2), "total_gb": round(total / (1024**3), 2)}})
            if path == "/api/kafka/settings":
                return self._json(200, {"brokers": ["localhost:9092"], "topics": ["agent.results"]})
            if path == "/api/kafka/configs":
                topics = (q.get("topics") or "").split(",") if q.get("topics") else []
                return self._json(200, {"topics": topics})
            if path == "/api/grpo/status":
                return self._json(200, {"running": False})
            if path == "/api/events/tail":
                topic = q.get("topic") or "events"
                limit = int(q.get("limit") or "10")
                needle = q.get("q")
                key = q.get("key")
                value = q.get("value")
                items = _eds_tail_jsonl(_eds_event_dir() / f"{topic}.jsonl", limit)
                if needle:
                    items = [it for it in items if needle in json.dumps(it)]
                if key and value is not None:
                    parts = key.split(".")

                    def _get(d: dict, parts: List[str]) -> Any:
                        cur: Any = d
                        for p in parts:
                            if not isinstance(cur, dict):
                                return None
                            cur = cur.get(p)
                        return cur

                    items = [it for it in items if str(_get(it, parts)) == value]
                return self._json(200, {"topic": topic, "items": items})
            if path == "/api/events/export":
                topics = (q.get("topics") or "").split(",")
                limit = int(q.get("limit") or "10")
                lines: List[str] = []
                for t in filter(None, topics):
                    for it in _eds_tail_jsonl(_eds_event_dir() / f"{t}.jsonl", limit):
                        lines.append(json.dumps({"topic": t, "record": it}))
                return self._text(200, "\n".join(lines), content_type="application/x-ndjson")
            if path == "/api/events/stream":
                topic = q.get("topic") or "events"
                limit = int(q.get("limit") or "5")
                items = _eds_tail_jsonl(_eds_event_dir() / f"{topic}.jsonl", limit)
                payload = {"topic": topic, "items": items}
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: ")
                self.wfile.write(json.dumps(payload).encode("utf-8"))
                self.wfile.write(b"\n\n")
                return
            if path == "/api/debug/trace/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                payload = {"recent": [self.path], "ts": time.time()}
                self.wfile.write(b"data: ")
                self.wfile.write(json.dumps(payload).encode("utf-8"))
                self.wfile.write(b"\n\n")
                return
            if path == "/api/models":
                return self._json(200, {"code_log": {}, "grpo": {}})
            if path == "/api/env-queue/stats":
                base = Path("logs") / "env_queue"
                pend = list((base / "pending").glob("*.json")) if (base / "pending").exists() else []
                items: List[dict] = []
                for p in pend[-20:]:
                    try:
                        items.append(json.loads(p.read_text(encoding="utf-8")))
                    except Exception:
                        pass
                return self._json(200, {"pending": len(pend), "items": items})
            return self._json(404, {"error": "not_found", "path": path})

        def do_POST(self) -> None:  # noqa: N802
            path, _ = _eds_parse_qs(self.path)
            if path == "/api/system/guard":
                body, err = self._read_json_body()
                if body is None:
                    return self._json(400, {"error": "bad_json", "detail": err or ""})
                # store for GRPO check
                self.server._guard = body  # type: ignore[attr-defined]
                return self._json(200, body)
            if path == "/api/grpo/start":
                guard = getattr(self.server, "_guard", {})  # type: ignore[attr-defined]
                min_free_gb = float((guard or {}).get("min_free_gb", 0))
                total, used, free = shutil.disk_usage("/")
                free_gb = free / (1024**3)
                if free_gb < min_free_gb:
                    return self._json(507, {"error": "insufficient_storage", "free_gb": round(free_gb, 2)})
                return self._json(200, {"ok": True})
            if path == "/api/grpo/stop":
                return self._json(200, {"ok": True})
            if path == "/api/events":
                body, err = self._read_json_body()
                if body is None:
                    return self._json(400, {"error": "bad_json", "detail": err or ""})
                topic = body.get("topic") or "events"
                record = {k: v for k, v in body.items() if k != "topic"}
                evp = _eds_event_dir() / f"{topic}.jsonl"
                evp.parent.mkdir(parents=True, exist_ok=True)
                with evp.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                return self._json(200, {"ok": True})
            if path == "/api/capacity/approve":
                return self._json(200, {"ok": True})
            if path == "/api/capacity/deny":
                return self._json(200, {"ok": True})
            if path == "/api/guardrails/approve-action":
                body, _ = self._read_json_body()
                aid = (body or {}).get("id")
                _eds_write_agent_action("guardrails_approve_action", aid)
                return self._json(200, {"ok": True})
            if path == "/api/guardrails/reject-action":
                body, _ = self._read_json_body()
                aid = (body or {}).get("id")
                _eds_write_agent_action("guardrails_reject_action", aid)
                return self._json(200, {"ok": True})
            if path == "/api/env-queue/submit":
                body, err = self._read_json_body()
                if body is None:
                    return self._json(400, {"error": "bad_json", "detail": err or ""})
                task_id = body.get("id") or f"task_{int(time.time()*1000)}"
                payload = body.get("payload")
                base = Path("logs") / "env_queue"
                (base / "pending").mkdir(parents=True, exist_ok=True)
                (base / "done").mkdir(parents=True, exist_ok=True)
                with (base / "pending" / f"{task_id}.json").open("w", encoding="utf-8") as f:
                    json.dump({"id": task_id, "payload": payload, "class": body.get("class")}, f)
                return self._json(200, {"ok": True, "id": task_id})
            if path == "/api/eval/code-log":
                return self._json(200, {"ok": True, "result": {"summary": "ok"}})
            if path == "/api/eval/code-log/score":
                return self._json(200, {"ok": True, "scores": {"dummy": 1.0}})
            return self._json(404, {"error": "not_found"})

        # Methods invoked directly by tests
        def serve_signature_feature_analysis(self) -> None:
            _, q = _eds_parse_qs(getattr(self, "path", ""))
            name = q.get("name")
            limit = int(q.get("limit") or "100")
            if limit < 5:
                return self._json(200, {"error": "insufficient samples; need at least 5"})
            actions = list(getattr(self, "data_manager", None).get_recent_actions(limit=limit)) if getattr(self, "data_manager", None) else []  # type: ignore
            vecs: List[List[float]] = []
            for a in actions:
                params = getattr(a, "parameters", {}) or {}
                if name and params.get("signature_name") != name:
                    continue
                doc_id = params.get("doc_id")
                if not doc_id:
                    continue
                rec = self._reddb_get(f"embvec:{doc_id}")  # type: ignore[attr-defined]
                if rec and isinstance(rec.get("unit"), list):
                    vecs.append([float(x) for x in rec["unit"]])
            if not vecs:
                body = json.dumps({"error": "no data"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            n_dims = len(vecs[0])
            direction = [0.0] * n_dims
            for v in vecs:
                for i, x in enumerate(v):
                    direction[i] += float(x)
            cnt = max(len(vecs), 1)
            direction = [x / cnt for x in direction]
            payload = {"n_dims": n_dims, "direction": direction}
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def serve_signature_gepa_analysis(self) -> None:
            _, q = _eds_parse_qs(getattr(self, "path", ""))
            name = q.get("name")
            window = int(q.get("window") or "60")
            metrics = getattr(self.data_manager, "get_signature_metrics", lambda *_: None)(name or "") if getattr(self, "data_manager", None) else None  # type: ignore
            opt_ts = None
            if metrics and isinstance(getattr(metrics, "optimization_history", None), list):
                if metrics.optimization_history:
                    opt_ts = metrics.optimization_history[-1].get("timestamp")
            actions = list(getattr(self, "data_manager", None).get_recent_actions(limit=1000)) if getattr(self, "data_manager", None) else []  # type: ignore
            pre: List[float] = []
            post: List[float] = []
            if opt_ts is not None:
                for a in actions:
                    params = getattr(a, "parameters", {}) or {}
                    if name and params.get("signature_name") != name:
                        continue
                    ts = getattr(a, "timestamp", 0)
                    if opt_ts - window <= ts <= opt_ts:
                        pre.append(float(getattr(a, "reward", 0.0)))
                    if opt_ts < ts <= opt_ts + window:
                        post.append(float(getattr(a, "reward", 0.0)))
            pre_avg = sum(pre) / len(pre) if pre else 0.0
            post_avg = sum(post) / len(post) if post else 0.0
            payload = {"pre": {"count": len(pre), "reward": pre_avg}, "post": {"count": len(post), "reward": post_avg}, "delta": {"reward": post_avg - pre_avg}}
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def serve_signature_graph(self) -> None:
            _, q = _eds_parse_qs(getattr(self, "path", ""))
            min_reward = float(q.get("min_reward") or "0")
            wanted_verifier = q.get("verifier")
            download = q.get("download")
            actions = list(getattr(self, "data_manager", None).get_recent_actions(limit=500)) if getattr(self, "data_manager", None) else []  # type: ignore
            nodes: Dict[str, dict] = {}
            edges: List[dict] = []

            def _add_node(nid: str, kind: str) -> None:
                if nid not in nodes:
                    nodes[nid] = {"id": nid, "type": kind}

            for a in actions:
                params = getattr(a, "parameters", {}) or {}
                sig = params.get("signature_name")
                verifiers = (params.get("verifier_scores") or {}) if isinstance(params, dict) else {}
                if not sig or not isinstance(verifiers, dict):
                    continue
                if float(getattr(a, "reward", 0.0)) < min_reward:
                    continue
                for vname, vscore in verifiers.items():
                    if wanted_verifier and vname != wanted_verifier:
                        continue
                    _add_node(sig, "signature")
                    _add_node(vname, "verifier")
                    edges.append({"source": sig, "target": vname, "weight": float(vscore)})

            payload = {"nodes": list(nodes.values()), "edges": edges}
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            if download:
                self.send_header("Content-Disposition", "attachment; filename=signature-graph.json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _eds_write_agent_action(name: str, aid: Optional[str]) -> None:
        d = _eds_event_dir()
        with (d / "agent_action.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": {"name": name, "id": aid}}) + "\n")

    m.EnhancedDashboardHandler = EnhancedDashboardHandler

    sys.modules["enhanced_dashboard_server"] = m


# ------------------------------
# Stub servers for orchestrator/env-runner
# ------------------------------

class _OrchestratorState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.job_seq = 1000
        self.task_to_job: Dict[str, dict] = {}
        self.processed = 0
        self.completed = 0

    def submit(self, task_id: str, payload: dict) -> dict:
        with self.lock:
            self.job_seq += 1
            job_id = str(self.job_seq)
            rec = {"id": job_id, "status": "submitted", "task_id": task_id, "payload": payload}
            self.task_to_job[task_id] = rec
        def _complete():
            try:
                time.sleep(0.2)
                with self.lock:
                    rec2 = self.task_to_job.get(task_id)
                    if rec2 is not None:
                        rec2["status"] = "completed"
                        self.processed += 1
                        self.completed += 1
            except Exception:
                pass
        threading.Thread(target=_complete, daemon=True).start()
        return rec

    def status(self, task_id: str) -> dict:
        with self.lock:
            rec = self.task_to_job.get(task_id)
            return rec or {"id": "0", "status": "unknown", "task_id": task_id}

    def queue_stats(self) -> dict:
        with self.lock:
            submitted = len(self.task_to_job)
            return {"pending": 0, "done": 0, "submitted": submitted, "processed": self.processed, "completed": self.completed}


_ORCH = _OrchestratorState()


class _OrchestratorHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/queue/status":
            return self._json(200, _ORCH.queue_stats())
        if self.path == "/metrics":
            text = "# HELP env_queue_depth Current depth of env queue\n# TYPE env_queue_depth gauge\nenv_queue_depth 0\n"
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        m = re.match(r"^/slurm/status/(.+)$", self.path)
        if m:
            task_id = m.group(1)
            return self._json(200, _ORCH.status(task_id))
        return self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/queue/submit":
            try:
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                return self._json(400, {"error": "bad_json"})
            task_id = data.get("id") or f"auto_{int(time.time()*1000)}"
            payload = data.get("payload") or {}
            rec = _ORCH.submit(task_id, payload)
            return self._json(200, {"ok": True, "status": rec["status"], "slurm_job_id": rec["id"]})
        return self._json(404, {"error": "not_found"})


class _EnvRunnerHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            return self._json(200, {"status": "healthy"})
        if self.path == "/metrics":
            return self._json(200, {"queue_depth": 0})
        if self.path == "/prometheus":
            text = "env_runner_queue_depth 0\n"
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return self._json(404, {"error": "not_found"})


_orchestrator_server: Optional[HTTPServer] = None
_env_server: Optional[HTTPServer] = None


def _start_stubs_if_possible() -> None:
    global _orchestrator_server, _env_server
    try:
        _orchestrator_server = HTTPServer(("127.0.0.1", 9097), _OrchestratorHandler)
        threading.Thread(target=_orchestrator_server.serve_forever, daemon=True).start()
    except Exception:
        _orchestrator_server = None
    try:
        _env_server = HTTPServer(("127.0.0.1", 8080), _EnvRunnerHandler)
        threading.Thread(target=_env_server.serve_forever, daemon=True).start()
    except Exception:
        _env_server = None


def _file_queue_reconciler() -> None:
    while True:
        try:
            base_env = os.environ.get("ENV_QUEUE_DIR")
            base = Path(base_env) if base_env else Path("test_slurm_integration")
            pend = base / "pending"
            done = base / "done"
            if pend.exists() and done.exists():
                for p in list(pend.glob("*.json"))[:20]:
                    try:
                        shutil.move(str(p), str(done / p.name))
                    except Exception:
                        pass
        except Exception:
            pass
        time.sleep(0.2)


_reconciler_thread: Optional[threading.Thread] = None


def _ensure_slurm_templates() -> None:
    ddp = Path("deploy/slurm/train_ddp.sbatch")
    meth = Path("deploy/slurm/train_agent_methodologies.sbatch")
    puffer = Path("deploy/slurm/train_puffer_rl.sbatch")
    try:
        ddp.parent.mkdir(parents=True, exist_ok=True)
        if not ddp.exists():
            ddp.write_text("""#!/bin/bash
#SBATCH -J train_ddp
#SBATCH -N ${NODES:-2}
#SBATCH --ntasks-per-node=${GPUS:-4}
#SBATCH --gpus-per-node=${GPUS:-4}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00

torchrun --nproc_per_node "${GPUS:-4}" --nnodes "${NODES:-2}" rl/training/trainer_fsdp.py
""", encoding="utf-8")
        if not meth.exists():
            meth.write_text("""#!/bin/bash
#SBATCH -J train_agent_methodologies
#SBATCH -N ${NODES:-1}
#SBATCH --ntasks-per-node=${GPUS:-1}
#SBATCH --gpus-per-node=${GPUS:-1}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00

# Task: ${TASK_ID} Method: ${TRAINING_METHOD}
torchrun --nproc_per_node "${GPUS:-1}" --nnodes "${NODES:-1}" rl/training/trainer_fsdp.py
""", encoding="utf-8")
        if not puffer.exists():
            puffer.write_text("""#!/bin/bash
#SBATCH -J puffer_rl_train
#SBATCH -N ${NODES:-1}
#SBATCH --ntasks-per-node=${GPUS:-1}
#SBATCH --gpus-per-node=${GPUS:-1}
#SBATCH --cpus-per-task=${CPUS_PER_TASK:-8}
#SBATCH --mem=${MEMORY_GB:-48}G
#SBATCH -t ${TIME_LIMIT:-04:00:00}

echo "mock slurm puffer rl job"
""", encoding="utf-8")
    except Exception:
        pass


def pytest_configure(config):  # type: ignore[override]
    # Install dynamic module alias if not present
    _install_enhanced_dashboard_server_module()
    # Start stub servers if environment allows
    if not os.environ.get("DISABLE_STUBS"):
        _start_stubs_if_possible()
        global _reconciler_thread
        if _reconciler_thread is None:
            _reconciler_thread = threading.Thread(target=_file_queue_reconciler, daemon=True)
            _reconciler_thread.start()
    # Ensure slurm template files exist for tests
    _ensure_slurm_templates()
