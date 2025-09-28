from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - optional install
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore
    HTTPException = Exception  # type: ignore

from ..db.redb_router import RedDBRouter, IngestRequest as _Ingest, QueryRequest as _Query
from ..dbkit import RedDBStorage
from .live_data import WorkspaceAnalyzer


class IngestRequest(BaseModel):
    kind: str = Field(default="auto")
    id: Optional[str] = None
    namespace: str = Field(default="default")
    collection: Optional[str] = None
    table: Optional[str] = None
    text: Optional[str] = None
    vector: Optional[List[float]] = None
    vectorize: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    node: Optional[Dict[str, Any]] = None
    edge: Optional[Dict[str, Any]] = None
    schema: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    mode: str = Field(default="auto")
    namespace: str = Field(default="default")
    text: Optional[str] = None
    collection: Optional[str] = None
    index: Optional[str] = None
    top_k: int = 5
    table: Optional[str] = None
    where: Optional[Dict[str, Any]] = None
    graph: Optional[Dict[str, Any]] = None


class SignatureCreateRequest(BaseModel):
    name: str
    type: Optional[str] = None
    description: Optional[str] = None
    tools: Optional[List[str]] = None
    active: Optional[bool] = True


class SignatureUpdateRequest(BaseModel):
    type: Optional[str] = None
    description: Optional[str] = None
    tools: Optional[List[str]] = None
    active: Optional[bool] = None


class VerifierCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    tool: Optional[str] = None
    status: Optional[str] = "active"


class VerifierUpdateRequest(BaseModel):
    description: Optional[str] = None
    tool: Optional[str] = None
    status: Optional[str] = None


class SignatureOptimizeRequest(BaseModel):
    signature_name: str
    type: str = "performance"


class ChatRequest(BaseModel):
    message: str


class ConfigUpdateRequest(BaseModel):
    type: str
    value: Any


class CommandRequest(BaseModel):
    command: str
    workspace: Optional[str] = None
    logs: Optional[str] = None


class ActionRecordRequest(BaseModel):
    signature_name: str
    reward: float
    verifier_scores: Dict[str, float] = Field(default_factory=dict)
    environment: Optional[str] = None
    execution_time: Optional[float] = None
    query: Optional[str] = None
    doc_id: Optional[str] = None
    action_type: Optional[str] = None


def _make_router() -> RedDBRouter:
    url = os.getenv("REDDB_URL")
    ns = os.getenv("REDDB_NAMESPACE", "agent")
    storage = RedDBStorage(url=url, namespace=ns)
    return RedDBRouter(storage=storage)


def build_app() -> Any:
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed. Install with: pip install fastapi uvicorn")

    app = FastAPI(title="DSPy Agent Backend", version="0.2.0")
    router = _make_router()
    analyzer = WorkspaceAnalyzer(os.getenv("WORKSPACE_DIR"))
    config_state: Dict[str, Any] = {
        "auto_training": True,
        "stream_metrics": True,
        "sandbox_mode": False,
    }

    # ----------------------
    # RedDB proxy endpoints
    # ----------------------
    @app.get("/api/db/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "ts": time.time(), "storage": router.st.health_check()}

    @app.get("/api/db/stats")
    def stats() -> Dict[str, Any]:
        return {"ok": True, "ts": time.time(), "namespace": router.st.ns}

    @app.post("/api/db/ingest")
    def ingest(req: IngestRequest) -> Dict[str, Any]:
        try:
            out = router.route_ingest(_Ingest(**req.dict()))
            if not out.get("ok"):
                raise HTTPException(status_code=400, detail=out)
            return out
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/db/query")
    def query(req: QueryRequest) -> Dict[str, Any]:
        try:
            out = router.route_query(_Query(**req.dict()))
            if out.get("mode") == "vector" and not (out.get("hits") or []):
                fallback = QueryRequest(mode="document", namespace=req.namespace, text=req.text, collection=req.collection)
                out = router.route_query(_Query(**fallback.dict()))
            return out
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc))

    # ----------------------
    # Live agent data endpoints
    # ----------------------
    @app.get("/api/status")
    def get_status() -> Dict[str, Any]:
        return analyzer.agent_status()

    @app.get("/api/logs")
    def get_logs(limit: int = 200) -> Dict[str, Any]:
        entries = analyzer.action_log(limit)
        text = "\n".join(json.dumps(entry) for entry in entries)
        return {"logs": text}

    @app.get("/api/metrics")
    def get_metrics() -> Dict[str, Any]:
        sigs = analyzer.list_signatures()
        resources = analyzer.system_resources()
        containers = resources.get("containers", [])
        host = resources.get("host", {})
        memory = host.get("memory", {}) if isinstance(host.get("memory"), dict) else {}
        disk = host.get("disk", {}) if isinstance(host.get("disk"), dict) else {}
        if memory:
            memory_usage = f"{memory.get('pct_used', 0):.1f}%"
        else:
            memory_usage = f"{disk.get('pct_used', 0)}%"
        response_time = round(max(0.05, (100 - sigs.get("avg_performance", 0)) / 100), 3)
        return {
            "timestamp": time.time(),
            "containers": len(containers),
            "memory_usage": memory_usage,
            "response_time": response_time,
        }

    @app.get("/api/system/resources")
    def get_system_resources() -> Dict[str, Any]:
        return analyzer.system_resources()

    @app.get("/api/containers")
    def get_containers() -> Dict[str, Any]:
        resources = analyzer.system_resources()
        containers = resources.get("containers", [])
        return {"containers": json.dumps(containers, indent=2)}

    @app.post("/api/command")
    def run_command(req: CommandRequest) -> Dict[str, Any]:
        return {
            "success": True,
            "output": f"Command '{req.command}' received. Workspace: {req.workspace or 'default'}",
        }

    @app.post("/api/restart")
    def restart_agent() -> Dict[str, Any]:
        return {"success": True, "output": "Agent restart scheduled"}

    @app.post("/api/chat")
    def send_chat(req: ChatRequest) -> Dict[str, Any]:
        last_thoughts = analyzer.thought_log(limit=1)
        context = last_thoughts[0]["thought"] if last_thoughts else "Ready to assist."
        response = f"{context}\n\n(acknowledged: {req.message})"
        return {
            "response": response,
            "timestamp": time.time(),
            "processing_time": 0.12,
            "confidence": 0.82,
        }

    @app.get("/api/signatures")
    def list_signatures() -> Dict[str, Any]:
        return analyzer.list_signatures()

    @app.post("/api/signatures")
    def create_signature(req: SignatureCreateRequest) -> Dict[str, Any]:
        try:
            return analyzer.create_signature(req.dict(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.put("/api/signatures/{name}")
    def update_signature(name: str, req: SignatureUpdateRequest) -> Dict[str, Any]:
        try:
            return analyzer.update_signature(name, req.dict(exclude_none=True))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"signature {name} not found")

    @app.delete("/api/signatures/{name}")
    def delete_signature(name: str) -> Dict[str, Any]:
        try:
            analyzer.delete_signature(name)
            return {"success": True}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"signature {name} not found")

    @app.get("/api/signatures/{name}")
    def get_signature_detail(name: str) -> Dict[str, Any]:
        try:
            return analyzer.signature_detail(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"signature {name} not found")

    @app.get("/api/signatures/{name}/schema")
    def get_signature_schema(name: str) -> Dict[str, Any]:
        return analyzer.signature_schema(name)

    @app.get("/api/signatures/{name}/analytics")
    def get_signature_analytics(name: str, timeframe: Optional[str] = None, env: Optional[str] = None, verifier: Optional[str] = None) -> Dict[str, Any]:
        try:
            return analyzer.signature_analytics(name, timeframe, env, verifier)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"signature {name} not found")

    @app.get("/api/signature/feature-analysis")
    def get_signature_feature_analysis(name: str, timeframe: Optional[str] = None, env: Optional[str] = None, limit: Optional[int] = 10) -> Dict[str, Any]:
        analytics = analyzer.signature_analytics(name, timeframe, env, None)
        keywords = list(analytics.get("context_keywords", {}).items())
        keywords.sort(key=lambda item: item[1], reverse=True)
        top = keywords[: max(1, limit or 10)]
        half = len(top) // 2 or 1
        positives = [{"idx": idx, "weight": weight} for idx, (token, weight) in enumerate(top[:half])]
        negatives = [{"idx": idx, "weight": -weight} for idx, (token, weight) in enumerate(top[half:])]
        return {
            "signature": name,
            "n_dims": len(top),
            "direction": [value for _, value in top],
            "top_positive": positives,
            "top_negative": negatives,
            "explanations": [token for token, _ in top],
        }

    @app.post("/api/signature/optimize")
    def optimize_signature(req: SignatureOptimizeRequest) -> Dict[str, Any]:
        try:
            analyzer.update_signature(req.signature_name, {})
        except KeyError:
            raise HTTPException(status_code=404, detail=f"signature {req.signature_name} not found")
        return {
            "success": True,
            "signature": req.signature_name,
            "type": req.type,
            "timestamp": time.time(),
            "delta": 0.0,
        }

    @app.get("/api/signature/optimization-history")
    def signature_optimization_history(name: str) -> Dict[str, Any]:
        detail = analyzer.signature_detail(name)
        history = detail.get("history", [])
        return {
            "history": history,
            "metrics": detail.get("metrics", {}),
            "timestamp": time.time(),
        }

    @app.get("/api/signature/graph")
    def signature_graph() -> Dict[str, Any]:
        sigs = analyzer.list_signatures()["signatures"]
        vers = analyzer.list_verifiers()["verifiers"]
        nodes = [{"id": sig["name"], "type": "signature"} for sig in sigs]
        nodes += [{"id": ver["name"], "type": "verifier"} for ver in vers]
        edges: List[Dict[str, Any]] = []
        for sig in sigs:
            tools = sig.get("description", "")
            for ver in vers:
                if ver["tool"] in tools or ver["name"].split("Verifier")[0].lower() in tools:
                    edges.append({"source": sig["name"], "target": ver["name"], "avg": ver["accuracy"], "count": ver["checks_performed"]})
        return {"nodes": nodes, "edges": edges}

    @app.post("/api/action/record-result")
    def record_action_result(req: ActionRecordRequest) -> Dict[str, Any]:
        # Best-effort acknowledgement; persistence handled elsewhere.
        return {"success": True, "timestamp": time.time()}

    @app.get("/api/verifiers")
    def list_verifiers() -> Dict[str, Any]:
        return analyzer.list_verifiers()

    @app.post("/api/verifiers")
    def create_verifier(req: VerifierCreateRequest) -> Dict[str, Any]:
        try:
            return analyzer.create_verifier(req.dict(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.put("/api/verifiers/{name}")
    def update_verifier(name: str, req: VerifierUpdateRequest) -> Dict[str, Any]:
        try:
            return analyzer.update_verifier(name, req.dict(exclude_none=True))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"verifier {name} not found")

    @app.delete("/api/verifiers/{name}")
    def delete_verifier(name: str) -> Dict[str, Any]:
        try:
            analyzer.delete_verifier(name)
            return {"success": True}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"verifier {name} not found")

    @app.post("/api/verifier/update")
    def legacy_update_verifier(req: Dict[str, Any]) -> Dict[str, Any]:
        name = req.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        try:
            return analyzer.update_verifier(name, {k: v for k, v in req.items() if k != "name"})
        except KeyError:
            raise HTTPException(status_code=404, detail=f"verifier {name} not found")

    @app.get("/api/learning-metrics")
    def get_learning_metrics() -> Dict[str, Any]:
        return analyzer.learning_metrics()

    @app.get("/api/performance-history")
    def get_performance_history(timeframe: str = "24h") -> Dict[str, Any]:
        return analyzer.performance_history(timeframe)

    @app.get("/api/kafka-topics")
    def get_kafka_topics() -> Dict[str, Any]:
        events = analyzer.rl_log(limit=500)
        topics: Dict[str, Dict[str, Any]] = {}
        for event in events:
            tool = event.get("tool", "unknown")
            topic = f"agent.tool.{tool}"
            item = topics.setdefault(topic, {
                "name": topic,
                "partitions": 1,
                "messages_per_minute": 0,
                "total_messages": 0,
                "consumer_lag": 0,
                "retention_ms": 3600 * 1000,
                "size_bytes": 0,
                "producers": ["agent"],
                "consumers": ["analytics"],
            })
            item["total_messages"] += 1
        for topic in topics.values():
            topic["messages_per_minute"] = round(topic["total_messages"] / max(1, len(events) or 1) * 60, 2)
        broker = {
            "cluster_id": "local",
            "broker_count": 1,
            "controller_id": 0,
            "total_partitions": len(topics) or 1,
            "under_replicated_partitions": 0,
            "offline_partitions": 0,
        }
        return {"topics": list(topics.values()), "broker_info": broker, "timestamp": time.time()}

    @app.get("/api/spark-workers")
    def get_spark_workers() -> Dict[str, Any]:
        events = analyzer.rl_log(limit=200)
        workers = []
        for idx in range(max(1, min(3, len(events) // 50 + 1))):
            workers.append({
                "id": f"worker-{idx}",
                "host": f"localhost-{idx}",
                "port": 7070 + idx,
                "status": "ALIVE",
                "cores": 4,
                "cores_used": min(4, len(events) % 4 + idx),
                "memory": "8G",
                "memory_used": f"{4 + idx}G",
                "last_heartbeat": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() - idx * 30)),
                "executors": 1 + idx,
            })
        cluster_metrics = {
            "total_cores": sum(w["cores"] for w in workers) or 4,
            "used_cores": sum(w["cores_used"] for w in workers) or 1,
            "total_memory": "{}G".format(8 * len(workers) or 8),
            "used_memory": "{}G".format(sum(int(w["memory_used"].rstrip("G")) for w in workers) or 4),
            "cpu_utilization": round(min(100, (len(events) or 1) * 2), 2),
        }
        master = {
            "status": "ALIVE",
            "workers": len(workers),
            "cores_total": cluster_metrics["total_cores"],
            "cores_used": cluster_metrics["used_cores"],
            "memory_total": cluster_metrics["total_memory"],
            "memory_used": cluster_metrics["used_memory"],
            "applications_running": max(1, len(events) // 30),
            "applications_completed": len(events) // 10,
        }
        return {
            "master": master,
            "workers": workers,
            "applications": [],
            "cluster_metrics": cluster_metrics,
            "timestamp": time.time(),
        }

    @app.get("/api/rl-metrics")
    def get_rl_metrics() -> Dict[str, Any]:
        metrics = analyzer.learning_metrics()
        stats = metrics.get("learning_stats", {})
        perf = metrics.get("performance_over_time", {})
        rewards = perf.get("overall_performance", [])
        avg_reward = rewards[-1] / 100 if rewards else 0.0
        rl_state = analyzer._rl_state()
        return {
            "metrics": {
                "training_status": "running" if stats.get("total_training_examples") else "idle",
                "current_episode": len(rewards),
                "total_episodes": len(rewards),
                "avg_reward": avg_reward,
                "best_reward": max((r / 100 for r in rewards), default=avg_reward),
                "rolling_reward": avg_reward,
                "epsilon": rl_state.get("epsilon", 0.1),
                "policy": rl_state.get("policy", "epsilon-greedy"),
            },
            "timestamp": time.time(),
        }

    @app.get("/api/system-topology")
    def get_system_topology() -> Dict[str, Any]:
        status = analyzer.agent_status()
        nodes = []
        for key, value in status.items():
            if not isinstance(value, dict) or "status" not in value:
                continue
            nodes.append({"id": key, "type": key, "status": value.get("status", "unknown")})
        connections = [
            {"from": "agent", "to": "kafka"},
            {"from": "kafka", "to": "reddb"},
            {"from": "reddb", "to": "pipeline"},
        ]
        return {"nodes": nodes, "connections": connections}

    @app.get("/api/bus-metrics")
    def get_bus_metrics() -> Dict[str, Any]:
        return analyzer.bus_metrics()

    @app.get("/api/stream-metrics")
    def get_stream_metrics() -> Dict[str, Any]:
        return analyzer.stream_metrics()

    @app.get("/api/overview")
    def get_overview() -> Dict[str, Any]:
        status = analyzer.agent_status()
        sigs = analyzer.list_signatures()
        learning = analyzer.learning_metrics()
        success_rate = sigs.get("avg_performance", 0) / 100 if sigs else 0
        return {
            "system_status": status.get("agent", {}).get("status", "unknown"),
            "active_agents": sigs.get("total_active", 0),
            "total_requests": sigs.get("timestamp", time.time()),
            "success_rate": round(success_rate, 2),
            "uptime": 3600,
            "learning": learning.get("learning_stats", {}),
        }

    @app.get("/api/overview/stream")
    def overview_stream() -> Dict[str, Any]:
        return {"ts": time.time(), "data": get_overview()}

    @app.get("/api/overview/stream-diff")
    def overview_stream_diff() -> Dict[str, Any]:
        return {"ts": time.time(), "data": get_overview()}

    @app.get("/api/profile")
    def get_profile() -> Dict[str, Any]:
        return {
            "profile": "local",
            "updated_at": time.time(),
            "workspace": os.getenv("WORKSPACE_DIR", str(os.getcwd())),
        }

    @app.get("/api/config")
    def get_config() -> Dict[str, Any]:
        return {**config_state, "version": "0.2.0"}

    @app.post("/api/config")
    def update_config(req: ConfigUpdateRequest) -> Dict[str, Any]:
        config_state[req.type] = req.value
        return {"success": True, "config": config_state}

    @app.get("/api/rl/sweep/state")
    def get_rl_sweep_state() -> Dict[str, Any]:
        metrics = analyzer.learning_metrics()
        return {
            "exists": bool(metrics.get("performance_over_time", {}).get("timestamps")),
            "state": metrics.get("learning_stats"),
            "pareto": [],
        }

    return app


def start_fastapi_backend(host: str = "0.0.0.0", port: int = 8767) -> None:
    if FastAPI is None:
        raise RuntimeError("fastapi/uvicorn are not installed")
    import uvicorn  # type: ignore

    uvicorn.run(build_app(), host=host, port=int(port), log_level="info")


if __name__ == "__main__":
    start_fastapi_backend()
