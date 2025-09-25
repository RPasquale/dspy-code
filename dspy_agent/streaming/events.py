from __future__ import annotations

import os
import socket
import time
from typing import Any, Dict, Optional

from .event_bus import get_event_bus, EventBus


# Canonical topics (keep flat and simple)
UI_ACTION = "ui.action"
BACKEND_API = "backend.api_call"
AGENT_ACTION = "agent.action"
INGEST_DECISION = "ingest.decision"
INGEST_FILE = "ingest.file"
TRAINING_TRIGGER = "training.trigger"
TRAINING_RESULT = "training.result"
TRAINING_DATASET = "training.dataset"
SPARK_APP = "spark.app"
SPARK_LOG = "spark.log"

ALLOWED_TOPICS = {
    UI_ACTION,
    BACKEND_API,
    AGENT_ACTION,
    INGEST_DECISION,
    INGEST_FILE,
    TRAINING_TRIGGER,
    TRAINING_RESULT,
    TRAINING_DATASET,
    SPARK_APP,
    SPARK_LOG,
}


def _context_meta(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a small, consistent envelope for every event."""
    return {
        "ts": time.time(),
        "service": os.getenv("SERVICE_NAME") or os.getenv("APP_NAME") or "dspy-agent",
        "hostname": socket.gethostname(),
        "pod": os.getenv("POD_NAME") or os.getenv("HOSTNAME") or "",
        "container": os.getenv("CONTAINER_NAME") or "",
        "app_version": os.getenv("APP_VERSION") or os.getenv("GIT_SHA") or "",
        **(extra or {}),
    }


def publish_event(topic: str, payload: Dict[str, Any], *, bus: Optional[EventBus] = None) -> None:
    """Publish an event to Kafka/RedDB/file with minimal code.

    Ensures a consistent metadata envelope and topic whitelist.
    """
    if topic not in ALLOWED_TOPICS:
        # Avoid noisy failures; route to a generic stream if needed
        topic = "events.other"
    # Capture caller (file:function:line) to enable code→log dataset mining
    caller = None
    try:
        import inspect, os as _os
        # Find first frame outside this module
        for frame, filename, lineno, func, _, _ in inspect.stack()[1:]:
            if __file__.rstrip('c') not in filename:
                # Normalize path
                f = filename
                try:
                    # Prefer relative to repo root when possible
                    from pathlib import Path
                    repo = Path(__file__).resolve().parents[2]
                    f = str(Path(filename).resolve())
                    try:
                        f = str(Path(f).resolve().relative_to(repo))
                    except Exception:
                        pass
                except Exception:
                    pass
                caller = {"file": f, "function": func, "line": lineno}
                break
    except Exception:
        caller = None
    meta = _context_meta(payload.pop("meta", None))
    if caller and 'caller' not in meta:
        meta['caller'] = caller
    rec = {"topic": topic, **meta, "event": payload}
    (bus or get_event_bus()).publish(topic, rec)


# Convenience helpers (thin wrappers)
def log_ui_action(action: str, props: Optional[Dict[str, Any]] = None, *, bus: Optional[EventBus] = None) -> None:
    publish_event(UI_ACTION, {"action": action, **(props or {})}, bus=bus)


def log_backend_api(route: str, method: str, status: int, duration_ms: float, *, bus: Optional[EventBus] = None, meta: Optional[Dict[str, Any]] = None) -> None:
    publish_event(BACKEND_API, {"route": route, "method": method, "status": status, "duration_ms": duration_ms, "meta": meta or {}}, bus=bus)


def log_agent_action(name: str, result: str = "", reward: Optional[float] = None, *, bus: Optional[EventBus] = None, **kwargs: Any) -> None:
    data = {"name": name, "result": result}
    if reward is not None:
        data["reward"] = reward
    data.update(kwargs)
    publish_event(AGENT_ACTION, data, bus=bus)


def log_ingest_decision(source_file: str, decision: str, reason: str = "", *, bus: Optional[EventBus] = None) -> None:
    publish_event(INGEST_DECISION, {"source_file": source_file, "decision": decision, "reason": reason}, bus=bus)


def log_training_trigger(trainer: str, args: Dict[str, Any], *, bus: Optional[EventBus] = None) -> None:
    publish_event(TRAINING_TRIGGER, {"trainer": trainer, "args": args}, bus=bus)


def log_training_result(status: str, metrics: Optional[Dict[str, Any]] = None, *, bus: Optional[EventBus] = None) -> None:
    publish_event(TRAINING_RESULT, {"status": status, "metrics": metrics or {}}, bus=bus)


def log_spark_app(event: str, name: str, state: str = "", namespace: str = "default", *, bus: Optional[EventBus] = None) -> None:
    publish_event(SPARK_APP, {"event": event, "name": name, "state": state, "namespace": namespace}, bus=bus)


def log_spark_log(app: str, pod: str, line: str, *, bus: Optional[EventBus] = None) -> None:
    publish_event(SPARK_LOG, {"app": app, "pod": pod, "line": line}, bus=bus)


def log_training_dataset(kind: str, detail: Dict[str, Any], *, bus: Optional[EventBus] = None) -> None:
    publish_event(TRAINING_DATASET, {"kind": kind, **detail}, bus=bus)
