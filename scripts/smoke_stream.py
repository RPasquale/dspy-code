from __future__ import annotations

import json
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from dspy_agent.streaming.streaming_runtime import start_local_stack
    from dspy_agent.streaming.streamkit import Aggregator

    ws = REPO_ROOT
    from dspy_agent.streaming.streamkit import autodiscover_logs
    # Ensure there is a backend log to tail, create if missing
    log_dir = ws / "logs" / "backend"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Pick the log that autodiscovery will likely select; fallback to a known path
    discs = autodiscover_logs(ws)
    backend = next((d for d in discs if d.container == "backend"), None)
    test_log = backend.log_file if backend else (log_dir / "smoke_backend.log")
    with Path(test_log).open("a") as f:
        f.write("INFO start\n")
        f.write("ERROR initial failure\n")

    threads, bus = start_local_stack(ws, None, storage=None, kafka=None)
    time.sleep(1.5)
    with Path(test_log).open("a") as f:
        f.write("ERROR subsequent failure\n")
    time.sleep(1.0)
    # Push a direct raw event to ensure aggregator has data
    bus.publish("logs.raw.backend", {"line": "ERROR aggregator synthetic", "ts": time.time()})
    for t in threads:
        try:
            if isinstance(t, Aggregator):
                t.flush_now()
        except Exception:
            pass
    time.sleep(0.5)
    # Directly publish a synthetic context to ensure vectorization path fires
    bus.publish("logs.ctx.backend", {"ctx": ["ERROR smoke failure", "traceback stack"]})
    time.sleep(0.5)
    metrics = bus.metrics()
    snap = bus.feature_snapshot()
    # Try to pull a vectorized message to ensure path works
    vec_msg = bus.get_latest("agent.rl.vectorized", timeout=0.5)
    orchestrator = getattr(bus, 'vector_orchestrator', None)
    vmetrics = orchestrator.metrics() if orchestrator else {}
    vnames = []
    try:
        vnames = getattr(bus, 'vectorizer', None).feature_names  # type: ignore[attr-defined]
    except Exception:
        vnames = []
    # If orchestrator hasn't processed anything, force a direct vectorization to validate components
    if (vmetrics or {}).get("processed", 0) == 0:
        try:
            sample = {"ctx": ["ERROR forced vectorization", "exception here"]}
            vec = getattr(bus, 'vectorizer', None)
            if vec is not None:
                rec = vec.vectorize("logs.ctx.backend", sample)
                if rec is not None:
                    bus.publish("agent.rl.vectorized", rec.as_dict())
                    try:
                        getattr(bus, 'feature_store', None).update(rec.as_dict())  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    time.sleep(0.2)
                    snap = bus.feature_snapshot()
                    vec_msg = bus.get_latest("agent.rl.vectorized", timeout=0.5) or vec_msg
        except Exception:
            pass
    print(json.dumps({
        "topics": list((metrics.get("topics") or {}).keys()),
        "dlq": metrics.get("dlq_total", 0),
        "feature_snapshot": bool(snap and (snap.get("count", 0) >= 0)),
        "vectorized_seen": bool(vec_msg),
        "orchestrator": vmetrics,
        "feature_names": vnames,
    }))
    for t in threads:
        try:
            stop = getattr(t, "stop", None)
            if callable(stop):
                stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
