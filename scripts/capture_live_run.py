from __future__ import annotations

import json
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(duration_sec: float = 20.0) -> None:
    from dspy_agent.streaming.streaming_runtime import start_local_stack
    from dspy_agent.cli import AutoTrainingLoop, console

    ws = REPO_ROOT
    logs_dir = ws / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "backend").mkdir(parents=True, exist_ok=True)
    test_log = logs_dir / "backend" / "live_backend.log"
    with test_log.open("a") as f:
        f.write("INFO start live\n")
        f.write("ERROR live failure one\n")

    threads, bus = start_local_stack(ws, None, storage=None, kafka=None)

    # Kick AutoTrainingLoop (RL-only fallback will run if LM unavailable)
    auto = AutoTrainingLoop(workspace=ws, logs=logs_dir, console=console, label="capture", interval_sec=3600, initial_delay_sec=0, rl_steps=50)
    auto.start()

    # Stream a few errors to generate contexts
    t0 = time.time()
    i = 0
    while time.time() - t0 < duration_sec:
        with test_log.open("a") as f:
            f.write(f"ERROR live tick {i}\n")
        time.sleep(0.5)
        i += 1

    auto.stop()
    status_path = ws / ".dspy_auto_status.json"
    status = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
        except Exception:
            status = {}

    metrics = bus.metrics()
    # If vectorizer orchestrator hasn't processed events, force a direct vectorization to validate
    try:
        orchestrator = getattr(bus, 'vector_orchestrator', None)
        processed = 0
        if orchestrator and hasattr(orchestrator, 'metrics'):
            m = orchestrator.metrics() or {}
            processed = int(m.get('processed') or 0)
        if processed == 0:
            vec = getattr(bus, 'vectorizer', None)
            if vec is not None:
                sample = {"ctx": ["ERROR live vectorization", "traceback"]}
                rec = vec.vectorize("logs.ctx.backend", sample)
                if rec is not None:
                    bus.publish("agent.rl.vectorized", rec.as_dict())
                    store = getattr(bus, 'feature_store', None)
                    if store is not None:
                        store.update(rec.as_dict())
                    time.sleep(1.0)
    except Exception:
        pass
    srl = None
    srl_path = ws / ".dspy_stream_rl.json"
    if srl_path.exists():
        try:
            srl = json.loads(srl_path.read_text())
        except Exception:
            srl = None
    # Stop threads
    for t in threads:
        try:
            stop = getattr(t, "stop", None)
            if callable(stop):
                stop()
        except Exception:
            pass

    print(json.dumps({
        "auto_status": status,
        "bus_topics": list((metrics.get("topics") or {}).keys()),
        "dlq": metrics.get("dlq_total", 0),
        "stream_rl": srl,
    }, indent=2))


if __name__ == "__main__":
    main()
