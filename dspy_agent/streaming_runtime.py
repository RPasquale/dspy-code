from __future__ import annotations

import os
import re
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, List, Optional, Tuple, Any

from .streaming_config import StreamConfig, DEFAULT_CONFIG_PATH, save_config as save_stream_cfg, ContainerTopic
from .log_reader import extract_key_events
from .code_snapshot import build_code_snapshot
from .skills.context_builder import ContextBuilder
from .skills.task_agent import TaskAgent
from .llm import configure_lm
from .autogen_dataset import extract_error_phrases
from .train_gepa import run_gepa


class LocalBus:
    def __init__(self, storage: Optional[object] = None, kafka: Optional[object] = None) -> None:
        self._topics: Dict[str, List[Queue]] = {}
        self._lock = threading.Lock()
        # Optional pluggable storage implementing dspy_agent.db.base.Storage
        self._storage = storage
        # Optional Kafka logger with .send(topic, value)
        self._kafka = kafka

    def publish(self, topic: str, message: Any) -> None:
        with self._lock:
            subs = list(self._topics.get(topic, []))
        for q in subs:
            q.put(message)
        # Best-effort persist to storage (fire-and-forget)
        try:
            if self._storage is not None and hasattr(self._storage, "append"):
                # Stream name matches topic
                self._storage.append(topic, message)  # type: ignore[attr-defined]
        except Exception:
            # Never break local flow due to persistence issues
            pass
        # Best-effort publish to Kafka
        try:
            if self._kafka is not None:
                self._kafka.send(topic, message)  # type: ignore[attr-defined]
        except Exception:
            pass

    def subscribe(self, topic: str) -> Queue:
        q: Queue = Queue()
        with self._lock:
            self._topics.setdefault(topic, []).append(q)
        return q


@dataclass
class Discovered:
    container: str
    service: str
    log_file: Path


def autodiscover_logs(root: Path) -> List[Discovered]:
    candidates: List[Tuple[str, str, Path]] = []
    # Heuristics: look for logs/ folders and *.log files
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".log" or p.name.lower().endswith((".out", ".err")) or p.parts[-2:] == ("logs", p.name):
            # Determine container and service from path segments
            parts = [x for x in p.parts if x not in (".",)]
            container = "backend" if any("back" in seg.lower() for seg in parts) else (
                "frontend" if any("front" in seg.lower() for seg in parts) else "app"
            )
            # Service guess: immediate parent folder name or 'core'
            service = p.parent.name or "core"
            candidates.append((container, service, p))
    # Reduce: pick one file per container as primary (as per current rule)
    chosen: Dict[str, Discovered] = {}
    for container, service, path in candidates:
        if container not in chosen:
            chosen[container] = Discovered(container, service, path)
    return list(chosen.values())


class FileTailer(threading.Thread):
    def __init__(self, path: Path, bus: LocalBus, topic: str, poll_interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.bus = bus
        self.topic = topic
        self.poll_interval = poll_interval
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            with self.path.open("r", errors="ignore") as f:
                # start at end
                f.seek(0, os.SEEK_END)
                while not self._stop.is_set():
                    line = f.readline()
                    if not line:
                        time.sleep(self.poll_interval)
                        continue
                    self.bus.publish(self.topic, {"line": line.rstrip("\n"), "ts": time.time()})
        except Exception:
            # ignore tailer failure silently to keep the agent alive
            pass


class Aggregator(threading.Thread):
    def __init__(self, bus: LocalBus, in_topic: str, out_topic: str, window_sec: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.bus = bus
        self.in_q = bus.subscribe(in_topic)
        self.out_topic = out_topic
        self.window_sec = window_sec
        self._stop = threading.Event()
        self._buf: List[str] = []
        self._last_flush = time.time()
        self._re = re.compile(r"error|warn|traceback|exception|failed|timeout", re.I)

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            now = time.time()
            try:
                item = self.in_q.get(timeout=0.2)
            except Empty:
                item = None
            if item:
                line = str(item.get("line", ""))
                if self._re.search(line):
                    self._buf.append(line)
            if (now - self._last_flush) >= self.window_sec and self._buf:
                ctx = {"ctx": list(self._buf), "ts": now}
                self.bus.publish(self.out_topic, ctx)
                self._buf.clear()
                self._last_flush = now

    # For tests or manual flushes
    def flush_now(self):
        if self._buf:
            ctx = {"ctx": list(self._buf), "ts": time.time()}
            self.bus.publish(self.out_topic, ctx)
            self._buf.clear()
            self._last_flush = time.time()


class Worker(threading.Thread):
    def __init__(self, container: str, bus: LocalBus, ctx_topic: str, results_topic: str) -> None:
        super().__init__(daemon=True)
        self.container = container
        self.bus = bus
        self.in_q = bus.subscribe(ctx_topic)
        self.results_topic = results_topic
        self._stop = threading.Event()
        # Try to configure LM; if not available, fallback to heuristic
        self.lm = configure_lm(provider="ollama", model_name=None, base_url=None, api_key=None)
        self.builder = ContextBuilder()
        self.agent = TaskAgent()

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try:
                item = self.in_q.get(timeout=0.5)
            except Empty:
                continue
            lines = item.get("ctx", [])
            if not lines:
                continue
            text = "\n".join(lines)
            result = process_ctx(self.container, text, self.lm, self.builder, self.agent)
            # Publish to results topic and also print a compact line
            self.bus.publish(self.results_topic, result)
            print(f"[dspy-worker] {self.container} | len={len(lines)} | summary_len={len(result['summary'])}")
            # Optional KV persistence for quick retrieval
            try:
                storage = getattr(self.bus, "_storage", None)
                if storage is not None and hasattr(storage, "put"):
                    prefix = f"last:{self.container}:"
                    storage.put(prefix + "summary", result.get("summary", ""))  # type: ignore[attr-defined]
                    storage.put(prefix + "key_points", result.get("key_points", ""))  # type: ignore[attr-defined]
                    storage.put(prefix + "plan", result.get("plan", ""))  # type: ignore[attr-defined]
                    storage.put(prefix + "ts", result.get("ts", 0))  # type: ignore[attr-defined]
            except Exception:
                pass


def start_local_stack(root: Path, cfg: Optional[StreamConfig] = None, storage: Optional[object] = None, kafka: Optional[object] = None) -> Tuple[List[threading.Thread], LocalBus]:
    """Start autodiscovery tailers + aggregators + workers in-process.

    Returns threads list and the local bus.
    """
    bus = LocalBus(storage=storage, kafka=kafka)
    if cfg is None:
        discs = autodiscover_logs(root)
        # Derive a default StreamConfig and persist
        containers = {}
        for d in discs:
            containers.setdefault(d.container, []).append(d.service)
        cfg = StreamConfig.default()
        cfg.containers = [
            # One service list per container. Use real dataclass for JSON serializable config.
            ContainerTopic(container=k, services=v)
            for k, v in containers.items()
        ]
        save_stream_cfg(cfg)

    threads: List[threading.Thread] = []
    # Map container to chosen log file via autodiscover
    chosen = {d.container: d.log_file for d in autodiscover_logs(root)}
    for ct in cfg.containers:
        container = getattr(ct, 'container')
        log_file = chosen.get(container)
        if not log_file or not log_file.exists():
            continue
        raw_topic = f"logs.raw.{container}"
        ctx_topic = f"logs.ctx.{container}"
        results_topic = f"agent.results.{container}"
        t1 = FileTailer(log_file, bus, raw_topic)
        t2 = Aggregator(bus, raw_topic, ctx_topic, window_sec=5.0)
        t3 = Worker(container, bus, ctx_topic, results_topic)
        t1.start(); t2.start(); t3.start()
        threads.extend([t1, t2, t3])
    return threads, bus


def process_ctx(container: str, text: str, lm: Optional[object], builder: ContextBuilder, agent: TaskAgent) -> Dict[str, Any]:
    try:
        if lm is not None:
            pred = builder(task=f"Summarize {container} errors", logs_preview=text)
            plan = agent(task=f"Plan steps for {container}", context=f"{pred.context}\n\n{pred.key_points}")
            return {
                "container": container,
                "summary": pred.context,
                "key_points": pred.key_points,
                "plan": plan.plan,
                "ts": time.time(),
            }
        else:
            summary = extract_key_events(text)
            return {
                "container": container,
                "summary": summary,
                "key_points": "",
                "plan": "",
                "ts": time.time(),
            }
    except Exception:
        summary = extract_key_events(text)
        return {
            "container": container,
            "summary": summary,
            "key_points": "",
            "plan": "",
            "ts": time.time(),
        }


def make_context_example(lines: List[str]) -> Dict[str, Any]:
    text = "\n".join(lines)
    errs = extract_error_phrases(text)
    return {
        "task": "Summarize logs for debugging",
        "logs_preview": text[:4000],
        "context_keywords": errs[:5],
        "key_points_keywords": errs[5:10],
    }


class Trainer(threading.Thread):
    def __init__(self, root: Path, bus: LocalBus, containers: List[str], min_batch: int = 3, interval_sec: float = 60.0) -> None:
        super().__init__(daemon=True)
        self.root = root
        self.bus = bus
        self.min_batch = min_batch
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._last_train = 0.0
        self._new_examples = 0
        self._notified_no_lm = False
        self.queues: List[Queue] = []
        for c in containers:
            self.queues.append(bus.subscribe(f"logs.ctx.{c}"))

    def stop(self):
        self._stop.set()

    def dataset_path(self) -> Path:
        out = self.root / ".dspy_data"
        out.mkdir(parents=True, exist_ok=True)
        return out / "context_train.jsonl"

    def append_example(self, ex: Dict[str, Any]) -> None:
        p = self.dataset_path()
        with p.open("a") as f:
            import json as _j
            f.write(_j.dumps(ex) + "\n")
        self._new_examples += 1

    def maybe_train(self) -> None:
        now = time.time()
        if self._new_examples >= self.min_batch or (now - self._last_train) >= self.interval_sec:
            p = self.dataset_path()
            if p.exists():
                try:
                    # Only run GEPA if a reflection LM is configured
                    lm = configure_lm(provider="ollama", model_name=None, base_url=None, api_key=None)
                    if lm is None:
                        if not self._notified_no_lm:
                            print("[trainer] No LM configured; skipping training")
                            self._notified_no_lm = True
                    else:
                        run_gepa(module="context", train_jsonl=p, auto="light", reflection_lm=lm)
                        print("[trainer] GEPA training pass complete on context dataset")
                except Exception:
                    print("[trainer] GEPA call failed; continuing")
            self._last_train = now
            self._new_examples = 0

    def run(self):
        while not self._stop.is_set():
            got = False
            for q in self.queues:
                try:
                    item = q.get(timeout=0.2)
                    got = True
                except Empty:
                    item = None
                if item:
                    lines = item.get("ctx", [])
                    if lines:
                        ex = make_context_example(lines)
                        self.append_example(ex)
            if got:
                self.maybe_train()
            else:
                # periodic train check
                self.maybe_train()
                time.sleep(0.2)
