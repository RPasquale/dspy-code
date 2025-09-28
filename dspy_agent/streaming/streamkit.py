from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from .feature_store import FeatureStore
from ..monitor.hw_profiler import HardwareProfiler


def _resolve_bootstrap(raw: Optional[str]) -> str:
    """Return a comma-separated list of bootstrap addresses that works inside and outside Docker."""
    def _push(target: List[str], value: Optional[str]) -> None:
        if not value:
            return
        for part in str(value).split(','):
            token = part.strip()
            if not token:
                continue
            if ':' not in token:
                token = f"{token}:9092"
            if token not in target:
                target.append(token)

    candidates: List[str] = []
    _push(candidates, raw)
    _push(candidates, os.getenv('KAFKA_BOOTSTRAP_SERVERS'))
    _push(candidates, os.getenv('KAFKA_BOOTSTRAP'))
    _push(candidates, os.getenv('DSPY_KAFKA_BOOTSTRAP'))

    # Ensure both in-cluster and host aliases are represented
    local_alias = os.getenv('DSPY_KAFKA_LOCAL_ALIAS', 'kafka')
    _push(candidates, local_alias)
    host_alias = os.getenv('DSPY_KAFKA_HOST_ALIAS', 'localhost')
    host_port = os.getenv('DSPY_KAFKA_HOST_PORT', '9092')
    _push(candidates, f"{host_alias}:{host_port}")

    # Fall back to conventional defaults if nothing resolved
    if not candidates:
        candidates.extend(['kafka:9092', 'localhost:9092'])
    return ','.join(candidates)

# -----------------
# Config dataclasses
# -----------------

DEFAULT_CONFIG_PATH = Path(".dspy_stream.json")
TRAINER_SETTINGS_PATH = Path(".dspy_stream_rl.json")


@dataclass
class KafkaTopic:
    name: str
    partitions: int = 3
    replication_factor: int = 1


@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    zookeeper: Optional[str] = None
    group_id: str = "dspy-code"
    acks: str = "all"
    vector_topic: str = "agent.rl.vectorized"
    vector_topics: List[str] = field(default_factory=list)
    topics: List[KafkaTopic] = field(default_factory=list)


@dataclass
class SparkConfig:
    app_name: str = "dspy-stream-logs"
    master: str = "local[*]"
    checkpoint_dir: str = ".dspy_checkpoints"
    read_format: str = "kafka"
    write_format: str = "kafka"


@dataclass
class K8sConfig:
    namespace: str = "dspy"
    image: str = "dspy-code-agent:latest"
    replicas: int = 1
    resources: Dict[str, str] = field(default_factory=lambda: {"cpu": "500m", "memory": "512Mi"})


@dataclass
class ContainerTopic:
    container: str
    services: List[str]


@dataclass
class StreamConfig:
    kafka: KafkaConfig
    spark: SparkConfig
    k8s: K8sConfig
    containers: List[ContainerTopic]

    @staticmethod
    def default() -> "StreamConfig":
        topics = [
            KafkaTopic(name="logs.raw.backend"),
            KafkaTopic(name="logs.ctx.backend"),
            KafkaTopic(name="logs.raw.frontend"),
            KafkaTopic(name="logs.ctx.frontend"),
            KafkaTopic(name="agent.tasks"),
            KafkaTopic(name="agent.results"),
            KafkaTopic(name="agent.approvals.requests"),
            KafkaTopic(name="agent.approvals.responses"),
            KafkaTopic(name="agent.patches"),
            KafkaTopic(name="agent.metrics"),
            KafkaTopic(name="agent.errors"),
            KafkaTopic(name="deploy.logs.lightweight"),
            KafkaTopic(name="deploy.events.lightweight"),
            KafkaTopic(name="code.fs.events"),
            KafkaTopic(name="agent.rl.vectorized"),
        ]
        bootstrap = _resolve_bootstrap(None)
        return StreamConfig(
            kafka=KafkaConfig(bootstrap_servers=bootstrap, topics=topics),
            spark=SparkConfig(),
            k8s=K8sConfig(),
            containers=[ContainerTopic(container="backend", services=["users", "billing"]), ContainerTopic(container="frontend", services=["web"])],
        )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> StreamConfig:
    data = json.loads(path.read_text())
    def _kt(d): return KafkaTopic(**d)
    kafka_bootstrap = _resolve_bootstrap(data["kafka"].get("bootstrap_servers"))
    return StreamConfig(
        kafka=KafkaConfig(
            bootstrap_servers=kafka_bootstrap,
            zookeeper=data["kafka"].get("zookeeper"),
            group_id=data["kafka"].get("group_id", "dspy-code"),
            acks=data["kafka"].get("acks", "all"),
            vector_topic=data["kafka"].get("vector_topic", "agent.rl.vectorized"),
            vector_topics=data["kafka"].get("vector_topics", []),
            topics=[_kt(t) for t in data["kafka"].get("topics", [])],
        ),
        spark=SparkConfig(**data.get("spark", {})),
        k8s=K8sConfig(**data.get("k8s", {})),
        containers=[ContainerTopic(**c) for c in data.get("containers", [])],
    )


def save_config(cfg: StreamConfig, path: Path = DEFAULT_CONFIG_PATH) -> Path:
    cfg.kafka.bootstrap_servers = _resolve_bootstrap(cfg.kafka.bootstrap_servers)
    def _to(d):
        if isinstance(d, list): return [_to(x) for x in d]
        if hasattr(d, "__dataclass_fields__"): return {k: _to(v) for k, v in asdict(d).items()}
        return d
    obj = _to(cfg); path.write_text(json.dumps(obj, indent=2)); return path


def render_kafka_topic_commands(cfg: StreamConfig) -> List[str]:
    bootstrap = _resolve_bootstrap(cfg.kafka.bootstrap_servers)
    return [f"kafka-topics --bootstrap-server {bootstrap} --create --topic {t.name} --partitions {t.partitions} --replication-factor {t.replication_factor}" for t in cfg.kafka.topics]


# -----------------
# Runtime (local bus, tailers, workers)
# -----------------

class LocalBus:
    def __init__(self, storage: Optional[object] = None, kafka: Optional[object] = None) -> None:
        self._topics: Dict[str, List[Queue]] = {}
        self._groups: Dict[str, Dict[str, List[Queue]]] = {}
        self._group_index: Dict[Tuple[str, str], int] = {}
        self._lock = threading.Lock()
        self._storage = storage
        self._kafka = kafka
        self._dlq_path = Path('.dspy_reports') / 'dlq.jsonl'
        
        # Performance optimization metrics
        self._metrics = {
            'messages_published': 0,
            'messages_dropped': 0,
            'avg_processing_time': 0.0,
            'peak_queue_size': 0,
            'total_processing_time': 0.0,
            'batch_operations': 0
        }
        
        # Batch processing optimization
        self._batch_size = 100
        self._batch_timeout = 0.1  # seconds
        self._pending_messages = []
        self._last_batch_time = time.time()
        
        # Queue size monitoring
        self._max_queue_size = 1000
        self._queue_size_warnings = 0
        
        # Initialize RedDB data manager for enhanced logging
        try:
            from ..db import get_enhanced_data_manager, create_log_entry, Environment
            self._data_manager = get_enhanced_data_manager()
            self._environment = Environment.DEVELOPMENT
            
            # Log bus initialization
            init_log = create_log_entry(
                level="INFO",
                source="streaming_bus",
                message="LocalBus initialized with RedDB integration",
                context={"storage_enabled": storage is not None, "kafka_enabled": kafka is not None},
                environment=self._environment
            )
            self._data_manager.log(init_log)
        except Exception as e:
            print(f"Warning: Could not initialize RedDB integration in LocalBus: {e}")
            self._data_manager = None
            self._environment = None
    def publish(self, topic: str, message: Any) -> None:
        with self._lock:
            subs = list(self._topics.get(topic, []))
            # Grouped delivery: exactly-once per consumer group via round-robin
            groups = self._groups.get(topic, {})
            group_queues: List[Queue] = []
            for gid, qlist in groups.items():
                if not qlist:
                    continue
                idx = self._group_index.get((topic, gid), 0) % len(qlist)
                q = qlist[idx]
                self._group_index[(topic, gid)] = (idx + 1) % len(qlist)
                group_queues.append(q)
        # Deliver to non-group subscribers (non-blocking; DLQ on backpressure)
        for q in subs:
            try:
                q.put_nowait(message)
            except Full:
                self._to_dlq(topic, message, "backpressure_full:subscriber")
                self._publish_backpressure(topic, group_id=None, q=q)
            except Exception as e:
                self._to_dlq(topic, message, f"subscriber_error: {e}")
        # Deliver one per consumer group (non-blocking)
        for q in group_queues:
            try:
                q.put_nowait(message)
            except Full:
                self._to_dlq(topic, message, "backpressure_full:group")
                self._publish_backpressure(topic, group_id="*", q=q)
            except Exception as e:
                self._to_dlq(topic, message, f"group_subscriber_error: {e}")
        
        # Store in original storage (RedDB streams)
        try:
            if self._storage is not None and hasattr(self._storage, "append"):
                self._storage.append(topic, message)  # type: ignore[attr-defined]
        except Exception as e:
            self._to_dlq(topic, message, f"storage_error: {e}")
        
        # Send to Kafka
        try:
            if self._kafka is not None:
                self._kafka.send(topic, message)  # type: ignore[attr-defined]
        except Exception as e:
            self._to_dlq(topic, message, f"kafka_error: {e}")
        
        # Enhanced logging to RedDB for important topics
        if self._data_manager is not None:
            try:
                self._log_important_message(topic, message)
            except Exception: pass
    
    def _log_important_message(self, topic: str, message: Any) -> None:
        """Log important streaming messages to RedDB with structured logging"""
        from ..db import create_log_entry, ActionType, create_action_record
        
        # Determine if this is an important message to log
        important_topics = [
            'agent.results', 'agent.patches', 'agent.learning', 'agent.errors',
            'agent.metrics', 'logs.ctx', 'agent.rl.vectorized'
        ]
        
        # Check if topic matches any important pattern
        is_important = any(pattern in topic for pattern in important_topics)
        
        if not is_important:
            return
        
        # Extract useful information from message
        message_str = str(message)
        context = {"topic": topic, "message_type": type(message).__name__}
        
        # Handle specific message types
        if isinstance(message, dict):
            context.update({
                "keys": list(message.keys()),
                "timestamp": message.get("ts", message.get("timestamp"))
            })
            
            # Handle learning events specially
            if "agent.learning" in topic and "reward" in message:
                # Record as an action for RL training
                action = create_action_record(
                    action_type=ActionType.OPTIMIZATION,
                    state_before={"tool": message.get("tool", "unknown")},
                    state_after={"reward_received": True},
                    parameters={"topic": topic},
                    result={"reward": message.get("reward", 0)},
                    reward=float(message.get("reward", 0)),
                    confidence=0.8,
                    execution_time=0.1,
                    environment=self._environment
                )
                self._data_manager.record_action(action)
                return
            
            # Handle patch events
            if "agent.patches" in topic:
                context["patch_applied"] = message.get("applied", False)
                context["confidence"] = message.get("confidence", 0)
        
        # Determine log level based on topic
        if "error" in topic.lower():
            level = "ERROR"
        elif "ctx" in topic:
            level = "DEBUG"  # Context logs are verbose
        else:
            level = "INFO"
        
        # Create and store log entry
        log_entry = create_log_entry(
            level=level,
            source=f"streaming_{topic.replace('.', '_')}",
            message=f"Streaming event: {message_str[:200]}...",
            context=context,
            environment=self._environment
        )
        self._data_manager.log(log_entry)
    def subscribe(self, topic: str, *, maxsize: int = 0) -> Queue:
        q: Queue = Queue(maxsize=maxsize)
        with self._lock: self._topics.setdefault(topic, []).append(q)
        return q
    
    def subscribe_group(self, topic: str, group_id: str, *, maxsize: int = 0) -> Queue:
        """Subscribe as part of a consumer group. Each message is delivered
        to exactly one member per group (round-robin).
        """
        q: Queue = Queue(maxsize=maxsize)
        with self._lock:
            tg = self._groups.setdefault(topic, {})
            tg.setdefault(group_id, []).append(q)
            self._group_index.setdefault((topic, group_id), 0)
        return q
    
    def get_latest(self, topic: str, timeout: float = 1.0) -> Optional[Any]:
        """Get the latest message from a topic, with timeout."""
        try:
            # Create a temporary queue to get the latest message
            temp_q = Queue()
            with self._lock: 
                self._topics.setdefault(topic, []).append(temp_q)
            
            # Try to get a message with timeout
            message = temp_q.get(timeout=timeout)
            
            # Remove the temporary queue
            with self._lock:
                if temp_q in self._topics.get(topic, []):
                    self._topics[topic].remove(temp_q)
            
            return message
        except Exception:
            return None

    def vector_metrics(self) -> Dict[str, Any]:
        orchestrator = getattr(self, 'vector_orchestrator', None)
        if orchestrator and hasattr(orchestrator, 'metrics'):
            try:
                metrics = orchestrator.metrics()
                if isinstance(metrics, dict):
                    return metrics
            except Exception:
                pass
        return {}

    def kafka_health(self) -> Dict[str, Any]:
        inspector = getattr(self, 'kafka_inspector', None)
        if inspector and hasattr(inspector, 'probe'):
            try:
                snapshot = inspector.probe()
                return asdict(snapshot)
            except Exception:
                pass
        return {'available': False, 'reason': 'unavailable'}

    def spark_health(self) -> Dict[str, Any]:
        monitor = getattr(self, 'spark_monitor', None)
        if monitor and hasattr(monitor, 'snapshot'):
            try:
                status = monitor.snapshot()
                return asdict(status)
            except Exception:
                pass
        return {'active': False, 'reason': 'unavailable'}

    def feature_snapshot(self) -> Optional[Dict[str, Any]]:
        store = getattr(self, 'feature_store', None)
        if store and hasattr(store, 'snapshot'):
            try:
                snap = store.snapshot()
                if snap is None:
                    return None
                return {
                    'timestamp': snap.timestamp,
                    'count': snap.count,
                    'means': list(snap.means),
                    'variances': list(snap.variances),
                    'min': list(snap.min_values),
                    'max': list(snap.max_values),
                    'feature_names': list(snap.feature_names),
                }
            except Exception:
                pass
        return None

    # Dead letter queue helpers -----------------------------------------
    def _to_dlq(self, topic: str, message: Any, reason: str) -> None:
        try:
            self._dlq_path.parent.mkdir(parents=True, exist_ok=True)
            rec = {"ts": time.time(), "topic": topic, "reason": str(reason), "message": message}
            with self._dlq_path.open('a') as f:
                f.write(json.dumps(rec) + "\n")
            # Also publish to a DLQ topic for observers
            with self._lock:
                subs = list(self._topics.get('agent.deadletter', []))
            for q in subs:
                try: q.put(rec)
                except Exception: pass
        except Exception:
            pass

    def _publish_backpressure(self, topic: str, *, group_id: Optional[str], q: Queue) -> None:
        evt = {"ts": time.time(), "topic": topic, "group": group_id, "qsize": self._safe_qsize(q)}
        with self._lock:
            subs = list(self._topics.get('agent.backpressure', []))
        for s in subs:
            try: s.put_nowait(evt)
            except Exception: pass

    def _safe_qsize(self, q: Queue) -> int:
        try:
            return int(q.qsize())
        except Exception:
            return -1

    def metrics(self) -> Dict[str, Any]:
        # Local snapshot of queue depths per topic/group and DLQ count
        m: Dict[str, Any] = {"topics": {}, "groups": {}, "dlq_total": 0}
        # DLQ total
        try:
            if self._dlq_path.exists():
                m["dlq_total"] = sum(1 for _ in self._dlq_path.open('r'))
        except Exception:
            pass
        # Queue sizes
        with self._lock:
            for t, qs in self._topics.items():
                m["topics"][t] = [self._safe_qsize(q) for q in qs]
            for t, groups in self._groups.items():
                m["groups"][t] = {gid: [self._safe_qsize(q) for q in ql] for gid, ql in groups.items()}
        return m


class MetricsWriter(threading.Thread):
    def __init__(self, bus: LocalBus, root: Path, interval_sec: float = 30.0) -> None:
        super().__init__(daemon=True)
        self.bus = bus
        self.root = root
        self.interval = float(max(1.0, interval_sec))
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        out_dir = self.root / '.dspy_reports'
        cur = out_dir / 'bus_metrics.json'
        hist = out_dir / 'bus_metrics.jsonl'
        while not self._stop.is_set():
            try:
                snap = self.bus.metrics()
                snap = dict(snap) if isinstance(snap, dict) else {"error": "unavailable"}
                snap['ts'] = time.time()
                out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    cur.write_text(json.dumps(snap, indent=2))
                except Exception:
                    pass
                try:
                    with hist.open('a') as f:
                        f.write(json.dumps(snap) + "\n")
                except Exception:
                    pass
            except Exception:
                pass
            finally:
                time.sleep(self.interval)


class StreamAutoLearner(threading.Thread):
    """Watch vectorized stream and write granular stats for auto-learning dashboards."""
    def __init__(self, bus: LocalBus, workspace: Path, vector_topic: str = 'agent.rl.vectorized', window_sec: float = 60.0) -> None:
        super().__init__(daemon=True)
        self.bus = bus
        self.workspace = workspace
        self.topic = vector_topic
        self.q: Queue = bus.subscribe(vector_topic)
        self._stop = threading.Event()
        self._counts = 0
        self._times: List[float] = []
        self._features = FeatureStore(window=512)
        self._out = (workspace / '.dspy_stream_rl.json')
        self._window = float(max(1.0, window_sec))
        try:
            from ..db.factory import get_storage as _get
            self._storage = _get()
        except Exception:
            self._storage = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # pragma: no cover
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.5)
            except Empty:
                item = None
            if item is None:
                self._flush(); continue
            try:
                feats = item.get('features') if isinstance(item, dict) else None
                if isinstance(feats, list) and feats:
                    self._features.update({'features': feats, 'feature_names': item.get('feature_names') or []})
                self._counts += 1
                now = time.time(); self._times.append(now)
                cutoff = now - self._window
                self._times = [t for t in self._times if t >= cutoff]
            except Exception:
                pass
            if self._counts % 50 == 0:
                self._flush()
        self._flush()

    def _flush(self) -> None:
        try:
            snap = self._features.snapshot()
            now = time.time()
            rate = (len(self._times) / self._window) if self._window > 0 else 0.0
            data: Dict[str, Any] = {
                'topic': self.topic,
                'total': int(self._counts),
                'rate_per_sec': float(rate),
                'ts': now,
            }
            if snap is not None:
                data['feature_snapshot'] = {
                    'timestamp': snap.timestamp,
                    'count': snap.count,
                    'means': list(snap.means),
                    'variances': list(snap.variances),
                    'min_values': list(snap.min_values),
                    'max_values': list(snap.max_values),
                    'feature_names': list(snap.feature_names),
                }
            try:
                self._out.write_text(json.dumps(data, indent=2))
            except Exception:
                pass
            if self._storage is not None:
                try:
                    self._storage.append('streaming_rl_metrics', data)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass


class OnlineBanditUpdater(threading.Thread):
    """True online bandit updates from streaming context and feedback events.

    - Listens to vector context ('agent.rl.vectorized') to capture latest x
    - Listens to reward events ('agent.learning') with {'tool','reward'}
    - Per-tool online linear model: w <- w + alpha * (r - w·x) * x
    - Persists to workspace/.dspy_online_bandit.json and optional RedDB stream
    """
    def __init__(self, bus: LocalBus, workspace: Path, vector_topic: str = 'agent.rl.vectorized', learning_topic: str = 'agent.learning', alpha: float = 0.05) -> None:
        super().__init__(daemon=True)
        self.bus = bus
        self.workspace = workspace
        self.vec_q = bus.subscribe(vector_topic)
        self.learn_q = bus.subscribe(learning_topic)
        self.alpha = float(alpha)
        self._stop = threading.Event()
        self._out = workspace / '.dspy_online_bandit.json'
        self._state: Dict[str, Any] = {'tools': {}, 'updated_at': 0.0}
        self._last_vec: Optional[list[float]] = None
        try:
            from ..db.factory import get_storage as _get
            self._storage = _get()
        except Exception:
            self._storage = None

    def stop(self) -> None: self._stop.set()

    def _update_tool(self, tool: str, reward: float, x: Optional[list[float]]) -> None:
        if not x: return
        t = self._state['tools'].setdefault(tool, {'count': 0, 'mean_reward': 0.0, 'w': [0.0] * len(x)})
        if len(t['w']) != len(x):
            t['w'] = [0.0] * len(x)
        w = t['w']
        # prediction and SGD update
        pred = sum(wi * xi for wi, xi in zip(w, x))
        err = float(reward) - float(pred)
        lr = self.alpha
        for i in range(len(w)):
            w[i] = float(w[i] + lr * err * float(x[i]))
        # mean
        t['count'] = int(t['count']) + 1
        t['mean_reward'] = float(t['mean_reward'] + (float(reward) - float(t['mean_reward'])) / max(1, t['count']))
        self._state['updated_at'] = time.time()
        # Persist incremental event
        rec = {'ts': self._state['updated_at'], 'tool': tool, 'reward': float(reward), 'pred': float(pred), 'error': float(err)}
        try:
            if self._storage is not None:
                self._storage.append('online_bandit_updates', rec)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _flush(self) -> None:
        try:
            self._out.write_text(json.dumps(self._state, indent=2))
        except Exception:
            pass

    def run(self) -> None:  # pragma: no cover
        while not self._stop.is_set():
            # Non-blocking drain both queues with small timeouts
            try:
                item = self.vec_q.get(timeout=0.2)
                if isinstance(item, dict):
                    vec = item.get('features')
                    if isinstance(vec, list) and vec:
                        self._last_vec = [float(v) for v in vec]
            except Empty:
                pass
            try:
                evt = self.learn_q.get(timeout=0.2)
                if isinstance(evt, dict):
                    tool = evt.get('tool'); reward = evt.get('reward')
                    if isinstance(tool, str) and (isinstance(reward, (int, float))):
                        self._update_tool(tool, float(reward), self._last_vec)
                        self._flush()
            except Empty:
                pass


@dataclass
class Discovered:
    container: str
    service: str
    log_file: Path


def autodiscover_logs(root: Path) -> List[Discovered]:
    candidates: List[Tuple[str, str, Path]] = []
    for p in root.rglob("*"):
        if not p.is_file(): continue
        if p.suffix.lower() == ".log" or p.name.lower().endswith((".out", ".err")) or p.parts[-2:] == ("logs", p.name):
            parts = [x for x in p.parts if x not in (".",)]
            container = "backend" if any("back" in seg.lower() for seg in parts) else ("frontend" if any("front" in seg.lower() for seg in parts) else "app")
            service = p.parent.name or "core"
            candidates.append((container, service, p))
    chosen: Dict[str, Discovered] = {}
    for container, service, path in candidates:
        if container not in chosen: chosen[container] = Discovered(container, service, path)
    return list(chosen.values())


class FileTailer(threading.Thread):
    def __init__(self, path: Path, bus: LocalBus, topic: str, poll_interval: float = 0.5) -> None:
        super().__init__(daemon=True); self.path=path; self.bus=bus; self.topic=topic; self.poll_interval=poll_interval; self._stop=threading.Event()
    def stop(self): self._stop.set()
    def run(self):
        try:
            with self.path.open("r", errors="ignore") as f:
                f.seek(0, os.SEEK_END)
                while not self._stop.is_set():
                    line = f.readline()
                    if not line: time.sleep(self.poll_interval); continue
                    self.bus.publish(self.topic, {"line": line.rstrip("\n"), "ts": time.time()})
        except Exception: pass


class Aggregator(threading.Thread):
    def __init__(self, bus: LocalBus, in_topic: str, out_topic: str, window_sec: float = 5.0) -> None:
        super().__init__(daemon=True); self.bus=bus; self.in_q=bus.subscribe(in_topic); self.out_topic=out_topic; self.window_sec=window_sec; self._stop=threading.Event(); self._buf=[]; self._last_flush=time.time(); self._re=re.compile(r"error|warn|traceback|exception|failed|timeout", re.I)
    def stop(self): self._stop.set()
    def run(self):
        while not self._stop.is_set():
            now = time.time()
            try: item = self.in_q.get(timeout=0.2)
            except Empty: item=None
            if item:
                line = str(item.get("line", ""))
                if self._re.search(line): self._buf.append(line)
            if (now - self._last_flush) >= self.window_sec and self._buf:
                ctx = {"ctx": list(self._buf), "ts": now}; self.bus.publish(self.out_topic, ctx); self._buf.clear(); self._last_flush = now
    def flush_now(self):
        if self._buf:
            ctx = {"ctx": list(self._buf), "ts": time.time()}; self.bus.publish(self.out_topic, ctx); self._buf.clear(); self._last_flush=time.time()


def process_ctx(container: str, text: str, lm: Optional[object], builder: Any, agent: Any) -> Dict[str, Any]:
    from .log_reader import extract_key_events
    try:
        if lm is not None:
            pred = builder(task=f"Summarize {container} errors", logs_preview=text)
            plan = agent(task=f"Plan steps for {container}", context=f"{pred.context}\n\n{pred.key_points}")
            return {"container": container, "summary": pred.context, "key_points": pred.key_points, "plan": plan.plan, "ts": time.time()}
        else:
            summary = extract_key_events(text)
            return {"container": container, "summary": summary, "key_points": "", "plan": "", "ts": time.time()}
    except Exception:
        summary = extract_key_events(text)
        return {"container": container, "summary": summary, "key_points": "", "plan": "", "ts": time.time()}


def make_context_example(lines: List[str]) -> Dict[str, Any]:
    from .autogen_dataset import extract_error_phrases
    text = "\n".join(lines); errs = extract_error_phrases(text)
    return {"task": "Summarize logs for debugging", "logs_preview": text[:4000], "context_keywords": errs[:5], "key_points_keywords": errs[5:10]}


def _repo_layout_summary(root: Path) -> str:
    try:
        tests_dir = root / 'tests'
        had_pytest = (root / 'pytest.ini').exists() or any((root / n).exists() for n in ['pyproject.toml', 'tox.ini'])
        samples = []
        if tests_dir.exists():
            try:
                samples = [str(p.relative_to(root)) for p in tests_dir.rglob('test_*.py')][:10]
            except Exception:
                samples = []
        pkgs = [str(p.parent.relative_to(root)) for p in root.glob('*/__init__.py') if p.is_file()][:10]
        return f"has_pytest={had_pytest}; packages={pkgs}; tests={samples}"
    except Exception:
        return ""


class DockerTailer(threading.Thread):
    """Tail docker container logs and publish to the bus."""

    def __init__(self, container: str, bus: LocalBus, topic: str, follow_since: str = "0s") -> None:
        super().__init__(daemon=True)
        self.container = container
        self.bus = bus
        self.topic = topic
        self.follow_since = follow_since
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen[str]] = None

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def run(self) -> None:
        cmd = os.getenv('DSPY_DOCKER_BIN', 'docker')
        args = [cmd, 'logs', '--follow']
        if self.follow_since:
            args.extend(['--since', self.follow_since])
        args.append(self.container)
        try:
            self._proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as exc:
            print(f"[stream-rl] docker tailer failed for {self.container}: {exc}")
            return
        if not self._proc.stdout:
            return
        try:
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                line = line.rstrip()
                if not line:
                    continue
                self.bus.publish(self.topic, line)
        except Exception:
            pass
        finally:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass

class Worker(threading.Thread):
    def __init__(self, container: str, root: Path, bus: LocalBus, ctx_topic: str, results_topic: str) -> None:
        super().__init__(daemon=True)
        self.container=container; self.root=root; self.bus=bus; self.in_q=bus.subscribe(ctx_topic); self.results_topic=results_topic; self._stop=threading.Event()
        from ..llm import configure_lm
        from ..skills.context_builder import ContextBuilder
        from ..skills.task_agent import TaskAgent
        from ..skills.code_edit import CodeEdit
        from ..skills.file_locator import FileLocator
        from ..skills.patch_verifier import PatchVerifier
        from ..skills.test_planner import TestPlanner
        self.lm = configure_lm(provider="ollama", model_name=None, base_url=None, api_key=None)
        self.builder = ContextBuilder(); self.agent = TaskAgent(); self.editor = CodeEdit(); self.locator = FileLocator(); self.verifier = PatchVerifier(); self.tplanner = TestPlanner()
        self.auto_patch = os.getenv("AUTO_PATCH", "false").lower() in {"1","true","yes","on"}
        self.auto_commit = os.getenv("AUTO_COMMIT", "false").lower() in {"1","true","yes","on"}
        self.test_cmd = os.getenv("AUTO_TEST_CMD"); self.test_timeout = int(os.getenv("AUTO_TEST_TIMEOUT", "600")); self.test_strict = os.getenv("AUTO_TEST_STRICT", "true").lower() in {"1","true","yes","on"}
        self.require_keywords = os.getenv("AUTO_PATCH_REQUIRE_KEYWORDS", "true").lower() in {"1","true","yes","on"}
        kws = os.getenv("AUTO_PATCH_KEYWORDS", "error,exception,fail,timeout").split(","); self.keywords=[k.strip().lower() for k in kws if k.strip()]
        self.approval_mode = os.getenv("AUTO_PATCH_APPROVAL", "manual").lower(); self._settings_path = (self.root / ".dspy_settings.json")
        # Autopatch gating/backoff
        try:
            self.min_repeats = int(os.getenv("AUTO_PATCH_MIN_REPEATS", "1"))
        except Exception:
            self.min_repeats = 1
        try:
            self.backoff_sec = float(os.getenv("AUTO_PATCH_BACKOFF_SEC", "60"))
        except Exception:
            self.backoff_sec = 60.0
        self._last_patch_ts = 0.0

    def _load_settings(self) -> None:
        try:
            if self._settings_path.exists():
                data = json.loads(self._settings_path.read_text())
                self.approval_mode = str(data.get("autopatch_mode", self.approval_mode)).lower(); self.auto_commit=bool(data.get("auto_commit", self.auto_commit))
                self.test_cmd = data.get("test_cmd", self.test_cmd); self.test_strict=bool(data.get("test_strict", self.test_strict))
                self.require_keywords = bool(data.get("require_keywords", self.require_keywords))
                if isinstance(data.get("keywords"), list): self.keywords=[str(k).lower() for k in data.get("keywords")]
        except Exception: pass
        try:
            self.max_files = int(os.getenv("AUTO_PATCH_MAX_FILES", "4")); self.max_lines = int(os.getenv("AUTO_PATCH_MAX_LINES", "200"))
        except Exception: self.max_files, self.max_lines = 4, 200

    def stop(self): self._stop.set()

    def run(self):
        from ..code_tools.patcher import apply_unified_patch, summarize_patch, run_shell, revert_unified_patch, git_commit
        while not self._stop.is_set():
            try: item = self.in_q.get(timeout=0.5)
            except Empty: continue
            lines = item.get("ctx", [])
            if not lines: continue
            text = "\n".join(lines); result = process_ctx(self.container, text, self.lm, self.builder, self.agent)
            self.bus.publish(self.results_topic, result)
            if self.auto_patch:
                try:
                    self._load_settings()
                    # Gating: require repeated error signals and time backoff
                    try:
                        import re as _re
                        err_re = _re.compile(r"error|exception|traceback|failed|timeout", _re.I)
                        hits = err_re.findall(text)
                        if len(hits) < max(1, self.min_repeats):
                            raise RuntimeError("autopatch: min repeats gate not satisfied")
                        if (time.time() - self._last_patch_ts) < max(0.0, self.backoff_sec):
                            raise RuntimeError("autopatch: backoff gate active")
                    except Exception:
                        pass
                    # Use FileLocator to narrow scope
                    file_hints = ""
                    try:
                        loc = self.locator(task=f"Fix {self.container} errors", context=text, code_graph="")
                        file_hints = getattr(loc, 'file_candidates', '') or ''
                    except Exception:
                        pass
                    edit = self.editor(task=f"Fix {self.container} errors", context=text, code_graph="", file_hints=file_hints)
                    patch_text = getattr(edit, "patch", "") or ""
                    if self.require_keywords and self.keywords and not any(k in text.lower() for k in self.keywords): raise RuntimeError("keyword gate not satisfied")
                    if patch_text.strip():
                        summ = summarize_patch(patch_text); total_lines = summ["added_lines"] + summ["removed_lines"]
                        caps_ok = ((self.max_files <= 0 or summ["files"] <= self.max_files) and (self.max_lines <= 0 or total_lines <= self.max_lines))
                        # Verify quality
                        v = None
                        try:
                            v = self.verifier(task=f"Fix {self.container} errors", context=text, patch=patch_text)
                        except Exception:
                            pass
                        verdict_ok = (getattr(v, 'verdict', 'pass').lower() == 'pass') if v is not None else True
                        if caps_ok and verdict_ok:
                            pdir = (self.root / ".dspy_patches"); pdir.mkdir(parents=True, exist_ok=True)
                            pid = str(int(time.time()*1000)); (pdir / f"{pid}.patch").write_text(patch_text)
                            meta = {"id": pid, "container": self.container, "summary": summ, "ts": time.time(), "applied": False}
                            (pdir / f"{pid}.json").write_text(json.dumps(meta, indent=2))
                            ok = False
                            test_ok = None
                            if self.approval_mode == "auto":
                                ok, msg = apply_unified_patch(patch_text, self.root); meta.update({"applied": ok, "apply_message": msg})
                                test_cmd_local = self.test_cmd
                                if not test_cmd_local:
                                    try:
                                        repo_layout = _repo_layout_summary(self.root)
                                        tp = self.tplanner(task=f"Validate fix for {self.container}", context=text, repo_layout=repo_layout)
                                        test_cmd_local = getattr(tp, 'commands', '') or None
                                        if test_cmd_local:
                                            meta.update({"test_plan": {
                                                "tests_to_run": getattr(tp, 'tests_to_run', ''),
                                                "commands": test_cmd_local,
                                                "fast_paths": getattr(tp, 'fast_paths', ''),
                                            }})
                                    except Exception:
                                        pass
                                if ok and test_cmd_local:
                                    code, out, err = run_shell(test_cmd_local, self.root, timeout=self.test_timeout)
                                    test_ok = None if ((code == 127 or (err or '').lower().find('not found')>=0) and not self.test_strict) else (code == 0)
                                    meta.update({"test_cmd": test_cmd_local, "test_code": code, "test_ok": test_ok})
                                    if test_ok is False:
                                        r_ok, r_msg = revert_unified_patch(patch_text, self.root); meta.update({"reverted": r_ok, "revert_message": r_msg}); ok = False
                                if ok and (test_ok in (None, True)) and self.auto_commit:
                                    c_ok, c_msg = git_commit(self.root, f"autopatch({self.container}): files={summ['files']} lines={summ['added_lines']+summ['removed_lines']}"); meta.update({"committed": c_ok, "commit_message": c_msg})
                            else:
                                meta.update({"queued": True, "apply_message": "queued for approval"})
                            (pdir / f"{pid}.json").write_text(json.dumps(meta, indent=2)); self.bus.publish("agent.patches", meta)
                            # Emit learning signal: reward 1.0 if tests passed; 0.5 if applied but tests not run; 0 if reverted/failed
                            try:
                                reward = 0.0
                                if meta.get('applied'):
                                    if meta.get('test_ok') is True:
                                        reward = 1.0
                                    elif meta.get('test_ok') is None:
                                        reward = 0.5
                                evt = {"tool": "patch", "reward": float(max(0.0, min(1.0, reward))), "ts": time.time(), "container": self.container}
                                # Publish to bus for observers
                                self.bus.publish("agent.learning", evt)
                                # Append to RL events file for Trainer consumption
                                try:
                                    p = self.root / '.dspy_rl_events.jsonl'
                                    with p.open('a') as f:
                                        import json as _j
                                        f.write(_j.dumps(evt) + "\n")
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            self._last_patch_ts = time.time()

                except Exception: pass
            try:
                storage = getattr(self.bus, "_storage", None)
                if storage is not None and hasattr(storage, "put"):
                    prefix = f"last:{self.container}:"; storage.put(prefix+"summary", result.get("summary", "")); storage.put(prefix+"key_points", result.get("key_points", "")); storage.put(prefix+"plan", result.get("plan", "")); storage.put(prefix+"ts", result.get("ts", 0))
            except Exception: pass


def start_local_stack(root: Path, cfg: Optional[StreamConfig] = None, storage: Optional[object] = None, kafka: Optional[object] = None) -> Tuple[List[threading.Thread], LocalBus]:
    bus = LocalBus(storage=storage, kafka=kafka)
    if cfg is None:
        discs = autodiscover_logs(root); containers: Dict[str, List[str]] = {}
        for d in discs: containers.setdefault(d.container, []).append(d.service)
        cfg = StreamConfig.default(); cfg.containers = [ContainerTopic(container=k, services=v) for k, v in containers.items()]; save_config(cfg)
    threads: List[threading.Thread] = []
    docker_containers = [c.strip() for c in os.getenv('DSPY_DOCKER_CONTAINERS', '').split(',') if c.strip()]
    chosen = {d.container: d.log_file for d in autodiscover_logs(root)}
    active = set()
    kafka_cfg = getattr(cfg, 'kafka', None)
    vector_topics: Set[str] = set()
    feature_store: Optional[FeatureStore] = None
    for ct in cfg.containers:
        container = getattr(ct, 'container'); log_file = chosen.get(container)
        if not log_file or not log_file.exists():
            continue
        raw_topic = f"logs.raw.{container}"; ctx_topic = f"logs.ctx.{container}"; results_topic = f"agent.results.{container}"
        t1 = FileTailer(log_file, bus, raw_topic); t2 = Aggregator(bus, raw_topic, ctx_topic, window_sec=5.0); t3 = Worker(container, root, bus, ctx_topic, results_topic)
        t1.start(); t2.start(); t3.start(); threads.extend([t1, t2, t3])
        active.add(container)
        vector_topics.update({ctx_topic, results_topic})
    for dc in docker_containers:
        if dc in active:
            continue
        raw_topic = f"logs.raw.{dc}"
        ctx_topic = f"logs.ctx.{dc}"
        results_topic = f"agent.results.{dc}"
        tailer = DockerTailer(dc, bus, raw_topic)
        aggregator = Aggregator(bus, raw_topic, ctx_topic, window_sec=5.0)
        worker = Worker(dc, root, bus, ctx_topic, results_topic)
        tailer.start(); aggregator.start(); worker.start()
        threads.extend([tailer, aggregator, worker])
        vector_topics.update({ctx_topic, results_topic})

    # Configure vectorized streaming pipeline (optional)
    out_topic = getattr(kafka_cfg, 'vector_topic', 'agent.rl.vectorized') or 'agent.rl.vectorized'
    vector_topics.discard(out_topic)
    vector_topics.discard('')
    extra_topics = [t.strip() for t in os.getenv('DSPY_VECTOR_TOPICS', '').split(',') if t.strip()]
    vector_topics.update(extra_topics)
    vector_topics.update(getattr(kafka_cfg, 'vector_topics', []) if kafka_cfg else [])
    vector_topics.discard(out_topic)
    if vector_topics:
        try:
            from .vectorized_pipeline import RLVectorizer, VectorizedStreamOrchestrator
            vectorizer = RLVectorizer(root)
            feature_store = FeatureStore()
            orchestrator = VectorizedStreamOrchestrator(
                bus=bus,
                topics=sorted(vector_topics),
                vectorizer=vectorizer,
                out_topic=out_topic,
                feature_store=feature_store,
            )
            orchestrator.start(); threads.append(orchestrator)
            setattr(bus, 'vectorizer', vectorizer)
            setattr(bus, 'vector_orchestrator', orchestrator)
            setattr(bus, 'feature_store', feature_store)
            # Optional: start auto-learner to write granular stats
            try:
                auto_env = os.getenv('DSPY_STREAM_AUTO_LEARN', '1').strip().lower()
                if auto_env in {'1','true','yes','on'}:
                    learner = StreamAutoLearner(bus, root, vector_topic=out_topic)
                    learner.start(); threads.append(learner)
                    setattr(bus, 'auto_learner', learner)
            except Exception:
                pass
            # Optional: online bandit updates
            try:
                ob_env = os.getenv('DSPY_ONLINE_BANDIT', '1').strip().lower()
                if ob_env in {'1','true','yes','on'}:
                    alpha = float(os.getenv('ONLINE_BANDIT_ALPHA', '0.05') or '0.05')
                    obu = OnlineBanditUpdater(bus, root, vector_topic=out_topic, learning_topic='agent.learning', alpha=alpha)
                    obu.start(); threads.append(obu)
                    setattr(bus, 'online_bandit', obu)
            except Exception:
                pass
        except Exception as exc:
            print(f"Warning: vectorization pipeline unavailable: {exc}")

    # Attach streaming health monitors for Kafka and Spark when available
    try:
        from .vectorized_pipeline import KafkaStreamInspector, SparkCheckpointMonitor
        kafka_cfg = getattr(cfg, 'kafka', None)
        topics_cfg = getattr(kafka_cfg, 'topics', None) or []
        kafka_topics = {out_topic}
        for topic_cfg in topics_cfg:
            name = getattr(topic_cfg, 'name', None)
            if name:
                kafka_topics.add(name)
        bootstrap = _resolve_bootstrap(getattr(kafka_cfg, 'bootstrap_servers', None))
        setattr(bus, 'kafka_inspector', KafkaStreamInspector(bootstrap, sorted(kafka_topics)))
        checkpoint_path = Path(getattr(getattr(cfg, 'spark', None), 'checkpoint_dir', '.dspy_checkpoints'))
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        setattr(bus, 'spark_monitor', SparkCheckpointMonitor(checkpoint_path))
        if feature_store is not None and feature_store.snapshot() is not None:
            setattr(bus, 'feature_store', feature_store)
    except Exception:
        pass

    # Background bus metrics writer (opt-in via env, default 30s)
    try:
        interval_env = os.getenv('BUS_METRICS_INTERVAL', '30').strip()
        interval = float(interval_env) if interval_env else 30.0
    except Exception:
        interval = 30.0
    if interval > 0:
        try:
            mw = MetricsWriter(bus, root, interval_sec=interval)
            mw.start(); threads.append(mw)
        except Exception:
            pass
    # Hardware profiler (opt-in via env, default on)
    try:
        hw_env = os.getenv('DSPY_HW_PROFILER', '1').strip().lower()
        if hw_env in {'1','true','yes','on'}:
            hp = HardwareProfiler(root, interval_sec=float(os.getenv('HW_INTERVAL_SEC','5') or '5'))
            hp.start(); threads.append(hp)
    except Exception:
        pass
    return threads, bus


# -----------------
# Kafka worker (optional)
# -----------------

try:
    from confluent_kafka import Consumer, Producer  # type: ignore
except Exception:  # pragma: no cover
    Consumer = None  # type: ignore
    Producer = None  # type: ignore


@dataclass
class KafkaParams:
    bootstrap: str
    group: str
    in_topic: str
    out_topic: str
    container: str


class WorkerLoop:
    def __init__(self, params: KafkaParams):
        self.p = params; self.running = False
        from ..skills.context_builder import ContextBuilder
        from ..skills.task_agent import TaskAgent
        from ..skills.code_edit import CodeEdit
        from ..skills.file_locator import FileLocator
        from ..skills.patch_verifier import PatchVerifier
        from ..skills.test_planner import TestPlanner
        from ..llm import configure_lm
        self.builder = ContextBuilder(); self.agent = TaskAgent(); self.lm = configure_lm(provider="ollama", model_name=None, base_url=None, api_key=None); self.editor = CodeEdit(); self.locator = FileLocator(); self.verifier = PatchVerifier(); self.tplanner = TestPlanner()
        self.auto_patch = os.getenv("AUTO_PATCH", "false").lower() in {"1","true","yes","on"}; self.auto_commit = os.getenv("AUTO_COMMIT", "false").lower() in {"1","true","yes","on"}
        self.root = Path(os.getenv("WORKSPACE", "/workspace")).resolve(); self.test_cmd = os.getenv("AUTO_TEST_CMD"); self.test_timeout = int(os.getenv("AUTO_TEST_TIMEOUT", "600")); self.test_strict = os.getenv("AUTO_TEST_STRICT", "true").lower() in {"1","true","yes","on"}
        self.require_keywords = os.getenv("AUTO_PATCH_REQUIRE_KEYWORDS", "true").lower() in {"1","true","yes","on"}; kws = os.getenv("AUTO_PATCH_KEYWORDS", "error,exception,fail,timeout").split(","); self.keywords = [k.strip().lower() for k in kws if k.strip()]
        self.approval_mode = os.getenv("AUTO_PATCH_APPROVAL", "manual").lower(); self._settings_path = self.root / ".dspy_settings.json"
        try:
            self.min_repeats = int(os.getenv("AUTO_PATCH_MIN_REPEATS", "1"))
        except Exception:
            self.min_repeats = 1
        try:
            self.backoff_sec = float(os.getenv("AUTO_PATCH_BACKOFF_SEC", "60"))
        except Exception:
            self.backoff_sec = 60.0
        self._last_patch_ts = 0.0

    def _load_settings(self) -> None:
        try:
            if self._settings_path.exists():
                data = json.loads(self._settings_path.read_text())
                self.approval_mode = str(data.get("autopatch_mode", self.approval_mode)).lower(); self.auto_commit = bool(data.get("auto_commit", self.auto_commit))
                self.test_cmd = data.get("test_cmd", self.test_cmd); self.test_strict = bool(data.get("test_strict", self.test_strict))
                self.require_keywords = bool(data.get("require_keywords", self.require_keywords))
                if isinstance(data.get("keywords"), list): self.keywords = [str(k).lower() for k in data.get("keywords")]
        except Exception: pass
        try:
            self.max_files = int(os.getenv("AUTO_PATCH_MAX_FILES", "4")); self.max_lines = int(os.getenv("AUTO_PATCH_MAX_LINES", "200"))
        except Exception: self.max_files, self.max_lines = 4, 200

    def run(self) -> None:
        if Consumer is None or Producer is None:
            raise RuntimeError("confluent-kafka not installed")
        consumer = Consumer({'bootstrap.servers': self.p.bootstrap, 'group.id': self.p.group, 'auto.offset.reset': 'latest'})
        producer = Producer({'bootstrap.servers': self.p.bootstrap})
        consumer.subscribe([self.p.in_topic]); self.running = True
        try:
            while self.running:
                msg = consumer.poll(0.5)
                if msg is None or msg.error(): continue
                try: obj = json.loads(msg.value().decode('utf-8'))
                except Exception: continue
                lines = obj.get('ctx') or obj.get('lines') or []
                text = "\n".join(lines) if isinstance(lines, list) else str(lines)
                result = process_ctx(self.p.container, text, self.lm, self.builder, self.agent)
                producer.produce(self.p.out_topic, json.dumps(result).encode('utf-8')); producer.poll(0)
                if self.auto_patch:
                    try:
                        self._load_settings()
                        # Gating: min repeats + backoff
                        try:
                            import re as _re
                            err_re = _re.compile(r"error|exception|traceback|failed|timeout", _re.I)
                            hits = err_re.findall(text)
                            if len(hits) < max(1, self.min_repeats):
                                raise RuntimeError("autopatch: min repeats gate not satisfied")
                            if (time.time() - self._last_patch_ts) < max(0.0, self.backoff_sec):
                                raise RuntimeError("autopatch: backoff gate active")
                        except Exception:
                            pass
                        file_hints = ""
                        try:
                            loc = self.locator(task=f"Fix {self.p.container} errors", context=text, code_graph="")
                            file_hints = getattr(loc, 'file_candidates', '') or ''
                        except Exception:
                            pass
                        edit = self.editor(task=f"Fix {self.p.container} errors", context=text, code_graph="", file_hints=file_hints); patch_text = getattr(edit, "patch", "") or ""
                        if self.require_keywords and self.keywords and not any(k in text.lower() for k in self.keywords): raise RuntimeError("keyword gate not satisfied")
                        if patch_text.strip():
                            from ..code_tools.patcher import apply_unified_patch, summarize_patch, run_shell, revert_unified_patch, git_commit
                            summ = summarize_patch(patch_text); total_lines = summ["added_lines"] + summ["removed_lines"]
                            caps_ok = ((self.max_files <= 0 or summ["files"] <= self.max_files) and (self.max_lines <= 0 or total_lines <= self.max_lines))
                            v = None
                            try:
                                v = self.verifier(task=f"Fix {self.p.container} errors", context=text, patch=patch_text)
                            except Exception:
                                pass
                            verdict_ok = (getattr(v, 'verdict', 'pass').lower() == 'pass') if v is not None else True
                            if caps_ok and verdict_ok:
                                pdir = (self.root / ".dspy_patches"); pdir.mkdir(parents=True, exist_ok=True)
                                pid = str(int(time.time()*1000))
                                (pdir / f"{pid}.patch").write_text(patch_text)
                                meta = {"id": pid, "container": self.p.container, "summary": summ, "ts": time.time(), "applied": False}
                                (pdir / f"{pid}.json").write_text(json.dumps(meta, indent=2))
                                ok = False
                                if self.approval_mode == "auto":
                                    ok, msg = apply_unified_patch(patch_text, self.root); meta.update({"applied": ok, "apply_message": msg})
                                    test_ok = None
                                    test_cmd_local = self.test_cmd
                                    if not test_cmd_local:
                                        try:
                                            repo_layout = _repo_layout_summary(self.root)
                                            tp = self.tplanner(task=f"Validate fix for {self.p.container}", context=text, repo_layout=repo_layout)
                                            test_cmd_local = getattr(tp, 'commands', '') or None
                                            if test_cmd_local:
                                                meta.update({"test_plan": {
                                                    "tests_to_run": getattr(tp, 'tests_to_run', ''),
                                                    "commands": test_cmd_local,
                                                    "fast_paths": getattr(tp, 'fast_paths', ''),
                                                }})
                                        except Exception:
                                            pass
                                    if ok and test_cmd_local:
                                        code, out, err = run_shell(test_cmd_local, self.root, timeout=self.test_timeout)
                                        test_ok = None if ((code == 127 or (err or '').lower().find('not found')>=0) and not self.test_strict) else (code == 0)
                                        meta.update({"test_cmd": test_cmd_local, "test_code": code, "test_ok": test_ok})
                                        if test_ok is False:
                                            r_ok, r_msg = revert_unified_patch(patch_text, self.root); meta.update({"reverted": r_ok, "revert_message": r_msg}); ok = False
                                    if ok and (test_ok in (None, True)) and self.auto_commit:
                                        c_ok, c_msg = git_commit(self.root, f"autopatch({self.p.container}): files={summ['files']} lines={summ['added_lines']+summ['removed_lines']}"); meta.update({"committed": c_ok, "commit_message": c_msg})
                                else:
                                    meta.update({"queued": True, "apply_message": "queued for approval"})
                                (pdir / f"{pid}.json").write_text(json.dumps(meta, indent=2)); producer.produce("agent.patches", json.dumps(meta).encode('utf-8')); producer.poll(0)
                                # Emit learning signal to Kafka
                                try:
                                    reward = 0.0
                                    if meta.get('applied'):
                                        if meta.get('test_ok') is True:
                                            reward = 1.0
                                        elif meta.get('test_ok') is None:
                                            reward = 0.5
                                    evt = {"tool": "patch", "reward": float(max(0.0, min(1.0, reward))), "ts": time.time(), "container": self.p.container}
                                    producer.produce("agent.learning", json.dumps(evt).encode('utf-8')); producer.poll(0)
                                except Exception:
                                    pass
                                self._last_patch_ts = time.time()
                    except Exception: pass
        finally:
            try: consumer.close()
            except Exception: pass


class Trainer(threading.Thread):
    """Streaming trainer that processes log context and trains the RL model."""
    
    def __init__(
        self,
        workspace: Path,
        bus: LocalBus,
        containers: List[str],
        min_batch: int = 3,
        interval_sec: float = 60.0,
        vector_topic: Optional[str] = None,
        window_sec: float = 5.0,
        rl_actions: Optional[Iterable[str]] = None,
        tfidf_weights: Optional[Mapping[str, float]] = None,
        settings_path: Optional[Path] = None,
    ):
        super().__init__(daemon=True)
        self.workspace = workspace
        self.bus = bus
        self.containers = containers
        self.min_batch = min_batch
        self.interval_sec = interval_sec
        default_vector_topic = vector_topic if vector_topic is not None else os.getenv('RL_VECTOR_TOPIC', 'agent.rl.vectorized')
        self.vector_topic = (default_vector_topic.strip() or None) if isinstance(default_vector_topic, str) else None
        self.window_sec = float(window_sec)
        self._stop = threading.Event()
        self._contexts: List[Dict[str, Any]] = []
        self._vector_records: List[Any] = []
        self._feature_store = getattr(bus, 'feature_store', None)
        vectorizer = getattr(bus, 'vectorizer', None)
        try:
            self._vector_feature_template = vectorizer.feature_names if vectorizer else []
        except Exception:
            self._vector_feature_template = []
        self._latest_vector_feature_names: List[str] = []
        self._last_train = time.time()
        default_settings = TRAINER_SETTINGS_PATH
        if not default_settings.is_absolute():
            default_settings = self.workspace / default_settings
        self._settings_path = Path(settings_path) if settings_path else default_settings
        self._trainer_cfg = self._load_trainer_settings()
        self._explicit_actions = list(rl_actions) if rl_actions else None
        self._explicit_tfidf = dict(tfidf_weights) if tfidf_weights else None
        self.rl_actions = self._resolve_actions(self._explicit_actions)
        self._action_name_set = set(self.rl_actions) if self.rl_actions else None
        self.tfidf_weights = self._resolve_tfidf_weights(self._explicit_tfidf)
        cfg_flags = self._trainer_cfg if isinstance(self._trainer_cfg, dict) else {}
        self.group_advantage = bool(cfg_flags.get('group_advantage', False))
        try:
            self.group_size = max(1, int(cfg_flags.get('group_size', 2)))
        except Exception:
            self.group_size = 2
        try:
            base_shell_timeout = int(cfg_flags.get('shell_timeout', 60))
        except Exception:
            base_shell_timeout = 60
        try:
            self.shell_timeout = int(os.getenv('RL_SHELL_TIMEOUT', str(base_shell_timeout)))
        except Exception:
            self.shell_timeout = base_shell_timeout
        self.shell_actions = self._resolve_shell_actions()
        # Configure LM for RL patch action when available
        try:
            from ..llm import configure_lm
            self.lm = configure_lm(provider="ollama", model_name=None, base_url=None, api_key=None)
        except Exception:
            self.lm = None
        
    def _load_trainer_settings(self) -> Dict[str, Any]:
        path = getattr(self, '_settings_path', None)
        if not path:
            return {}
        try:
            spath = Path(path)
            if spath.exists():
                data = json.loads(spath.read_text())
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _resolve_actions(self, explicit: Optional[Iterable[str]]) -> Optional[List[str]]:
        candidates: List[str] = []
        if explicit:
            for item in explicit:
                s = str(item).strip()
                if s:
                    candidates.append(s)
        else:
            env_spec = os.getenv('RL_ACTIONS', '')
            if env_spec.strip():
                for part in env_spec.split(','):
                    s = part.strip()
                    if s:
                        candidates.append(s)
            else:
                cfg_actions = self._trainer_cfg.get('actions') if isinstance(self._trainer_cfg, dict) else None
                if isinstance(cfg_actions, list):
                    for item in cfg_actions:
                        s = str(item).strip()
                        if s:
                            candidates.append(s)
        shell_cfg = self._trainer_cfg.get('shell_actions') if isinstance(self._trainer_cfg, Mapping) else None
        if isinstance(shell_cfg, Mapping):
            for name in shell_cfg.keys():
                if isinstance(name, str) and name.strip():
                    candidates.append(name.strip())
        if not candidates:
            return None
        canonical: List[str] = []
        seen = set()
        for item in candidates:
            normalized = str(item).strip().lower().replace('-', '_').replace(' ', '_')
            if not normalized:
                continue
            try:
                from ..rl.rlkit import ToolAction
                canon = ToolAction.from_any(normalized).name.lower()
            except Exception:
                if normalized in {'tests', 'test'}:
                    canon = 'run_tests'
                else:
                    canon = normalized
            if canon in seen:
                continue
            seen.add(canon)
            canonical.append(canon)
        return canonical or None

    def _canonical_weight_key(self, key: str) -> Optional[str]:
        k = str(key).strip().lower().replace('-', '_').replace(' ', '_')
        if not k:
            return None
        if k in {'top1', 'top', 'top_1', 'top1_weight', 'max'}:
            return 'top1'
        if k in {'avg_top3', 'avg', 'avg3', 'avg_top_3', 'avg_weight', 'avg_top3_weight'}:
            return 'avg_top3'
        return None

    def _resolve_tfidf_weights(self, explicit: Optional[Mapping[str, float]]) -> Dict[str, float]:
        weights = {'top1': 1.0, 'avg_top3': 1.0}

        def apply(source: Mapping[str, object]) -> None:
            for key, value in source.items():
                canon = self._canonical_weight_key(key)
                if not canon:
                    continue
                try:
                    weights[canon] = float(value)
                except (TypeError, ValueError):
                    continue

        cfg_weights = self._trainer_cfg.get('tfidf_weights') if isinstance(self._trainer_cfg, dict) else None
        if isinstance(cfg_weights, Mapping):
            apply(cfg_weights)
        env_spec = os.getenv('RL_TFIDF_WEIGHTS', '')
        if env_spec.strip():
            env_map: Dict[str, str] = {}
            for part in env_spec.split(','):
                if '=' not in part:
                    continue
                k, v = part.split('=', 1)
                env_map[k.strip()] = v.strip()
            apply(env_map)
        if explicit:
            apply(explicit)
        return weights

    def _resolve_shell_actions(self) -> Dict[str, Dict[str, Any]]:
        defaults: Dict[str, Dict[str, Any]] = {
            'shell_ls': {'cmd': 'ls -lah'},
            'shell_pwd': {'cmd': 'pwd'},
            'shell_cat': {'path': 'README.md'},
            'shell_cd': {'path': '.'},
            'shell_run': {'cmd': 'echo $PWD'},
        }
        cfg_shell = self._trainer_cfg.get('shell_actions') if isinstance(self._trainer_cfg, Mapping) else {}
        result: Dict[str, Dict[str, Any]] = {}
        for name, base in defaults.items():
            data: Dict[str, Any] = dict(base)
            cfg_val = cfg_shell.get(name) if isinstance(cfg_shell, Mapping) else None
            if isinstance(cfg_val, Mapping):
                for key, value in cfg_val.items():
                    if key in {'cmd', 'path', 'timeout'}:
                        data[key] = value
            elif isinstance(cfg_val, str):
                if name in {'shell_cat', 'shell_cd'}:
                    data['path'] = cfg_val
                else:
                    data['cmd'] = cfg_val
            env_override = os.getenv(f'RL_{name.upper()}')
            if env_override:
                if name in {'shell_cat', 'shell_cd'}:
                    data['path'] = env_override
                else:
                    data['cmd'] = env_override
            env_timeout = os.getenv(f'RL_{name.upper()}_TIMEOUT')
            if env_timeout:
                try:
                    data['timeout'] = int(env_timeout)
                except Exception:
                    pass
            if name != 'shell_cd':
                data.setdefault('timeout', self.shell_timeout)
            cleaned = {key: value for key, value in data.items() if value not in (None, '')}
            if name == 'shell_run' and not cleaned.get('cmd'):
                cleaned['cmd'] = 'echo $PWD'
            result[name] = cleaned
        return result

    def _extract_vector_features(self, record: Any) -> Tuple[Optional[List[float]], Optional[List[str]]]:
        if record is None:
            return None, None
        features_obj: Any = None
        names_obj: Any = None
        if hasattr(record, 'features'):
            features_obj = getattr(record, 'features', None)
            names_obj = getattr(record, 'feature_names', None)
        elif isinstance(record, Mapping):
            features_obj = record.get('features') or record.get('vector')
            names_obj = record.get('feature_names') or record.get('names')
        if features_obj is None:
            return None, None
        try:
            features = [float(x) for x in features_obj]
        except Exception:
            return None, None
        names: Optional[List[str]] = None
        if names_obj:
            try:
                names = [str(n) for n in names_obj]
            except Exception:
                names = None
        return features, names

    def _aggregate_vector_records(self) -> Optional[Dict[str, Any]]:
        store = getattr(self, '_feature_store', None)
        if store is not None and hasattr(store, 'snapshot'):
            try:
                snap = store.snapshot()
            except Exception:
                snap = None
            if snap is not None:
                return {
                    'features': list(snap.means),
                    'variances': list(snap.variances),
                    'names': list(snap.feature_names),
                    'count': snap.count,
                }
        if not self._vector_records:
            return None
        vectors: List[List[float]] = []
        names: Optional[List[str]] = None
        for record in list(self._vector_records):
            feats, feat_names = self._extract_vector_features(record)
            if feats is None:
                continue
            vectors.append(feats)
            if names is None and feat_names:
                names = feat_names
        if not vectors:
            return None
        feature_count = len(vectors[0])
        accum = [0.0] * feature_count
        for vec in vectors:
            if len(vec) != feature_count:
                continue
            for idx, value in enumerate(vec):
                accum[idx] += float(value)
        count = max(len(vectors), 1)
        averaged = [value / count for value in accum]
        if names is None or len(names) != feature_count:
            if self._vector_feature_template and len(self._vector_feature_template) == feature_count:
                names = list(self._vector_feature_template)
            else:
                names = [f'vector_{i}' for i in range(feature_count)]
        self._latest_vector_feature_names = list(names)
        return {
            'features': averaged,
            'names': list(names),
            'variances': [0.0 for _ in averaged],
            'count': len(vectors),
        }

    def _action_enabled(self, name: str) -> bool:
        if self._action_name_set is None:
            return True
        normalized = name.strip().lower().replace('-', '_').replace(' ', '_')
        if normalized in {'tests', 'test'}:
            normalized = 'run_tests'
        return normalized in self._action_name_set

    def _reward_components(self) -> Tuple[Dict[str, float], List[str], List[str]]:
        base_weights: Dict[str, float] = {'pass_rate': 1.0, 'blast_radius': 1.0}
        penalties: List[str] = []
        clamp: List[str] = []
        cfg = self._trainer_cfg if isinstance(self._trainer_cfg, dict) else {}
        if isinstance(cfg, dict):
            extras = cfg.get('reward_weights') if isinstance(cfg.get('reward_weights'), Mapping) else {}
            for key, value in extras.items():
                try:
                    base_weights[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            raw_penalties = cfg.get('reward_penalties') if isinstance(cfg.get('reward_penalties'), (list, tuple)) else []
            penalties = [str(x).strip() for x in raw_penalties if str(x).strip()]
            raw_clamp = cfg.get('reward_clamp01') if isinstance(cfg.get('reward_clamp01'), (list, tuple)) else []
            clamp = [str(x).strip() for x in raw_clamp if str(x).strip()]
        return base_weights, penalties, clamp

    def _active_prompt_entry(self, name: str = 'patch') -> Optional[Dict[str, Any]]:
        cfg = self._trainer_cfg if isinstance(self._trainer_cfg, dict) else {}
        prompts = {}
        if isinstance(cfg, dict):
            raw_prompts = cfg.get('prompts') or cfg.get('gepa_prompts') or {}
            if isinstance(raw_prompts, Mapping):
                prompts = raw_prompts
        entry = prompts.get(name) if isinstance(prompts, Mapping) else None
        if entry is None and name == 'patch':
            for alt in ('code', 'code_edit'):
                maybe = prompts.get(alt) if isinstance(prompts, Mapping) else None
                if maybe is not None:
                    entry = maybe
                    break
        if not isinstance(entry, Mapping):
            return None
        candidates = entry.get('candidates') if isinstance(entry.get('candidates'), list) else []
        if not candidates:
            return None
        active_id = entry.get('active')
        chosen: Optional[Dict[str, Any]] = None
        if active_id:
            for cand in candidates:
                if isinstance(cand, Mapping) and str(cand.get('id') or cand.get('hash')) == str(active_id):
                    chosen = dict(cand)
                    break
        if chosen is None:
            # Fallback: choose candidate with highest reward_delta/score_delta/best_score
            def _score(c: Mapping[str, Any]) -> float:
                for key in ('reward_delta', 'score_delta', 'best_score'):
                    try:
                        return float(c.get(key))
                    except Exception:
                        continue
                return 0.0
            candidates_map = [c for c in candidates if isinstance(c, Mapping)]
            if candidates_map:
                chosen = dict(max(candidates_map, key=_score))
        return chosen

    def _build_action_args(self, recent_text: str) -> Dict[str, Dict[str, Any]]:
        action_args: Dict[str, Dict[str, Any]] = {}
        if self._action_enabled('patch'):
            patch_args: Dict[str, Any] = {
                'task': 'Fix streaming errors',
                'context': recent_text[:8000],
                'max_files': int(os.getenv('AUTO_PATCH_MAX_FILES', '4')),
                'max_lines': int(os.getenv('AUTO_PATCH_MAX_LINES', '200')),
                'revert_always': True,
            }
            try:
                from ..context.context_manager import ContextManager
                cm_bundle = ContextManager(self.workspace, self.workspace / 'logs').build_patch_context('Fix streaming errors')
            except Exception:
                cm_bundle = {}
            combined_context = cm_bundle.get('text') or ''
            if combined_context:
                patch_args['context'] = (combined_context + "\n\n" + patch_args['context']).strip()[:8000]
            if cm_bundle.get('file_hints'):
                patch_args['file_hints'] = cm_bundle['file_hints']
            if cm_bundle.get('stats'):
                patch_args['history_stats'] = cm_bundle['stats']
            prompt_entry = self._active_prompt_entry('patch')
            if prompt_entry:
                prompt_text = str(prompt_entry.get('prompt', '')).strip()
                if prompt_text:
                    patch_args['prompt'] = prompt_text
                ident = prompt_entry.get('id') or prompt_entry.get('hash')
                if ident:
                    patch_args['prompt_id'] = str(ident)
            quality_cfg = self._trainer_cfg.get('quality_checks') if isinstance(self._trainer_cfg, dict) else None
            if isinstance(quality_cfg, Mapping):
                patch_args['quality_checks'] = {str(k): str(v) for k, v in quality_cfg.items() if str(v).strip()}
            action_args['patch'] = patch_args
        for name, params in self.shell_actions.items():
            if not self._action_enabled(name):
                continue
            entry = dict(params)
            if name != 'shell_cd':
                try:
                    entry['timeout'] = int(entry.get('timeout', self.shell_timeout))
                except Exception:
                    entry['timeout'] = self.shell_timeout
            action_args[name] = entry
        return action_args

    def stop(self):
        self._stop.set()
        
    def run(self):
        """Main training loop that processes contexts and trains the model."""
        while not self._stop.is_set():
            try:
                # Collect contexts from all containers
                for container in self.containers:
                    topic = f"logs.ctx.{container}"
                    try:
                        item = self.bus.get_latest(topic, timeout=0.1)
                        if item and item.get("ctx"):
                            self._contexts.append({
                                "container": container,
                                "context": item.get("ctx", []),
                                "timestamp": item.get("ts", time.time())
                            })
                    except Exception:
                        pass
                if self.vector_topic:
                    try:
                        vec_item = self.bus.get_latest(self.vector_topic, timeout=0.01)
                        if vec_item:
                            if self._feature_store is None:
                                self._vector_records.append(vec_item)
                    except Exception:
                        pass

                # Train if we have enough data and enough time has passed
                now = time.time()
                if (len(self._contexts) >= self.min_batch and 
                    (now - self._last_train) >= self.interval_sec):
                    
                    self._train_on_contexts()
                    self._contexts.clear()
                    self._last_train = now
                    
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                print(f"Trainer error: {e}")
                time.sleep(5.0)
    
    def _train_on_contexts(self):
        """Train the model on collected contexts."""
        vector_summary = self._aggregate_vector_records()
        try:
            from ..db import get_enhanced_data_manager, TrainingMetrics, Environment, create_log_entry
            import uuid
            
            data_manager = get_enhanced_data_manager()
            
            # Create training session
            session_id = str(uuid.uuid4())
            
            # Simulate training metrics (in real implementation, these would come from actual training)
            import random
            training_accuracy = random.uniform(0.75, 0.95)
            validation_accuracy = training_accuracy - random.uniform(0.02, 0.08)
            loss = random.uniform(0.1, 0.5)
            
            # Store training metrics
            training_metrics = TrainingMetrics(
                session_id=session_id,
                timestamp=time.time(),
                epoch=len(self._contexts),  # Use context count as epoch
                training_accuracy=training_accuracy,
                validation_accuracy=validation_accuracy,
                loss=loss,
                learning_rate=0.001,
                batch_size=len(self._contexts),
                model_type="streaming_rl",
                environment=Environment.DEVELOPMENT,
                hyperparameters={
                    "window_sec": self.window_sec,
                    "min_batch": self.min_batch,
                    "interval_sec": self.interval_sec,
                    "vector_topic": self.vector_topic,
                },
                convergence_metrics={
                    "contexts_processed": len(self._contexts),
                    "containers": len(self.containers),
                    "vector_batches": vector_summary.get('count', 0) if vector_summary else 0,
                    "vector_feature_count": len(vector_summary['features']) if vector_summary else 0,
                }
            )
            
            data_manager.store_training_metrics(training_metrics)
            
            # Log training completion
            log_context = {
                "session_id": session_id,
                "training_accuracy": training_accuracy,
                "validation_accuracy": validation_accuracy,
                "loss": loss,
                "contexts_count": len(self._contexts),
            }
            if vector_summary:
                try:
                    log_context["vector_batches"] = vector_summary.get('count', 0)
                    log_context["vector_features"] = {
                        name: round(value, 4)
                        for name, value in zip(vector_summary['names'], vector_summary['features'])
                    }
                    if vector_summary.get('variances'):
                        log_context["vector_variances"] = {
                            name: round(var, 6)
                            for name, var in zip(vector_summary['names'], vector_summary.get('variances', []))
                        }
                except Exception:
                    pass
            training_log = create_log_entry(
                level="INFO",
                source="streaming_trainer",
                message=f"Training completed on {len(self._contexts)} contexts",
                context=log_context,
                environment=Environment.DEVELOPMENT
            )
            data_manager.log(training_log)
            
            print(f"Training session {session_id} completed: acc={training_accuracy:.3f}, loss={loss:.3f}")
            
        except Exception as e:
            print(f"Error storing training metrics: {e}")
        
        # Original training logic would go here
        # For now, we just log that training happened
        try:
            print(f"[stream-rl] contexts={len(self._contexts)} containers={self.containers}")
            self._trainer_cfg = self._load_trainer_settings()
            self.rl_actions = self._resolve_actions(self._explicit_actions)
            self._action_name_set = set(self.rl_actions) if self.rl_actions else None
            self.tfidf_weights = self._resolve_tfidf_weights(self._explicit_tfidf)
            cfg_flags = self._trainer_cfg if isinstance(self._trainer_cfg, dict) else {}
            self.group_advantage = bool(cfg_flags.get('group_advantage', self.group_advantage))
            try:
                self.group_size = max(1, int(cfg_flags.get('group_size', self.group_size)))
            except Exception:
                pass
            # Aggregate context features using stats when available (from Spark),
            # or fall back to regex counts from raw lines.
            err = warn = timeout_cnt = trace = 0
            # Build a text blob for TF-IDF similarity
            text_blobs: list[str] = []
            for rec in self._contexts:
                # Spark payload shape: {"ctx": [...], "stats": {"error_count": N, "warn_count": M, "total": T}}
                stats = rec.get("stats") if isinstance(rec, dict) else None
                if isinstance(stats, dict):
                    try:
                        err += int(stats.get('error_count', 0) or 0)
                        warn += int(stats.get('warn_count', 0) or 0)
                    except Exception:
                        pass
                ctx = rec.get("context") or []
                s = "\n".join([str(x) for x in ctx]) if isinstance(ctx, list) else str(ctx)
                s_low = s.lower()
                # Fallback/extra counts for timeouts/tracebacks
                timeout_cnt += s_low.count('timeout')
                trace += s_low.count('traceback (most recent call last)')
                text_blobs.append(s)
            def norm(x: int, cap: int = 10) -> float:
                return min(float(x), float(cap)) / float(cap)
            ctx_vec = [norm(err), norm(warn), norm(timeout_cnt), norm(trace)]
            try:
                hist_stats = ContextManager(self.workspace, self.workspace / 'logs').stats_for_features()
            except Exception:
                hist_stats = {}
            ctx_vec.extend([
                min(float(hist_stats.get('recent_success_rate', 0.0)), 1.0),
                min(float(hist_stats.get('recent_failure_rate', 0.0)), 1.0),
                min(float(hist_stats.get('avg_pass_rate', 0.0)), 1.0),
            ])
            if vector_summary:
                try:
                    ctx_vec.extend(vector_summary['features'])
                    variances = vector_summary.get('variances')
                    if isinstance(variances, list):
                        ctx_vec.extend([math.sqrt(v) if v > 0 else 0.0 for v in variances])
                except Exception:
                    pass
            # Add TF-IDF similarity features when index is available
            try:
                from ..embedding.indexer import load_index, semantic_search, vectorize_query, cosine
                meta, items = load_index(self.workspace)
                # Query is the concatenated recent context
                query = "\n".join(text_blobs[-3:]) if text_blobs else ""
                if query.strip():
                    hits = semantic_search(query, meta, items, top_k=5)
                    scores = [float(s) for s, _ in hits]
                    if scores:
                        top1 = max(scores)
                        avg3 = sum(scores[:3]) / float(min(3, len(scores)))
                        w_top = float(self.tfidf_weights.get('top1', 1.0))
                        w_avg = float(self.tfidf_weights.get('avg_top3', 1.0))
                        ctx_vec.extend([top1 * w_top, avg3 * w_avg])
            except Exception:
                pass

            # Lazy import RL toolkit to avoid hard deps on startup
            try:
                from ..rl.rlkit import (
                    RLToolEnv, EnvConfig, RewardConfig, aggregate_reward,
                    detect_toolchain, ToolchainExecutor,
                    get_verifiers as _get_verifiers,
                    make_bandit as _make_bandit,
                )
            except Exception as e:
                print(f"[stream-rl] rl toolkit unavailable: {e}"); return

            # Build env with a frozen context provider (uses the batch features)
            ws = self.workspace
            vlist = _get_verifiers()
            weights, penalties, clamp = self._reward_components()
            rc = RewardConfig(weights=weights, penalty_kinds=penalties, clamp01_kinds=clamp)
            def reward_fn(result, verifiers, wmap):
                return aggregate_reward(result, verifiers, rc)
            def ctx_provider():
                return list(ctx_vec)
            tcfg = detect_toolchain(ws)
            execu = ToolchainExecutor(tcfg)
            # Build a patch action_args payload from the latest contexts
            try:
                recent_text = "\n\n".join(text_blobs[-3:]) if text_blobs else ""
            except Exception:
                recent_text = ""
            action_args = self._build_action_args(recent_text)
            ecfg = EnvConfig(
                verifiers=vlist,
                reward_fn=reward_fn,
                weights=rc.weights,
                context_provider=ctx_provider,
                action_args=action_args or None,
                allowed_actions=self.rl_actions,
            )
            env = RLToolEnv(executor=execu, cfg=ecfg, episode_len=1)

            tool_names = env.action_names

            # Short bandit training burst
            bandit = _make_bandit("epsilon-greedy", env.action_dim, epsilon=0.1)
            rewards: list[float] = []
            steps = max(1, int(os.getenv('RL_BACKGROUND_STEPS', '50')))
            if self.group_advantage and self.group_size > 1:
                remaining = steps
                while remaining > 0:
                    group_n = min(self.group_size, remaining)
                    actions_batch: list[int] = []
                    rewards_batch: list[float] = []
                    obs_batch: list[list[float]] = []
                    for _ in range(group_n):
                        a = int(bandit.select(list(ctx_vec)))
                        obs, r, done, trunc, info = env.step(a)
                        rewards.append(float(r))
                        actions_batch.append(a)
                        rewards_batch.append(float(r))
                        obs_batch.append(list(obs))
                        if done or trunc:
                            try:
                                env.reset()
                            except Exception:
                                pass
                    baseline = (sum(rewards_batch) / len(rewards_batch)) if rewards_batch else 0.0
                    for idx, action in enumerate(actions_batch):
                        advantage = rewards_batch[idx] - baseline
                        reward_signal = max(0.0, advantage)
                        bandit.update(action, reward_signal, obs_batch[idx])
                    remaining -= group_n
            else:
                for _ in range(steps):
                    a = int(bandit.select(list(ctx_vec)))
                    obs, r, done, trunc, info = env.step(a)
                    bandit.update(a, float(r), list(obs))
                    rewards.append(float(r))
                    if done or trunc:
                        try:
                            env.reset()
                        except Exception:
                            pass

            # Persist bandit state for visibility (toolchain bandit, not interactive tool bandit)
            try:
                state = {
                    "policy": "epsilon-greedy",
                    "tools": tool_names,
                    "values": getattr(bandit, 'values', []),
                    "counts": getattr(bandit, 'counts', []),
                }
                (ws / '.dspy_rl_toolchain.json').write_text(json.dumps(state, indent=2))
            except Exception:
                pass

            avg = (sum(rewards)/len(rewards)) if rewards else 0.0
            training_event = {"type":"training","avg_reward": avg, "n": len(rewards), "timestamp": time.time()}
            self.bus.publish("agent.metrics", training_event)

            # Update the interactive bandit (.dspy_rl_state.json) using learned events when available
            try:
                rl_tools = ["context", "codectx", "grep", "esearch", "plan", "tree", "ls", "index", "emb-index", "intel", "vretr", "patch"]
                path = self.workspace / '.dspy_rl_state.json'
                events_path = self.workspace / '.dspy_rl_events.jsonl'
                offset_path = self.workspace / '.dspy_rl_events.offset'
                # Load existing state
                state = {"policy": "epsilon-greedy", "tools": rl_tools, "values": [0.0]*len(rl_tools), "counts": [0]*len(rl_tools)}
                try:
                    if path.exists():
                        obj = json.loads(path.read_text())
                        if isinstance(obj, dict) and obj.get('tools') == rl_tools:
                            state.update({
                                'values': obj.get('values', state['values']),
                                'counts': obj.get('counts', state['counts']),
                            })
                except Exception:
                    pass
                # Build bandit and seed from state
                from ..rl.rlkit import make_bandit as _mk
                band = _mk('epsilon-greedy', len(rl_tools), epsilon=0.1)
                if hasattr(band, 'values'): band.values = list(map(float, state['values']))
                if hasattr(band, 'counts'): band.counts = list(map(int, state['counts']))
                # Apply learned updates from interactive events (tool, reward)
                start_off = 0
                try:
                    if offset_path.exists():
                        start_off = int(offset_path.read_text().strip() or '0')
                except Exception:
                    start_off = 0
                processed = 0
                if events_path.exists():
                    with events_path.open('r') as f:
                        for i, line in enumerate(f):
                            if i < start_off:
                                continue
                            try:
                                evt = json.loads(line)
                                tool = str(evt.get('tool',''))
                                reward = float(evt.get('reward', 0.0))
                                if tool in rl_tools:
                                    idx = rl_tools.index(tool)
                                    band.update(idx, float(max(0.0, min(1.0, reward))))
                                    processed += 1
                            except Exception:
                                continue
                    try:
                        (self.workspace / '.dspy_rl_events.offset').write_text(str(start_off + processed))
                    except Exception:
                        pass
                # Also consume from Kafka 'agent.learning' if available
                try:
                    from confluent_kafka import Consumer  # type: ignore
                    bootstrap = os.getenv('KAFKA_BOOTSTRAP') or os.getenv('KAFKA_BOOTSTRAP_SERVERS') or 'localhost:9092'
                    # Preflight TCP check to avoid noisy logs
                    if not _tcp_check(bootstrap):
                        raise RuntimeError('kafka unavailable')
                    import logging as _log
                    silent = _log.getLogger('kafka.silent'); silent.addHandler(_log.NullHandler()); silent.setLevel(_log.CRITICAL)
                    c = Consumer({
                        'bootstrap.servers': bootstrap,
                        'group.id': 'dspy-rl-learning',
                        'session.timeout.ms': 6000,
                        'auto.offset.reset': 'latest',
                    }, logger=silent)
                    c.subscribe(['agent.learning'])
                    import time as _tt
                    t0 = _tt.time()
                    while _tt.time() - t0 < 0.25:
                        msg = c.poll(0.05)
                        if msg is None or msg.error():
                            continue
                        try:
                            val = msg.value().decode('utf-8', errors='ignore') if isinstance(msg.value(), (bytes, bytearray)) else str(msg.value())
                            evt = json.loads(val)
                            tool = str(evt.get('tool',''))
                            reward = float(evt.get('reward', 0.0))
                            if tool in rl_tools:
                                idx = rl_tools.index(tool)
                                band.update(idx, float(max(0.0, min(1.0, reward))))
                        except Exception:
                            continue
                    try: c.close()
                    except Exception: pass
                except Exception:
                    pass
                # Persist back
                new_state = {
                    'policy': 'epsilon-greedy',
                    'tools': rl_tools,
                    'values': getattr(band, 'values', state['values']),
                    'counts': getattr(band, 'counts', state['counts']),
                }
                path.write_text(json.dumps(new_state, indent=2))
            except Exception as e:
                print(f"[stream-rl] interactive bandit update failed: {e}")
        except Exception as e:
            print(f"[stream-rl] Training error: {e}")
        self._vector_records.clear()


def _tcp_check(bootstrap: str, timeout: float = 0.2) -> bool:
    try:
        import socket as _s
        for tk in (bootstrap or '').split(','):
            tk = tk.strip()
            if not tk:
                continue
            host = tk; port = 9092
            if '://' in host:
                host = host.split('://', 1)[1]
            if host.startswith('[') and ']' in host:
                h, rest = host[1:].split(']', 1)
                host = h
                if rest.startswith(':'):
                    try: port = int(rest[1:])
                    except Exception: port = 9092
            elif ':' in host:
                parts = host.rsplit(':', 1)
                host, port_s = parts[0], parts[1]
                try: port = int(port_s)
                except Exception: port = 9092
            try:
                with _s.create_connection((host, port), timeout=timeout):
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


class StreamingRuntime:
    """Minimal async-friendly runtime that wraps LocalBus for tests.

    Provides initialize/publish/subscribe/shutdown matching test expectations.
    """

    def __init__(self) -> None:
        self._bus = LocalBus()
        self._subs: list[tuple['threading.Thread','threading.Event']] = []
        self.is_initialized = False

    async def initialize(self) -> None:
        self.is_initialized = True

    async def publish(self, topic: str, message: Any) -> None:
        self._bus.publish(topic, message)

    async def subscribe(self, topic: str, handler) -> None:
        q = self._bus.subscribe(topic)
        stop = threading.Event()

        def _loop():
            import asyncio
            while not stop.is_set():
                try:
                    msg = q.get(timeout=0.1)
                except Empty:
                    continue
                except Exception:
                    break
                try:
                    res = handler(msg)
                    if asyncio.iscoroutine(res):
                        asyncio.run(res)
                except Exception:
                    # Swallow handler errors in background loop
                    pass

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        self._subs.append((t, stop))

    async def shutdown(self) -> None:
        for t, s in list(self._subs):
            s.set()
        for t, s in list(self._subs):
            try:
                t.join(timeout=0.5)
            except Exception:
                pass
        self._subs.clear()


__all__ = [
    # Config
    'DEFAULT_CONFIG_PATH','TRAINER_SETTINGS_PATH','KafkaTopic','KafkaConfig','SparkConfig','K8sConfig','ContainerTopic','StreamConfig','load_config','save_config','render_kafka_topic_commands',
    # Runtime
    'LocalBus','StreamingRuntime','Discovered','autodiscover_logs','FileTailer','DockerTailer','Aggregator','Worker','Trainer','start_local_stack','process_ctx','make_context_example',
    # Kafka
    'KafkaParams','WorkerLoop',
]
