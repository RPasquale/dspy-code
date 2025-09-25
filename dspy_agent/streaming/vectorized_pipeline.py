from __future__ import annotations

import hashlib
import json
import math
import string
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

# Avoid importing confluent-kafka at module import time
AdminClient = None  # type: ignore

from ..embedding.indexer import load_index, semantic_search, tokenize

if TYPE_CHECKING:  # pragma: no cover
    from .streamkit import LocalBus
    from .feature_store import FeatureStore


FEATURE_NAMES: List[str] = [
    'char_norm',
    'line_norm',
    'token_diversity',
    'avg_token_len_norm',
    'entropy_norm',
    'rare_token_ratio',
    'error_density',
    'warning_density',
    'exception_density',
    'timeout_density',
    'uppercase_ratio',
    'digit_ratio',
    'punctuation_ratio',
    'tfidf_top1',
    'tfidf_avg_top3',
    'novelty_score',
]


def _tcp_check(bootstrap: str, timeout: float = 0.2) -> bool:
    """Best-effort TCP connectivity check against Kafka bootstrap servers."""
    import socket

    try:
        for target in (bootstrap or '').split(','):
            host = target.strip()
            if not host:
                continue
            port = 9092
            if '://' in host:
                host = host.split('://', 1)[1]
            if host.startswith('[') and ']' in host:
                host, rest = host[1:].split(']', 1)
                if rest.startswith(':'):
                    try:
                        port = int(rest[1:])
                    except ValueError:
                        port = 9092
            elif ':' in host:
                host, port_s = host.rsplit(':', 1)
                try:
                    port = int(port_s)
                except ValueError:
                    port = 9092
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    return True
            except OSError:
                continue
    except Exception:
        return False
    return False


@dataclass
class VectorizedRecord:
    """Normalized vector representation of a streaming event."""

    topic: str
    timestamp: float
    features: List[float]
    feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            'topic': self.topic,
            'ts': self.timestamp,
            'features': list(self.features),
            'feature_names': list(self.feature_names),
            'meta': self.metadata,
        }


class RLVectorizer:
    """Transforms raw streaming payloads into fixed-length RL-ready feature vectors."""

    def __init__(
        self,
        workspace: Path,
        max_chars: int = 4000,
        refresh_interval: float = 300.0,
        max_index_hits: int = 3,
    ) -> None:
        self.workspace = Path(workspace)
        self.max_chars = max_chars
        self.refresh_interval = refresh_interval
        self.max_index_hits = max_index_hits
        self._feature_names = FEATURE_NAMES
        self._index_cache: Optional[Tuple[Any, Any]] = None
        self._last_index_refresh = 0.0
        self._last_index_error: Optional[str] = None

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def feature_size(self) -> int:
        return len(self._feature_names)

    def vectorize(self, topic: str, payload: Any) -> Optional[VectorizedRecord]:
        text, base_meta = self._coerce_text(payload)
        text = (text or '').strip()
        if not text:
            return None
        text = text[: self.max_chars]
        lines = text.splitlines()
        tokens = tokenize(text)
        token_count = len(tokens)
        counts = Counter(tokens)
        unique_tokens = len(counts)
        chars = len(text)

        total_tokens = max(token_count, 1)
        char_norm = min(chars / float(self.max_chars), 1.0)
        line_norm = min(len(lines) / 200.0, 1.0)
        token_diversity = min(unique_tokens / float(total_tokens), 1.0)
        avg_token_len = sum(len(tok) for tok in tokens) / float(total_tokens) if tokens else 0.0
        avg_token_len_norm = min(avg_token_len / 12.0, 1.0)

        entropy_norm = 0.0
        if total_tokens > 1:
            entropy = 0.0
            for value in counts.values():
                p = value / float(total_tokens)
                entropy -= p * math.log(p, 2)
            entropy_norm = min(entropy / math.log(total_tokens, 2), 1.0) if total_tokens > 1 else 0.0

        rare_token_ratio = sum(1 for v in counts.values() if v == 1) / float(total_tokens)

        def _density(keys: Iterable[str]) -> float:
            return sum(counts.get(k, 0) for k in keys) / float(total_tokens)

        error_density = _density(('error', 'failed', 'failure', 'fatal', 'panic'))
        warning_density = _density(('warn', 'warning'))
        exception_density = _density(('exception', 'traceback'))
        timeout_density = _density(('timeout', 'timed', 'deadline', 'timeouterror'))

        letters = [c for c in text if c.isalpha()]
        uppercase_ratio = (sum(1 for c in letters if c.isupper()) / float(len(letters))) if letters else 0.0
        digit_ratio = sum(1 for c in text if c.isdigit()) / float(max(chars, 1))
        punctuation_ratio = sum(1 for c in text if c in string.punctuation) / float(max(chars, 1))

        top1 = 0.0
        avg_top3 = 0.0
        novelty = 1.0
        semantic_hits: List[Dict[str, Any]] = []

        meta_bundle = self._ensure_index()
        if meta_bundle is not None:
            meta, items = meta_bundle
            try:
                hits = semantic_search(text, meta, items, top_k=self.max_index_hits)
                if hits:
                    scores = [float(score) for score, _ in hits]
                    top1 = max(scores)
                    avg_top3 = sum(scores[:3]) / float(min(3, len(scores)))
                    novelty = max(0.0, 1.0 - top1)
                    for score, item in hits[:3]:
                        semantic_hits.append({
                            'path': getattr(item, 'path', ''),
                            'start_line': getattr(item, 'start_line', 0),
                            'end_line': getattr(item, 'end_line', 0),
                            'score': float(score),
                        })
            except Exception as exc:  # pragma: no cover - defensive
                self._last_index_error = str(exc)

        features = [
            float(max(0.0, min(1.0, char_norm))),
            float(max(0.0, min(1.0, line_norm))),
            float(max(0.0, min(1.0, token_diversity))),
            float(max(0.0, min(1.0, avg_token_len_norm))),
            float(max(0.0, min(1.0, entropy_norm))),
            float(max(0.0, min(1.0, rare_token_ratio))),
            float(max(0.0, min(1.0, error_density))),
            float(max(0.0, min(1.0, warning_density))),
            float(max(0.0, min(1.0, exception_density))),
            float(max(0.0, min(1.0, timeout_density))),
            float(max(0.0, min(1.0, uppercase_ratio))),
            float(max(0.0, min(1.0, digit_ratio))),
            float(max(0.0, min(1.0, punctuation_ratio))),
            float(max(0.0, min(1.0, top1))),
            float(max(0.0, min(1.0, avg_top3))),
            float(max(0.0, min(1.0, novelty))),
        ]

        digest = hashlib.sha1(text.encode('utf-8', errors='ignore')).hexdigest()
        metadata = {
            'source_topic': topic,
            'sample': text[:240],
            'chars': chars,
            'lines': len(lines),
            'token_count': token_count,
            'unique_tokens': unique_tokens,
            'semantic_hits': semantic_hits,
            'tfidf_top1': top1,
            'tfidf_avg_top3': avg_top3,
            'novelty': novelty,
            'digest': digest,
        }
        metadata.update(base_meta)
        record = VectorizedRecord(
            topic=topic,
            timestamp=time.time(),
            features=features,
            feature_names=self.feature_names,
            metadata=metadata,
        )
        return record

    def _ensure_index(self) -> Optional[Tuple[Any, Any]]:
        now = time.time()
        if self._index_cache is not None and (now - self._last_index_refresh) < self.refresh_interval:
            return self._index_cache
        try:
            self._index_cache = load_index(self.workspace)
            self._last_index_refresh = now
            self._last_index_error = None
        except Exception as exc:
            self._index_cache = None
            self._last_index_refresh = now
            self._last_index_error = str(exc)
        return self._index_cache

    def _coerce_text(self, payload: Any) -> Tuple[str, Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        if payload is None:
            return '', meta
        if isinstance(payload, str):
            return payload, meta
        if isinstance(payload, bytes):
            try:
                return payload.decode('utf-8', errors='ignore'), meta
            except Exception:
                return payload.decode('latin-1', errors='ignore'), meta
        if isinstance(payload, Mapping):
            meta['payload_keys'] = list(payload.keys())
            if 'ts' in payload:
                meta['source_ts'] = payload.get('ts')
            if 'ctx' in payload and isinstance(payload['ctx'], list):
                return '\n'.join(str(x) for x in payload['ctx']), meta
            if 'lines' in payload and isinstance(payload['lines'], list):
                return '\n'.join(str(x) for x in payload['lines']), meta
            if 'message' in payload:
                return str(payload['message']), meta
            if 'value' in payload:
                return str(payload['value']), meta
            try:
                return json.dumps(payload, default=str), meta
            except Exception:
                return str(payload), meta
        if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
            return '\n'.join(str(x) for x in payload), meta
        return str(payload), meta


class VectorizedStreamOrchestrator(Thread):
    """Background worker that vectorizes selected bus topics into RL features."""

    def __init__(
        self,
        bus: 'LocalBus',
        topics: Sequence[str],
        vectorizer: RLVectorizer,
        out_topic: str,
        feature_store: Optional['FeatureStore'] = None,
        idle_sleep: float = 0.1,
    ) -> None:
        super().__init__(daemon=True)
        self.bus = bus
        self.topics = [t for t in dict.fromkeys(topics) if t and t != out_topic]
        self.vectorizer = vectorizer
        self.out_topic = out_topic
        self.idle_sleep = idle_sleep
        self._stop_event = Event()
        self._feature_store = feature_store
        self._subscriptions: List[Tuple[str, Queue]] = [(topic, bus.subscribe(topic)) for topic in self.topics]
        self._metrics: Dict[str, Any] = {
            'processed': 0,
            'dropped': 0,
            'errors': 0,
            'last_error': None,
            'last_emission_ts': None,
            'per_topic': {topic: 0 for topic in self.topics},
        }

    def stop(self) -> None:
        self._stop_event.set()

    def metrics(self) -> Dict[str, Any]:
        state = dict(self._metrics)
        state['per_topic'] = dict(self._metrics.get('per_topic', {}))
        state['topics'] = list(self.topics)
        state['out_topic'] = self.out_topic
        return state

    def run(self) -> None:  # pragma: no cover - exercised indirectly via tests
        while not self._stop_event.is_set():
            had_event = False
            for topic, queue in self._subscriptions:
                if self._stop_event.is_set():
                    break
                try:
                    payload = queue.get_nowait()
                except Empty:
                    continue
                had_event = True
                try:
                    record = self.vectorizer.vectorize(topic, payload)
                except Exception as exc:  # pragma: no cover - defensive
                    self._metrics['errors'] += 1
                    self._metrics['last_error'] = str(exc)
                    continue
                if record is None:
                    self._metrics['dropped'] += 1
                    continue
                if self._feature_store is not None:
                    try:
                        self._feature_store.update(record.as_dict())
                    except Exception:
                        pass
                self.bus.publish(self.out_topic, record.as_dict())
                self._metrics['processed'] += 1
                per_topic = self._metrics.setdefault('per_topic', {})
                per_topic[topic] = per_topic.get(topic, 0) + 1
                self._metrics['last_emission_ts'] = record.timestamp
            if not had_event:
                self._stop_event.wait(self.idle_sleep)


@dataclass
class KafkaTopicStatus:
    name: str
    partitions: int
    replication_factor: int
    has_leader: bool


@dataclass
class KafkaHealthSnapshot:
    available: bool
    reason: Optional[str]
    timestamp: float
    topics: List[KafkaTopicStatus] = field(default_factory=list)


class KafkaStreamInspector:
    """Light-weight Kafka health checker for dashboards and diagnostics."""

    def __init__(self, bootstrap: str, topics: Sequence[str], timeout: float = 1.0) -> None:
        self.bootstrap = bootstrap
        self.topics = [t for t in topics if t]
        self.timeout = timeout
        self._last_snapshot: Optional[KafkaHealthSnapshot] = None

    def probe(self) -> KafkaHealthSnapshot:
        if not self.bootstrap:
            snapshot = KafkaHealthSnapshot(False, 'bootstrap not configured', time.time())
            self._last_snapshot = snapshot
            return snapshot

        if not _tcp_check(self.bootstrap, timeout=self.timeout):
            snapshot = KafkaHealthSnapshot(False, 'bootstrap unreachable', time.time())
            self._last_snapshot = snapshot
            return snapshot

        topics: List[KafkaTopicStatus] = []
        reason: Optional[str] = None
        # Lazy import AdminClient only when explicitly enabled
        if os.getenv("DSPY_ENABLE_KAFKA_ADMIN", "0").lower() in {"1", "true", "yes"}:
            try:
                from confluent_kafka.admin import AdminClient as _AdminClient  # type: ignore
                admin = _AdminClient({'bootstrap.servers': self.bootstrap})
                metadata = admin.list_topics(timeout=self.timeout)
                for topic_name in self.topics:
                    topic_meta = metadata.topics.get(topic_name)
                    if topic_meta is None or getattr(topic_meta, 'error', None):
                        topics.append(KafkaTopicStatus(topic_name, 0, 0, False))
                        continue
                    partitions = getattr(topic_meta, 'partitions', {})
                    partition_count = len(partitions)
                    replication_factor = 0
                    has_leader = True
                    for pdata in partitions.values():
                        replicas = getattr(pdata, 'replicas', [])
                        replication_factor = max(replication_factor, len(replicas))
                        if getattr(pdata, 'leader', -1) == -1:
                            has_leader = False
                    topics.append(KafkaTopicStatus(topic_name, partition_count, replication_factor, has_leader))
            except Exception as exc:  # pragma: no cover - network/path dependent
                reason = f'metadata error: {exc}'
        snapshot = KafkaHealthSnapshot(True, reason, time.time(), topics=topics)
        self._last_snapshot = snapshot
        return snapshot

    def last_snapshot(self) -> Optional[KafkaHealthSnapshot]:
        return self._last_snapshot


@dataclass
class SparkCheckpointStatus:
    active: bool
    timestamp: float
    staleness: float
    reason: Optional[str] = None
    latest_file: Optional[str] = None


class SparkCheckpointMonitor:
    """Monitors Spark Structured Streaming checkpoints for activity."""

    def __init__(self, checkpoint_dir: Path, freshness_threshold: float = 120.0) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.freshness_threshold = freshness_threshold
        self._last_status: Optional[SparkCheckpointStatus] = None

    def snapshot(self) -> SparkCheckpointStatus:
        now = time.time()
        if not self.checkpoint_dir.exists():
            status = SparkCheckpointStatus(False, now, float('inf'), reason='checkpoint missing')
            self._last_status = status
            return status

        latest_mtime = None
        latest_path = None
        try:
            for path in self.checkpoint_dir.rglob('*'):
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_path = str(path)
        except Exception:
            latest_mtime = None
        if latest_mtime is None:
            latest_mtime = self.checkpoint_dir.stat().st_mtime
            latest_path = str(self.checkpoint_dir)

        staleness = now - latest_mtime
        active = staleness <= self.freshness_threshold
        reason = None if active else f'stale by {staleness:.1f}s'
        status = SparkCheckpointStatus(active, now, staleness, reason=reason, latest_file=latest_path)
        self._last_status = status
        return status

    def last_status(self) -> Optional[SparkCheckpointStatus]:
        return self._last_status


__all__ = [
    'VectorizedRecord',
    'RLVectorizer',
    'VectorizedStreamOrchestrator',
    'KafkaTopicStatus',
    'KafkaHealthSnapshot',
    'KafkaStreamInspector',
    'SparkCheckpointStatus',
    'SparkCheckpointMonitor',
]
