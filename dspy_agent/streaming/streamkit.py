from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------
# Config dataclasses
# -----------------

DEFAULT_CONFIG_PATH = Path(".dspy_stream.json")


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
        ]
        return StreamConfig(
            kafka=KafkaConfig(topics=topics),
            spark=SparkConfig(),
            k8s=K8sConfig(),
            containers=[ContainerTopic(container="backend", services=["users", "billing"]), ContainerTopic(container="frontend", services=["web"])],
        )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> StreamConfig:
    data = json.loads(path.read_text())
    def _kt(d): return KafkaTopic(**d)
    return StreamConfig(
        kafka=KafkaConfig(
            bootstrap_servers=data["kafka"].get("bootstrap_servers", "localhost:9092"),
            zookeeper=data["kafka"].get("zookeeper"),
            group_id=data["kafka"].get("group_id", "dspy-code"),
            acks=data["kafka"].get("acks", "all"),
            topics=[_kt(t) for t in data["kafka"].get("topics", [])],
        ),
        spark=SparkConfig(**data.get("spark", {})),
        k8s=K8sConfig(**data.get("k8s", {})),
        containers=[ContainerTopic(**c) for c in data.get("containers", [])],
    )


def save_config(cfg: StreamConfig, path: Path = DEFAULT_CONFIG_PATH) -> Path:
    def _to(d):
        if isinstance(d, list): return [_to(x) for x in d]
        if hasattr(d, "__dataclass_fields__"): return {k: _to(v) for k, v in asdict(d).items()}
        return d
    obj = _to(cfg); path.write_text(json.dumps(obj, indent=2)); return path


def render_kafka_topic_commands(cfg: StreamConfig) -> List[str]:
    return [f"kafka-topics --bootstrap-server {cfg.kafka.bootstrap_servers} --create --topic {t.name} --partitions {t.partitions} --replication-factor {t.replication_factor}" for t in cfg.kafka.topics]


# -----------------
# Runtime (local bus, tailers, workers)
# -----------------

class LocalBus:
    def __init__(self, storage: Optional[object] = None, kafka: Optional[object] = None) -> None:
        self._topics: Dict[str, List[Queue]] = {}
        self._lock = threading.Lock()
        self._storage = storage
        self._kafka = kafka
    def publish(self, topic: str, message: Any) -> None:
        with self._lock: subs = list(self._topics.get(topic, []))
        for q in subs: q.put(message)
        try:
            if self._storage is not None and hasattr(self._storage, "append"):
                self._storage.append(topic, message)  # type: ignore[attr-defined]
        except Exception: pass
        try:
            if self._kafka is not None:
                self._kafka.send(topic, message)  # type: ignore[attr-defined]
        except Exception: pass
    def subscribe(self, topic: str) -> Queue:
        q: Queue = Queue()
        with self._lock: self._topics.setdefault(topic, []).append(q)
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


class Worker(threading.Thread):
    def __init__(self, container: str, root: Path, bus: LocalBus, ctx_topic: str, results_topic: str) -> None:
        super().__init__(daemon=True)
        self.container=container; self.root=root; self.bus=bus; self.in_q=bus.subscribe(ctx_topic); self.results_topic=results_topic; self._stop=threading.Event()
        from ..llm import configure_lm
        from .skills.context_builder import ContextBuilder
        from .skills.task_agent import TaskAgent
        from .skills.code_edit import CodeEdit
        self.lm = configure_lm(provider="ollama", model_name=None, base_url=None, api_key=None)
        self.builder = ContextBuilder(); self.agent = TaskAgent(); self.editor = CodeEdit()
        self.auto_patch = os.getenv("AUTO_PATCH", "false").lower() in {"1","true","yes","on"}
        self.auto_commit = os.getenv("AUTO_COMMIT", "false").lower() in {"1","true","yes","on"}
        self.test_cmd = os.getenv("AUTO_TEST_CMD"); self.test_timeout = int(os.getenv("AUTO_TEST_TIMEOUT", "600")); self.test_strict = os.getenv("AUTO_TEST_STRICT", "true").lower() in {"1","true","yes","on"}
        self.require_keywords = os.getenv("AUTO_PATCH_REQUIRE_KEYWORDS", "true").lower() in {"1","true","yes","on"}
        kws = os.getenv("AUTO_PATCH_KEYWORDS", "error,exception,fail,timeout").split(","); self.keywords=[k.strip().lower() for k in kws if k.strip()]
        self.approval_mode = os.getenv("AUTO_PATCH_APPROVAL", "manual").lower(); self._settings_path = (self.root / ".dspy_settings.json")

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
        from .patcher import apply_unified_patch, summarize_patch, run_shell, revert_unified_patch, git_commit
        while not self._stop.is_set():
            try: item = self.in_q.get(timeout=0.5)
            except Empty: continue
            lines = item.get("ctx", [])
            if not lines: continue
            text = "\n".join(lines); result = process_ctx(self.container, text, self.lm, self.builder, self.agent)
            self.bus.publish(self.results_topic, result)
            if self.auto_patch:
                try:
                    self._load_settings(); edit = self.editor(task=f"Fix {self.container} errors", context=text, code_graph="", file_hints=""); patch_text = getattr(edit, "patch", "") or ""
                    if self.require_keywords and self.keywords and not any(k in text.lower() for k in self.keywords): raise RuntimeError("keyword gate not satisfied")
                    if patch_text.strip():
                        summ = summarize_patch(patch_text); total_lines = summ["added_lines"] + summ["removed_lines"]
                        caps_ok = ((self.max_files <= 0 or summ["files"] <= self.max_files) and (self.max_lines <= 0 or total_lines <= self.max_lines))
                        if caps_ok:
                            pdir = (self.root / ".dspy_patches"); pdir.mkdir(parents=True, exist_ok=True)
                            pid = str(int(time.time()*1000)); (pdir / f"{pid}.patch").write_text(patch_text)
                            meta = {"id": pid, "container": self.container, "summary": summ, "ts": time.time(), "applied": False}
                            (pdir / f"{pid}.json").write_text(json.dumps(meta, indent=2))
                            ok = False
                            if self.approval_mode == "auto":
                                ok, msg = apply_unified_patch(patch_text, self.root); meta.update({"applied": ok, "apply_message": msg})
                                test_ok = None
                                if ok and self.test_cmd:
                                    code, out, err = run_shell(self.test_cmd, self.root, timeout=self.test_timeout)
                                    test_ok = None if ((code == 127 or (err or '').lower().find('not found')>=0) and not self.test_strict) else (code == 0)
                                    meta.update({"test_cmd": self.test_cmd, "test_code": code, "test_ok": test_ok})
                                    if test_ok is False:
                                        r_ok, r_msg = revert_unified_patch(patch_text, self.root); meta.update({"reverted": r_ok, "revert_message": r_msg}); ok = False
                                if ok and (test_ok in (None, True)) and self.auto_commit:
                                    c_ok, c_msg = git_commit(self.root, f"autopatch({self.container}): files={summ['files']} lines={summ['added_lines']+summ['removed_lines']}"); meta.update({"committed": c_ok, "commit_message": c_msg})
                            else:
                                meta.update({"queued": True, "apply_message": "queued for approval"})
                            (pdir / f"{pid}.json").write_text(json.dumps(meta, indent=2)); self.bus.publish("agent.patches", meta)
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
    chosen = {d.container: d.log_file for d in autodiscover_logs(root)}
    for ct in cfg.containers:
        container = getattr(ct, 'container'); log_file = chosen.get(container)
        if not log_file or not log_file.exists(): continue
        raw_topic = f"logs.raw.{container}"; ctx_topic = f"logs.ctx.{container}"; results_topic = f"agent.results.{container}"
        t1 = FileTailer(log_file, bus, raw_topic); t2 = Aggregator(bus, raw_topic, ctx_topic, window_sec=5.0); t3 = Worker(container, root, bus, ctx_topic, results_topic)
        t1.start(); t2.start(); t3.start(); threads.extend([t1, t2, t3])
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
        from .skills.context_builder import ContextBuilder
        from .skills.task_agent import TaskAgent
        from .skills.code_edit import CodeEdit
        from ..llm import configure_lm
        self.builder = ContextBuilder(); self.agent = TaskAgent(); self.lm = configure_lm(provider="ollama", model_name=None, base_url=None, api_key=None); self.editor = CodeEdit()
        self.auto_patch = os.getenv("AUTO_PATCH", "false").lower() in {"1","true","yes","on"}; self.auto_commit = os.getenv("AUTO_COMMIT", "false").lower() in {"1","true","yes","on"}
        self.root = Path(os.getenv("WORKSPACE", "/workspace")).resolve(); self.test_cmd = os.getenv("AUTO_TEST_CMD"); self.test_timeout = int(os.getenv("AUTO_TEST_TIMEOUT", "600")); self.test_strict = os.getenv("AUTO_TEST_STRICT", "true").lower() in {"1","true","yes","on"}
        self.require_keywords = os.getenv("AUTO_PATCH_REQUIRE_KEYWORDS", "true").lower() in {"1","true","yes","on"}; kws = os.getenv("AUTO_PATCH_KEYWORDS", "error,exception,fail,timeout").split(","); self.keywords = [k.strip().lower() for k in kws if k.strip()]
        self.approval_mode = os.getenv("AUTO_PATCH_APPROVAL", "manual").lower(); self._settings_path = self.root / ".dspy_settings.json"

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
                        self._load_settings(); edit = self.editor(task=f"Fix {self.p.container} errors", context=text, code_graph="", file_hints=""); patch_text = getattr(edit, "patch", "") or ""
                        if self.require_keywords and self.keywords and not any(k in text.lower() for k in self.keywords): raise RuntimeError("keyword gate not satisfied")
                        if patch_text.strip():
                            from .patcher import apply_unified_patch, summarize_patch, run_shell, revert_unified_patch, git_commit
                            summ = summarize_patch(patch_text); total_lines = summ["added_lines"] + summ["removed_lines"]
                            caps_ok = ((self.max_files <= 0 or summ["files"] <= self.max_files) and (self.max_lines <= 0 or total_lines <= self.max_lines))
                            if caps_ok:
                                pdir = (self.root / ".dspy_patches"); pdir.mkdir(parents=True, exist_ok=True)
                                pid = str(int(time.time()*1000))
                                (pdir / f"{pid}.patch").write_text(patch_text)
                                meta = {"id": pid, "container": self.p.container, "summary": summ, "ts": time.time(), "applied": False}
                                (pdir / f"{pid}.json").write_text(json.dumps(meta, indent=2))
                                ok = False
                                if self.approval_mode == "auto":
                                    ok, msg = apply_unified_patch(patch_text, self.root); meta.update({"applied": ok, "apply_message": msg})
                                    test_ok = None
                                    if ok and self.test_cmd:
                                        code, out, err = run_shell(self.test_cmd, self.root, timeout=self.test_timeout)
                                        test_ok = None if ((code == 127 or (err or '').lower().find('not found')>=0) and not self.test_strict) else (code == 0)
                                        meta.update({"test_cmd": self.test_cmd, "test_code": code, "test_ok": test_ok})
                                        if test_ok is False:
                                            r_ok, r_msg = revert_unified_patch(patch_text, self.root); meta.update({"reverted": r_ok, "revert_message": r_msg}); ok = False
                                    if ok and (test_ok in (None, True)) and self.auto_commit:
                                        c_ok, c_msg = git_commit(self.root, f"autopatch({self.p.container}): files={summ['files']} lines={summ['added_lines']+summ['removed_lines']}"); meta.update({"committed": c_ok, "commit_message": c_msg})
                                else:
                                    meta.update({"queued": True, "apply_message": "queued for approval"})
                                (pdir / f"{pid}.json").write_text(json.dumps(meta, indent=2)); producer.produce("agent.patches", json.dumps(meta).encode('utf-8')); producer.poll(0)
                    except Exception: pass
        finally:
            try: consumer.close()
            except Exception: pass


class Trainer(threading.Thread):
    """Streaming trainer that processes log context and trains the RL model."""
    
    def __init__(self, workspace: Path, bus: LocalBus, containers: List[str], min_batch: int = 3, interval_sec: float = 60.0):
        super().__init__(daemon=True)
        self.workspace = workspace
        self.bus = bus
        self.containers = containers
        self.min_batch = min_batch
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._contexts = []
        self._last_train = time.time()
        
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
        try:
            print(f"[stream-rl] contexts={len(self._contexts)} containers={self.containers}")
            # Aggregate context features: error/warn/timeout/traceback counts
            err = warn = timeout_cnt = trace = 0
            for rec in self._contexts:
                ctx = rec.get("context") or []
                s = "\n".join([str(x) for x in ctx]).lower() if isinstance(ctx, list) else str(ctx).lower()
                err += s.count(' error ')
                warn += s.count(' warn')
                timeout_cnt += s.count('timeout')
                trace += s.count('traceback (most recent call last)')
            def norm(x: int, cap: int = 10) -> float:
                return min(float(x), float(cap)) / float(cap)
            ctx_vec = [norm(err), norm(warn), norm(timeout_cnt), norm(trace)]

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
            rc = RewardConfig(weights={"pass_rate": 1.0, "blast_radius": 1.0})
            def reward_fn(result, verifiers, wmap):
                return aggregate_reward(result, verifiers, rc)
            def ctx_provider():
                return list(ctx_vec)
            tcfg = detect_toolchain(ws)
            execu = ToolchainExecutor(tcfg)
            ecfg = EnvConfig(verifiers=vlist, reward_fn=reward_fn, weights=rc.weights, context_provider=ctx_provider, action_args=None)
            env = RLToolEnv(executor=execu, cfg=ecfg, episode_len=1)

            # Short bandit training burst
            bandit = _make_bandit("epsilon-greedy", env.action_dim, epsilon=0.1)
            rewards = []
            import random as _r
            steps = int(os.getenv('RL_BACKGROUND_STEPS', '50'))
            for _ in range(max(1, steps)):
                a = bandit.select(list(ctx_vec))
                obs, r, done, trunc, info = env.step(int(a))
                bandit.update(int(a), float(r), list(obs))
                rewards.append(float(r))

            # Persist bandit state for visibility (toolchain bandit, not interactive tool bandit)
            try:
                state = {
                    "policy": "epsilon-greedy",
                    "tools": ["run_tests","lint","build"],
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
                rl_tools = ["context", "codectx", "grep", "esearch", "plan", "tree", "ls", "index", "emb-index", "intel", "vretr"]
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


__all__ = [
    # Config
    'DEFAULT_CONFIG_PATH','KafkaTopic','KafkaConfig','SparkConfig','K8sConfig','ContainerTopic','StreamConfig','load_config','save_config','render_kafka_topic_commands',
    # Runtime
    'LocalBus','Discovered','autodiscover_logs','FileTailer','Aggregator','Worker','Trainer','start_local_stack','process_ctx','make_context_example',
    # Kafka
    'KafkaParams','WorkerLoop',
]
