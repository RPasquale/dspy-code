from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional


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
            containers=[
                ContainerTopic(container="backend", services=["users", "billing"]),
                ContainerTopic(container="frontend", services=["web"]),
            ],
        )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> StreamConfig:
    data = json.loads(path.read_text())
    def _kt(d):
        return KafkaTopic(**d)
    cfg = StreamConfig(
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
    return cfg


def save_config(cfg: StreamConfig, path: Path = DEFAULT_CONFIG_PATH) -> Path:
    def _to(d):
        if isinstance(d, list):
            return [_to(x) for x in d]
        if hasattr(d, "__dataclass_fields__"):
            return {k: _to(v) for k, v in asdict(d).items()}
        return d
    obj = _to(cfg)
    path.write_text(json.dumps(obj, indent=2))
    return path


def render_kafka_topic_commands(cfg: StreamConfig) -> List[str]:
    cmds: List[str] = []
    for t in cfg.kafka.topics:
        cmds.append(
            f"kafka-topics --bootstrap-server {cfg.kafka.bootstrap_servers} --create --topic {t.name} --partitions {t.partitions} --replication-factor {t.replication_factor}"
        )
    return cmds
