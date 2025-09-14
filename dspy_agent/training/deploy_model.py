from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Stream names for deployment logs/events (used with Storage.append)
DEPLOY_LOG_STREAM = "deploy.logs.lightweight"
DEPLOY_EVENT_STREAM = "deploy.events.lightweight"
DEPLOY_LOG_TOPIC = DEPLOY_LOG_STREAM
DEPLOY_EVENT_TOPIC = DEPLOY_EVENT_STREAM

# KV keys for latest deployment snapshot
KV_LAST_STATUS = "deploy:last:lightweight:status"
KV_LAST_IMAGE = "deploy:last:lightweight:image"
KV_LAST_COMPOSE_HASH = "deploy:last:lightweight:compose_hash"
KV_LAST_TS = "deploy:last:lightweight:ts"


Status = Literal["pending", "building", "up", "down", "error", "done"]


@dataclass
class DeployEvent:
    ts: float
    phase: str
    level: str
    message: str
