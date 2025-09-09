from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    model_name: str
    openai_api_key: str | None
    openai_base_url: str | None
    local_mode: bool
    max_log_bytes: int  # 500 KB per file cap
    # Tool approval mode: "auto" (default) or "manual" (require confirmation before tool calls)
    tool_approval_mode: str


def get_settings() -> Settings:
    return Settings(
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        local_mode=os.getenv("LOCAL_MODE", "false").lower() in {"1", "true", "yes", "on"},
        max_log_bytes=int(os.getenv("MAX_LOG_BYTES", "512000")),
        tool_approval_mode=os.getenv("TOOL_APPROVAL", "auto").lower(),
    )
