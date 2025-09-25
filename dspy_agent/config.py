from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json as _json


@dataclass
class Settings:
    model_name: str
    openai_api_key: str | None
    openai_base_url: str | None
    use_ollama: bool
    local_mode: bool
    max_log_bytes: int  # 500 KB per file cap
    # Tool approval mode: "auto" (default) or "manual" (require confirmation before tool calls)
    tool_approval_mode: str
    # Database configuration (local/dev/prod)
    db_backend: str  # e.g., "reddb", "none"
    reddb_url: str | None
    reddb_namespace: str | None
    # RL configuration
    rl_policy: str
    rl_epsilon: float
    rl_ucb_c: float
    rl_n_envs: int
    rl_verifiers_module: Optional[str]
    rl_puffer: bool
    rl_weights: Dict[str, float]
    rl_penalty_kinds: List[str]
    rl_clamp01_kinds: List[str]
    rl_scales: Dict[str, Tuple[float, float]]


def _parse_json_map(env_key: str, default: Dict) -> Dict:
    try:
        raw = os.getenv(env_key)
        if not raw:
            return default
        val = _json.loads(raw)
        return dict(val)
    except Exception:
        return default


def _parse_csv(env_key: str) -> List[str]:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(',') if x.strip()]


def _parse_scales(env_key: str) -> Dict[str, Tuple[float, float]]:
    raw = os.getenv(env_key)
    if not raw:
        return {}
    try:
        data = _json.loads(raw)
        out: Dict[str, Tuple[float, float]] = {}
        for k, v in dict(data).items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                out[str(k)] = (float(v[0]), float(v[1]))
        return out
    except Exception:
        return {}


def get_settings() -> Settings:
    return Settings(
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        use_ollama=os.getenv("USE_OLLAMA", "true").lower() in {"1", "true", "yes", "on"},
        local_mode=os.getenv("LOCAL_MODE", "false").lower() in {"1", "true", "yes", "on"},
        max_log_bytes=int(os.getenv("MAX_LOG_BYTES", "512000")),
        tool_approval_mode=os.getenv("TOOL_APPROVAL", "auto").lower(),
        db_backend=os.getenv("DB_BACKEND", os.getenv("REDDB_URL") and "reddb" or "none"),
        reddb_url=os.getenv("REDDB_URL"),
        reddb_namespace=os.getenv("REDDB_NAMESPACE"),
        rl_policy=os.getenv("RL_POLICY", "epsilon-greedy"),
        rl_epsilon=float(os.getenv("RL_EPSILON", "0.1")),
        rl_ucb_c=float(os.getenv("RL_UCB_C", "2.0")),
        rl_n_envs=int(os.getenv("RL_N_ENVS", "2")),
        rl_verifiers_module=os.getenv("RL_VERIFIERS_MODULE"),
        rl_puffer=os.getenv("RL_PUFFER", "false").lower() in {"1", "true", "yes", "on"},
        rl_weights=_parse_json_map("RL_WEIGHTS", {}),
        rl_penalty_kinds=_parse_csv("RL_PENALTY_KINDS"),
        rl_clamp01_kinds=_parse_csv("RL_CLAMP01_KINDS"),
        rl_scales=_parse_scales("RL_SCALES"),
    )


# Backward-compatibility shim for older tests that expect Config
class Config:
    """Simple configuration facade for tests expecting `Config`.

    Exposes a minimal surface used by tests:
    - `workspace`: resolved from WORKSPACE_DIR or CWD
    - `log_level`: normalized log level from DSPY_LOG_LEVEL
    - `auto_train`: boolean from DSPY_AUTO_TRAIN
    """

    _ALLOWED_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}

    def __init__(self, workspace: Optional[Path] = None):
        try:
            ws_env = os.getenv("WORKSPACE_DIR")
            base = Path(ws_env) if ws_env else (workspace or Path.cwd())
            self.workspace = Path(base).resolve()
        except Exception:
            # Fall back to current directory if resolution fails
            self.workspace = Path.cwd()

        raw_level = (os.getenv("DSPY_LOG_LEVEL", "INFO") or "INFO").upper()
        self.log_level = raw_level if raw_level in self._ALLOWED_LEVELS else "INFO"

        auto = os.getenv("DSPY_AUTO_TRAIN", "false").strip().lower()
        self.auto_train = auto in {"1", "true", "yes", "on"}
