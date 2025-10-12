from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

from ..skills.task_agent import TaskAgent


@dataclass
class ReasoningBundle:
    """Structured reasoning output for downstream modules."""

    plan: str
    commands: str
    assumptions: str
    risks: str
    rationale: str
    generated_at: float

    def to_prompt(self, *, max_chars: int = 600) -> str:
        """Render a compact prompt segment."""
        sections = []
        if self.plan.strip():
            sections.append(f"Plan: {self.plan.strip()}")
        if self.commands.strip():
            sections.append(f"Commands: {self.commands.strip()}")
        if self.assumptions.strip():
            sections.append(f"Assumptions: {self.assumptions.strip()}")
        if self.risks.strip():
            sections.append(f"Risks: {self.risks.strip()}")
        if self.rationale.strip():
            sections.append(f"Rationale: {self.rationale.strip()}")
        text = " | ".join(sections)
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        return text

    def to_dict(self) -> Dict[str, str]:
        out = asdict(self)
        out.pop("generated_at", None)
        return out


class ReasoningHarness:
    """Reusable wrapper around TaskAgent to supply explicit reasoning plans."""

    def __init__(self, workspace: Optional[Path] = None, *, cache_ttl: float = 60.0) -> None:
        self.workspace = workspace.resolve() if isinstance(workspace, Path) else None
        self.cache_ttl = max(5.0, float(cache_ttl))
        self._cache: Dict[str, ReasoningBundle] = {}
        try:
            self.agent = TaskAgent(use_cot=None)
        except Exception:
            self.agent = TaskAgent()

    @staticmethod
    def _cache_key(task: str, context: str) -> str:
        src = (task or "").strip() + "\n" + (context or "").strip()
        return hashlib.md5(src.encode("utf-8", errors="ignore")).hexdigest()

    def run(self, task: str, *, context: str = "", force_refresh: bool = False) -> ReasoningBundle:
        key = self._cache_key(task, context)
        now = time.time()
        if not force_refresh:
            cached = self._cache.get(key)
            if cached and (now - cached.generated_at) < self.cache_ttl:
                return cached
        try:
            result = self.agent(task=task, context=context)
        except Exception:
            result = None
        bundle = ReasoningBundle(
            plan=(getattr(result, "plan", "") or "").strip(),
            commands=(getattr(result, "commands", "") or "").strip(),
            assumptions=(getattr(result, "assumptions", "") or "").strip(),
            risks=(getattr(result, "risks", "") or "").strip(),
            rationale=(getattr(result, "rationale", "") or "").strip(),
            generated_at=now,
        )
        if not bundle.plan:
            bundle.plan = f"Rephrase goal succinctly: {task.strip()[:180]}"
        self._cache[key] = bundle
        return bundle
