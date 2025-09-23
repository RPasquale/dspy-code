from __future__ import annotations

"""User preferences and policy checks to steer the agent.

This module lets users declare preferences for how the agent should operate
(e.g., avoid certain commands or code patterns). The checks are best‑effort and
non‑fatal: if no preferences file exists, they are skipped.

File: `.dspy_preferences.json` at the workspace root.
Schema (JSON):
{
  "forbidden_commands": ["rm -rf", "git reset --hard"],
  "forbidden_code_patterns": ["except Exception:\\s*pass"],
  "max_blast_radius": 800
}
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import json
import re


PREFS_FILENAME = ".dspy_preferences.json"


@dataclass
class Preferences:
    forbidden_commands: List[str] = field(default_factory=list)
    forbidden_code_patterns: List[str] = field(default_factory=list)
    max_blast_radius: int = 0  # 0 disables

    @staticmethod
    def load(workspace: Path) -> "Preferences | None":
        p = (workspace / PREFS_FILENAME)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            return Preferences(
                forbidden_commands=list(data.get("forbidden_commands", []) or []),
                forbidden_code_patterns=list(data.get("forbidden_code_patterns", []) or []),
                max_blast_radius=int(data.get("max_blast_radius", 0) or 0),
            )
        except Exception:
            return None


def check_patch_against_prefs(patch_text: str, prefs: Preferences) -> List[str]:
    issues: List[str] = []
    text = patch_text or ""
    # Command lines (in tests or scripts) in added hunks
    added_lines = [ln[1:] for ln in text.splitlines() if ln.startswith('+')]
    joined = "\n".join(added_lines)
    for pattern in prefs.forbidden_commands:
        try:
            if re.search(pattern, joined, re.IGNORECASE):
                issues.append(f"forbidden command: /{pattern}/")
        except re.error:
            # Fallback to substring
            if pattern.lower() in joined.lower():
                issues.append(f"forbidden command (substr): {pattern}")
    for pattern in prefs.forbidden_code_patterns:
        try:
            if re.search(pattern, joined):
                issues.append(f"forbidden code pattern: /{pattern}/")
        except re.error:
            if pattern in joined:
                issues.append(f"forbidden code pattern (substr): {pattern}")
    return issues


def write_default_preferences(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj: Dict[str, Any] = {
        "forbidden_commands": ["rm -rf", "git reset --hard", "chmod -R 777 /"],
        "forbidden_code_patterns": [r"except Exception:\\s*pass", r"subprocess\.run\(.*shell=True.*\)"],
        "max_blast_radius": 800,
    }
    path.write_text(json.dumps(obj, indent=2))
    return path

