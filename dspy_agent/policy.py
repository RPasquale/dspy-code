from __future__ import annotations

"""Policy registry: compact do/don'ts to nudge routing & patch suggestions.

File formats supported:
- YAML: .dspy_policy.yaml (preferred)
- JSON: .dspy_policy.json (fallback)

Schema (YAML/JSON):
policy:
  prefer_tools: ["esearch", "patch"]
  deny_tools: ["open", "watch"]
  rules:
    - regex: "(?i)semantic|search"
      prefer_tools: ["esearch"]
    - regex: "(?i)danger|delete"
      deny_tools: ["open", "watch"]
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re


POLICY_YAML = ".dspy_policy.yaml"
POLICY_JSON = ".dspy_policy.json"


@dataclass
class Rule:
    regex: str
    prefer_tools: List[str] = field(default_factory=list)
    deny_tools: List[str] = field(default_factory=list)


@dataclass
class Policy:
    prefer_tools: List[str] = field(default_factory=list)
    deny_tools: List[str] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)

    @staticmethod
    def load(workspace: Path) -> Optional["Policy"]:
        data: Optional[Dict[str, Any]] = None
        ypath = workspace / POLICY_YAML
        jpath = workspace / POLICY_JSON
        if ypath.exists():
            try:
                import yaml  # type: ignore
                data = yaml.safe_load(ypath.read_text())
            except Exception:
                data = None
        if data is None and jpath.exists():
            try:
                data = json.loads(jpath.read_text())
            except Exception:
                data = None
        if not data:
            return None
        root = data.get('policy') or {}
        rules_data = root.get('rules') or []
        rules: List[Rule] = []
        for r in rules_data:
            rules.append(Rule(regex=str(r.get('regex', '')), prefer_tools=list(r.get('prefer_tools', []) or []), deny_tools=list(r.get('deny_tools', []) or []))
                        )
        return Policy(
            prefer_tools=list(root.get('prefer_tools', []) or []),
            deny_tools=list(root.get('deny_tools', []) or []),
            rules=rules,
        )


def apply_policy_to_state(query: str, state: str, policy: Policy) -> str:
    parts = []
    if policy.prefer_tools:
        parts.append(f"Prefer tools: {', '.join(policy.prefer_tools)}")
    if policy.deny_tools:
        parts.append(f"Avoid tools: {', '.join(policy.deny_tools)}")
    for r in policy.rules:
        try:
            if r.regex and re.search(r.regex, query):
                if r.prefer_tools:
                    parts.append(f"For '{r.regex}': prefer {', '.join(r.prefer_tools)}")
                if r.deny_tools:
                    parts.append(f"For '{r.regex}': avoid {', '.join(r.deny_tools)}")
        except re.error:
            continue
    if not parts:
        return state
    return f"{state} | Policy: {'; '.join(parts)}"


def enforce_policy_on_tool(query: str, tool: str, policy: Policy) -> tuple[str, Optional[str]]:
    t = (tool or '').strip()
    if not t:
        return tool, None
    # Global deny
    if t in set(policy.deny_tools or []):
        return "plan", f"policy-deny: {t}"
    # Rule-specific deny
    for r in policy.rules:
        try:
            if r.regex and re.search(r.regex, query) and t in set(r.deny_tools or []):
                return "plan", f"policy-deny({r.regex}): {t}"
        except re.error:
            continue
    return tool, None


def update_policy_with_feedback(workspace: Path, *, prefer_tool: Optional[str] = None, deny_tool: Optional[str] = None, regex: Optional[str] = None) -> Path:
    """Append a simple rule to the YAML policy file.

    If the YAML parser is unavailable, falls back to JSON file.
    """
    policy = Policy.load(workspace) or Policy()
    # Update base lists
    if prefer_tool and prefer_tool not in policy.prefer_tools:
        policy.prefer_tools.append(prefer_tool)
    if deny_tool and deny_tool not in policy.deny_tools:
        policy.deny_tools.append(deny_tool)
    # Append rule when regex provided
    if regex:
        policy.rules.append(Rule(regex=regex, prefer_tools=[prefer_tool] if prefer_tool else [], deny_tools=[deny_tool] if deny_tool else []))
    obj = {
        'policy': {
            'prefer_tools': list(policy.prefer_tools),
            'deny_tools': list(policy.deny_tools),
            'rules': [
                {'regex': r.regex, 'prefer_tools': r.prefer_tools, 'deny_tools': r.deny_tools}
                for r in policy.rules
            ],
        }
    }
    ypath = workspace / POLICY_YAML
    try:
        import yaml  # type: ignore
        ypath.write_text((yaml.safe_dump(obj)))
        return ypath
    except Exception:
        jpath = workspace / POLICY_JSON
        jpath.write_text(json.dumps(obj, indent=2))
        return jpath

