from __future__ import annotations

"""Train a PreferenceRewriter from feedback via GEPA and emit a policy prompt.

This consumes .dspy_feedback.jsonl and produces a compact text prompt stored in
`.dspy_policy_prompt.txt`, which the orchestrator appends to its state.
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
import dspy

from ..skills.preference_rewriter import PreferenceRewriter


def _read_feedback(path: Path) -> List[dict]:
    items: List[dict] = []
    if not path.exists():
        return items
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            continue
    return items


def _examples_from_feedback(fb: List[dict]) -> List[dspy.Example]:
    out: List[dspy.Example] = []
    for rec in fb:
        q = rec.get('query', '') or ''
        exp_tool = ((rec.get('expected') or {}).get('tool') or '')
        obs_tool = ((rec.get('observed') or {}).get('tool') or '')
        rgx = rec.get('regex', '') or ''
        policy_text = []
        if exp_tool:
            policy_text.append(f"Prefer tools: {exp_tool}")
        if obs_tool:
            policy_text.append(f"Avoid tools: {obs_tool}")
        if rgx:
            policy_text.append(f"For '{rgx}': prefer {exp_tool} avoid {obs_tool}")
        text = '; '.join([p for p in policy_text if p]) or 'Prefer safe, precise tools.'
        ex = dspy.Example(
            query=q,
            state="Agent state: routing tools.",
            policy_text=text,
            adjusted_state=f"{text} | For: {q}",
            prefer_tools=exp_tool,
            blocked_tools=obs_tool,
            rationale="Bias routing toward expected tools; avoid observed mistakes."
        ).with_inputs('query', 'state', 'policy_text')
        out.append(ex)
    return out


def _metric(gold: dspy.Example, pred: dspy.Prediction, *args, **kwargs) -> float:
    exp = (getattr(gold, 'prefer_tools', '') or '').strip().lower()
    got = (getattr(pred, 'prefer_tools', '') or '').strip().lower()
    if not exp:
        return 1.0 if got else 0.5  # neutral
    return 1.0 if exp in got else 0.0


def train_preference_rewriter(workspace: Path, *, feedback_file: Optional[Path] = None, out_prompt: Optional[Path] = None, auto: str = 'light') -> Path:
    ws = workspace
    fb_path = feedback_file or (ws / '.dspy_feedback.jsonl')
    out_path = out_prompt or (ws / '.dspy_policy_prompt.txt')
    fb = _read_feedback(fb_path)
    if not fb:
        raise RuntimeError(f"No feedback found at {fb_path}")
    examples = _examples_from_feedback(fb)
    # Train
    rewriter = PreferenceRewriter()
    gepa = dspy.GEPA(metric=_metric, auto=auto)
    optimized = gepa.compile(rewriter, trainset=examples, valset=examples)
    # Synthesize a compact prompt from the optimized module outputs
    # Use the last example to generate a unified policy prompt
    sample = examples[-1]
    pred = optimized(query=sample.query, state=sample.state, policy_text=sample.policy_text)
    prompt_lines = []
    if getattr(pred, 'prefer_tools', ''):
        prompt_lines.append(f"Prefer tools: {getattr(pred, 'prefer_tools', '').strip()}")
    if getattr(pred, 'blocked_tools', ''):
        prompt_lines.append(f"Avoid tools: {getattr(pred, 'blocked_tools', '').strip()}")
    if getattr(pred, 'rationale', ''):
        prompt_lines.append(f"Why: {getattr(pred, 'rationale', '').strip()}")
    prompt_text = '\n'.join(prompt_lines) or 'Prefer safe, precise tools; avoid destructive actions.'
    out_path.write_text(prompt_text)
    return out_path

