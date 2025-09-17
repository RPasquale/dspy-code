from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import dspy


class CodeEditSig(dspy.Signature):
    """Propose a minimal, safe code change as a unified diff.

    Constraints:
    - Output a standard unified diff with correct headers (---, +++), no extra prose.
    - Keep the change minimal and directly tied to the task/context.
    - Prefer focused fixes over refactors; avoid unrelated formatting/noise.
    - If unsure, narrow scope and add a short rationale.
    """

    task: str = dspy.InputField()
    context: str = dspy.InputField()
    code_graph: str = dspy.InputField(desc="Optional code graph summary", default="")
    file_hints: str = dspy.InputField(desc="Optional hints: files/modules to touch", default="")

    patch: str = dspy.OutputField(desc="Unified diff patch (git-style)")
    rationale: str = dspy.OutputField(desc="Brief reasoning and safety checks")


class EditCriticSig(dspy.Signature):
    """Critique a proposed patch for safety, minimality, and correctness.

    Identify problems (oversized changes, unrelated edits, missing headers), and propose
    concrete revision instructions to satisfy constraints and size caps.
    """

    task: str = dspy.InputField()
    context: str = dspy.InputField()
    patch: str = dspy.InputField()
    constraints: str = dspy.InputField(desc="Size caps and format constraints")

    problems: str = dspy.OutputField(desc="Bulleted list of issues/risks")
    revise_instructions: str = dspy.OutputField(desc="Concrete guidance to fix the patch")


class ReviseEditSig(dspy.Signature):
    """Revise a patch according to critique and constraints, keeping changes minimal."""

    task: str = dspy.InputField()
    context: str = dspy.InputField()
    patch: str = dspy.InputField()
    critique: str = dspy.InputField()
    constraints: str = dspy.InputField()

    patch: str = dspy.OutputField(desc="Unified diff patch (git-style)")
    rationale: str = dspy.OutputField(desc="Brief reasoning focusing on applied critique")


def _summarize_patch(patch_text: str) -> Tuple[int, int, int]:
    """Return (files, added, removed) for a unified diff string.

    Lightweight summarizer mirroring code_tools.patcher.summarize_patch behavior.
    """
    files = set()
    added = 0
    removed = 0
    for line in (patch_text or "").splitlines():
        if line.startswith('+++ ') or line.startswith('--- '):
            parts = line.split()
            if len(parts) >= 2 and parts[1] not in {'/dev/null'}:
                files.add(parts[1])
        elif line.startswith('+') and not line.startswith('+++'):
            added += 1
        elif line.startswith('-') and not line.startswith('---'):
            removed += 1
    # Each file contributes a pair of +++/--- lines
    file_count = max(0, len(files) // 2) if files else 0
    return file_count, added, removed


def _has_unified_headers(patch_text: str) -> bool:
    pt = patch_text or ""
    return ('--- ' in pt) and ('+++ ' in pt) and ('@@' in pt)


class CodeEdit(dspy.Module):
    def __init__(self, use_cot: bool = True, *, max_files: int = 4, max_lines: int = 200, attempts: int = 2):
        super().__init__()
        # Prefer Chain-of-Thought for code edits; fallback to Predict when unavailable
        self.propose = dspy.ChainOfThought(CodeEditSig) if use_cot else dspy.Predict(CodeEditSig)
        # Backward-compat: some callers expect `self.predict`
        self.predict = self.propose
        self.critic = dspy.Predict(EditCriticSig)
        self.reviser = dspy.Predict(ReviseEditSig)
        self.max_files = int(max_files)
        self.max_lines = int(max_lines)
        self.attempts = max(1, int(attempts))

    def forward(self, task: str, context: str, code_graph: str = "", file_hints: str = ""):
        constraints = (
            f"Caps: files <= {self.max_files}, total changed lines (add+remove) <= {self.max_lines}. "
            "Output pure unified diff (no prose), with correct headers and minimal scope."
        )

        pred = self.propose(task=task, context=context, code_graph=code_graph, file_hints=file_hints)
        patch = getattr(pred, 'patch', '') or ''
        rationale = getattr(pred, 'rationale', '') or ''

        files, add, rem = _summarize_patch(patch)
        total = add + rem
        ok = bool(patch.strip()) and _has_unified_headers(patch) and (
            (self.max_files <= 0 or files <= self.max_files) and
            (self.max_lines <= 0 or total <= self.max_lines)
        )

        attempts = 0
        last_critique = ""
        while not ok and attempts < self.attempts:
            attempts += 1
            critique = self.critic(task=task, context=context, patch=patch, constraints=constraints)
            last_critique = f"{getattr(critique, 'problems', '')}\n\n{getattr(critique, 'revise_instructions', '')}"
            revised = self.reviser(task=task, context=context, patch=patch, critique=last_critique, constraints=constraints)
            patch = getattr(revised, 'patch', '') or ''
            rationale = getattr(revised, 'rationale', '') or rationale
            files, add, rem = _summarize_patch(patch)
            total = add + rem
            ok = bool(patch.strip()) and _has_unified_headers(patch) and (
                (self.max_files <= 0 or files <= self.max_files) and
                (self.max_lines <= 0 or total <= self.max_lines)
            )

        # Return a simple object with expected fields
        return SimpleNamespace(patch=patch, rationale=rationale, attempts=attempts, ok=ok, files=files, added=add, removed=rem, critique=last_critique)
