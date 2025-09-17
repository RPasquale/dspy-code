from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import dspy


def _summarize_patch(patch_text: str) -> Tuple[int, int, int]:
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
    return max(0, len(files) // 2) if files else 0, added, removed


def _has_unified_headers(patch_text: str) -> bool:
    pt = patch_text or ""
    return ('--- ' in pt) and ('+++ ' in pt) and ('@@' in pt)


class PatchVerifierSig(dspy.Signature):
    """Assess whether a proposed patch is safe and appropriate.

    Provide a pass/fail verdict, a coarse risk level (low/medium/high), brief reasons,
    and specific suggestions for fixes if failing or risky.
    """

    task: str = dspy.InputField()
    context: str = dspy.InputField()
    patch: str = dspy.InputField()

    verdict: str = dspy.OutputField(desc="pass or fail")
    risk_level: str = dspy.OutputField(desc="low, medium, or high")
    reasons: str = dspy.OutputField(desc="Brief reasons and evidence")
    fix_suggestions: str = dspy.OutputField(desc="Concrete next steps to improve")


class PatchVerifier(dspy.Module):
    def __init__(self, max_files: int = 4, max_lines: int = 200):
        super().__init__()
        self.predict = dspy.Predict(PatchVerifierSig)
        self.max_files = int(max_files)
        self.max_lines = int(max_lines)

    def forward(self, task: str, context: str, patch: str):
        # Quick local checks to catch malformed or obviously risky patches
        if not (patch or "").strip():
            return SimpleNamespace(
                verdict="fail",
                risk_level="high",
                reasons="Empty patch",
                fix_suggestions="Return a valid unified diff with the minimal change to address the task.",
            )
        if not _has_unified_headers(patch):
            return SimpleNamespace(
                verdict="fail",
                risk_level="high",
                reasons="Patch missing unified diff headers (---, +++, @@)",
                fix_suggestions="Emit a proper git-style unified diff and keep it minimal.",
            )

        files, add, rem = _summarize_patch(patch)
        total = add + rem
        if (self.max_files > 0 and files > self.max_files) or (self.max_lines > 0 and total > self.max_lines):
            return SimpleNamespace(
                verdict="fail",
                risk_level="high",
                reasons=f"Patch too large: files={files}, lines={total} (caps: files<={self.max_files}, lines<={self.max_lines})",
                fix_suggestions="Split into smaller, task-focused patches and limit scope to the smallest viable change.",
            )

        # Delegate nuanced assessment to the LM
        return self.predict(task=task, context=context, patch=patch)

