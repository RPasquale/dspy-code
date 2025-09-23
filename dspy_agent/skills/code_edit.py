from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple, Optional

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
    def __init__(self, use_cot: Optional[bool] = None, *, max_files: int = 4, max_lines: int = 200, attempts: int = 2, beam_k: int = 1):
        super().__init__()
        # Light gating: fast Predict first; escalate to CoT if malformed/low-signal
        self.propose_fast = dspy.Predict(CodeEditSig)
        self.propose_slow = dspy.ChainOfThought(CodeEditSig)
        # Backward-compat: some callers expect `self.predict` attribute
        self.predict = self.propose_slow if use_cot else self.propose_fast
        self.use_cot = use_cot
        self.critic = dspy.Predict(EditCriticSig)
        self.reviser = dspy.Predict(ReviseEditSig)
        self.max_files = int(max_files)
        self.max_lines = int(max_lines)
        self.attempts = max(1, int(attempts))
        self.beam_k = max(1, int(beam_k))

    def forward(self, task: str, context: str, code_graph: str = "", file_hints: str = ""):
        constraints = (
            f"Caps: files <= {self.max_files}, total changed lines (add+remove) <= {self.max_lines}. "
            "Output pure unified diff (no prose), with correct headers and minimal scope."
        )

        # Propose patch (fast → slow gating unless explicitly forced)
        if self.use_cot is True:
            pred = self.propose_slow(task=task, context=context, code_graph=code_graph, file_hints=file_hints)
        else:
            pred = self.propose_fast(task=task, context=context, code_graph=code_graph, file_hints=file_hints)
        patch = getattr(pred, 'patch', '') or ''
        rationale = getattr(pred, 'rationale', '') or ''

        files, add, rem = _summarize_patch(patch)
        total = add + rem
        ok = bool(patch.strip()) and _has_unified_headers(patch) and (
            (self.max_files <= 0 or files <= self.max_files) and
            (self.max_lines <= 0 or total <= self.max_lines)
        )

        # Escalate to CoT if fast path was low-signal
        if not ok and self.use_cot is None:
            pred = self.propose_slow(task=task, context=context, code_graph=code_graph, file_hints=file_hints)
            patch = getattr(pred, 'patch', '') or ''
            rationale = getattr(pred, 'rationale', '') or rationale
            files, add, rem = _summarize_patch(patch)
            total = add + rem
            ok = bool(patch.strip()) and _has_unified_headers(patch) and (
                (self.max_files <= 0 or files <= self.max_files) and
                (self.max_lines <= 0 or total <= self.max_lines)
            )

        # Beam search over proposals (choose best valid candidate by minimal scope)
        def _score_candidate(p: str) -> float:
            if not (p or '').strip() or not _has_unified_headers(p):
                return float('-inf')
            f, a, r = _summarize_patch(p)
            t = a + r
            if (self.max_files > 0 and f > self.max_files) or (self.max_lines > 0 and t > self.max_lines):
                return float('-inf')
            return 10.0 - (f * 1.5 + t * 0.02)

        best_patch = patch
        best_rationale = rationale
        best_score = _score_candidate(best_patch)
        # Decide which proposer to use for beam
        proposer = self.propose_slow if (self.use_cot is True or (self.use_cot is None and not ok)) else self.propose_fast
        if self.beam_k > 1:
            for _ in range(self.beam_k - 1):
                cand = proposer(task=task, context=context, code_graph=code_graph, file_hints=file_hints)
                cpatch = getattr(cand, 'patch', '') or ''
                cscore = _score_candidate(cpatch)
                if cscore > best_score:
                    best_patch = cpatch
                    best_rationale = getattr(cand, 'rationale', '') or best_rationale
                    best_score = cscore
        patch = best_patch
        rationale = best_rationale
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
