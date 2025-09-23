from __future__ import annotations

from typing import Optional

import dspy


class TestPlannerSig(dspy.Signature):
    """Plan targeted tests to validate a change.

    Given a task/context and a repo layout (tests path, packages), select the most
    relevant tests, commands to run them, and any fast paths to save time.
    """

    task: str = dspy.InputField()
    context: str = dspy.InputField()
    repo_layout: str = dspy.InputField(desc="Overview: test dir, packages, tooling")

    tests_to_run: str = dspy.OutputField(desc="List or JSON of prioritized tests")
    commands: str = dspy.OutputField(desc="Shell commands to run tests")
    fast_paths: str = dspy.OutputField(desc="Strategies to run fewer/faster tests")
    assumptions: str = dspy.OutputField(desc="Assumptions/prereqs for commands", default="")
    rationale: str = dspy.OutputField(desc="Why these tests/commands", default="")


class TestPlanner(dspy.Module):
    def __init__(self, use_cot: Optional[bool] = None, *, beam_k: int = 1):
        super().__init__()
        self.fast = dspy.Predict(TestPlannerSig)
        self.slow = dspy.ChainOfThought(TestPlannerSig)
        self.use_cot = use_cot
        self.beam_k = max(1, int(beam_k))

    def forward(self, task: str, context: str, repo_layout: str):
        if self.use_cot is True:
            return self.slow(task=task, context=context, repo_layout=repo_layout)
        pred = self.fast(task=task, context=context, repo_layout=repo_layout)
        if self.use_cot is False:
            return pred
        tests = (getattr(pred, 'tests_to_run', '') or '').strip()
        cmds = (getattr(pred, 'commands', '') or '').strip()
        low_signal = (not tests) or (not cmds)
        if low_signal:
            pred = self.slow(task=task, context=context, repo_layout=repo_layout)
        # Beam select
        def _score(o) -> float:
            t = len(((getattr(o, 'tests_to_run', '') or '')).strip())
            c = len(((getattr(o, 'commands', '') or '')).strip())
            return 0.7 * t + 0.3 * c
        best = pred; best_score = _score(pred)
        if self.beam_k > 1:
            proposer = self.slow if (self.use_cot is True or low_signal) else self.fast
            for _ in range(self.beam_k - 1):
                cand = proposer(task=task, context=context, repo_layout=repo_layout)
                sc = _score(cand)
                if sc > best_score:
                    best, best_score = cand, sc
        return best
