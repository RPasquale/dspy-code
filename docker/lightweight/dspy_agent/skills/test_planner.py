from __future__ import annotations

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


class TestPlanner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(TestPlannerSig)

    def forward(self, task: str, context: str, repo_layout: str):
        return self.predict(task=task, context=context, repo_layout=repo_layout)

