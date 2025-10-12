from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from ..skills.task_agent import TaskAgent
from ..skills.file_locator import FileLocator
from ..skills.test_planner import TestPlanner


@dataclass
class CodingPlan:
    """Structured reasoning payload for code-generation signatures."""

    plan: str
    target_files: str
    test_plan: str
    notes: str
    raw_file_candidates: str

    def as_context_block(self) -> str:
        sections: list[str] = []
        if self.plan.strip():
            sections.append(f"Implementation plan:\n{self.plan.strip()}")
        if self.target_files.strip():
            sections.append(f"Target files:\n{self.target_files.strip()}")
        if self.test_plan.strip():
            sections.append(f"Tests/commands:\n{self.test_plan.strip()}")
        if self.notes.strip():
            sections.append(f"Additional notes:\n{self.notes.strip()}")
        return "\n\n".join(sections).strip()

    def merged_file_hints(self, existing: str = "") -> str:
        hints: list[str] = []
        for chunk in (existing or "").replace("\n", ",").split(","):
            item = chunk.strip()
            if item:
                hints.append(item)
        for line in self.target_files.splitlines():
            frag = line.strip()
            if not frag:
                continue
            # handle "path (score=..)" formatting
            candidate = frag.split(" ", 1)[0]
            if candidate and candidate not in hints:
                hints.append(candidate)
        return ",".join(dict.fromkeys(hints))


class CodingReasoner:
    """Aggregates planning, file location, and test strategy for code edits."""

    def __init__(
        self,
        workspace: Optional[Path],
        *,
        task_agent: Optional[TaskAgent] = None,
        file_locator: Optional[FileLocator] = None,
        test_planner: Optional[TestPlanner] = None,
    ) -> None:
        self.workspace = workspace.resolve() if isinstance(workspace, Path) else None
        self._task_agent = task_agent or TaskAgent(use_cot=None)
        self._file_locator = file_locator or FileLocator()
        self._test_planner = test_planner or TestPlanner()

    # ------------------------------------------------------------------
    def _repo_layout_summary(self) -> str:
        if not self.workspace:
            return ""
        tests_dir = None
        src_dirs: list[str] = []
        try:
            for entry in self.workspace.iterdir():
                if entry.name.startswith(".") or not entry.is_dir():
                    continue
                if entry.name in {"tests", "test", "pytest"}:
                    tests_dir = entry.name
                if any(entry.joinpath(fname).is_file() for fname in ("__init__.py", "pyproject.toml", "setup.py")):
                    src_dirs.append(entry.name)
        except Exception:
            pass
        pieces = []
        if src_dirs:
            pieces.append(f"src={','.join(src_dirs[:5])}")
        if tests_dir:
            pieces.append(f"tests={tests_dir}")
        if not pieces:
            return ""
        return "; ".join(pieces)

    def _format_candidates(self, candidates_json: str) -> Tuple[str, str]:
        if not candidates_json:
            return "", ""
        try:
            data = json.loads(candidates_json)
        except Exception:
            return "", ""
        if not isinstance(data, Sequence):
            return "", ""
        lines: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            if not path:
                continue
            score = item.get("score")
            reason = str(item.get("reason") or "").strip()
            score_txt = f" score={float(score):.2f}" if isinstance(score, (int, float)) else ""
            if reason:
                lines.append(f"{path}{score_txt} – {reason}")
            else:
                lines.append(f"{path}{score_txt}")
        pretty = "\n".join(lines)
        top_paths = "\n".join(item.split(" ", 1)[0] for item in lines)
        return pretty, top_paths

    # ------------------------------------------------------------------
    def build(
        self,
        task: str,
        context: str,
        *,
        code_graph: str = "",
        existing_file_hints: str = "",
    ) -> CodingPlan:
        plan_text = ""
        notes: list[str] = []
        try:
            plan = self._task_agent(task=task, context=context)
            plan_text = (getattr(plan, "plan", "") or "").strip()
            commands = (getattr(plan, "commands", "") or "").strip()
            assumptions = (getattr(plan, "assumptions", "") or "").strip()
            risks = (getattr(plan, "risks", "") or "").strip()
            rationale = (getattr(plan, "rationale", "") or "").strip()
            for label, value in (
                ("commands", commands),
                ("assumptions", assumptions),
                ("risks", risks),
                ("rationale", rationale),
            ):
                if value:
                    notes.append(f"{label}: {value}")
        except Exception:
            pass

        file_candidates_raw = ""
        target_files_formatted = ""
        try:
            locator = self._file_locator(task=task, context=context, code_graph=code_graph)
            file_candidates_raw = (getattr(locator, "file_candidates", "") or "").strip()
            locator_notes = (getattr(locator, "notes", "") or "").strip()
            if locator_notes:
                notes.append(f"file_locator: {locator_notes}")
            formatted, top_paths = self._format_candidates(file_candidates_raw)
            target_files_formatted = formatted or top_paths
        except Exception:
            pass

        test_plan = ""
        try:
            repo_layout = self._repo_layout_summary()
            planner = self._test_planner(task=task, context=context, repo_layout=repo_layout)
            tests_to_run = (getattr(planner, "tests_to_run", "") or "").strip()
            commands = (getattr(planner, "commands", "") or "").strip()
            rationale = (getattr(planner, "rationale", "") or "").strip()
            fast_paths = (getattr(planner, "fast_paths", "") or "").strip()
            fragments = []
            if tests_to_run:
                fragments.append(f"tests: {tests_to_run}")
            if commands:
                fragments.append(f"commands: {commands}")
            if fast_paths:
                fragments.append(f"fast_paths: {fast_paths}")
            if rationale:
                fragments.append(f"rationale: {rationale}")
            test_plan = "\n".join(fragments)
        except Exception:
            pass

        plan_summary = plan_text or "Clarify requirements, inspect relevant modules, apply minimal diff, validate with tests."
        notes_text = "\n".join(notes)

        return CodingPlan(
            plan=plan_summary,
            target_files=target_files_formatted,
            test_plan=test_plan,
            notes=notes_text,
            raw_file_candidates=file_candidates_raw or existing_file_hints or "",
        )
