import json
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

import pytest

from dspy_agent.agents import orchestrator_runtime as runtime


@pytest.mark.unit
def test_json_arg_handles_dicts_and_strings() -> None:
    assert runtime._json_arg({"a": 1}) == {"a": 1}
    assert runtime._json_arg(json.dumps({"b": 2})) == {"b": 2}
    assert runtime._json_arg("not-json") == {}


@pytest.mark.unit
def test_evaluate_tool_choice_unknown_tool(workspace: Path) -> None:
    outcome = runtime.evaluate_tool_choice(
        tool="danger",
        args={"foo": "bar"},
        workspace=workspace,
    )

    assert outcome.score == pytest.approx(0.0)
    assert "not evaluated" in outcome.feedback


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.mark.unit
def test_evaluate_tool_choice_grep_hits(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    hit = SimpleNamespace(line="goal line", path=workspace / "file.py", line_no=1)

    def fake_search_text(*args, **kwargs):
        return [hit]

    monkeypatch.setattr(runtime, "search_text", fake_search_text)

    outcome = runtime.evaluate_tool_choice(
        tool="grep",
        args={"pattern": "goal", "globs": ["**/*.py"]},
        workspace=workspace,
        targets=["goal"],
    )

    assert outcome.score >= 1.5
    assert "grep hits=1" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_extract_symbol(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    sample_file = workspace / "mod.py"
    sample_file.write_text("def target():\n    return 1\n")

    def fake_python_extract_symbol(path: Path, symbol: str):
        assert path == sample_file
        assert symbol == "target"
        return 1, 2, "needle contents"

    monkeypatch.setattr(runtime, "python_extract_symbol", fake_python_extract_symbol)

    outcome = runtime.evaluate_tool_choice(
        tool="extract",
        args={"file": "mod.py", "symbol": "target"},
        workspace=workspace,
        targets=["needle"],
    )

    assert outcome.score >= 1.5
    assert "extract segment_len=" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_context(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    monkeypatch.setattr(runtime, "load_logs", lambda paths: ("error needle", 1))
    monkeypatch.setattr(runtime, "extract_key_events", lambda bundle: bundle)

    outcome = runtime.evaluate_tool_choice(
        tool="context",
        args={},
        workspace=workspace,
        targets=["needle"],
    )

    assert outcome.score >= 1.5
    assert "context logs_found=True" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_codectx(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    monkeypatch.setattr(runtime, "build_code_snapshot", lambda root: "needle summary")

    outcome = runtime.evaluate_tool_choice(
        tool="codectx",
        args={},
        workspace=workspace,
        targets=["needle"],
    )

    assert outcome.score >= 1.5
    assert "codectx snapshot_len=" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_index(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    captured = {}

    def fake_build_index(root: Path, smart: bool = False):
        assert root == workspace
        captured["built"] = True
        return {"count": 1}, ["chunk"]

    monkeypatch.setattr(runtime, "build_index", fake_build_index)
    monkeypatch.setattr(runtime, "save_index", lambda *args, **kwargs: captured.setdefault("saved", True))

    outcome = runtime.evaluate_tool_choice(
        tool="index",
        args={},
        workspace=workspace,
    )

    assert captured == {"built": True, "saved": True}
    assert outcome.score == pytest.approx(0.5)
    assert "index chunks=1" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_esearch(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    snippet_path = workspace / "snippet.py"
    snippet_path.write_text("needle present\n")
    meta = {"version": 1}
    items = [SimpleNamespace(path=str(snippet_path), start_line=1, end_line=1)]

    monkeypatch.setattr(runtime, "load_index", lambda root: (meta, items))
    monkeypatch.setattr(runtime, "semantic_search", lambda q, m, it, top_k=5: [(0.9, items[0])])

    outcome = runtime.evaluate_tool_choice(
        tool="esearch",
        args={"query": "topic"},
        workspace=workspace,
        targets=["needle"],
    )

    assert outcome.score >= 1.5
    assert "esearch hits=1" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_plan(workspace: Path) -> None:
    outcome = runtime.evaluate_tool_choice(
        tool="plan",
        args={"plan_text": "- step one\n- step two"},
        workspace=workspace,
    )

    assert outcome.score >= 1.0
    assert "plan steps_detected=2" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_edit(workspace: Path) -> None:
    outcome = runtime.evaluate_tool_choice(
        tool="edit",
        args={"patch": "diff\n+ needle"},
        workspace=workspace,
    )

    assert outcome.score >= 1.0
    assert "edit patch_lines=2" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_patch_metrics(workspace: Path) -> None:
    outcome = runtime.evaluate_tool_choice(
        tool="patch",
        args={},
        workspace=workspace,
        result_metrics={"applied": True, "pass_rate": 1.0, "blast_radius": 10},
    )

    assert outcome.score >= 1.8
    assert "patch applied=True" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_run_tests(workspace: Path) -> None:
    outcome = runtime.evaluate_tool_choice(
        tool="run_tests",
        args={},
        workspace=workspace,
        result_metrics={"tests_total": 4, "tests_passed": 4, "tests_failed": 0, "pass_rate": 1.0},
    )

    assert outcome.score >= 1.2
    assert "run_tests total=4" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_lint(workspace: Path) -> None:
    outcome = runtime.evaluate_tool_choice(
        tool="lint",
        args={},
        workspace=workspace,
        result_metrics={"lint_ok": True, "lint_issues": 0},
    )

    assert outcome.score >= 1.0
    assert "lint ok=True" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_build(workspace: Path) -> None:
    outcome = runtime.evaluate_tool_choice(
        tool="build",
        args={},
        workspace=workspace,
        result_metrics={"build_ok": True},
    )

    assert outcome.score >= 1.0
    assert "build ok=True" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_knowledge(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    class DummyStorage:
        def get(self, key: str):
            if key == "code:graph":
                return {
                    "files": [
                        {"path": str(workspace / "pkg" / "mod.py"), "classes": ["Foo"], "functions": ["bar"], "imports": []}
                    ]
                }
            return None

    monkeypatch.setattr("dspy_agent.db.factory.get_storage", lambda: DummyStorage())

    outcome = runtime.evaluate_tool_choice(
        tool="knowledge",
        args={"class": "Foo"},
        workspace=workspace,
        targets=["Foo"],
    )

    assert outcome.score >= 1.5
    assert "knowledge matches=1" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_vretr(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    snippet_path = workspace / "vector.py"
    snippet_path.write_text("needle content\n")

    emb_module = ModuleType("dspy_agent.agents.embeddings_index")

    def load_emb_index(root: Path):
        return [SimpleNamespace(path=str(snippet_path), start_line=1, end_line=1)]

    def embed_query(embedder, query: str):
        return [0.1]

    def emb_search(qvec, items, top_k: int = 5):
        return [(0.8, items[0])]

    emb_module.load_emb_index = load_emb_index
    emb_module.embed_query = embed_query
    emb_module.emb_search = emb_search
    monkeypatch.setitem(sys.modules, "dspy_agent.agents.embeddings_index", emb_module)

    dspy_mod = ModuleType("dspy")

    class DummyEmbeddings:
        def __init__(self, model: str) -> None:
            self.model = model

    dspy_mod.Embeddings = DummyEmbeddings
    monkeypatch.setitem(sys.modules, "dspy", dspy_mod)

    outcome = runtime.evaluate_tool_choice(
        tool="vretr",
        args={"query": "needle"},
        workspace=workspace,
        targets=["needle"],
    )

    assert outcome.score >= 1.5
    assert "vretr hits=1" in outcome.evidence


@pytest.mark.unit
def test_evaluate_tool_choice_intel(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    vector_file = workspace / "intel.py"
    vector_file.write_text("needle intel details\n")

    class DummyStorage:
        def get(self, key: str):
            if key == "code:graph":
                return {
                    "files": [
                        {
                            "path": str(vector_file),
                            "classes": ["IntelClass"],
                            "functions": ["intel_fn"],
                            "imports": ["json"],
                        }
                    ]
                }
            return None

    emb_module = ModuleType("dspy_agent.agents.embeddings_index")

    def load_emb_index(root: Path):
        return [SimpleNamespace(path=str(vector_file), start_line=1, end_line=1)]

    def embed_query(embedder, query: str):
        return [0.5]

    def emb_search(qvec, items, top_k: int = 5):
        return [(0.9, items[0])]

    emb_module.load_emb_index = load_emb_index
    emb_module.embed_query = embed_query
    emb_module.emb_search = emb_search
    monkeypatch.setitem(sys.modules, "dspy_agent.agents.embeddings_index", emb_module)

    dspy_mod = ModuleType("dspy")

    class DummyEmbeddings:
        def __init__(self, model: str) -> None:
            self.model = model

    dspy_mod.Embeddings = DummyEmbeddings
    monkeypatch.setitem(sys.modules, "dspy", dspy_mod)
    monkeypatch.setattr("dspy_agent.db.factory.get_storage", lambda: DummyStorage())

    outcome = runtime.evaluate_tool_choice(
        tool="intel",
        args={"query": "IntelClass intel_fn"},
        workspace=workspace,
        targets=["needle"],
    )

    assert outcome.score >= 2.0
    assert "intel kn=1" in outcome.evidence
    assert "vretr=1" in outcome.evidence
