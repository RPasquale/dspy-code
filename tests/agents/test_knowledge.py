import textwrap
from pathlib import Path

import pytest

from dspy_agent.agents.knowledge import FileFacts, _py_facts, build_code_graph, summarize_code_graph


@pytest.mark.unit
def test_py_facts_extracts_imports_classes_and_functions(tmp_path: Path) -> None:
    src = textwrap.dedent(
        """
        import os
        from math import sqrt

        class Foo:
            pass

        def bar():
            return sqrt(4)

        def __private__():
            return os.name
        """
    )
    path = tmp_path / "sample.py"
    path.write_text(src)

    facts = _py_facts(path)

    assert facts == FileFacts(
        path=str(path),
        lines=len(src.splitlines()),
        imports=["math", "os"],
        classes=["Foo"],
        functions=["bar"],
    )


@pytest.mark.unit
def test_py_facts_handles_syntax_errors(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    path.write_text("def broken(:\n    pass\n")

    facts = _py_facts(path)

    assert facts.imports == []
    assert facts.classes == []
    assert facts.functions == []
    assert facts.lines == 2


@pytest.mark.unit
def test_build_code_graph_aggregates_repository_stats(tmp_path: Path) -> None:
    good_dir = tmp_path / "pkg"
    good_dir.mkdir()
    (good_dir / "alpha.py").write_text("class Alpha:\n    pass\n")
    (good_dir / "beta.py").write_text("def beta():\n    return 'ok'\n")
    ignored_dir = tmp_path / ".venv"
    ignored_dir.mkdir()
    (ignored_dir / "ghost.py").write_text("def ghost():\n    return 0\n")

    graph = build_code_graph(tmp_path)

    assert graph["stats"] == {
        "files": 2,
        "lines": 4,
        "classes": 1,
        "functions": 1,
    }
    recorded_paths = {Path(entry["path"]) for entry in graph["files"]}
    assert (good_dir / "alpha.py").resolve() in recorded_paths
    assert (good_dir / "beta.py").resolve() in recorded_paths
    assert all(".venv" not in str(path) for path in recorded_paths)
    # Ensure the ranking prefers files with more definitions
    top_paths = [Path(entry["path"]) for entry in graph["top_by_defs"]]
    assert (good_dir / "alpha.py").resolve() in top_paths


@pytest.mark.unit
def test_summarize_code_graph_formats_human_readable_output() -> None:
    graph = {
        "root": "/repo",
        "stats": {"files": 3, "lines": 120, "classes": 2, "functions": 5},
        "top_by_defs": [
            {"path": "/repo/mod/a.py", "classes": ["Foo"], "functions": ["bar"], "lines": 10},
            {"path": "/repo/mod/b.py", "classes": [], "functions": ["baz", "qux"], "lines": 15},
        ],
    }

    output = summarize_code_graph(graph, max_files=2)

    assert "Files: 3 | Lines: 120 | Classes: 2 | Functions: 5" in output
    assert "Top files by definitions:" in output
    assert "mod/a.py" in output
    assert "classes=1" in output
    assert "funcs=2" in output
