import json
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
        language="python",
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

    assert facts.language == "python"
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


@pytest.mark.unit
def test_build_code_graph_handles_typescript_imports(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    entry = src_dir / "App.tsx"
    util = src_dir / "lib" / "util.ts"
    util.parent.mkdir()
    util.write_text("export default function helper() { return 42; }\n")
    entry.write_text("import helper from '@/lib/util';\nexport const App = () => helper();\n")
    (tmp_path / "tsconfig.json").write_text(json.dumps({
        "compilerOptions": {
            "baseUrl": ".",
            "paths": {
                "@/*": ["src/*"],
            },
        }
    }))

    graph = build_code_graph(tmp_path)

    edge_matches = [
        e
        for e in graph["edges"]
        if e["source"].endswith("App.tsx")
        and e["target"].endswith("lib/util.ts")
        and e["kind"].endswith("import")
    ]
    assert edge_matches, f"expected import edge from App.tsx to util.ts, got {graph['edges']}"



@pytest.mark.unit
def test_build_code_graph_tracks_multiple_languages(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()

    (tmp_path / "src" / "lib.rs").write_text('mod helpers;\nuse crate::helpers::value;\n')
    (tmp_path / "src" / "helpers.rs").write_text('pub fn value() -> i32 { 1 }\n')
    (tmp_path / "main.go").write_text('package main\nimport "./util"\nfunc main() { util.Run() }\n')
    (tmp_path / "util.go").write_text('package util\nfunc Run() {}\n')
    (tmp_path / "scripts" / "build.sh").write_text('#!/usr/bin/env bash\nsource ./env.sh\n')
    (tmp_path / "scripts" / "env.sh").write_text('export VAR=1\n')
    (tmp_path / "index.html").write_text('<link rel="stylesheet" href="styles.css" />')
    (tmp_path / "styles.css").write_text('@import "theme.css";')
    (tmp_path / "theme.css").write_text('body { color: #000; }')
    (tmp_path / "App.java").write_text('import java.util.List; class App {}')

    graph = build_code_graph(tmp_path)

    languages = {entry['language'] for entry in graph['files']}
    expected = {"rust", "go", "bash", "html", "css", "java"}
    assert expected.issubset(languages)

    rust_edges = [e for e in graph['edges'] if e['kind'].endswith('import') and e['source'].endswith('lib.rs')]
    assert any('helpers.rs' in e['target'] for e in rust_edges)

    bash_edges = [e for e in graph['edges'] if e['kind'].startswith('bash_')]
    assert bash_edges, "expected bash reference edges"

    html_edges = [e for e in graph['edges'] if e['kind'].startswith('html_')]
    assert any(e['target'].endswith('styles.css') for e in html_edges)

    css_edges = [e for e in graph['edges'] if e['kind'].startswith('css_')]
    assert any(e['target'].endswith('theme.css') for e in css_edges)
