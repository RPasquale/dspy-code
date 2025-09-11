from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


@dataclass
class FileFacts:
    path: str
    lines: int
    imports: List[str]
    classes: List[str]
    functions: List[str]


def _py_facts(path: Path) -> FileFacts:
    text = path.read_text(errors="ignore")
    try:
        tree = ast.parse(text)
    except Exception:
        return FileFacts(str(path), len(text.splitlines()), [], [], [])
    imports: Set[str] = set()
    classes: List[str] = []
    functions: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            # skip dunder trivial
            if not (node.name.startswith("__") and node.name.endswith("__")):
                functions.append(node.name)
    return FileFacts(str(path), len(text.splitlines()), sorted(imports), classes, functions)


def build_code_graph(root: Path) -> Dict[str, Any]:
    """Build a lightweight code graph for the repository (Python-only for now).

    Returns a dict with per-file facts and aggregated stats.
    """
    root = root.resolve()
    files: List[Path] = [p for p in root.rglob("*.py") if ".venv" not in p.parts and ".git" not in p.parts]
    facts: List[FileFacts] = []
    for p in files:
        try:
            facts.append(_py_facts(p))
        except Exception:
            continue
    total_lines = sum(f.lines for f in facts)
    total_classes = sum(len(f.classes) for f in facts)
    total_funcs = sum(len(f.functions) for f in facts)
    by_defs = sorted(facts, key=lambda f: (len(f.classes) + len(f.functions), f.lines), reverse=True)[:20]
    return {
        "root": str(root),
        "files": [asdict(f) for f in facts],
        "stats": {
            "files": len(facts),
            "lines": total_lines,
            "classes": total_classes,
            "functions": total_funcs,
        },
        "top_by_defs": [asdict(f) for f in by_defs],
    }


def summarize_code_graph(graph: Dict[str, Any], max_files: int = 10) -> str:
    st = graph.get("stats", {})
    lines = [
        f"Files: {st.get('files', 0)} | Lines: {st.get('lines', 0)} | Classes: {st.get('classes', 0)} | Functions: {st.get('functions', 0)}",
        "Top files by definitions:",
    ]
    for f in graph.get("top_by_defs", [])[:max_files]:
        lines.append(
            f"- {Path(f['path']).relative_to(Path(graph.get('root','/')))} (classes={len(f['classes'])}, funcs={len(f['functions'])}, lines={f['lines']})"
        )
    return "\n".join(lines)

