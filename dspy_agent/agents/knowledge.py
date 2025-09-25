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
    skip_dirs = {".venv", "venv", ".git", "node_modules", "dist", "build", ".mypy_cache", ".pytest_cache", "__pycache__"}
    files: List[Path] = [
        p for p in root.rglob("*.py")
        if not any(sd in p.parts for sd in skip_dirs)
    ]
    facts: List[FileFacts] = []
    for p in files:
        try:
            facts.append(_py_facts(p))
        except Exception:
            continue
    # Build import edges to packages/modules within the repo
    # Map package/module base names to canonical file paths
    pkg_init: Dict[str, Path] = {}
    mod_file: Dict[str, Path] = {}
    for p in files:
        rel = p.relative_to(root)
        if p.name == "__init__.py":
            pkg_init[rel.parent.name] = p
        else:
            mod_file[p.stem] = p

    import_edges: List[Tuple[str, str]] = []  # (src, dst)
    for f in facts:
        src = f.path
        for m in f.imports:
            tgt_path: Path | None = None
            if m in mod_file:
                tgt_path = mod_file[m]
            elif m in pkg_init:
                tgt_path = pkg_init[m]
            if tgt_path is not None:
                import_edges.append((src, str(tgt_path)))

    # Build simple call edges for uniquely-named functions
    # Map function name -> defining file (only if unique across repo)
    def_map: Dict[str, str] = {}
    seen_multi: set[str] = set()
    for f in facts:
        for fn in f.functions:
            if fn in seen_multi:
                continue
            if fn in def_map:
                # now ambiguous
                seen_multi.add(fn); def_map.pop(fn, None)
            else:
                def_map[fn] = f.path

    call_edges: List[Tuple[str, str]] = []
    for p in files:
        try:
            src_text = p.read_text(errors="ignore")
            tree = ast.parse(src_text)
        except Exception:
            continue
        calls: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = None
                if isinstance(f, ast.Name):
                    name = f.id
                elif isinstance(f, ast.Attribute):
                    name = f.attr
                if name:
                    calls.append(name)
        uniques = {c for c in calls if c in def_map}
        for c in uniques:
            dst = def_map.get(c)
            if dst and str(p) != dst:
                call_edges.append((str(p), dst))
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
        "edges": [
            {"source": s, "target": t, "kind": "import"} for (s, t) in import_edges
        ] + [
            {"source": s, "target": t, "kind": "call"} for (s, t) in call_edges
        ],
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


def neighbors(graph: Dict[str, Any], file_path: str) -> Dict[str, list[str]]:
    """Return inbound/outbound neighbors for the given file path.

    Keys: imports_out, imports_in, calls_out, calls_in
    """
    src = str(Path(file_path).resolve())
    # Graph stores absolute paths from build_code_graph
    edges = graph.get("edges", []) or []
    imp_out = sorted({e["target"] for e in edges if e.get("kind") == "import" and e.get("source") == src})
    imp_in = sorted({e["source"] for e in edges if e.get("kind") == "import" and e.get("target") == src})
    call_out = sorted({e["target"] for e in edges if e.get("kind") == "call" and e.get("source") == src})
    call_in = sorted({e["source"] for e in edges if e.get("kind") == "call" and e.get("target") == src})
    return {
        "imports_out": imp_out,
        "imports_in": imp_in,
        "calls_out": call_out,
        "calls_in": call_in,
    }


class KnowledgeAgent:
    """Minimal knowledge agent for tests.

    Stores simple documents in-memory and supports basic search & summarize.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()
        self._docs: List[Dict[str, Any]] = []

    def ingest(self, doc: Dict[str, Any]) -> None:
        try:
            self._docs.append(dict(doc))
        except Exception:
            pass

    def search(self, query: str) -> List[Dict[str, Any]]:
        q = (query or "").lower()
        out: List[Dict[str, Any]] = []
        for d in self._docs:
            content = str(d.get("content", "")).lower()
            title = str(d.get("title", "")).lower()
            if q and (q in content or q in title):
                out.append(d)
        # Fallback to code graph facts if no docs match
        if not out:
            try:
                graph = build_code_graph(self.workspace)
                for f in graph.get("files", [])[:3]:
                    out.append({"title": f.get("path"), "content": " ".join((f.get("classes") or []) + (f.get("functions") or []))})
            except Exception:
                pass
        return out

    def summarize(self, query: str) -> str:
        hits = self.search(query)
        if not hits:
            return ""
        titles = [str(h.get("title", "doc")) for h in hits[:5]]
        return f"Found {len(hits)} document(s): " + ", ".join(titles)
