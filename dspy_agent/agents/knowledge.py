from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional


@dataclass
class FileFacts:
    path: str
    language: str
    lines: int
    imports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


def _py_facts(path: Path) -> FileFacts:
    text = path.read_text(errors="ignore")
    try:
        tree = ast.parse(text)
    except Exception:
        return FileFacts(
            path=str(path),
            language="python",
            lines=len(text.splitlines()),
        )
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
    return FileFacts(
        path=str(path),
        language="python",
        lines=len(text.splitlines()),
        imports=sorted(imports),
        classes=classes,
        functions=functions,
    )


def _basic_facts(path: Path, language: str) -> FileFacts:
    text = path.read_text(errors="ignore")
    return FileFacts(path=str(path), language=language, lines=len(text.splitlines()))


def _rust_facts(path: Path) -> FileFacts:
    facts = _basic_facts(path, "rust")
    text = path.read_text(errors="ignore")
    uses = re.findall(r"^\s*use\s+([A-Za-z0-9_:]+)", text, re.MULTILINE)
    mods = re.findall(r"^\s*(?:pub\s+)?mod\s+([A-Za-z0-9_]+)\s*;", text, re.MULTILINE)
    refs: Set[str] = set()
    for item in uses:
        target = item
        if target.startswith("crate::"):
            target = target[len("crate::"):]
            refs.add(f"src/{target.replace('::', '/')}")
        elif target.startswith("super::"):
            # Approximate by keeping relative expression; resolver handles it
            refs.add(target.replace("::", "/"))
        elif target.startswith("self::"):
            refs.add(target.replace("::", "/"))
    for module in mods:
        refs.add(f"{module}")
        refs.add(f"src/{module}")
    facts.imports = sorted({item.split("::")[0] for item in uses})
    facts.references = sorted(refs)
    return facts


def _go_facts(path: Path) -> FileFacts:
    facts = _basic_facts(path, "go")
    text = path.read_text(errors="ignore")
    block_imports = re.findall(r"import\s*\((.*?)\)", text, re.DOTALL)
    single_imports = re.findall(r"^\s*import\s+\"([^\"]+)\"", text, re.MULTILINE)
    refs: Set[str] = set(single_imports)
    for block in block_imports:
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            match = re.search(r"\"([^\"]+)\"", line)
            if match:
                refs.add(match.group(1))
    facts.imports = sorted({r.split("/")[-1] for r in refs})
    facts.references = sorted(refs)
    return facts


def _ts_js_facts(path: Path, language: str) -> FileFacts:
    facts = _basic_facts(path, language)
    text = path.read_text(errors="ignore")
    imports = re.findall(r"import\s+(?:[^;]*?from\s+)?['\"]([^'\"]+)['\"]", text)
    requires = re.findall(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", text)
    dynamic = re.findall(r"import\(\s*['\"]([^'\"]+)['\"]\s*\)", text)
    refs = set(imports) | set(requires) | set(dynamic)
    facts.imports = sorted({ref.split('/')[-1] for ref in refs})
    facts.references = sorted(refs)
    return facts


def _bash_facts(path: Path) -> FileFacts:
    facts = _basic_facts(path, "bash")
    text = path.read_text(errors="ignore")
    refs = re.findall(r"^\s*(?:source|\.)\s+([\w./-]+)", text, re.MULTILINE)
    facts.references = sorted(set(refs))
    facts.imports = sorted({Path(ref).stem for ref in refs})
    return facts


def _html_facts(path: Path) -> FileFacts:
    facts = _basic_facts(path, "html")
    text = path.read_text(errors="ignore")
    refs = re.findall(r"(?:src|href)=['\"]([^'\"]+)['\"]", text)
    facts.references = sorted({ref for ref in refs if ref})
    return facts


def _css_facts(path: Path) -> FileFacts:
    facts = _basic_facts(path, "css")
    text = path.read_text(errors="ignore")
    refs = re.findall(r"@import\s+['\"]([^'\"]+)['\"]", text)
    facts.references = sorted(set(refs))
    return facts


def _java_facts(path: Path) -> FileFacts:
    facts = _basic_facts(path, "java")
    text = path.read_text(errors="ignore")
    imports = re.findall(r"^\s*import\s+([A-Za-z0-9_.]+);", text, re.MULTILINE)
    refs: Set[str] = set()
    for item in imports:
        refs.add(item.replace('.', '/'))
    facts.imports = sorted({item.split('.')[-1] for item in imports})
    facts.references = sorted(refs)
    return facts


def _generic_text_facts(path: Path) -> FileFacts:
    return _basic_facts(path, "text")


def _detect_language(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.endswith('.d.ts'):
        return 'typescript'
    if suffix == '.py':
        return 'python'
    if suffix == '.rs':
        return 'rust'
    if suffix == '.go':
        return 'go'
    if suffix in {'.tsx', '.ts'}:
        return 'typescript'
    if suffix in {'.jsx', '.mjs', '.cjs', '.js'}:
        return 'javascript'
    if suffix in {'.sh', '.bash'}:
        return 'bash'
    if suffix in {'.html', '.htm'}:
        return 'html'
    if suffix in {'.css', '.scss'}:
        return 'css'
    if suffix == '.java':
        return 'java'
    return 'text'


LANGUAGE_PARSERS = {
    'python': _py_facts,
    'rust': _rust_facts,
    'go': _go_facts,
    'typescript': lambda p: _ts_js_facts(p, 'typescript'),
    'javascript': lambda p: _ts_js_facts(p, 'javascript'),
    'bash': _bash_facts,
    'html': _html_facts,
    'css': _css_facts,
    'java': _java_facts,
}


REFERENCE_EXTENSION_CANDIDATES: Dict[str, List[str]] = {
    'python': ['', '.py', '/__init__.py'],
    'rust': ['', '.rs', '/mod.rs', '/lib.rs'],
    'go': ['', '.go'],
    'typescript': ['', '.ts', '.tsx', '.js', '.jsx', '/index.ts', '/index.tsx', '/index.js', '/index.jsx'],
    'javascript': ['', '.js', '.jsx', '.ts', '.tsx', '/index.js', '/index.jsx', '/index.ts', '/index.tsx'],
    'bash': ['', '.sh'],
    'html': ['', '.html', '.htm', '.js', '.css'],
    'css': ['', '.css', '.scss'],
    'java': ['', '.java'],
    'text': [''],
}


def _candidate_paths(base: Path, patterns: List[str]) -> List[Path]:
    base = base.resolve()
    candidates: Set[Path] = {base}
    for pattern in patterns:
        if not pattern:
            continue
        if pattern.startswith('/'):
            candidates.add((base / pattern.lstrip('/')).resolve())
            continue
        if pattern.startswith('index.'):
            candidates.add((base / pattern).resolve())
            continue
        try:
            if base.suffix:
                candidates.add(base.with_suffix(pattern))
            else:
                candidates.add((base / pattern.lstrip('/')).resolve())
                candidates.add((base.parent / f"{base.name}{pattern}").resolve())
        except (ValueError, RuntimeError):
            candidates.add((base / pattern.lstrip('/')).resolve())
    return list(candidates)


AliasMap = Dict[str, List[Tuple[str, bool]]]


def _resolve_reference(
    reference: str,
    current_path: Path,
    root: Path,
    rel_to_abs: Dict[str, str],
    basename_index: Dict[str, List[str]],
    language: str,
    alias_map: AliasMap,
) -> Optional[str]:
    ref = reference.strip().strip('"').strip("'")
    if not ref or ref.startswith(('http://', 'https://', 'data:')):
        return None

    patterns = REFERENCE_EXTENSION_CANDIDATES.get(language, [''])
    candidates: List[Path] = []
    current_dir = current_path.parent

    # Alias handling (TypeScript/JavaScript path mappings)
    for alias, targets in alias_map.items():
        if ref == alias or ref.startswith(f"{alias}/"):
            remainder = ref[len(alias):].lstrip('/')
            for pattern, has_wildcard in targets:
                if has_wildcard:
                    target_path = pattern.replace('*', remainder)
                else:
                    target_path = pattern.rstrip('/')
                    if remainder:
                        target_path = f"{target_path}/{remainder}" if target_path else remainder
                base = (root / target_path).resolve()
                candidates.extend(_candidate_paths(base, patterns))
            break

    if ref.startswith(('./', '../')):
        base = (current_dir / ref).resolve()
        candidates.extend(_candidate_paths(base, patterns))
    elif ref.startswith('/'):
        base = (root / ref.lstrip('/')).resolve()
        candidates.extend(_candidate_paths(base, patterns))
    else:
        base = (current_dir / ref).resolve()
        candidates.extend(_candidate_paths(base, patterns))
        root_based = (root / ref).resolve()
        candidates.extend(_candidate_paths(root_based, patterns))

    for cand in candidates:
        try:
            rel = str(cand.relative_to(root))
        except ValueError:
            continue
        if rel in rel_to_abs:
            return rel_to_abs[rel]

    # Fallback: match by suffix on relative path
    suffix = ref.replace('.', '/').strip('/')
    if suffix:
        for rel, abs_path in rel_to_abs.items():
            if rel.endswith(suffix) or rel.endswith(f"{suffix}.py"):
                return abs_path

    # Fallback: by basename
    base_name = Path(ref).stem
    if base_name in basename_index:
        return basename_index[base_name][0]

    return None


def _load_path_aliases(root: Path) -> AliasMap:
    """Load TypeScript/JavaScript path aliases from tsconfig/jsconfig/package.json."""

    alias_map: AliasMap = {}

    def _merge(paths: Dict[str, Any], base_url: str = '.'):
        for alias, targets in paths.items():
            if not isinstance(targets, list):
                continue
            cleaned = alias.rstrip('*').rstrip('/').rstrip('\n')
            if not cleaned:
                continue
            for target in targets:
                if not isinstance(target, str):
                    continue
                full = (Path(base_url) / target).as_posix()
                alias_map.setdefault(cleaned, []).append((full, '*' in target))

    for config_name in ('tsconfig.json', 'jsconfig.json'):
        config_path = root / config_name
        if not config_path.exists():
            continue
        try:
            data = json.loads(config_path.read_text())
        except Exception:
            continue
        compiler = data.get('compilerOptions') or {}
        base_url = compiler.get('baseUrl') or '.'
        paths = compiler.get('paths') or {}
        if isinstance(paths, dict):
            _merge(paths, base_url)

    # Support optional package.json "imports" field (Node >=16)
    pkg_path = root / 'package.json'
    if pkg_path.exists():
        try:
            data = json.loads(pkg_path.read_text())
            imports = data.get('imports') or {}
            if isinstance(imports, dict):
                normalized = {}
                for alias, target in imports.items():
                    alias_clean = alias.rstrip('*').rstrip('/')
                    targets = []
                    if isinstance(target, str):
                        targets = [target]
                    elif isinstance(target, dict):
                        targets = [v for v in target.values() if isinstance(v, str)]
                    if alias_clean and targets:
                        normalized[alias_clean] = targets
                if normalized:
                    _merge(normalized, '.')
        except Exception:
            pass

    deduped: AliasMap = {}
    for alias, entries in alias_map.items():
        seen: Set[Tuple[str, bool]] = set()
        uniq: List[Tuple[str, bool]] = []
        for entry in entries:
            if entry not in seen:
                uniq.append(entry)
                seen.add(entry)
        deduped[alias] = uniq
    return deduped


def build_code_graph(root: Path) -> Dict[str, Any]:
    """Build a multi-language code graph for the repository.

    Supported languages: Python, Rust, Go, TypeScript/JavaScript (incl. React), CSS, HTML, Bash, Java.
    """
    root = root.resolve()
    skip_dirs = {
        ".venv",
        "venv",
        ".git",
        "node_modules",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        "target",
    }

    candidate_files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(sd in path.parts for sd in skip_dirs):
            continue
        language = _detect_language(path)
        if language == "text":
            continue
        candidate_files.append(path.resolve())

    alias_map = _load_path_aliases(root)

    facts: List[FileFacts] = []
    python_paths: List[Path] = []
    for path in candidate_files:
        language = _detect_language(path)
        parser = LANGUAGE_PARSERS.get(language, _generic_text_facts)
        try:
            fact = parser(path)
            facts.append(fact)
            if language == "python":
                python_paths.append(path)
        except Exception:
            continue

    rel_to_abs: Dict[str, str] = {}
    abs_to_rel: Dict[str, str] = {}
    basename_index: Dict[str, List[str]] = {}
    for path in candidate_files:
        rel = str(path.relative_to(root))
        abs_str = str(path)
        rel_to_abs[rel] = abs_str
        abs_to_rel[abs_str] = rel
        stem = Path(rel).stem
        basename_index.setdefault(stem, []).append(abs_str)

    facts_by_path = {f.path: f for f in facts}

    # Python-specific import edges (module-based)
    pkg_init: Dict[str, Path] = {}
    mod_file: Dict[str, Path] = {}
    for p in python_paths:
        rel = p.relative_to(root)
        if p.name == "__init__.py":
            pkg_init[rel.parent.name] = p
        else:
            mod_file[p.stem] = p

    edges: Set[Tuple[str, str, str]] = set()

    for fact in facts:
        if fact.language != "python":
            continue
        src = fact.path
        for module in fact.imports:
            tgt_path: Optional[Path] = None
            if module in mod_file:
                tgt_path = mod_file[module]
            elif module in pkg_init:
                tgt_path = pkg_init[module]
            if tgt_path is not None:
                edges.add((src, str(tgt_path), "import"))

    # Python call edges using AST on python files
    def_map: Dict[str, str] = {}
    seen_multi: set[str] = set()
    for fact in facts:
        if fact.language != "python":
            continue
        for fn in fact.functions:
            if fn in seen_multi:
                continue
            if fn in def_map:
                seen_multi.add(fn)
                def_map.pop(fn, None)
            else:
                def_map[fn] = fact.path

    for p in python_paths:
        try:
            src_text = p.read_text(errors="ignore")
            tree = ast.parse(src_text)
        except Exception:
            continue
        calls: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name:
                    calls.append(name)
        uniques = {c for c in calls if c in def_map}
        src_path = str(p)
        for callee in uniques:
            dst = def_map.get(callee)
            if dst and src_path != dst:
                edges.add((src_path, dst, "call"))

    # Generic reference edges for all languages
    for fact in facts:
        if not fact.references:
            continue
        src_path = Path(fact.path)
        for ref in fact.references:
            resolved = _resolve_reference(
                ref,
                src_path,
                root,
                rel_to_abs,
                basename_index,
                fact.language,
                alias_map,
            )
            if resolved and resolved != fact.path:
                edges.add((fact.path, resolved, f"{fact.language}_import"))

    total_lines = sum(f.lines for f in facts)
    total_classes = sum(len(f.classes) for f in facts)
    total_funcs = sum(len(f.functions) for f in facts)
    by_defs = sorted(
        facts,
        key=lambda f: (len(f.classes) + len(f.functions), f.lines),
        reverse=True,
    )[:20]

    edge_list = [
        {"source": src, "target": dst, "kind": kind}
        for (src, dst, kind) in sorted(edges)
    ]

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
        "edges": edge_list,
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
    imp_out = sorted({
        e["target"]
        for e in edges
        if (e.get("kind") or "").endswith("import") and e.get("source") == src
    })
    imp_in = sorted({
        e["source"]
        for e in edges
        if (e.get("kind") or "").endswith("import") and e.get("target") == src
    })
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
