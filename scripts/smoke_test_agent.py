#!/usr/bin/env python3
"""
Agent smoke test (no external LLM/deps required).

Exercises core components end-to-end without network or DSPy runtime:
 - code search + snapshot
 - classical TF-IDF index build + semantic search
 - orchestrator_runtime safe tool evaluations (grep/extract/context/codectx/index/esearch/plan)
 - retrieval event logging + agentic features
 - vectorized pipeline features on synthetic payload
 - enhanced data manager basic log/metrics accessors

Prints a compact summary and exits nonzero on critical failure.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    ok = True
    workspace = ROOT / ".smoke_ws"
    logs_dir = workspace / "logs"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Seed sample files
    (workspace / "sample.py").write_text("""
def target():
    return 42
""".strip()+"\n")
    (workspace / "text.txt").write_text("hello needle world\n")
    (logs_dir / "app.log").write_text("[ERROR] needle failure in pipeline\n")

    results: Dict[str, Any] = {}

    try:
        _section("Code tools: search + snapshot")
        import runpy as _rp
        ns_cs = _rp.run_path(str(ROOT / 'dspy_agent' / 'code_tools' / 'code_search.py'))
        ns_snap = _rp.run_path(str(ROOT / 'dspy_agent' / 'code_tools' / 'code_snapshot.py'))
        hits = ns_cs['search_text'](workspace, r"needle", regex=True)
        snap = ns_snap['build_code_snapshot'](workspace / "sample.py")
        print(f"search hits: {len(hits)} | snapshot chars: {len(snap)}")
        results["search_hits"] = len(hits)
        results["snapshot_len"] = len(snap)
    except Exception as e:
        print(f"[FAIL] code tools: {e}")
        ok = False

    try:
        _section("Index build + semantic search (TF-IDF)")
        from dspy_agent.embedding.indexer import build_index, save_index, load_index, semantic_search
        meta, items = build_index(workspace, smart=True)
        save_index(workspace, meta, items)
        meta2, items2 = load_index(workspace)
        hits = semantic_search("needle", meta2, items2, top_k=5)
        print(f"index chunks: {len(items)} | esearch hits: {len(hits)}")
        results["index_chunks"] = len(items)
        results["esearch_hits"] = len(hits)
    except Exception as e:
        print(f"[FAIL] index/search: {e}")
        ok = False

    try:
        _section("Orchestrator runtime: safe tool evaluations")
        # Provide a minimal 'dspy' stub to satisfy optional imports during module init
        import types as _types
        import sys as _sys
        if 'dspy' not in _sys.modules:
            dspy_stub = _types.ModuleType('dspy')
            class _Emb:
                def __init__(self, model: str, *a, **k): pass
                def embed(self, texts): return [[0.0] * 4 for _ in texts]
            setattr(dspy_stub, 'Embeddings', _Emb)
            _sys.modules['dspy'] = dspy_stub
        from dspy_agent.agents.orchestrator_runtime import evaluate_tool_choice
        # grep
        out_g = evaluate_tool_choice("grep", {"pattern": "needle"}, workspace=workspace)
        # extract by regex line in file
        out_x = evaluate_tool_choice("extract", {"file": "sample.py", "regex": "def target"}, workspace=workspace)
        # context
        out_c = evaluate_tool_choice("context", {}, workspace=workspace, logs_path=logs_dir)
        # codectx
        out_cc = evaluate_tool_choice("codectx", {}, workspace=workspace)
        # index/esearch reuse built index
        out_i = evaluate_tool_choice("index", {}, workspace=workspace)
        out_s = evaluate_tool_choice("esearch", {"query": "needle"}, workspace=workspace)
        # plan heuristic
        out_p = evaluate_tool_choice("plan", {"plan_text": "- step one\n- step two"}, workspace=workspace)
        print("scores:", {k: round(v.score, 2) for k, v in {
            'grep': out_g, 'extract': out_x, 'context': out_c, 'codectx': out_cc,
            'index': out_i, 'esearch': out_s, 'plan': out_p
        }.items()})
        results["orchestrator_scores"] = {k: v.score for k, v in {
            'grep': out_g, 'extract': out_x, 'context': out_c, 'codectx': out_cc,
            'index': out_i, 'esearch': out_s, 'plan': out_p
        }.items()}
    except Exception as e:
        print(f"[FAIL] orchestrator_runtime: {e}")
        ok = False

    try:
        _section("Retrieval events + agentic features")
        from dspy_agent.agentic import log_retrieval_event
        from dspy_agent.context.context_manager import ContextManager
        # Log two retrieval hits
        log_retrieval_event(workspace, "needle", [{"path": str(workspace / "text.txt"), "score": 0.9, "source": "esearch"}])
        time.sleep(0.01)
        log_retrieval_event(workspace, "target", [{"path": str(workspace / "sample.py"), "score": 0.8, "source": "esearch"}])
        cm = ContextManager(workspace)
        feats = cm.agentic_features()
        ctx = cm.build_patch_context("fix test")
        print(f"agentic_features: {len(feats)} dims | ctx keys: {sorted(ctx.keys())[:5]}")
        results["agentic_dims"] = len(feats)
        results["ctx_has_text"] = bool(ctx.get("text"))
    except Exception as e:
        print(f"[FAIL] retrieval/agentic: {e}")
        ok = False

    try:
        _section("Vectorized pipeline features")
        from dspy_agent.streaming.vectorized_pipeline import RLVectorizer
        v = RLVectorizer(workspace)
        rec = v.vectorize("logs.ctx.demo", {"text": "ERROR needle occurred in handler"})
        if rec is None:
            raise RuntimeError("vectorize returned None")
        print(f"vector features: {len(rec.features)} | top1={rec.metadata.get('tfidf_top1'):.3f}")
        results["vec_features"] = len(rec.features)
    except Exception as e:
        print(f"[FAIL] vectorizer: {e}")
        ok = False

    try:
        _section("Enhanced data manager")
        from dspy_agent.db.enhanced_storage import get_enhanced_data_manager
        from dspy_agent.db import create_log_entry, Environment
        dm = get_enhanced_data_manager()
        entry = create_log_entry(level="INFO", source="smoke", message="ping", context={"ok": True}, environment=Environment.DEVELOPMENT)
        dm.log(entry)
        logs = dm.get_recent_logs(level="INFO", limit=5)
        stats = dm.get_cache_stats()
        print(f"recent logs: {len(logs)} | cache main size: {stats['main_cache']['size']}")
        results["dm_logs"] = len(logs)
    except Exception as e:
        print(f"[FAIL] enhanced data manager: {e}")
        ok = False

    # Summarize
    _section("Smoke test summary")
    print(json.dumps(results, indent=2))
    if not ok:
        print("\n⚠️  One or more checks failed.")
        return 1
    print("\n✅ All smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
