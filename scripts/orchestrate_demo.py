#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dspy_agent.skills.tools.db_tools import db_ingest, db_multi_head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns', '--namespace', dest='ns', default='default')
    ap.add_argument('--workspace', dest='workspace', default='.')
    args = ap.parse_args()

    ns = args.ns
    ws = Path(args.workspace).resolve()

    print(f"[demo] workspace={ws} ns={ns}")
    print("[demo] ingest a couple of docs")
    db_ingest({"kind": "document", "collection": "notes", "id": "n1", "text": "Payment API returns 500 when JSON invalid"}, namespace=ns)
    db_ingest({"kind": "document", "collection": "notes", "id": "n2", "text": "Checkout service timeout on high load"}, namespace=ns)

    print("[demo] multi-head search (non-LLM)")
    out = db_multi_head("neighbors of svc_checkout", namespace=ns, top_k=5, use_lm=False)
    print(json.dumps(out, indent=2)[:4000])
    print("[demo] done")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

