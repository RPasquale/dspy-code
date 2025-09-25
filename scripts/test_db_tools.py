#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from dspy_agent.skills.tools.db_tools import db_ingest, db_query, db_multi_head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns', '--namespace', dest='ns', default='default')
    ap.add_argument('--no-server', action='store_true', help='Run without any HTTP server (pure in-memory)')
    args = ap.parse_args()

    ns = args.ns
    if args.no_server:
        os.environ.pop('REDDB_URL', None)

    print(f"[db] ingesting sample documents into namespace={ns}")
    db_ingest({"kind": "document", "collection": "notes", "id": "n1", "text": "Payment API returns 500 when JSON invalid"}, namespace=ns)
    db_ingest({"kind": "document", "collection": "notes", "id": "n2", "text": "Checkout service timeout on high load"}, namespace=ns)

    print("[db] simple query (auto)")
    q1 = db_query({"mode": "auto", "text": "payment 500", "collection": "notes", "top_k": 5}, namespace=ns)
    print(json.dumps(q1, indent=2))

    print("[db] multi-head (fusion)")
    q2 = db_multi_head("neighbors of svc_checkout", namespace=ns, use_lm=False)
    print(json.dumps(q2, indent=2)[:4000])

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

