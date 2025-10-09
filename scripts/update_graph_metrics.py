#!/usr/bin/env python3
"""Ingest runtime edge counts and coverage summaries into RedDB for graph metrics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dspy_agent.graph.feeds import (
    store_runtime_edge_counts_from_file,
    store_coverage_summary_from_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Update graph runtime/coverage metrics in RedDB.')
    parser.add_argument('--namespace', default=os.getenv('DSPY_NAMESPACE', 'dspy_agent'))
    parser.add_argument('--workspace', default=os.getenv('DSPY_WORKSPACE', '.'))
    parser.add_argument('--runtime', type=str, help='Path to JSON with runtime edge counts (list or dict).')
    parser.add_argument('--coverage', type=str, help='Path to coverage-summary.json (coverage/coverage-summary.json).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    namespace = args.namespace
    if args.runtime:
        store_runtime_edge_counts_from_file(namespace, Path(args.runtime))
        print(f'Updated runtime edge counts from {args.runtime} into namespace {namespace}.')
    if args.coverage:
        store_coverage_summary_from_file(namespace, Path(args.coverage))
        print(f'Updated coverage summary from {args.coverage} into namespace {namespace}.')
    if not args.runtime and not args.coverage:
        print('No inputs provided. Specify --runtime and/or --coverage files.')


if __name__ == '__main__':
    main()
