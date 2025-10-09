#!/usr/bin/env python3
"""CLI to generate project activity graph and persist it to RedDB."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dspy_agent.project.activity_graph import build_project_activity_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate project activity graph snapshot.')
    parser.add_argument('--namespace', default=os.getenv('DSPY_NAMESPACE', 'dspy_agent'))
    parser.add_argument('--workspace', default=os.getenv('DSPY_WORKSPACE'))
    parser.add_argument('--actions', type=int, default=500, help='Recent actions to sample.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_project_activity_graph(args.namespace, workspace=args.workspace, action_limit=args.actions)
    print(f"Project graph snapshot generated with {len(payload['nodes'])} nodes and {len(payload['edges'])} edges.")


if __name__ == '__main__':
    main()
