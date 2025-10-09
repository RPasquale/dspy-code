#!/usr/bin/env python3
"""CLI entrypoint to run the graph memory refresh MCTS."""

from __future__ import annotations

import argparse
import os

from dspy_agent.agents.memory_mcts import run_mcts_memory_refresh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run MCTS-based graph memory refresh.')
    parser.add_argument('--namespace', default=os.getenv('DSPY_NAMESPACE', 'dspy_agent'))
    parser.add_argument('--iterations', type=int, default=400)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--penalty', type=float, default=1.0, help='Reward boost for successful fixes (unused placeholder).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    priorities = run_mcts_memory_refresh(args.namespace, iterations=args.iterations, depth=args.depth)
    count = len(priorities)
    print(f"Updated MCTS priorities for {count} nodes in namespace '{args.namespace}'.")


if __name__ == '__main__':
    main()
