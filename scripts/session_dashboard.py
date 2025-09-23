from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from typing import Dict

from rich.live import Live
from rich.panel import Panel
from rich.table import Table


def main() -> None:
    # Aggregate from RedDB streams
    from dspy_agent.db.factory import get_storage
    st = get_storage()
    actions_off = 0
    metrics_off = 0
    tool_counts: Dict[str, int] = defaultdict(int)
    tool_scores: Dict[str, float] = defaultdict(float)
    tool_seen: Dict[str, int] = defaultdict(int)
    last_session = {}
    # Rolling windows
    tool_end_times = deque(maxlen=300)  # (ts)
    session_timeline = deque(maxlen=200)  # (ts, steps, avg_score)

    def render():
        # Tools panel
        t = Table(title="Tool Usage")
        t.add_column("Tool")
        t.add_column("Count", justify="right")
        t.add_column("Avg Score", justify="right")
        for k in sorted(tool_counts.keys()):
            cnt = tool_counts[k]
            avg = (tool_scores[k] / max(1, tool_seen[k])) if tool_seen[k] else 0.0
            t.add_row(k, str(cnt), f"{avg:.3f}")
        # Rate panel
        now = time.time()
        rate_1m = len([ts for ts in tool_end_times if now - ts <= 60.0])
        rate_5m = len([ts for ts in tool_end_times if now - ts <= 300.0]) / max(1.0, 5.0)
        rp = Table(title="Rates (per min)")
        rp.add_column("Window")
        rp.add_column("Rate", justify="right")
        rp.add_row("1m", str(rate_1m))
        rp.add_row("5m avg", f"{rate_5m:.1f}")
        # Timeline panel (last N session summaries)
        tp = Table(title="Session Timeline")
        tp.add_column("t (s)")
        tp.add_column("steps", justify="right")
        tp.add_column("avg_score", justify="right")
        base = session_timeline[0][0] if session_timeline else now
        for ts, steps, avg in list(session_timeline)[-15:]:
            tp.add_row(f"{int(ts - base)}", str(int(steps)), f"{float(avg):.3f}")
        # Compose
        left = Panel.fit(t, title="Tools")
        right = Panel.fit(rp, title="Rates")
        bottom = Panel.fit(tp, title="Timeline")
        # Rich can't easily make complex grids without Layout; simple stack
        return Panel.fit(Panel.fit(left), title="Session Dashboard")

    with Live(render(), refresh_per_second=2) as live:
        while True:
            # Read metrics
            rows = list(st.read('agent.metrics', start=metrics_off, count=200))  # type: ignore
            if rows:
                metrics_off = rows[-1][0] + 1
                for _, rec in rows:
                    if not isinstance(rec, dict):
                        continue
                    name = rec.get('metric')
                    data = rec.get('data') or {}
                    if name == 'tool.end':
                        tool = str(data.get('tool', ''))
                        score = float(data.get('score', 0.0) or 0.0)
                        tool_counts[tool] += 1
                        tool_scores[tool] += score
                        tool_seen[tool] += 1
                        ts = float(rec.get('ts', time.time()) or time.time())
                        tool_end_times.append(ts)
                    elif name == 'session.summary':
                        # Update session snapshot
                        last_session = data
                        ts = float(rec.get('ts', time.time()) or time.time())
                        session_timeline.append((ts, int(data.get('steps', 0) or 0), float(data.get('avg_score', 0.0) or 0.0)))
                live.update(render())
            time.sleep(0.5)


if __name__ == '__main__':
    main()
