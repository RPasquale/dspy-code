from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


@dataclass
class ToolStats:
    count: int = 0
    success: int = 0
    sum_score: float = 0.0
    sum_dur: float = 0.0

    def update(self, success: bool, score: float, dur: float) -> None:
        self.count += 1
        if success:
            self.success += 1
        self.sum_score += float(score)
        self.sum_dur += float(dur)

    @property
    def avg_score(self) -> float:
        return (self.sum_score / self.count) if self.count else 0.0

    @property
    def avg_dur(self) -> float:
        return (self.sum_dur / self.count) if self.count else 0.0


def render_tool_table(stats: Dict[str, ToolStats]) -> Table:
    table = Table(title="Per-Tool Stats", show_header=True, header_style="bold cyan")
    table.add_column("tool", style="magenta")
    table.add_column("count", justify="right")
    table.add_column("success", justify="right")
    table.add_column("avg_score", justify="right")
    table.add_column("avg_dur_s", justify="right")
    for tool in sorted(stats.keys()):
        s = stats[tool]
        table.add_row(tool, str(s.count), str(s.success), f"{s.avg_score:.3f}", f"{s.avg_dur:.2f}")
    return table


def render_events_table(events: deque[dict]) -> Table:
    table = Table(title="Recent tool.end", show_header=True, header_style="bold cyan")
    table.add_column("tool", style="green")
    table.add_column("score", justify="right")
    table.add_column("dur_s", justify="right")
    table.add_column("ok", style="magenta")
    table.add_column("session", style="dim")
    for e in list(events)[-20:]:  # last 20
        data = e.get("data", {})
        table.add_row(
            str(data.get("tool")),
            f"{float(data.get('score', 0.0)):.3f}",
            f"{float(data.get('duration_sec', 0.0)):.2f}",
            "yes" if bool(data.get("success", False)) else "no",
            str(data.get("session") or "-"),
        )
    return table


def render_session_panel(summary: Optional[dict]) -> Panel:
    if not summary:
        return Panel("(no session summary yet)", title="Session", border_style="yellow")
    body = []
    for key in ("session_id", "steps", "last_tool", "avg_score"):
        if key in summary:
            body.append(f"{key}: {summary.get(key)}")
    text = "\n".join(body) if body else "(partial)"
    return Panel(text, title="Session", border_style="green")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rich dashboard for agent.metrics + session summary (RedDB)")
    ap.add_argument("--start", type=int, default=0, help="Start offset for agent.metrics")
    ap.add_argument("--interval", type=float, default=1.0, help="Refresh interval seconds")
    ap.add_argument("--workspace", default=None, help="Filter by workspace path (optional)")
    ap.add_argument("--window", type=int, default=500, help="Max events to retain in memory")
    args = ap.parse_args()

    from dspy_agent.db.factory import get_storage
    st = get_storage()

    console = Console()
    offset = int(args.start)
    tool_stats: Dict[str, ToolStats] = defaultdict(ToolStats)
    last_events: deque[dict] = deque(maxlen=max(50, int(args.window)))
    last_session: Optional[dict] = None

    def tick() -> Panel:
        nonlocal offset, last_session
        rows = list(st.read("agent.metrics", start=offset, count=100))  # type: ignore
        if rows:
            offset = rows[-1][0] + 1
        for off, evt in rows:
            if args.workspace and str(evt.get("workspace")).strip() != str(args.workspace).strip():
                continue
            name = evt.get("metric")
            if name == "tool.end":
                data = evt.get("data", {})
                tool = str(data.get("tool") or "?")
                success = bool(data.get("success", False))
                score = float(data.get("score", 0.0))
                dur = float(data.get("duration_sec", 0.0))
                tool_stats[tool].update(success, score, dur)
                last_events.append(evt)
            elif name == "session.summary":
                # Keep latest summary
                last_events.append(evt)
        # KV session summary (latest)
        kv = st.get("agent.session.summary")  # type: ignore
        if isinstance(kv, dict):
            last_session = kv

        group = Group(
            render_tool_table(tool_stats),
            render_events_table(last_events),
            render_session_panel(last_session),
        )
        return Panel(group, title="Agent Dashboard", border_style="cyan")

    with Live(console=console, refresh_per_second=max(1, int(1/args.interval))):
        while True:
            panel = tick()
            console.clear()
            console.print(panel)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()

