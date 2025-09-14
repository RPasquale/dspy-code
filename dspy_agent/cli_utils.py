from __future__ import annotations

from pathlib import Path
from typing import List

from rich.panel import Panel


def banner_text() -> str:
    return (
        "\n"
        "██████╗ ███████╗██████╗ ██╗   ██╗   ██████╗ ██████╗ ██████╗  ███████╗\n"
        "██╔══██╗██╔════╝██╔══██╗╚██╗ ██╔╝  ██╔════╝██╔══██╗██╔══██╗ ██╔════╝\n"
        "██║  ██║███████╗██████╔╝ ╚████╔╝   ██║     ██║  ██║██║  ██║ █████╗   \n"
        "██║  ██║╚════██║██╔═══╝   ╚██╔╝     ██║     ██║  ██║██║  ██║ ██╔══╝   \n"
        "██████╔╝███████║██║        ██║      ╚██████╗██████╔╝██████╔╝ ███████╗\n"
        "╚═════╝ ╚══════╝╚═╝        ╚═╝       ╚═════╝╚═════╝ ╚═════╝  ╚══════╝\n"
        "\n                 DSPY-CODE — Trainable Coding Agent\n"
    )


def print_banner(console) -> None:
    if console is None:
        return
    from os import getenv
    if getenv("DSPY_NO_BANNER"):
        return
    console.print(Panel.fit(banner_text(), border_style="magenta", title="[banner]DSPY-CODE[/banner]", subtitle="[accent]Booting neural synths...[/accent]"))


def print_header(console, title: str) -> None:
    console.rule(f"[accent]{title}")


def render_tree(root: Path, max_depth: int = 2, show_hidden: bool = False) -> str:
    def walk(dir_path: Path, prefix: str, depth: int, lines: List[str]):
        if depth > max_depth:
            return
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception as e:
            lines.append(prefix + f"[error: {e}]")
            return
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith('.')]
        for i, p in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(prefix + connector + p.name + ("/" if p.is_dir() else ""))
            if p.is_dir() and depth < max_depth:
                extension = "    " if i == len(entries) - 1 else "│   "
                walk(p, prefix + extension, depth + 1, lines)

    if root.is_file():
        return root.name
    lines: List[str] = [root.as_posix()]
    walk(root, "", 1, lines)
    return "\n".join(lines)


def sparkline(values: List[float], width: int = 28) -> str:
    if not values:
        return ""
    symbols = "▁▂▃▄▅▆▇█"
    vmin = min(values); vmax = max(values); span = (vmax - vmin) or 1.0
    pts = values[-width:]
    s = []
    for x in pts:
        norm = (x - vmin) / span
        idx = min(len(symbols)-1, int(norm * (len(symbols)-1)))
        s.append(symbols[idx])
    return "".join(s)

