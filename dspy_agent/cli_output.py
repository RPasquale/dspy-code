from __future__ import annotations

import sys
from typing import Any

try:
    from rich.panel import Panel
except Exception:  # pragma: no cover - optional
    Panel = None  # type: ignore


class PlainConsole:
    """Very small console shim for CI/plain output.

    It accepts the same .print(...) calls used by rich.Console but writes
    a simplified text-only representation to stdout.
    """

    def _to_text(self, obj: Any) -> str:
        try:
            # Unwrap rich Panel
            if Panel is not None and isinstance(obj, Panel):  # type: ignore
                inner = getattr(obj, "renderable", obj)
                return str(inner)
        except Exception:
            pass
        try:
            # Rich Text and other objects often stringify well
            return str(obj)
        except Exception:
            return repr(obj)

    def print(self, *objs: Any, **kwargs: Any) -> None:  # noqa: D401
        for obj in objs:
            txt = self._to_text(obj)
            # Strip very common rich tags
            for tag in ("[green]", "[/green]", "[yellow]", "[/yellow]", "[red]", "[/red]", "[cyan]", "[/cyan]", "[bold]", "[/bold]"):
                txt = txt.replace(tag, "")
            sys.stdout.write(txt + ("\n" if not txt.endswith("\n") else ""))
            sys.stdout.flush()

