from __future__ import annotations

import typer
from pathlib import Path
from typing import Optional



patch_app = typer.Typer(no_args_is_help=True, help="Patching and code editing commands")


@patch_app.command("patch")
def patch(
    patch_file: Optional[Path] = typer.Option(None, '--file', exists=True, help="Unified diff file"),
):
    from . import cli as _cli
    _cli.patch(patch_file=patch_file)


@patch_app.command("diff")
def diff(
    file: Path = typer.Option(..., '--file', exists=True, help="Existing file to diff against"),
    new: Optional[Path] = typer.Option(None, '--new', exists=True, help="New file to compare; if omitted, reads from STDIN"),
    unified: int = typer.Option(3, '--unified', help="Context lines in diff"),
    out: Optional[Path] = typer.Option(None, '--out', help="Write patch to this file"),
):
    from . import cli as _cli
    _cli.diff(file=file, new=new, unified=unified, out=out)


@patch_app.command("edit")
def edit(
    task: str = typer.Argument(..., help="Describe the code change you want"),
    workspace: Path = typer.Option(Path.cwd(), '--workspace', exists=True, dir_okay=True),
    context: Optional[str] = typer.Option(None, '--context', help="Optional context (errors/logs). If omitted, built from logs"),
    file_hints: Optional[str] = typer.Option(None, '--files', help="Optional file/module hints"),
    apply: bool = typer.Option(False, '--apply/--no-apply', help="Apply the generated patch"),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama'),
    model: Optional[str] = typer.Option("qwen3:1.7b", '--model'),
    base_url: Optional[str] = typer.Option(None, '--base-url'),
    api_key: Optional[str] = typer.Option(None, '--api-key'),
    show_rationale: bool = typer.Option(False, '--show-rationale/--no-show-rationale', help='Show/hide model rationales (debug)'),
    beam_k: int = typer.Option(1, '--beam', help='Beam size for proposals'),
    speculative: bool = typer.Option(False, '--speculative/--no-speculative', help='Use draft model for speculative pass'),
    draft_model: Optional[str] = typer.Option(None, '--draft-model', help='Draft LM (e.g., qwen2:0.5b)'),
    profile: Optional[str] = typer.Option(None, '--profile', help='Performance profile: fast|balanced|maxquality'),
):
    from . import cli as _cli
    _cli.code_edit(
        task=task,
        workspace=workspace,
        context=context,
        file_hints=file_hints,
        apply=apply,
        ollama=ollama,
        model=model,
        base_url=base_url,
        api_key=api_key,
        show_rationale=show_rationale,
        beam_k=beam_k,
        speculative=speculative,
        draft_model=draft_model,
        profile=profile,
    )
