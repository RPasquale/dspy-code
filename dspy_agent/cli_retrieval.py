from __future__ import annotations

import typer
from pathlib import Path
from typing import Optional, List

# Reuse implementations from main CLI to avoid duplication


retrieval_app = typer.Typer(no_args_is_help=True, help="Retrieval and embeddings commands")


@retrieval_app.command("index")
def index(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace to index"),
    glob: Optional[List[str]] = typer.Option(None, '--glob', help="Include glob. Can repeat."),
    exclude: Optional[List[str]] = typer.Option(None, '--exclude', help="Exclude glob. Can repeat."),
    lines: int = typer.Option(200, '--chunk-lines', help="Lines per chunk (non-Python)"),
    smart: bool = typer.Option(True, '--smart/--no-smart', help="Code-aware chunking (Python)"),
):
    from . import cli as _cli
    _cli.index(workspace=workspace, glob=glob, exclude=exclude, lines=lines, smart=smart)


@retrieval_app.command("esearch")
def esearch(
    query: str = typer.Argument(..., help="Semantic query"),
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace to search"),
    k: int = typer.Option(5, '--k', help="Top-K results"),
    context: int = typer.Option(4, '--context', help="Lines of context to show around chunk bounds"),
):
    from . import cli as _cli
    _cli.esearch(query=query, workspace=workspace, k=k, context=context)


@retrieval_app.command("emb-index")
def emb_index(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace"),
    model: str = typer.Option(..., '--model', help="Embeddings model (e.g., openai/text-embedding-3-small or Qwen/Qwen3-Embedding-0.6B)"),
    base_url: Optional[str] = typer.Option(None, '--base-url', help="Embeddings API base (for DSPy providers)"),
    api_key: Optional[str] = typer.Option(None, '--api-key', help="Embeddings API key (for DSPy providers)"),
    hf: bool = typer.Option(False, '--hf/--no-hf', help="Use sentence-transformers (HuggingFace) for local embeddings"),
    device: Optional[str] = typer.Option(None, '--device', help="HF device map (e.g., 'auto' or 'cpu')"),
    flash: bool = typer.Option(False, '--flash/--no-flash', help="Enable flash_attention_2 in HF model_kwargs"),
    lines: int = typer.Option(200, '--chunk-lines', help="Lines per chunk for non-Python"),
    smart: bool = typer.Option(True, '--smart/--no-smart', help="Code-aware chunking (Python)"),
    persist: bool = typer.Option(False, '--persist/--no-persist', help="Also persist embeddings and code chunks to RedDB"),
    infermesh_url: Optional[str] = typer.Option(None, '--infermesh-url', help="Use InferMesh embedding service at this base URL (overrides other embed options)"),
    infermesh_model: Optional[str] = typer.Option(None, '--infermesh-model', help="InferMesh model id (defaults to --model)"),
    infermesh_api_key: Optional[str] = typer.Option(None, '--infermesh-api-key', help="Bearer token for InferMesh (optional)"),
    force: bool = typer.Option(False, '--force/--no-force', help="Override config if embeddings are disabled"),
):
    from . import cli as _cli
    _cli.emb_index(
        workspace=workspace,
        model=model,
        base_url=base_url,
        api_key=api_key,
        hf=hf,
        device=device,
        flash=flash,
        lines=lines,
        smart=smart,
        persist=persist,
        infermesh_url=infermesh_url,
        infermesh_model=infermesh_model,
        infermesh_api_key=infermesh_api_key,
        force=force,
    )


@retrieval_app.command("embeddings_inspect")
def embeddings_inspect(
    start: int = typer.Option(0, '--start', help="Start offset in emb.index stream"),
    count: int = typer.Option(10, '--count', help="Number of entries to show"),
):
    from . import cli as _cli
    _cli.embeddings_inspect(start=start, count=count)


@retrieval_app.command("chunks_compact")
def chunks_compact(
    start: int = typer.Option(0, '--start', help="Start offset in code.chunks stream"),
    count: int = typer.Option(10, '--count', help="Number of entries to show"),
):
    from . import cli as _cli
    _cli.chunks_compact(start=start, count=count)
