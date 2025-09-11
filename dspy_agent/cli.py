from __future__ import annotations

import os
from pathlib import Path
import sys
import shlex
import time
from datetime import datetime
from typing import Optional, List

import typer
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table
from rich.align import Align
from rich.console import Group
from rich.columns import Columns

import dspy
from .config import get_settings
from .llm import configure_lm
from .log_reader import extract_key_events, load_logs
from .skills.context_builder import ContextBuilder
from .skills.code_context import CodeContext
from .skills.task_agent import TaskAgent
from .skills.orchestrator import Orchestrator
from .code_search import (
    search_text,
    search_file as file_search,
    extract_context,
    python_extract_symbol,
    run_ast_grep,
    ast_grep_available,
)
from .code_snapshot import build_code_snapshot
from .patcher import apply_unified_patch, summarize_patch
from .indexer import build_index, save_index, load_index, semantic_search
from .train_gepa import run_gepa, run_gepa_with_val, evaluate_on_set
from .train_orchestrator import run_gepa_orchestrator, run_gepa_orchestrator_with_val, evaluate_orchestrator
from .autogen_dataset import bootstrap_datasets, bootstrap_datasets_with_splits
from .orchestrator_runtime import evaluate_tool_choice
from .indexer import tokenize
from .embeddings_index import (
    build_emb_index,
    save_emb_index,
    load_emb_index,
    emb_search as emb_search_fn,
    embed_query,
)
from .diffutil import unified_diff_file_vs_text
from .streaming_config import (
    StreamConfig,
    load_config as load_stream_cfg,
    save_config as save_stream_cfg,
    DEFAULT_CONFIG_PATH as STREAM_CFG_PATH,
    render_kafka_topic_commands,
)
from .streaming_runtime import start_local_stack, autodiscover_logs
from .kafka_log import get_kafka_logger
from .deploy import DeploymentLogger
from .status_http import start_status_server
from .streaming_kafka import WorkerLoop, KafkaParams
import threading
import json as _json


CYBER_THEME = Theme({
    "banner": "bold magenta",
    "accent": "bright_cyan",
    "ok": "bright_green",
    "warn": "yellow",
    "err": "bright_red",
    "dim": "dim",
})
app = typer.Typer(add_completion=False, help="DSPy-based local coding agent")
console = Console(theme=CYBER_THEME)
LIGHTWEIGHT_DIR = Path('docker/lightweight')


def _print_header(title: str):
    console.rule(f"[accent]{title}")


def _banner_text() -> str:
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


def _print_banner():
    if os.environ.get("DSPY_NO_BANNER"):
        return
    console.print(Panel.fit(_banner_text(), border_style="magenta", title="[banner]DSPY-CODE[/banner]", subtitle="[accent]Booting neural synths...[/accent]"))


@app.callback(invoke_without_command=True)
def _entry(ctx: typer.Context):
    _print_banner()
    if ctx.invoked_subcommand is None:
        console.print(Panel.fit(
            "Quickstart:\n"
            "  dspy-code              Launch interactive agent (recommended)\n"
            "  dspy-agent live        Start live training with sensible defaults\n"
            "  dspy-agent dataset     Build dataset + splits from ./ and ./logs\n\n"
            "Tips:\n"
            "  - Interactive 'start' shows all commands and usage.\n"
            "  - Most commands pick smart defaults (workspace=., logs=./logs).\n",
            title="Welcome",
            border_style="accent",
        ))


def _maybe_configure_lm(use_lm: bool, ollama: bool, model: Optional[str], base_url: Optional[str], api_key: Optional[str]):
    settings = get_settings()
    if not use_lm or settings.local_mode:
        return None
    return configure_lm(
        provider="ollama" if ollama else None,
        model_name=model,
        base_url=base_url,
        api_key=api_key,
    )


def _render_tree(root: Path, max_depth: int = 2, show_hidden: bool = False) -> str:
    def walk(dir_path: Path, prefix: str, depth: int, lines: list[str]):
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
    lines: list[str] = [root.as_posix()]
    walk(root, "", 1, lines)
    return "\n".join(lines)


def _watch_logs(target: Path, tail_lines: int = 0, interval: float = 2.0):
    from .log_reader import iter_log_paths, read_capped  # local import to avoid circular
    last_sizes: dict[Path, int] = {}
    console.print(Panel.fit(
        f"Watching {target} (interval {interval}s). Ctrl-C to stop.",
        title="watch logs", border_style="cyan"
    ))
    try:
        while True:
            changed = False
            for p in list(iter_log_paths([target])):
                try:
                    sz = p.stat().st_size
                except Exception:
                    continue
                prev = last_sizes.get(p, 0)
                if sz != prev:
                    changed = True
                    last_sizes[p] = sz
                    if tail_lines > 0:
                        try:
                            text = p.read_text(errors="ignore")
                            tail = "\n".join(text.splitlines()[-tail_lines:])
                            stamp = datetime.now().strftime("%H:%M:%S")
                            console.print(Panel(tail or "(empty)", title=f"{p} tail({tail_lines}) @ {stamp}", border_style="magenta"))
                        except Exception:
                            pass
            if changed:
                # Show key events snapshot
                bundle, _ = load_logs([target])
                key = extract_key_events(bundle) if bundle else "(no logs)"
                stamp = datetime.now().strftime("%H:%M:%S")
                console.print(Panel(key, title=f"Key Events @ {stamp}", border_style="blue"))
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("[dim]Stopped watching.[/dim]")


def _logs_mtime_sum(root: Path) -> int:
    total = 0
    try:
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    total += int(p.stat().st_mtime)
                except Exception:
                    pass
    except Exception:
        pass
    return total


def _sparkline(values: list[float], width: int = 28) -> str:
    if not values:
        return ""
    symbols = "▁▂▃▄▅▆▇█"
    vmin = min(values)
    vmax = max(values)
    span = (vmax - vmin) or 1.0
    # Take last N points
    pts = values[-width:]
    s = []
    for x in pts:
        norm = (x - vmin) / span
        idx = min(len(symbols)-1, int(norm * (len(symbols)-1)))
        s.append(symbols[idx])
    return "".join(s)


def _progress_panel(module: str, auto: str, tr: list[float], va: list[float], title: str = "Training Progress", window: int = 20) -> Panel:
    avg_tr = sum(tr)/len(tr) if tr else 0.0
    avg_va = sum(va)/len(va) if va else 0.0
    r_tr = tr[-window:] if tr else []
    r_va = va[-window:] if va else []
    ravg_tr = sum(r_tr)/len(r_tr) if r_tr else 0.0
    ravg_va = sum(r_va)/len(r_va) if r_va else 0.0
    max_tr = max(tr) if tr else 0.0
    max_va = max(va) if va else 0.0
    rows = []
    rows.append(f"module: [accent]{module}[/accent]   budget: [accent]{auto}[/accent]")
    rows.append(f"train n={len(tr)} avg={avg_tr:.3f} (r{window}={ravg_tr:.3f}) max={max_tr:.3f}  {_sparkline(tr)}")
    rows.append(f"val   n={len(va)} avg={avg_va:.3f} (r{window}={ravg_va:.3f}) max={max_va:.3f}  {_sparkline(va)}")
    body = "\n".join(rows)
    return Panel.fit(body, title=title, border_style="accent")


@app.command()
def context(
    logs: Optional[Path] = typer.Option(
        None, '--logs', file_okay=True, dir_okay=True, exists=True,
        help="Logs file or directory",
    ),
    path: Optional[Path] = typer.Option(
        None, '--path', file_okay=True, dir_okay=True, exists=True,
        help="Alias of --logs for convenience",
    ),
    workspace: Optional[Path] = typer.Option(
        None, '--workspace', file_okay=False, dir_okay=True, exists=True,
        help="Workspace folder (used to default logs to <ws>/logs)",
    ),
    use_lm: bool = typer.Option(True, '--use-lm/--no-lm', help="Use LLM to enhance context"),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama', help="Use Ollama (OpenAI-compatible) backend"),
    model: Optional[str] = typer.Option("deepseek-coder:1.3b", '--model', help="Model name (e.g., llama3)"),
    base_url: Optional[str] = typer.Option(None, '--base-url', help="Override base URL for OpenAI-compatible server"),
    api_key: Optional[str] = typer.Option(None, '--api-key', help="API key; for Ollama any string is fine"),
    force_json: bool = typer.Option(False, '--force-json', help="Force simple JSON outputs; skip structured-outputs"),
    structured: bool = typer.Option(False, '--structured', help="Prefer structured-outputs when available (overrides --force-json)"),
):
    settings = get_settings()
    ws = workspace or Path.cwd()
    target = logs or path or (ws / "logs")
    # Runtime toggle for adapter behavior
    if structured:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'false'
    elif force_json:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'true'
    bundle, count = load_logs([target])
    if not bundle:
        console.print("[yellow]No logs found.[/yellow]")
        raise typer.Exit(0)

    _print_header("Log Key Events")
    key = extract_key_events(bundle)
    console.print(Panel.fit(key, title="Extracted Events", border_style="magenta"))

    if not use_lm or settings.local_mode:
        console.print("[dim]LOCAL_MODE or --no-lm: showing heuristics only.[/dim]")
        raise typer.Exit(0)

    lm = _maybe_configure_lm(use_lm, ollama, model, base_url, api_key)
    if lm is None:
        console.print("[yellow]No LM configured; skipping enhanced context.[/yellow]")
        raise typer.Exit(0)

    _print_header("Enhanced Context (DSPy)")
    builder = ContextBuilder()
    pred = builder(task="Summarize logs for debugging", logs_preview=key)
    console.print(Panel.fit(pred.context, title="Context", border_style="cyan"))
    console.print(Panel.fit(pred.key_points, title="Key Points", border_style="green"))


@app.command()
def tree(
    root: Optional[Path] = typer.Option(None, '--root', dir_okay=True, exists=True, help="Root folder"),
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Default root if --root not given"),
    depth: int = typer.Option(2, '--depth', min=1, help="Max depth to display"),
    hidden: bool = typer.Option(False, '--hidden/--no-hidden', help="Show hidden files"),
):
    base = root or workspace
    output = _render_tree(base, max_depth=depth, show_hidden=hidden)
    console.print(output)


@app.command()
def codectx(
    path: Path = typer.Option(..., '--path', exists=True, help="File or directory to summarize"),
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace root for relative paths"),
    use_lm: bool = typer.Option(True, '--use-lm/--no-lm', help="Use LLM to summarize"),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama', help="Use Ollama for LLM"),
    model: Optional[str] = typer.Option("deepseek-coder:1.3b", '--model', help="Model name for LLM"),
    base_url: Optional[str] = typer.Option(None, '--base-url', help="Override base URL"),
    api_key: Optional[str] = typer.Option(None, '--api-key', help="API key"),
):
    # Runtime toggle for adapter behavior
    if os.environ.get('DSPY_FORCE_JSON_OBJECT') is None:
        # Allow per-invocation override only when not explicitly set in env
        pass
    target = path if path.is_absolute() else (workspace / path)
    snap = build_code_snapshot(target)
    _print_header("Code Snapshot")
    console.print(Panel.fit(snap[:8000] + ("\n..." if len(snap) > 8000 else ""), title=str(target), border_style="magenta"))
    if not use_lm:
        raise typer.Exit(0)
    lm = _maybe_configure_lm(True, ollama, model, base_url, api_key)
    if lm is None:
        raise typer.Exit(0)
    _print_header("Code Context (DSPy)")
    cc = CodeContext()
    out = cc(snapshot=snap, ask="Summarize key components, APIs, and likely modification points.")
    console.print(Panel.fit(out.summary, title="Summary", border_style="cyan"))
    console.print(Panel.fit(out.bullets, title="Bullets", border_style="green"))


@app.command()
def dataset(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace to sample"),
    logs: Optional[Path] = typer.Option(None, '--logs', file_okay=True, dir_okay=True, exists=True, help="Logs path"),
    out_dir: Path = typer.Option(Path.cwd() / '.dspy_data', '--out', dir_okay=True, help="Output dataset dir"),
    seed: int = typer.Option(42, '--seed', help="Split seed"),
    split: bool = typer.Option(True, '--split/--no-split', help="Write train/val/test splits"),
    dedup: bool = typer.Option(True, '--dedup/--no-dedup', help="De-duplicate rows before splitting"),
    stratify_by: Optional[str] = typer.Option("task_type", '--stratify-by', help="Field to stratify by (e.g., task_type)"),
):
    _print_header("Build Dataset")
    if split:
        paths = bootstrap_datasets_with_splits(workspace, logs, out_dir, seed=seed, dedup=dedup, stratify_by=stratify_by)
        console.print(Panel.fit(f"Wrote splits to {out_dir}/splits\n{paths}", title="dataset", border_style="green"))
    else:
        paths = bootstrap_datasets(workspace, logs, out_dir)
        console.print(Panel.fit(f"Wrote raw JSONL to {out_dir}\n{paths}", title="dataset", border_style="green"))
    console.print(Panel.fit("Flags: --workspace --logs --out --seed --split/--no-split --dedup/--no-dedup --stratify-by", title="usage", border_style="dim"))


@app.command()
def tail_metrics(
    progress_file: Path = typer.Option(..., '--file', exists=False, help="Path to progress.jsonl"),
    interval: float = typer.Option(1.0, '--interval', help="Seconds between refresh"),
    module: str = typer.Option("module", '--module', help="Display label for module"),
    budget: str = typer.Option("light", '--auto', help="GEPA budget label"),
):
    """Follow a training progress JSONL file and render a live dashboard.

    Works with files produced by gepa_train/gepa_orchestrator.
    """
    tr: list[float] = []
    va: list[float] = []
    last_pos = 0
    console.print(Panel.fit(f"Tailing {progress_file}", title="tail-metrics", border_style="accent"))
    with Live(_progress_panel(module, budget, tr, va), refresh_per_second=4, console=console) as live:
        try:
            while True:
                try:
                    if progress_file.exists():
                        with progress_file.open('r') as f:
                            f.seek(last_pos)
                            for line in f:
                                try:
                                    rec = _json.loads(line)
                                    if rec.get('split') == 'val':
                                        va.append(float(rec.get('score', 0.0)))
                                    else:
                                        tr.append(float(rec.get('score', 0.0)))
                                except Exception:
                                    pass
                            last_pos = f.tell()
                    live.update(_progress_panel(module, budget, tr, va, title="Live Metrics"))
                except Exception:
                    pass
                time.sleep(interval)
        except KeyboardInterrupt:
            console.print("[dim]Stopped tailing.[/dim]")
    console.print(Panel.fit("Flags: --file --interval --module --auto", title="usage", border_style="dim"))


# Streaming + Infra helpers

@app.command()
def stream_init(
    out: Path = typer.Option(STREAM_CFG_PATH, '--out', help="Path to write stream config JSON"),
):
    cfg = StreamConfig.default()
    save_stream_cfg(cfg, out)
    console.print(Panel.fit(f"Wrote streaming config to {out}", title="stream config", border_style="accent"))
    console.print(Panel.fit("Edit containers, topics, and infra settings as needed.", title="next", border_style="dim"))


@app.command()
def stream_topics(
    cfg_path: Path = typer.Option(STREAM_CFG_PATH, '--config', exists=True, help="Streaming config path"),
):
    cfg = load_stream_cfg(cfg_path)
    cmds = render_kafka_topic_commands(cfg)
    console.print(Panel.fit("\n".join(cmds), title="kafka topic create commands", border_style="accent"))
    console.print(Panel.fit("Run these on your Kafka broker nodes or with docker-compose.", title="hint", border_style="dim"))


@app.command()
def deploy_topics(
    cfg_path: Path = typer.Option(STREAM_CFG_PATH, '--config', exists=False, help="Streaming config path (optional)"),
    bootstrap: Optional[str] = typer.Option(None, '--bootstrap', help="Kafka bootstrap servers (override)"),
):
    """Print topic creation commands for deployment-related topics (lightweight)."""
    from .streaming_config import StreamConfig
    cfg = None
    if cfg_path.exists():
        try:
            cfg = load_stream_cfg(cfg_path)
        except Exception:
            cfg = None
    cfg = cfg or StreamConfig.default()
    # Filter deploy.* topics
    deploy_topics = [t for t in cfg.kafka.topics if t.name.startswith('deploy.')]  # type: ignore[attr-defined]
    bs = bootstrap or cfg.kafka.bootstrap_servers
    lines = [
        f"kafka-topics --bootstrap-server {bs} --create --topic {t.name} --partitions {t.partitions} --replication-factor {t.replication_factor}"
        for t in deploy_topics
    ]
    console.print(Panel("\n".join(lines), title="deploy topics", border_style="accent"))
    console.print(Panel.fit("Copy/paste to your Kafka environment.", title="hint", border_style="dim"))


@app.command()
def last(
    container: str = typer.Option(..., '--container', help="Container name (e.g., backend, frontend, app)"),
    what: str = typer.Option('all', '--what', help="Which field: summary|plan|key_points|ts|all"),
):
    """Show the latest summary/plan for a container from storage (RedDB if configured)."""
    try:
        from .db.factory import get_storage as _get_storage
    except Exception as e:
        console.print(Panel(str(e), title="storage unavailable", border_style="red")); raise typer.Exit(1)
    st = _get_storage()
    if st is None:
        console.print(Panel("No storage configured. Set REDDB_URL or use --db reddb in 'up'.", title="storage", border_style="yellow"))
        raise typer.Exit(1)
    pref = f"last:{container}:"
    try:
        data = {
            'summary': st.get(pref + 'summary'),
            'key_points': st.get(pref + 'key_points'),
            'plan': st.get(pref + 'plan'),
            'ts': st.get(pref + 'ts'),
        }
    except Exception as e:
        console.print(Panel(str(e), title="read failed", border_style="red")); raise typer.Exit(1)
    if what != 'all':
        val = data.get(what)
        if val is None:
            console.print(Panel("(no data)", title=f"{container}:{what}", border_style="yellow")); raise typer.Exit(0)
        console.print(Panel(str(val), title=f"{container}:{what}", border_style="cyan")); raise typer.Exit(0)
    console.print(Panel(str(data.get('summary') or '(no summary)'), title=f"{container}:summary", border_style="cyan"))
    console.print(Panel(str(data.get('key_points') or '(no key points)'), title=f"{container}:key_points", border_style="green"))
    console.print(Panel(str(data.get('plan') or '(no plan)'), title=f"{container}:plan", border_style="blue"))
    console.print(Panel(str(data.get('ts') or '(no ts)'), title=f"{container}:ts", border_style="magenta"))


@app.command()
def spark_script(
    out: Path = typer.Option(Path("scripts/streaming/spark_logs.py"), '--out', help="Path to write PySpark job"),
):
    out.parent.mkdir(parents=True, exist_ok=True)
    code = '''#!/usr/bin/env python3
from pyspark.sql import SparkSession, functions as F
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bootstrap', default='localhost:9092')
    ap.add_argument('--raw-topic', required=True)
    ap.add_argument('--ctx-topic', required=True)
    ap.add_argument('--checkpoint', default='.dspy_checkpoints/spark_logs')
    args = ap.parse_args()

    spark = SparkSession.builder.appName('dspy-stream-logs').getOrCreate()
    df = (spark
          .readStream
          .format('kafka')
          .option('kafka.bootstrap.servers', args.bootstrap)
          .option('subscribe', args.raw_topic)
          .load())
    # Basic transform: parse value, extract error keywords, keep recent lines
    val = df.selectExpr("CAST(value AS STRING) AS line", "timestamp")
    # Example heuristic: keep ERROR/WARN/Traceback windows
    hits = val.where(F.lower(F.col('line')).rlike('error|warn|traceback|exception'))
    agg = (hits.groupBy(F.window('timestamp', '30 seconds'))
               .agg(F.collect_list('line').alias('lines'))
               .select(F.to_json(F.struct(F.col('lines').alias('ctx'))).alias('value')))
    q = (agg.writeStream
             .format('kafka')
             .option('kafka.bootstrap.servers', args.bootstrap)
             .option('topic', args.ctx_topic)
             .option('checkpointLocation', args.checkpoint)
             .outputMode('update')
             .start())
    q.awaitTermination()

if __name__ == '__main__':
    main()
'''
    out.write_text(code)
    console.print(Panel.fit(f"Wrote PySpark job to {out}", title="spark", border_style="accent"))
    console.print(Panel.fit("Example: spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 scripts/streaming/spark_logs.py --raw-topic logs.raw.backend --ctx-topic logs.ctx.backend", title="run", border_style="dim"))


@app.command()
def k8s_render(
    cfg_path: Path = typer.Option(STREAM_CFG_PATH, '--config', exists=True),
    out_dir: Path = typer.Option(Path('deploy/k8s'), '--out')
):
    cfg = load_stream_cfg(cfg_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Minimal Deployment per container topic (worker)
    for ct in cfg.containers:
        name = f"dspy-worker-{ct.container}"
        dep = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {'name': name, 'namespace': cfg.k8s.namespace},
            'spec': {
                'replicas': cfg.k8s.replicas,
                'selector': {'matchLabels': {'app': name}},
                'template': {
                    'metadata': {'labels': {'app': name}},
                    'spec': {
                        'containers': [{
                            'name': 'agent',
                            'image': cfg.k8s.image,
                            'env': [
                                {'name': 'KAFKA_BOOTSTRAP', 'value': cfg.kafka.bootstrap_servers},
                                {'name': 'DSPY_FORCE_JSON_OBJECT', 'value': 'true'},
                                {'name': 'TOPIC', 'value': ct.container},
                            ],
                            'args': ['dspy-agent', 'worker', '--topic', ct.container],
                            'resources': {'requests': cfg.k8s.resources, 'limits': cfg.k8s.resources},
                        }]
                    }
                }
            }
        }
        p = out_dir / f"{name}.yaml"
        try:
            try:
                import yaml as _y  # type: ignore
            except Exception:
                _y = None  # type: ignore
            if _y is not None:
                p.write_text(_y.safe_dump(dep, sort_keys=False))
            else:
                raise RuntimeError('yaml not available')
        except Exception:
            # fallback to json
            import json as _j
            p.write_text(_j.dumps(dep, indent=2))
        console.print(f"[green]Rendered {p}")
    console.print(Panel.fit(f"Render complete in {out_dir}", title="k8s", border_style="accent"))


@app.command()
def worker(
    topic: str = typer.Option(..., '--topic', help="Container/topic name (e.g., backend, frontend)"),
    bootstrap: Optional[str] = typer.Option(None, '--bootstrap', help="Kafka bootstrap servers"),
    group: Optional[str] = typer.Option(None, '--group', help="Kafka consumer group"),
    config: Path = typer.Option(STREAM_CFG_PATH, '--config', exists=False, help="Streaming config path"),
):
    """Run a streaming worker for a topic. Note: requires Kafka client (install confluent-kafka)."""
    # Minimal stub to avoid hard dependency; provide friendly guidance
    bs = bootstrap
    grp = group
    if config.exists():
        try:
            cfg = load_stream_cfg(config)
            bs = bs or cfg.kafka.bootstrap_servers
            grp = grp or cfg.kafka.group_id
        except Exception:
            pass
    console.print(Panel.fit(
        f"topic={topic}\nbootstrap={bs or 'localhost:9092'}\ngroup={grp or 'dspy-code'}\n",
        title="worker config", border_style="accent"
    ))
    try:
        params = KafkaParams(
            bootstrap=bs or 'localhost:9092',
            group=grp or 'dspy-code',
            in_topic=f'logs.ctx.{topic}',
            out_topic=f'agent.results.{topic}',
            container=topic,
        )
        loop = WorkerLoop(params)
        loop.run()
    except RuntimeError as e:
        console.print(Panel(str(e), title="worker", border_style="red"))
        raise typer.Exit(1)


@app.command()
def autodiscover(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Repo root to scan"),
    out: Path = typer.Option(STREAM_CFG_PATH, '--out', help="Path to write stream config JSON"),
):
    discs = autodiscover_logs(workspace)
    if not discs:
        console.print(Panel("No log files found. Ensure logs/ or *.log exist.", title="autodiscover", border_style="yellow"))
        raise typer.Exit(1)
    # Update default config with discovered containers/services
    cfg = StreamConfig.default()
    containers: Dict[str, List[str]] = {}
    for d in discs:
        containers.setdefault(d.container, []).append(d.service)
    cfg.containers = [type('CT', (), {'container': k, 'services': v}) for k, v in containers.items()]  # type: ignore
    save_stream_cfg(cfg, out)
    console.print(Panel.fit(f"Discovered {len(discs)} container(s). Wrote {out}", title="autodiscover", border_style="accent"))


@app.command()
def up(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Repo root"),
    train: bool = typer.Option(True, '--train/--no-train', help="Enable streaming trainer (context dataset)"),
    db: str = typer.Option("auto", '--db', help="Storage backend: auto|none|reddb"),
    status: bool = typer.Option(True, '--status/--no-status', help="Run status HTTP server"),
    status_host: str = typer.Option("0.0.0.0", '--status-host', help="Status server host"),
    status_port: int = typer.Option(8765, '--status-port', help="Status server port (0 to disable)"),
):
    """Start local tailers + aggregators + workers (no Kafka needed). Single command dev stack."""
    _print_banner()
    # Optionally wire storage + kafka for persistence
    storage = None
    kafka = None
    try:
        from .db.factory import get_storage as _get_storage
        if db.lower() == "none":
            storage = None
        elif db.lower() == "reddb":
            # Force RedDB regardless of env
            from .db.reddb import RedDBStorage
            from .config import get_settings
            s = get_settings()
            if not s.reddb_url:
                console.print("[yellow]--db reddb specified but REDDB_URL is not set; falling back to in-memory stub.[/yellow]")
            storage = RedDBStorage(url=s.reddb_url, namespace=s.reddb_namespace or "dspy")
        else:
            storage = _get_storage()
    except Exception:
        storage = None
    try:
        kafka = get_kafka_logger()
        if kafka is None:
            console.print("[dim]Kafka logging disabled (set KAFKA_BOOTSTRAP_SERVERS to enable).[/dim]")
    except Exception:
        kafka = None
    threads, bus = start_local_stack(workspace, None, storage=storage, kafka=kafka)
    if train:
        # Discover containers from .dspy_stream.json or autodiscovery
        try:
            cfg = load_stream_cfg(STREAM_CFG_PATH) if STREAM_CFG_PATH.exists() else None
        except Exception:
            cfg = None
        containers = [getattr(ct, 'container') for ct in getattr(cfg, 'containers', [])] if cfg else [d.container for d in autodiscover_logs(workspace)]
        from .streaming_runtime import Trainer
        trainer = Trainer(workspace, bus, containers, min_batch=3, interval_sec=60.0)
        trainer.start(); threads.append(trainer)
    if not threads:
        console.print(Panel("No tailers/workers started. Run 'dspy-agent autodiscover' and verify logs exist.", title="up", border_style="red"))
        raise typer.Exit(1)
    # Start status server (non-blocking)
    if status and status_port > 0:
        try:
            th = start_status_server(status_host, status_port, workspace)
            threads.append(th)
            console.print(f"[green]Status server: http://{status_host}:{status_port}[/green]")
        except Exception as e:
            console.print(Panel(f"Status server failed: {e}", title="status", border_style="yellow"))
    console.print(Panel.fit(f"Started {len(threads)} threads (tailers/aggregators/workers). Ctrl-C to stop.", title="local stack", border_style="cyan"))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("[dim]Stopping...[/dim]")


# -----------------------------
# Lightweight Containers (Local)
# -----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _validate_paths(ws: Path, logs: Optional[Path]) -> list[str]:
    errs: list[str] = []
    # Ensure workspace directory exists
    if not ws.exists():
        try:
            ws.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errs.append(f"failed to create workspace: {ws} ({e})")
    elif not ws.is_dir():
        errs.append(f"workspace is not a directory: {ws}")

    # Ensure logs directory exists if provided
    if logs:
        if not logs.exists():
            try:
                logs.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errs.append(f"failed to create logs dir: {logs} ({e})")
        elif not logs.is_dir():
            errs.append(f"logs path is not a directory: {logs}")
    return errs


def _docker_available() -> bool:
    import shutil
    return shutil.which('docker') is not None


def _compose_yaml(image: str, host_ws: Path, host_logs: Optional[Path], db_backend: str) -> str:
    logs_map = f"\n      - {host_logs.resolve()}:/workspace/logs:ro" if host_logs else ""
    return f"""
services:
  dspy-agent:
    image: {image}
    build:
      context: ../../
      dockerfile: docker/lightweight/Dockerfile
    environment:
      - LOCAL_MODE=false
      - USE_OLLAMA=true
      - DB_BACKEND={db_backend}
      - REDDB_URL
      - REDDB_NAMESPACE=dspy
      - REDDB_TOKEN
      - MODEL_NAME=deepseek-coder:1.3b
      - OPENAI_API_KEY=
      - OPENAI_BASE_URL=http://ollama:11434
      - OLLAMA_MODEL=deepseek-coder:1.3b
      - OLLAMA_API_KEY=
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - KAFKA_CLIENT_ID=dspy-agent
      - KAFKA_TOPIC_PREFIX
    working_dir: /app
    # ENTRYPOINT is dspy-agent; do not repeat binary in command
    command: ["up", "--workspace", "/workspace", "--db", "{db_backend}", "--status", "--status-port", "8765"]
    volumes:
      - {host_ws.resolve()}:/workspace:rw{logs_map}
    ports:
      - "127.0.0.1:8765:8765"  # reserved for future HTTP status
    restart: unless-stopped
    depends_on:
      - ollama
      - kafka

  ollama:
    image: ollama/ollama:latest
    entrypoint: ["/bin/sh", "-lc"]
    command: |
      ollama serve &
      sleep 3;
      ollama pull deepseek-coder:1.3b || true;
      wait
    ports:
      - "127.0.0.1:11435:11434"
    volumes:
      - ollama:/root/.ollama

  zookeeper:
    image: bitnami/zookeeper:3.9
    environment:
      - ALLOW_ANONYMOUS_LOGIN=yes
    ports:
      - "127.0.0.1:2181:2181"

  kafka:
    image: bitnami/kafka:3.6
    depends_on:
      - zookeeper
    environment:
      - KAFKA_CFG_ZOOKEEPER_CONNECT=zookeeper:2181
      - ALLOW_PLAINTEXT_LISTENER=yes
      - KAFKA_LISTENERS=PLAINTEXT://:9092
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:9092
      - KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      - KAFKA_INTER_BROKER_LISTENER_NAME=PLAINTEXT
    ports:
      - "127.0.0.1:9092:9092"

volumes:
  ollama: {{}}
""".strip()


def _dockerfile() -> str:
    return """
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc git curl librdkafka-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY dspy_agent /app/dspy_agent

RUN pip install --no-cache-dir uv && \
    uv pip install --system . && \
    pip install --no-cache-dir confluent-kafka || true

ENTRYPOINT ["dspy-agent"]
""".strip()


@app.command("lightweight_init")
def lightweight_init(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=False, help="Host workspace to mount (created if missing; falls back to CWD if unwritable)"),
    logs: Optional[Path] = typer.Option(None, '--logs', help="Host logs directory to mount read-only (optional; created if missing; disabled if unwritable)", file_okay=False, dir_okay=True),
    out_dir: Path = typer.Option(LIGHTWEIGHT_DIR, '--out-dir', help="Where to write Dockerfile/compose"),
    db: str = typer.Option("auto", '--db', help="Storage backend: auto|none|reddb"),
):
    # Ensure output dir
    _ensure_dir(out_dir)

    # Resolve workspace/logs with graceful fallbacks
    ws = workspace
    lg = logs
    warns: list[str] = []
    try:
        if not ws.exists():
            ws.mkdir(parents=True, exist_ok=True)
        if not ws.is_dir():
            raise RuntimeError("not a directory")
    except Exception as e:
        fallback = Path.cwd()
        warns.append(f"workspace not usable: {ws} ({e}); falling back to {fallback}")
        ws = fallback
    if lg is not None:
        try:
            if not lg.exists():
                lg.mkdir(parents=True, exist_ok=True)
            if not lg.is_dir():
                raise RuntimeError("not a directory")
        except Exception as e:
            warns.append(f"logs not usable: {lg} ({e}); disabling logs mount")
            lg = None

    df = out_dir / 'Dockerfile'
    dc = out_dir / 'docker-compose.yml'
    try:
        df.write_text(_dockerfile())
        dc.write_text(_compose_yaml(image='dspy-lightweight:latest', host_ws=ws, host_logs=lg, db_backend=db))
    except Exception as e:
        console.print(Panel(str(e), title="write failed", border_style="red"))
        raise typer.Exit(1)

    if warns:
        console.print(Panel("\n".join(f"- {w}" for w in warns), title="adjustments", border_style="yellow"))
    console.print(Panel.fit(
        f"Wrote:\n- {df}\n- {dc}", title="lightweight init", border_style="green"
    ))
    next_steps = [
        "1) Review docker/lightweight/docker-compose.yml and adjust env as needed (REDDB_URL, OPENAI_BASE_URL, etc.)",
        "2) Build the image: docker compose -f docker/lightweight/docker-compose.yml build",
        "3) Start the stack: docker compose -f docker/lightweight/docker-compose.yml up -d",
        "4) Logs: docker compose -f docker/lightweight/docker-compose.yml logs -f dspy-agent",
    ]
    console.print(Panel("\n".join(next_steps), title="next steps", border_style="cyan"))
    if not _docker_available():
        console.print("[yellow]Docker not detected in PATH. Install Docker Desktop or CLI first.[/yellow]")
    if db.lower() == 'reddb' and not os.getenv('REDDB_URL'):
        console.print("[yellow]You selected --db reddb but REDDB_URL is not set. The agent will fallback to in-memory until you provide REDDB_URL.[/yellow]")


@app.command("lightweight_up")
def lightweight_up(
    compose: Path = typer.Option(LIGHTWEIGHT_DIR / 'docker-compose.yml', '--compose', exists=False, help="Path to compose file"),
    build: bool = typer.Option(True, '--build/--no-build', help="Build image before up"),
):
    logger = DeploymentLogger(workspace=Path.cwd(), name="lightweight")
    if not compose.exists():
        msg = f"Compose file not found: {compose}. Run 'dspy-agent lightweight_init' first."
        logger.event("up", "error", msg)
        console.print(Panel(msg, title="lightweight up", border_style="red"))
        logger.close()
        raise typer.Exit(1)
    if not _docker_available():
        logger.event("up", "warn", "Docker not detected in PATH")
        console.print("[yellow]Docker not detected in PATH. Please run the following manually:[/yellow]")
        console.print(f"docker compose -f {compose} {'build && ' if build else ''}up -d")
        logger.close()
        raise typer.Exit(1)
    try:
        logger.status("building")
        logger.set_compose_hash(compose)
        if build:
            code = logger.run_stream(["docker", "compose", "-f", str(compose), "build"], phase="build")
            if code != 0:
                logger.status("error"); logger.close(); raise typer.Exit(1)
        logger.status("up")
        code = logger.run_stream(["docker", "compose", "-f", str(compose), "up", "-d"], phase="up")
        if code != 0:
            logger.status("error"); logger.close(); raise typer.Exit(1)
        console.print("[green]Lightweight stack is up.[/green]")
        logger.event("up", "info", "Lightweight stack is up")
    except Exception as e:
        logger.status("error")
        console.print(Panel(str(e), title="docker compose failed", border_style="red"))
    finally:
        logger.close()
    raise typer.Exit(0)


@app.command("lightweight_down")
def lightweight_down(
    compose: Path = typer.Option(LIGHTWEIGHT_DIR / 'docker-compose.yml', '--compose', exists=False),
):
    logger = DeploymentLogger(workspace=Path.cwd(), name="lightweight")
    if not compose.exists():
        msg = f"Compose file not found: {compose}"
        logger.event("down", "error", msg)
        console.print(Panel(msg, title="lightweight down", border_style="red"))
        logger.close()
        raise typer.Exit(1)
    if not _docker_available():
        logger.event("down", "warn", "Docker not detected in PATH")
        console.print("[yellow]Docker not detected. Run manually:[/yellow]")
        console.print(f"docker compose -f {compose} down")
        logger.close()
        raise typer.Exit(1)
    try:
        logger.status("down")
        code = logger.run_stream(["docker", "compose", "-f", str(compose), "down"], phase="down")
        if code != 0:
            logger.status("error"); logger.close(); raise typer.Exit(1)
        console.print("[green]Lightweight stack stopped.[/green]")
        logger.event("down", "info", "Lightweight stack stopped")
    except Exception as e:
        logger.status("error")
        console.print(Panel(str(e), title="docker compose failed", border_style="red"))
    finally:
        logger.close()
    raise typer.Exit(0)


@app.command("lightweight_status")
def lightweight_status(
    compose: Path = typer.Option(LIGHTWEIGHT_DIR / 'docker-compose.yml', '--compose', exists=False),
):
    logger = DeploymentLogger(workspace=Path.cwd(), name="lightweight")
    if not compose.exists():
        msg = f"Compose file not found: {compose}"
        logger.event("status", "error", msg)
        console.print(Panel(msg, title="lightweight status", border_style="red"))
        logger.close()
        raise typer.Exit(1)
    if not _docker_available():
        logger.event("status", "warn", "Docker not detected in PATH")
        console.print("[yellow]Docker not detected. Run manually:[/yellow]")
        console.print(f"docker compose -f {compose} ps")
        logger.close()
        raise typer.Exit(1)
    try:
        code = logger.run_stream(["docker", "compose", "-f", str(compose), "ps"], phase="status")
        if code != 0:
            logger.status("error")
    except Exception as e:
        logger.status("error")
        console.print(Panel(str(e), title="docker compose failed", border_style="red"))
    finally:
        logger.close()
    raise typer.Exit(0)


@app.command("lightweight_build")
def lightweight_build(
    compose: Path = typer.Option(LIGHTWEIGHT_DIR / 'docker-compose.yml', '--compose', exists=False, help="Path to compose file"),
    no_restart: bool = typer.Option(False, '--no-restart', help="Only build, do not restart containers"),
):
    """Rebuild the lightweight image and optionally restart, with comprehensive deployment logs.

    Logs are written to logs/deployments/<timestamp>.log and appended to the RedDB stream
    'deploy.logs.lightweight' when configured.
    """
    logger = DeploymentLogger(workspace=Path.cwd(), name="lightweight")
    if not compose.exists():
        msg = f"Compose file not found: {compose}. Run 'dspy-agent lightweight_init' first."
        logger.event("build", "error", msg)
        console.print(Panel(msg, title="lightweight build", border_style="red"))
        logger.close(); raise typer.Exit(1)
    if not _docker_available():
        logger.event("build", "warn", "Docker not detected in PATH")
        console.print("[yellow]Docker not detected in PATH. Run manually:[/yellow]")
        console.print(f"docker compose -f {compose} build && {'true' if no_restart else f'docker compose -f {compose} up -d'}")
        logger.close(); raise typer.Exit(1)
    try:
        logger.status("building")
        logger.set_compose_hash(compose)
        code = logger.run_stream(["docker", "compose", "-f", str(compose), "build"], phase="build")
        if code != 0:
            logger.status("error"); logger.close(); raise typer.Exit(1)
        if not no_restart:
            logger.status("up")
            code = logger.run_stream(["docker", "compose", "-f", str(compose), "up", "-d"], phase="up")
            if code != 0:
                logger.status("error"); logger.close(); raise typer.Exit(1)
        logger.status("done")
        logger.event("build", "info", "Build complete")
        console.print("[green]Lightweight build complete.[/green]")
    except Exception as e:
        logger.status("error")
        console.print(Panel(str(e), title="lightweight build failed", border_style="red"))
    finally:
        logger.close()
    raise typer.Exit(0)


@app.command()
def chat(
    task: str = typer.Argument(..., help="Natural instruction/task"),
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True),
    logs: Optional[Path] = typer.Option(None, '--logs', file_okay=True, dir_okay=True, exists=True),
    steps: int = typer.Option(4, '--steps', help="Max auto steps"),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama'),
    model: Optional[str] = typer.Option("deepseek-coder:1.3b", '--model'),
    base_url: Optional[str] = typer.Option(None, '--base-url'),
    api_key: Optional[str] = typer.Option(None, '--api-key'),
    force_json: bool = typer.Option(False, '--force-json', help="Force simple JSON outputs; skip structured-outputs"),
    structured: bool = typer.Option(False, '--structured', help="Prefer structured-outputs when available (overrides --force-json)"),
    approval: Optional[str] = typer.Option(None, '--approval', help='Tool approval mode: auto|manual'),
):
    if structured:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'false'
    elif force_json:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'true'
    # Local session variables
    ws = workspace
    logs_path = logs or (ws / 'logs')
    provider_is_ollama = ollama
    model_name = model
    base = base_url
    key = api_key
    # Resolve approval mode
    settings = get_settings()
    approval_mode = (approval or getattr(settings, 'tool_approval_mode', 'auto') or 'auto').lower()
    if approval_mode not in {"auto", "manual"}:
        approval_mode = "auto"

    console.print(Panel.fit(f"Workspace: {ws}\nLogs: {logs_path}\nApproval: {approval_mode}", title="chat session", border_style="cyan"))
    # Reuse orchestrate_chain
    # Since orchestrate_chain is nested inside start, replicate minimal logic here
    def _extract_targets_from_query(nl: str, k: int = 5) -> list[str]:
        toks = tokenize(nl)
        seen = []
        for t in toks:
            if len(t) > 2 and t not in seen:
                seen.append(t)
            if len(seen) >= k:
                break
        return seen
    targets = _extract_targets_from_query(task)
    history_summary = ""
    last_tool = None
    last_args = None
    for step in range(1, steps + 1):
        state = f"workspace={ws}, logs={logs_path} | history: {history_summary[:4000]}"
        tool = None; args = {}
        lm = _maybe_configure_lm(True, provider_is_ollama, model_name, base, key)
        if lm is not None:
            try:
                orch = Orchestrator(); pred = orch(query=task, state=state)
                tool = (pred.tool or "").strip(); import json as _json; args = _json.loads(pred.args_json or "{}")
            except Exception:
                tool = None; args = {}
        if not tool:
            tool = "context" if logs_path.exists() else "codectx"; args = {}
        console.print(Panel.fit(f"{tool} {args}", title=f"Step {step}: action", border_style="yellow"))
        if last_tool == tool and last_args == args:
            console.print("[dim]No new action; stopping.[/dim]")
            break
        last_tool, last_args = tool, dict(args)
        try:
            # If manual approval is required, confirm before executing tool
            if approval_mode == "manual":
                approved = typer.confirm(f"Approve tool '{tool}' with args {args}?", default=False)
                if not approved:
                    console.print("[dim]Tool execution skipped by user.[/dim]")
                    continue
            # dispatch_tool is nested in start; inline minimal relevant calls
            if tool == "context":
                bundle, _ = load_logs([logs_path]); key = extract_key_events(bundle) if bundle else ""; _print_header("Log Key Events"); console.print(Panel.fit(key, title="Extracted Events", border_style="magenta"))
            elif tool == "codectx":
                snap = build_code_snapshot(ws); _print_header("Code Snapshot"); console.print(Panel.fit(snap[:8000] + ("\n..." if len(snap)>8000 else ""), title=str(ws), border_style="magenta"))
            elif tool == "grep":
                hits = search_text(ws, args.get("pattern", task), regex=True); [console.print(f"{h.path}:{h.line_no}: {h.line}") for h in hits[:200]]
            elif tool == "esearch":
                try: meta, items = load_index(ws)
                except FileNotFoundError: meta, items = build_index(ws, smart=True); save_index(ws, meta, items)
                hits = semantic_search(task, meta, items, top_k=5)
                for score,it in hits:
                    p = Path(it.path); text = p.read_text(errors="ignore"); lines = text.splitlines(); s=max(1,it.start_line-3); e=min(len(lines), it.end_line+3); seg = "\n".join(lines[s-1:e]); console.print(Panel(seg, title=f"{p} score={score:.3f} lines {s}-{e}", border_style="blue"))
            elif tool == "extract":
                fp = (ws / args.get("file",".")).resolve(); sym=args.get("symbol"); rx=args.get("regex");
                if sym and fp.suffix==".py": res = python_extract_symbol(fp, sym); 
                elif rx: hits=file_search(fp, rx, regex=True); text=fp.read_text(errors="ignore"); s,e,seg=extract_context(text, hits[0].line_no, before=3, after=3); console.print(Panel(seg, title=f"{fp} lines {s}-{e}", border_style="green"))
            else:
                # default to context snapshot
                bundle, _ = load_logs([logs_path]); key = extract_key_events(bundle) if bundle else ""; console.print(Panel.fit(key or "(no logs)", title="Key Events", border_style="magenta"))
        except Exception as e:
            console.print(Panel(str(e), title=f"agent failed ({tool})", border_style="red")); break
        try:
            outcome = evaluate_tool_choice(tool, args, workspace=ws, logs_path=logs_path, targets=targets); piece=f"{tool}: score={outcome.score:.2f}; {outcome.evidence}"
        except Exception:
            piece=f"{tool}: done"
        history_summary = (history_summary + " | " + piece).strip()
        if step >= 2 and ("score=" in piece and float(piece.split("score=")[1].split(";")[0]) >= 1.2):
            break
def run(
    task: str = typer.Argument(..., help="What do you want to accomplish?"),
    logs: Optional[Path] = typer.Option(
        None, '--logs', file_okay=True, dir_okay=True, exists=True,
        help="Logs file or directory",
    ),
    path: Optional[Path] = typer.Option(
        None, '--path', file_okay=True, dir_okay=True, exists=True,
        help="Alias of --logs for convenience",
    ),
    workspace: Optional[Path] = typer.Option(
        None, '--workspace', file_okay=False, dir_okay=True, exists=True,
        help="Workspace folder (used to default logs to <ws>/logs)",
    ),
    use_lm: bool = typer.Option(True, '--use-lm/--no-lm', help="Use LLM for planning"),
    ollama: bool = typer.Option(False, '--ollama/--no-ollama', help="Use Ollama (OpenAI-compatible) backend"),
    model: Optional[str] = typer.Option(None, '--model', help="Model name (e.g., llama3)"),
    base_url: Optional[str] = typer.Option(None, '--base-url', help="Override base URL for OpenAI-compatible server"),
    api_key: Optional[str] = typer.Option(None, '--api-key', help="API key; for Ollama any string is fine"),
):
    settings = get_settings()
    ws = workspace or Path.cwd()
    target = logs or path or (ws / "logs")
    bundle, count = load_logs([target])
    key = extract_key_events(bundle) if bundle else ""

    if not use_lm or settings.local_mode:
        _print_header("Heuristic Context")
        console.print(Panel.fit(key or "(no logs)", title="Key Events", border_style="magenta"))
        console.print("[yellow]LOCAL_MODE or --no-lm: cannot generate a plan.[/yellow]")
        raise typer.Exit(0)

    lm = _maybe_configure_lm(use_lm, ollama, model, base_url, api_key)
    if lm is None:
        _print_header("Heuristic Context")
        console.print(Panel.fit(key or "(no logs)", title="Key Events", border_style="magenta"))
        console.print("[yellow]No LM configured; cannot generate a plan.[/yellow]")
        raise typer.Exit(1)

    builder = ContextBuilder()
    ctx = builder(task=task, logs_preview=key)

    _print_header("Agent Plan")
    agent = TaskAgent()
    out = agent(task=task, context=f"{ctx.context}\n\n{ctx.key_points}")

    console.print(Panel.fit(task, title="Task", border_style="white"))
    console.print(Panel.fit(ctx.context, title="Context", border_style="cyan"))
    console.print(Panel.fit(ctx.key_points, title="Key Points", border_style="green"))
    console.print(Panel.fit(out.plan, title="Proposed Plan", border_style="blue"))
    console.print(Panel.fit(out.commands or "(no commands)", title="Suggested Commands", border_style="yellow"))


@app.command()
def index(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace to index"),
    glob: Optional[List[str]] = typer.Option(None, '--glob', help="Include glob. Can repeat."),
    exclude: Optional[List[str]] = typer.Option(None, '--exclude', help="Exclude glob. Can repeat."),
    lines: int = typer.Option(200, '--chunk-lines', help="Lines per chunk (non-Python)"),
    smart: bool = typer.Option(True, '--smart/--no-smart', help="Code-aware chunking (Python)"),
):
    inc = list(glob) if glob else None
    exc = list(exclude) if exclude else None
    _print_header("Building index")
    meta, items = build_index(workspace, include_globs=inc, exclude_globs=exc, lines_per_chunk=lines, smart=smart)
    out_dir = save_index(workspace, meta, items)
    console.print(f"[green]Indexed {len(items)} chunks from {workspace}. Saved to {out_dir}[/green]")


@app.command()
def esearch(
    query: str = typer.Argument(..., help="Semantic query"),
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace to search"),
    k: int = typer.Option(5, '--k', help="Top-K results"),
    context: int = typer.Option(4, '--context', help="Lines of context to show around chunk bounds"),
):
    meta, items = load_index(workspace)
    hits = semantic_search(query, meta, items, top_k=k)
    if not hits:
        console.print("[yellow]No results.[/yellow]")
        raise typer.Exit(0)
    for score, it in hits:
        p = Path(it.path)
        try:
            text = p.read_text(errors="ignore")
            lines = text.splitlines()
            start = max(1, it.start_line - context)
            end = min(len(lines), it.end_line + context)
            seg = "\n".join(lines[start - 1 : end])
        except Exception:
            seg = "(unreadable)"
            start = it.start_line
            end = it.end_line
        title = f"{p}  score={score:.3f}  lines {start}-{end}"
        console.print(Panel(seg, title=title, border_style="blue"))


@app.command()
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
):
    if hf or model.startswith("Qwen/"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            console.print(Panel("Install sentence-transformers>=2.7.0 and transformers>=4.51.0", title="Missing deps", border_style="red"))
            raise typer.Exit(1)
        model_kwargs = {}
        tok_kwargs = {}
        if flash:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        if device:
            model_kwargs["device_map"] = device
        try:
            embedder = SentenceTransformer(model, model_kwargs=model_kwargs or None, tokenizer_kwargs=tok_kwargs or None)
        except Exception as e:
            console.print(Panel(str(e), title="Failed to load HF model", border_style="red"))
            raise typer.Exit(1)
    else:
        try:
            embedder = dspy.Embeddings(model=model, api_base=base_url, api_key=api_key)
        except Exception as e:
            console.print(Panel(str(e), title="Failed to init DSPy Embeddings", border_style="red"))
            raise typer.Exit(1)
    items = build_emb_index(workspace, embedder, lines_per_chunk=lines, smart=smart)
    out_dir = save_emb_index(workspace, items)
    console.print(f"[green]Embedded {len(items)} chunks. Saved to {out_dir}[/green]")


@app.command()
def emb_search(
    query: str = typer.Argument(..., help="Query text"),
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace"),
    model: str = typer.Option(..., '--model', help="Embeddings model (same used for emb-index)"),
    base_url: Optional[str] = typer.Option(None, '--base-url'),
    api_key: Optional[str] = typer.Option(None, '--api-key'),
    hf: bool = typer.Option(False, '--hf/--no-hf', help="Use sentence-transformers (HuggingFace) for local embeddings"),
    device: Optional[str] = typer.Option(None, '--device', help="HF device map (e.g., 'auto' or 'cpu')"),
    flash: bool = typer.Option(False, '--flash/--no-flash', help="Enable flash_attention_2 in HF model_kwargs"),
    k: int = typer.Option(5, '--k'),
    context: int = typer.Option(4, '--context'),
):
    if hf or model.startswith("Qwen/"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            console.print(Panel("Install sentence-transformers>=2.7.0 and transformers>=4.51.0", title="Missing deps", border_style="red"))
            raise typer.Exit(1)
        model_kwargs = {}
        tok_kwargs = {}
        if flash:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        if device:
            model_kwargs["device_map"] = device
        try:
            embedder = SentenceTransformer(model, model_kwargs=model_kwargs or None, tokenizer_kwargs=tok_kwargs or None)
        except Exception as e:
            console.print(Panel(str(e), title="Failed to load HF model", border_style="red"))
            raise typer.Exit(1)
    else:
        try:
            embedder = dspy.Embeddings(model=model, api_base=base_url, api_key=api_key)
        except Exception as e:
            console.print(Panel(str(e), title="Failed to init DSPy Embeddings", border_style="red"))
            raise typer.Exit(1)
    items = load_emb_index(workspace)
    qv = embed_query(embedder, query)
    hits = emb_search_fn(qv, items, top_k=k)
    if not hits:
        console.print("[yellow]No results.[/yellow]")
        raise typer.Exit(0)
    for score, it in hits:
        p = Path(it.path)
        try:
            text = p.read_text(errors="ignore")
            lines = text.splitlines()
            start = max(1, it.start_line - context)
            end = min(len(lines), it.end_line + context)
            seg = "\n".join(lines[start - 1 : end])
        except Exception:
            seg = "(unreadable)"
            start = it.start_line
            end = it.end_line
        title = f"{p}  score={score:.3f}  lines {start}-{end}"
        console.print(Panel(seg, title=title, border_style="cyan"))


@app.command()
def gepa_train(
    module: str = typer.Option(..., '--module', help="Which module to optimize: context|task|code"),
    train_jsonl: Optional[Path] = typer.Option(None, '--train-jsonl', exists=True, help="Train JSONL"),
    val_jsonl: Optional[Path] = typer.Option(None, '--val-jsonl', exists=True, help="Val JSONL"),
    test_jsonl: Optional[Path] = typer.Option(None, '--test-jsonl', exists=True, help="Test JSONL"),
    dataset_dir: Optional[Path] = typer.Option(None, '--dataset-dir', dir_okay=True, exists=True, help="Dir with {module}_{split}.jsonl"),
    auto: Optional[str] = typer.Option("light", '--auto', help="Budget: light|medium|heavy (mutually exclusive with max_* options)"),
    max_full_evals: Optional[int] = typer.Option(None, '--max-full-evals'),
    max_metric_calls: Optional[int] = typer.Option(None, '--max-metric-calls'),
    log_dir: Optional[Path] = typer.Option(None, '--log-dir', help="Directory to store GEPA logs"),
    track_stats: bool = typer.Option(True, '--track-stats/--no-track-stats'),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama', help="Use Ollama for reflection LM"),
    model: Optional[str] = typer.Option("deepseek-coder:1.3b", '--model', help="Reflection model name"),
    base_url: Optional[str] = typer.Option(None, '--base-url'),
    api_key: Optional[str] = typer.Option(None, '--api-key'),
    save_best: Optional[Path] = typer.Option(None, '--save-best', help="Write best candidate program mapping to this JSON file"),
    force_json: bool = typer.Option(False, '--force-json', help="Force simple JSON outputs; skip structured-outputs"),
    structured: bool = typer.Option(False, '--structured', help="Prefer structured-outputs when available (overrides --force-json)"),
    report_dir: Optional[Path] = typer.Option(None, '--report-dir', help="Directory to write evaluation reports"),
):
    if structured:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'false'
    elif force_json:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'true'
    module = module.lower().strip()
    if module not in {"context", "task", "code"}:
        console.print("[yellow]--module must be one of: context, task, code[/yellow]")
        raise typer.Exit(2)

    # Configure reflection LM
    lm = _maybe_configure_lm(True, ollama, model, base_url, api_key)
    if lm is None:
        console.print("[yellow]No LM configured for reflection. Configure Ollama or pass --no-ollama with OpenAI-compatible envs.[/yellow]")
        raise typer.Exit(1)

    _print_header(f"GEPA Train ({module})")
    # Resolve dataset paths from directory if provided
    if dataset_dir and not train_jsonl:
        cand = dataset_dir / f"{module}_train.jsonl"
        if cand.exists():
            train_jsonl = cand
    if dataset_dir and not val_jsonl:
        cand = dataset_dir / f"{module}_val.jsonl"
        if cand.exists():
            val_jsonl = cand
    if not train_jsonl:
        console.print("[yellow]No --train-jsonl or --dataset-dir provided.[/yellow]")
        raise typer.Exit(2)
    # Stream live training metrics
    if not log_dir:
        log_dir = Path.cwd() / f'.gepa_{module}'
    progress_path = (log_dir / 'progress.jsonl').resolve()
    result: dict = {}
    def _worker():
        try:
            if val_jsonl and val_jsonl.exists():
                result['prog'] = run_gepa_with_val(
                    module=module,
                    train_jsonl=train_jsonl,
                    val_jsonl=val_jsonl,
                    auto=auto,
                    max_full_evals=max_full_evals,
                    max_metric_calls=max_metric_calls,
                    reflection_lm=lm,
                    log_dir=str(log_dir) if log_dir else None,
                    track_stats=track_stats,
                    progress_path=str(progress_path),
                )
            else:
                result['prog'] = run_gepa(
                    module=module,
                    train_jsonl=train_jsonl,
                    auto=auto,
                    max_full_evals=max_full_evals,
                    max_metric_calls=max_metric_calls,
                    reflection_lm=lm,
                    log_dir=str(log_dir) if log_dir else None,
                    track_stats=track_stats,
                    progress_path=str(progress_path),
                )
        except Exception as e:
            result['error'] = e
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    last_pos = 0
    tr: list[float] = []
    va: list[float] = []
    with Live(_progress_panel(module, auto or 'light', tr, va), refresh_per_second=4, console=console) as live:
        while t.is_alive():
            try:
                if progress_path.exists():
                    with open(progress_path, 'r') as f:
                        f.seek(last_pos)
                        for line in f:
                            try:
                                rec = _json.loads(line)
                                if rec.get('split') == 'val':
                                    va.append(float(rec.get('score', 0.0)))
                                else:
                                    tr.append(float(rec.get('score', 0.0)))
                            except Exception:
                                pass
                        last_pos = f.tell()
                live.update(_progress_panel(module, auto or 'light', tr, va))
            except Exception:
                pass
            time.sleep(1)
    if 'error' in result:
        console.print(Panel(str(result['error']), title="GEPA failed", border_style="red"))
        raise typer.Exit(1)
    optimized = result.get('prog')

    console.print("[green]GEPA optimization complete.[/green]")
    # If track_stats, show brief stats
    if getattr(optimized, "detailed_results", None):
        dr = optimized.detailed_results
        try:
            best = dr.best_idx
            agg = dr.val_aggregate_scores[best] if dr.val_aggregate_scores else None
            console.print(f"Best candidate index: {best}, score: {agg}")
        except Exception:
            pass
        if save_best:
            try:
                from dspy.teleprompt.gepa.gepa import DspyGEPAResult  # type: ignore
            except Exception:
                pass
            try:
                best_prog = dr.best_candidate
                save_best.parent.mkdir(parents=True, exist_ok=True)
                Path(save_best).write_text(json.dumps(best_prog, indent=2))
                console.print(f"[green]Saved best candidate program to {save_best}[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not save best candidate: {e}[/yellow]")

    # Evaluate on test set if available
    if not test_jsonl and dataset_dir:
        cand = dataset_dir / f"{module}_test.jsonl"
        if cand.exists():
            test_jsonl = cand
    if test_jsonl and test_jsonl.exists():
        stats = evaluate_on_set(module, optimized, test_jsonl)
        console.print(Panel(f"Test n={int(stats['n'])} avg_score={stats['avg_score']:.3f}", title="Test Eval", border_style="cyan"))
        if report_dir:
            try:
                report_dir.mkdir(parents=True, exist_ok=True)
                from datetime import datetime as _dt
                import json as _json
                ts = _dt.now().strftime('%Y%m%d_%H%M%S')
                (report_dir / f"report_{module}_{ts}.json").write_text(_json.dumps({
                    'timestamp': ts,
                    'module': module,
                    'train_jsonl': str(train_jsonl) if train_jsonl else None,
                    'val_jsonl': str(val_jsonl) if val_jsonl else None,
                    'test_jsonl': str(test_jsonl),
                    'auto': auto,
                    'stats': stats,
                }, indent=2))
            except Exception as e:
                console.print(f"[yellow]Could not write report: {e}[/yellow]")
    console.print(Panel.fit("Flags: --module --train-jsonl --val-jsonl --test-jsonl --dataset-dir --auto --max-full-evals --max-metric-calls --log-dir --save-best --ollama/--no-ollama --model --base-url --api-key", title="usage", border_style="dim"))


@app.command()
def gepa_orchestrator(
    train_jsonl: Optional[Path] = typer.Option(None, '--train-jsonl', exists=True, help="JSONL dataset with query/workspace/logs/targets"),
    val_jsonl: Optional[Path] = typer.Option(None, '--val-jsonl', exists=True),
    test_jsonl: Optional[Path] = typer.Option(None, '--test-jsonl', exists=True),
    dataset_dir: Optional[Path] = typer.Option(None, '--dataset-dir', dir_okay=True, exists=True),
    auto: Optional[str] = typer.Option("light", '--auto', help="Budget: light|medium|heavy"),
    log_dir: Optional[Path] = typer.Option(None, '--log-dir'),
    track_stats: bool = typer.Option(True, '--track-stats/--no-track-stats'),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama'),
    model: Optional[str] = typer.Option("deepseek-coder:1.3b", '--model'),
    base_url: Optional[str] = typer.Option(None, '--base-url'),
    api_key: Optional[str] = typer.Option(None, '--api-key'),
    save_best: Optional[Path] = typer.Option(None, '--save-best', help="Write best candidate orchestration mapping to file"),
    report_dir: Optional[Path] = typer.Option(None, '--report-dir', help="Directory to write evaluation reports"),
    force_json: bool = typer.Option(False, '--force-json', help="Force simple JSON outputs; skip structured-outputs"),
    structured: bool = typer.Option(False, '--structured', help="Prefer structured-outputs when available (overrides --force-json)"),
):
    if structured:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'false'
    elif force_json:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'true'
    lm = _maybe_configure_lm(True, ollama, model, base_url, api_key)
    if lm is None:
        console.print("[yellow]No LM configured for reflection.[/yellow]")
        raise typer.Exit(1)
    _print_header("GEPA Train (Orchestrator)")
    # Resolve dataset from directory
    if dataset_dir and not train_jsonl:
        cand = dataset_dir / "orchestrator_train.jsonl"
        if cand.exists():
            train_jsonl = cand
    if dataset_dir and not val_jsonl:
        cand = dataset_dir / "orchestrator_val.jsonl"
        if cand.exists():
            val_jsonl = cand
    if not train_jsonl:
        console.print("[yellow]No --train-jsonl or --dataset-dir provided.[/yellow]")
        raise typer.Exit(2)
    # Stream live training metrics
    if not log_dir:
        log_dir = Path.cwd() / '.gepa_orch'
    progress_path = (log_dir / 'progress.jsonl').resolve()
    result: dict = {}
    def _worker():
        try:
            if val_jsonl and val_jsonl.exists():
                result['prog'] = run_gepa_orchestrator_with_val(train_jsonl=train_jsonl, val_jsonl=val_jsonl, auto=auto, reflection_lm=lm, log_dir=str(log_dir) if log_dir else None, track_stats=track_stats, progress_path=str(progress_path))
            else:
                result['prog'] = run_gepa_orchestrator(train_jsonl=train_jsonl, auto=auto, reflection_lm=lm, log_dir=str(log_dir) if log_dir else None, track_stats=track_stats, progress_path=str(progress_path))
        except Exception as e:
            result['error'] = e
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    last_pos = 0
    tr: list[float] = []
    va: list[float] = []
    with Live(_progress_panel('orchestrator', auto or 'light', tr, va), refresh_per_second=4, console=console) as live:
        while t.is_alive():
            try:
                if progress_path.exists():
                    with open(progress_path, 'r') as f:
                        f.seek(last_pos)
                        for line in f:
                            try:
                                rec = _json.loads(line)
                                if rec.get('split') == 'val':
                                    va.append(float(rec.get('score', 0.0)))
                                else:
                                    tr.append(float(rec.get('score', 0.0)))
                            except Exception:
                                pass
                        last_pos = f.tell()
                live.update(_progress_panel('orchestrator', auto or 'light', tr, va))
            except Exception:
                pass
            time.sleep(1)
    if 'error' in result:
        console.print(Panel(str(result['error']), title="GEPA failed", border_style="red"))
        raise typer.Exit(1)
    optimized = result.get('prog')
    console.print("[green]GEPA optimization complete for Orchestrator.[/green]")
    if getattr(optimized, "detailed_results", None) and save_best:
        try:
            dr = optimized.detailed_results
            best_prog = dr.best_candidate
            save_best.parent.mkdir(parents=True, exist_ok=True)
            Path(save_best).write_text(json.dumps(best_prog, indent=2))
            console.print(f"[green]Saved best candidate to {save_best}[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not save best candidate: {e}[/yellow]")

    # Evaluate on test set if available
    if not test_jsonl and dataset_dir:
        cand = dataset_dir / "orchestrator_test.jsonl"
        if cand.exists():
            test_jsonl = cand
    if test_jsonl and test_jsonl.exists():
        stats = evaluate_orchestrator(optimized, test_jsonl)
        console.print(Panel(f"Test n={int(stats['n'])} avg_score={stats['avg_score']:.3f}", title="Test Eval", border_style="cyan"))
        if report_dir:
            try:
                report_dir.mkdir(parents=True, exist_ok=True)
                from datetime import datetime as _dt
                import json as _json
                ts = _dt.now().strftime('%Y%m%d_%H%M%S')
                (report_dir / f"report_orchestrator_{ts}.json").write_text(_json.dumps({
                    'timestamp': ts,
                    'module': 'orchestrator',
                    'train_jsonl': str(train_jsonl) if train_jsonl else None,
                    'val_jsonl': str(val_jsonl) if val_jsonl else None,
                    'test_jsonl': str(test_jsonl),
                    'auto': auto,
                    'stats': stats,
                }, indent=2))
            except Exception as e:
                console.print(f"[yellow]Could not write report: {e}[/yellow]")
    console.print(Panel.fit("Flags: --train-jsonl --val-jsonl --test-jsonl --dataset-dir --auto --log-dir --save-best --ollama/--no-ollama --model --base-url --api-key", title="usage", border_style="dim"))


@app.command()
def init(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace to initialize"),
    logs: Optional[Path] = typer.Option(None, '--logs', file_okay=True, dir_okay=True, exists=True, help="Logs path (defaults to <ws>/logs)"),
    out_dir: Optional[Path] = typer.Option(None, '--out-dir', help="Where to write datasets (default <ws>/.dspy_data)"),
    train: bool = typer.Option(False, '--train/--no-train', help="Run a light GEPA training pass after bootstrapping"),
    budget: str = typer.Option('light', '--budget', help="GEPA budget: light|medium|heavy"),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama', help="Use Ollama for reflection LM"),
    model: Optional[str] = typer.Option("deepseek-coder:1.3b", '--model', help="Reflection model name"),
    base_url: Optional[str] = typer.Option(None, '--base-url'),
    api_key: Optional[str] = typer.Option(None, '--api-key'),
    force_json: bool = typer.Option(False, '--force-json', help="Force simple JSON outputs; skip structured-outputs"),
    structured: bool = typer.Option(False, '--structured', help="Prefer structured-outputs when available (overrides --force-json)"),
):
    if structured:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'false'
    elif force_json:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'true'
    ws = workspace
    lp = logs or (ws / 'logs')
    _print_header("Bootstrapping datasets")
    paths = bootstrap_datasets(ws, lp, out_dir)
    for name, p in paths.items():
        console.print(f"[green]{name}[/green]: {p}")

    if not train:
        return

    _print_header("Light GEPA Training")
    lm = _maybe_configure_lm(True, ollama, model, base_url, api_key)
    if lm is None:
        console.print("[yellow]No LM configured; skipping training.[/yellow]")
        return
    try:
        _ = run_gepa_orchestrator(train_jsonl=paths['orchestrator'], auto=budget, reflection_lm=lm, log_dir=str(ws / '.gepa_orch'), track_stats=True)
        console.print("[green]Orchestrator training complete.[/green]")
    except Exception as e:
        console.print(Panel(str(e), title="orchestrator training failed", border_style="red"))
    try:
        _ = run_gepa(module='context', train_jsonl=paths['context'], auto=budget, reflection_lm=lm, log_dir=str(ws / '.gepa_ctx'), track_stats=True)
        console.print("[green]Context module training complete.[/green]")
    except Exception as e:
        console.print(Panel(str(e), title="context training failed", border_style="red"))
    try:
        _ = run_gepa(module='code', train_jsonl=paths['code'], auto=budget, reflection_lm=lm, log_dir=str(ws / '.gepa_code'), track_stats=True)
        console.print("[green]Code module training complete.[/green]")
    except Exception as e:
        console.print(Panel(str(e), title="code training failed", border_style="red"))
    try:
        _ = run_gepa(module='task', train_jsonl=paths['task'], auto=budget, reflection_lm=lm, log_dir=str(ws / '.gepa_task'), track_stats=True)
        console.print("[green]Task module training complete.[/green]")
    except Exception as e:
        console.print(Panel(str(e), title="task training failed", border_style="red"))
    console.print(Panel.fit("Flags: --workspace --logs --out-dir --train/--no-train --budget --ollama/--no-ollama --model --base-url --api-key", title="usage", border_style="dim"))


@app.command()
def live_train(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True),
    logs: Path = typer.Option(Path.cwd() / 'logs', '--logs', file_okay=True, dir_okay=True, exists=False),
    out_dir: Path = typer.Option(Path.cwd() / '.dspy_data', '--out', dir_okay=True),
    interval: int = typer.Option(300, '--interval', help="Seconds between checks"),
    modules: str = typer.Option('all', '--modules', help="Comma-separated: orchestrator,context,task,code or 'all'"),
    auto: str = typer.Option('light', '--auto', help="GEPA budget: light|medium|heavy"),
    report_dir: Path = typer.Option(Path.cwd() / '.dspy_reports', '--report-dir', dir_okay=True, help="Where to write evaluation reports"),
    seed: int = typer.Option(42, '--seed', help="Split seed"),
    dedup: bool = typer.Option(True, '--dedup/--no-dedup', help="De-duplicate rows"),
    stratify_by: Optional[str] = typer.Option('task_type', '--stratify-by'),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama'),
    model: Optional[str] = typer.Option("deepseek-coder:1.3b", '--model'),
    base_url: Optional[str] = typer.Option(None, '--base-url'),
    api_key: Optional[str] = typer.Option(None, '--api-key'),
):
    """Watch logs and retrain on a cadence, preserving test set.

    - Rebuild datasets and stratified splits
    - Train with val set; evaluate on test
    - Persist evaluation reports under --report-dir
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    lm = _maybe_configure_lm(True, ollama, model, base_url, api_key)
    if lm is None:
        console.print("[yellow]No LM configured; cannot train. Exiting.")
        raise typer.Exit(1)

    def write_report(module: str, stats: dict):
        from datetime import datetime as _dt
        import json as _json
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        p = report_dir / f"report_{module}_{ts}.json"
        try:
            p.write_text(_json.dumps({
                'timestamp': ts,
                'module': module,
                'workspace': str(workspace),
                'logs': str(logs),
                'auto': auto,
                'stats': stats,
            }, indent=2))
            console.print(f"[green]Wrote report {p}")
        except Exception as e:
            console.print(f"[yellow]Could not write report: {e}")

    last_sig = -1
    console.print(Panel.fit(f"Live training every {interval}s. Ctrl-C to stop.", title="live-train", border_style="cyan"))
    try:
        while True:
            sig = _logs_mtime_sum(logs) if logs.exists() else 0
            changed = (sig != last_sig)
            if changed or last_sig == -1:
                last_sig = sig
                _print_header("Rebuilding datasets")
                _ = bootstrap_datasets_with_splits(workspace, logs if logs.exists() else None, out_dir, seed=seed, dedup=dedup, stratify_by=stratify_by)
                split_dir = (out_dir / 'splits')
                mods = [m.strip() for m in modules.split(',')] if modules != 'all' else ['orchestrator', 'context', 'task', 'code']

                # Prepare concurrent training with per-module progress files
                threads: dict[str, threading.Thread] = {}
                errors: dict[str, Exception] = {}
                results: dict[str, object] = {}
                progress_files: dict[str, Path] = {}
                for m in mods:
                    if m == 'orchestrator':
                        progress_files[m] = (workspace / '.gepa_orch' / 'progress.jsonl').resolve()
                    else:
                        progress_files[m] = (workspace / f'.gepa_{m}' / 'progress.jsonl').resolve()

                def _train_module(mod: str):
                    try:
                        if mod == 'orchestrator':
                            prog = run_gepa_orchestrator_with_val(
                                train_jsonl=split_dir / 'orchestrator_train.jsonl',
                                val_jsonl=split_dir / 'orchestrator_val.jsonl',
                                auto=auto, reflection_lm=lm, log_dir=str(workspace / '.gepa_orch'), track_stats=True, progress_path=str(progress_files[mod]),
                            )
                            results[mod] = prog
                        else:
                            prog = run_gepa_with_val(
                                module=mod,
                                train_jsonl=split_dir / f'{mod}_train.jsonl',
                                val_jsonl=split_dir / f'{mod}_val.jsonl',
                                auto=auto, reflection_lm=lm, log_dir=str(workspace / f'.gepa_{mod}'), track_stats=True, progress_path=str(progress_files[mod]),
                            )
                            results[mod] = prog
                    except Exception as e:
                        errors[mod] = e

                # Spawn threads
                for m in mods:
                    th = threading.Thread(target=_train_module, args=(m,), daemon=True)
                    th.start()
                    threads[m] = th

                # Tail all progress files into a combined dashboard
                last_pos: dict[str, int] = {m: 0 for m in mods}
                tr: dict[str, list[float]] = {m: [] for m in mods}
                va: dict[str, list[float]] = {m: [] for m in mods}

                def _render_group():
                    panels = []
                    for m in mods:
                        panels.append(_progress_panel(m, auto, tr[m], va[m], title=f"{m} progress"))
                    return Columns(panels, equal=True, expand=True)

                with Live(_render_group(), refresh_per_second=4, console=console) as live:
                    while any(th.is_alive() for th in threads.values()):
                        try:
                            for m in mods:
                                pf = progress_files[m]
                                if pf.exists():
                                    with pf.open('r') as f:
                                        f.seek(last_pos[m])
                                        for line in f:
                                            try:
                                                rec = _json.loads(line)
                                                if rec.get('split') == 'val':
                                                    va[m].append(float(rec.get('score', 0.0)))
                                                else:
                                                    tr[m].append(float(rec.get('score', 0.0)))
                                            except Exception:
                                                pass
                                        last_pos[m] = f.tell()
                            # Update combined dashboard after tailing
                            live.update(_render_group())
                        except Exception:
                            pass
                        time.sleep(1)

                # Join and evaluate; also export time series artifacts
                for m, th in threads.items():
                    th.join(timeout=0)
                for m in mods:
                    try:
                        if m in errors:
                            console.print(Panel(str(errors[m]), title=f"{m} training failed", border_style="red"))
                            continue
                        # Export time series CSV/JSON for notebooks
                        try:
                            from datetime import datetime as _dt
                            ts = _dt.now().strftime('%Y%m%d_%H%M%S')
                            series = []
                            pf = progress_files[m]
                            if pf.exists():
                                with pf.open('r') as f:
                                    for line in f:
                                        try:
                                            rec = _json.loads(line)
                                        except Exception:
                                            continue
                                        series.append(rec)
                                # Write artifacts
                                report_dir.mkdir(parents=True, exist_ok=True)
                                (report_dir / f'series_{m}_{ts}.json').write_text(_json.dumps(series, indent=2))
                                # CSV: ts,split,score
                                import csv
                                with (report_dir / f'series_{m}_{ts}.csv').open('w', newline='') as cf:
                                    w = csv.writer(cf)
                                    w.writerow(['ts','module','split','score'])
                                    for rec in series:
                                        w.writerow([rec.get('ts',''), rec.get('module',''), rec.get('split',''), rec.get('score',0.0)])
                        except Exception as e:
                            console.print(Panel(str(e), title=f"{m} export failed", border_style="yellow"))
                        if m not in results:
                            console.print(Panel("No trained program returned.", title=f"{m} eval skipped", border_style="yellow"))
                            continue
                        if m == 'orchestrator':
                            stats = evaluate_orchestrator(results[m], split_dir / 'orchestrator_test.jsonl')  # type: ignore[arg-type]
                        else:
                            stats = evaluate_on_set(m, results[m], split_dir / f'{m}_test.jsonl')  # type: ignore[arg-type]
                        write_report(m, stats)
                    except Exception as e:
                        console.print(Panel(str(e), title=f"{m} eval failed", border_style="red"))
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("[dim]Stopped live training.[/dim]")
    console.print(Panel.fit("Flags: --workspace --logs --out --interval --modules --auto --report-dir --seed --dedup/--no-dedup --stratify-by --ollama/--no-ollama --model --base-url --api-key", title="usage", border_style="dim"))


@app.command()
def live():
    """Start live training with sensible defaults (workspace=., logs=./logs)."""
    ws = Path.cwd()
    logs = ws / 'logs'
    live_train(
        workspace=ws,
        logs=logs if logs.exists() else ws / 'logs',
        out_dir=ws / '.dspy_data',
        interval=300,
        modules='all',
        auto='light',
        report_dir=ws / '.dspy_reports',
        seed=42,
        dedup=True,
        stratify_by='task_type',
        ollama=True,
        model="deepseek-coder:1.3b",
        base_url=None,
        api_key=None,
    )


@app.command(name="open")
def open_cmd(
    path: str = typer.Argument(..., help="File to open (supports :line[:col])"),
):
    line = None
    col = None
    parts = path.split(":")
    file_part = parts[0]
    if len(parts) >= 2 and parts[1].isdigit():
        line = int(parts[1])
    if len(parts) >= 3 and parts[2].isdigit():
        col = int(parts[2])
    target = Path(file_part)
    editor = os.environ.get("EDITOR")
    try:
        if editor:
            ed = os.path.basename(editor)
            quoted = shlex.quote(str(target))
            if line is not None and ("code" in ed):
                # VS Code
                loc = f"{quoted}:{line}:{col or 1}"
                os.system(f"{editor} -g {loc}")
            elif line is not None and ("subl" in ed or "sublime" in ed):
                os.system(f"{editor} {quoted}:{line}:{col or 1}")
            elif line is not None and ("vim" in ed or "nvim" in ed):
                os.system(f"{editor} +{line} {quoted}")
            elif line is not None and ("emacs" in ed):
                if col is not None:
                    os.system(f"{editor} +{line}:{col} {quoted}")
                else:
                    os.system(f"{editor} +{line} {quoted}")
            elif line is not None and ("idea" in ed):
                os.system(f"{editor} --line {line} {quoted}")
            else:
                os.system(f"{editor} {quoted}")
        else:
            if os.name == 'nt':
                os.system(f'start "" {shlex.quote(str(target))}')
            elif sys.platform == 'darwin':
                os.system(f'open {shlex.quote(str(target))}')
            else:
                os.system(f'xdg-open {shlex.quote(str(target))}')
        console.print(f"[green]Opened {target}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to open: {e}[/red]")


@app.command()
def patch(
    patch_file: Optional[Path] = typer.Option(None, '--file', exists=True, help="Unified diff file"),
):
    if not patch_file:
        console.print("[yellow]Usage: dspy-coder patch --file <PATCHFILE>[/yellow]")
        raise typer.Exit(2)
    text = patch_file.read_text(errors="ignore")
    ok, msg = apply_unified_patch(text, Path.cwd())
    if ok:
        console.print(f"[ok]{msg}[/ok]")
        summ = summarize_patch(text)
        console.print(Panel.fit(
            f"files: {summ['files']}  +lines: {summ['added_lines']}  -lines: {summ['removed_lines']}",
            title="patch metrics", border_style="accent"
        ))
    else:
        console.print(Panel(msg, title="patch failed", border_style="err"))


def code_entry():
    """Entry point for 'dspy-code' shortcut: launches interactive agent with defaults."""
    try:
        _print_banner()
        # Use cwd as workspace and ./logs if exists
        ws = Path.cwd()
        logs = ws / 'logs'
        start(workspace=ws, logs=logs if logs.exists() else None, ollama=True, model="deepseek-coder:1.3b", base_url=None, api_key=None, force_json=False, structured=False, approval=None)  # type: ignore[arg-type]
    except SystemExit:
        pass


@app.command()
def diff(
    file: Path = typer.Option(..., '--file', exists=True, help="Existing file to diff against"),
    new: Optional[Path] = typer.Option(None, '--new', exists=True, help="New file to compare; if omitted, reads from STDIN"),
    unified: int = typer.Option(3, '--unified', help="Context lines in diff"),
    out: Optional[Path] = typer.Option(None, '--out', help="Write patch to this file"),
):
    if new is not None:
        new_text = new.read_text(errors="ignore")
    else:
        try:
            new_text = sys.stdin.read()
        except Exception:
            console.print("[yellow]No input provided for --new or STDIN.[/yellow]")
            raise typer.Exit(2)
    from .diffutil import unified_diff_from_texts
    old_text = file.read_text(errors="ignore") if file.exists() else ""
    patch_text = unified_diff_from_texts(old_text, new_text, a_path=str(file), b_path=str(file), n=unified)
    if out:
        out.write_text(patch_text)
        console.print(f"[green]Wrote patch to {out}[/green]")
    else:
        console.print(patch_text or "(no differences)")


@app.command()
def start(
    workspace: Optional[Path] = typer.Option(None, '--workspace', dir_okay=True, exists=True, help="Initial workspace"),
    logs: Optional[Path] = typer.Option(None, '--logs', dir_okay=True, file_okay=True, exists=False, help="Initial logs path (defaults to <ws>/logs)"),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama', help="Use Ollama by default in session"),
    model: Optional[str] = typer.Option("deepseek-coder:1.3b", '--model', help="Default model for session"),
    base_url: Optional[str] = typer.Option(None, '--base-url', help="Override base URL"),
    api_key: Optional[str] = typer.Option(None, '--api-key', help="API key (unused for Ollama)"),
    force_json: bool = typer.Option(False, '--force-json', help="Force simple JSON outputs; skip structured-outputs"),
    structured: bool = typer.Option(False, '--structured', help="Prefer structured-outputs when available (overrides --force-json)"),
    approval: Optional[str] = typer.Option(None, '--approval', help='Tool approval mode: auto|manual'),
):
    """Interactive session to pick workspace/logs and run tasks."""
    ws = Path(workspace) if workspace else Path.cwd()
    logs_path = Path(logs) if logs else (ws / 'logs')
    use_lm = True
    provider_is_ollama = ollama
    last_extract: Optional[str] = None

    # Apply runtime toggle for adapter behavior
    if structured:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'false'
    elif force_json:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'true'

    # Resolve approval mode
    settings = get_settings()
    approval_mode = (approval or getattr(settings, 'tool_approval_mode', 'auto') or 'auto').lower()
    if approval_mode not in {"auto", "manual"}:
        approval_mode = "auto"

    console.print(Panel.fit(
        f"Workspace: {ws}\nLogs: {logs_path}\nLLM: {'Ollama ' + (model or '') if provider_is_ollama else 'OpenAI/compatible'}\nApproval: {approval_mode}\nTip: Type natural instructions — the agent will choose tools.",
        title="dspy-coder session",
        border_style="cyan"
    ))

    def show_help():
        console.print(Panel.fit(
            "Commands:\n"
            "  help                Show this help\n"
            "  ws                  Show workspace\n"
            "  cd <PATH>           Change workspace\n"
            "  logs [PATH]         Show or set logs path\n"
            "  ctx                 Show context from logs\n"
            "  plan <TASK>         Propose plan + commands\n"
            "  ls [PATH]           List files (relative to workspace)\n"
            "  tree [PATH] [-d N] [--hidden]  Show directory tree (default depth 2)\n"
            "  grep <PATTERN>      Search (flags: -f fixed, -c N, -g GLOB, -x GLOB, -F FILE)\n"
            "  extract --file F [--symbol NAME | --re REGEX --before N --after N]\n"
            "  sg [-p PATTERN] [-l LANG] [-r RULE.yaml] [--json] Run ast-grep\n"
            "  watch [-n SECS] [-t LINES]  Tail + refresh key events\n"
            "  codectx [PATH]      Summarize code file/dir\n"
            "  index               Build code index for semantic search\n"
            "  esearch <QUERY>     Semantic search over index\n"
            "  emb-index [-m MODEL] Build embedding index (requires embeddings provider)\n"
            "  emb-search <QUERY> [-m MODEL] Embedding search\n"
            "  open <PATH>         Open a file in $EDITOR / OS default\n"
            "  patch <PATCHFILE>   Apply unified diff patch\n"
            "  diff <FILE>         Diff last extract against FILE\n"
            "  write <PATH>        Save last extract to file\n"
            "  gstatus             Git status (short)\n"
            "  gadd <PATHS...>     Git add files\n"
            "  gcommit -m MSG      Git commit with message\n"
            "  ollama on|off       Toggle Ollama provider\n"
            "  model <NAME>        Set model name\n"
            "  exit|quit           Leave session",
            title="help",
            border_style="magenta"
        ))

    def build_state() -> str:
        # Summarize env for orchestrator
        idx_dir = ws / ".dspy_index"
        has_idx = (idx_dir / "index.jsonl").exists()
        has_emb = (idx_dir / "emb_index.jsonl").exists()
        has_logs = logs_path.exists()
        return (
            f"workspace={ws}, logs={logs_path}, logs_exist={has_logs}, "
            f"has_index={has_idx}, has_emb_index={has_emb}, last_extract={'yes' if last_extract else 'no'}"
        )

    def _extract_targets_from_query(nl: str, k: int = 5) -> list[str]:
        toks = tokenize(nl)
        # simple heuristic: take unique tokens >2 chars
        seen = []
        for t in toks:
            if len(t) > 2 and t not in seen:
                seen.append(t)
            if len(seen) >= k:
                break
        return seen

    def orchestrate_chain(nl: str, max_steps: int = 4):
        # Try LLM orchestrator, fall back to heuristics
        history_summary = ""
        last_tool = None
        last_args = None
        targets = _extract_targets_from_query(nl)
        for step in range(1, max_steps + 1):
            state = build_state() + (f" | history: {history_summary[:4000]}" if history_summary else "")
            tool = None
            args = {}
            lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key)
            if lm is not None:
                try:
                    orch = Orchestrator()
                    pred = orch(query=nl, state=state)
                    tool = (pred.tool or "").strip()
                    import json as _json
                    try:
                        args = _json.loads(pred.args_json or "{}")
                    except Exception:
                        args = {}
                except Exception:
                    tool = None
            if not tool:
                # fallback heuristic single shot
                tool = "context" if (logs_path.exists() if isinstance(logs_path, Path) else True) else "codectx"
                args = {}

            console.print(Panel.fit(f"{tool} {args}", title=f"Step {step}: action", border_style="yellow"))
            # Stop if repeating same choice
            if last_tool == tool and last_args == args:
                console.print("[dim]No new action; stopping.[/dim]")
                break
            last_tool, last_args = tool, dict(args)

            # Execute action for the user
            try:
                if approval_mode == "manual":
                    approved = typer.confirm(f"Approve tool '{tool}' with args {args}?", default=False)
                    if not approved:
                        console.print("[dim]Tool execution skipped by user.[/dim]")
                        # Do not update history summary, simply continue to next step
                        continue
                dispatch_tool(tool, args)
            except Exception as e:
                console.print(Panel(str(e), title=f"agent failed ({tool})", border_style="red"))
                break

            # Summarize outcome to feed back
            try:
                outcome = evaluate_tool_choice(tool, args, workspace=ws, logs_path=logs_path, targets=targets)
                piece = f"{tool}: score={outcome.score:.2f}; {outcome.evidence}"
            except Exception:
                piece = f"{tool}: done"
            history_summary = (history_summary + " | " + piece).strip()
            # Simple stop conditions
            if tool in {"esearch", "grep", "extract"} and "hits=0" in piece:
                continue
            if tool in {"context", "codectx"} and "events_len=0" in piece:
                continue
            # If good score, and we've run at least 2 steps, stop
            if step >= 2 and ("score=" in piece and float(piece.split("score=")[1].split(";")[0]) >= 1.2):
                break

    def dispatch_tool(tool: str, args: dict):
        nonlocal last_extract, logs_path, ws, model, provider_is_ollama, base_url, api_key
        t = tool.lower().strip()
        # Normalize common synonyms for better UX
        if t in {"summarize", "summary", "summarise", "describe", "overview", "summarize_repo", "repo_summary"}:
            t = "codectx"
        if t == "context":
            bundle, count = load_logs([logs_path])
            if not bundle:
                console.print("[yellow]No logs found.[/yellow]"); return
            _print_header("Log Key Events"); key = extract_key_events(bundle)
            console.print(Panel.fit(key, title="Extracted Events", border_style="magenta"))
        elif t == "plan":
            bundle, count = load_logs([logs_path]); key = extract_key_events(bundle) if bundle else ""
            lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key)
            if lm is None:
                console.print("[yellow]No LM configured for planning.[/yellow]"); return
            builder = ContextBuilder(); ctx = builder(task=args.get("task", ""), logs_preview=key)
            _print_header("Agent Plan"); agent = TaskAgent()
            out = agent(task=args.get("task", ""), context=f"{ctx.context}\n\n{ctx.key_points}")
            console.print(Panel.fit(out.plan, title="Proposed Plan", border_style="blue"))
            console.print(Panel.fit(out.commands or "(no commands)", title="Suggested Commands", border_style="yellow"))
        elif t == "grep":
            pattern = args.get("pattern") or args.get("query") or ""
            if not pattern: console.print("[yellow]No pattern provided.[/yellow]"); return
            hits = search_text(ws, pattern, regex=True)
            if not hits:
                console.print("[yellow]No matches found.[/yellow]"); return
            for h in hits[:200]:
                console.print(f"{h.path}:{h.line_no}: {h.line}")
        elif t == "extract":
            file = args.get("file"); symbol = args.get("symbol"); regex = args.get("regex")
            if not file: console.print("[yellow]extract needs a file.[/yellow]"); return
            fpath = (ws / file).resolve()
            if symbol and fpath.suffix == ".py":
                res = python_extract_symbol(fpath, symbol)
                if not res: console.print("[yellow]Symbol not found.[/yellow]"); return
                start, end, seg = res; last_extract = seg
                console.print(Panel(seg, title=f"{fpath}::{symbol} lines {start}-{end}", border_style="green"))
            elif regex:
                hits = file_search(fpath, regex, regex=True)
                if not hits: console.print("[yellow]No regex match.[/yellow]"); return
                hit = hits[0]; text = fpath.read_text(errors="ignore");
                s,e,seg = extract_context(text, hit.line_no, before=args.get("before",3), after=args.get("after",3)); last_extract=seg
                console.print(Panel(seg, title=f"{fpath} lines {s}-{e}", border_style="green"))
            else:
                console.print("[yellow]Provide --symbol or --regex for extract.[/yellow]")
        elif t == "tree":
            depth = int(args.get("depth", 2)); hidden = bool(args.get("hidden", False));
            console.print(_render_tree(ws, max_depth=depth, show_hidden=hidden))
        elif t == "ls":
            console.print(Panel("\n".join(p.name+("/" if p.is_dir() else "") for p in sorted(ws.iterdir())[:500]), title=str(ws), border_style="blue"))
        elif t == "codectx":
            path = args.get("path"); target = (ws / path).resolve() if path else ws
            snap = build_code_snapshot(target); _print_header("Code Snapshot")
            console.print(Panel.fit(snap[:8000] + ("\n..." if len(snap)>8000 else ""), title=str(target), border_style="magenta"))
            lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key)
            if lm:
                cc = CodeContext(); out = cc(snapshot=snap, ask="Summarize key parts and modification points.")
                console.print(Panel.fit(out.summary, title="Summary", border_style="cyan"))
                console.print(Panel.fit(out.bullets, title="Bullets", border_style="green"))
        elif t == "index":
            _print_header("Building index"); meta, items = build_index(ws, smart=True); out_dir = save_index(ws, meta, items)
            console.print(f"[green]Indexed {len(items)} chunks. Saved to {out_dir}[/green]")
        elif t == "esearch":
            q = args.get("query") or args.get("q");
            if not q: console.print("[yellow]No query[/yellow]"); return
            try: meta, items = load_index(ws)
            except FileNotFoundError: console.print("[yellow]No index. Run 'index' first.[/yellow]"); return
            hits = semantic_search(q, meta, items, top_k=int(args.get("k",5)))
            for score,it in hits:
                p = Path(it.path)
                try:
                    text = p.read_text(errors="ignore"); lines = text.splitlines(); s=max(1,it.start_line-3); e=min(len(lines), it.end_line+3)
                    seg = "\n".join(lines[s-1:e])
                except Exception: seg="(unreadable)"; s=it.start_line; e=it.end_line
                console.print(Panel(seg, title=f"{p} score={score:.3f} lines {s}-{e}", border_style="blue"))
        elif t == "open":
            spec = args.get("path");
            if not spec: console.print("[yellow]No path for open[/yellow]"); return
            parts = str(spec).split(":"); file_part = parts[0]; line = int(parts[1]) if len(parts)>=2 and str(parts[1]).isdigit() else None; col = int(parts[2]) if len(parts)>=3 and str(parts[2]).isdigit() else None
            target = (ws / file_part).resolve(); editor = os.environ.get("EDITOR"); quoted = shlex.quote(str(target))
            if editor:
                ed = os.path.basename(editor)
                if line is not None and ("code" in ed): os.system(f"{editor} -g {quoted}:{line}:{col or 1}")
                elif line is not None and ("subl" in ed or "sublime" in ed): os.system(f"{editor} {quoted}:{line}:{col or 1}")
                elif line is not None and ("vim" in ed or "nvim" in ed): os.system(f"{editor} +{line} {quoted}")
                elif line is not None and ("emacs" in ed): os.system(f"{editor} +{line}{(':'+str(col)) if col else ''} {quoted}")
                elif line is not None and ("idea" in ed): os.system(f"{editor} --line {line} {quoted}")
                else: os.system(f"{editor} {quoted}")
            else:
                if os.name=='nt': os.system(f'start "" {quoted}')
                elif sys.platform=='darwin': os.system(f'open {quoted}')
                else: os.system(f'xdg-open {quoted}')
            console.print(f"[green]Opened {target}[/green]")
        elif t == "watch":
            _watch_logs(logs_path, tail_lines=int(args.get("tail",20)), interval=float(args.get("interval",2)))
        elif t == "sg":
            pattern = args.get("pattern"); lang = args.get("lang"); rule=args.get("rule"); as_json=bool(args.get("json", False))
            code,out,err = run_ast_grep(root=ws, pattern=pattern, lang=lang, rule_file=(ws / rule) if rule else None, json=as_json)
            if err.strip(): console.print(Panel(err, title="ast-grep stderr", border_style="red"))
            if out.strip(): console.print(out)
        elif t == "patch":
            pf = args.get("file"); p = (ws / pf).resolve() if pf else None
            if not p or not p.exists(): console.print("[yellow]patch needs --file[/yellow]"); return
            text = p.read_text(errors="ignore"); ok,msg = apply_unified_patch(text, ws);
            if ok:
                console.print(f"[ok]{msg}[/ok]")
                summ = summarize_patch(text)
                console.print(Panel.fit(f"files: {summ['files']}  +lines: {summ['added_lines']}  -lines: {summ['removed_lines']}", title="patch metrics", border_style="accent"))
            else:
                console.print(Panel(msg, title="patch failed", border_style="err"))
        elif t == "diff":
            if not last_extract: console.print("[yellow]No extract buffer. Run extract first.[/yellow]"); return
            file = args.get("file"); p=(ws / file).resolve() if file else None
            if not p: console.print("[yellow]diff needs a file[/yellow]"); return
            from .diffutil import unified_diff_file_vs_text
            patch_text = unified_diff_file_vs_text(p, last_extract, n=3); console.print(patch_text or "(no differences)")
        elif t == "git_status":
            import subprocess; proc = subprocess.run(["git","-C",str(ws),"status","-s"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            console.print(proc.stdout or (Panel(proc.stderr or "git status failed", title="git", border_style="red")))
        elif t == "git_add":
            import subprocess; paths = args.get("paths"); paths = paths if isinstance(paths, list) else [paths] if paths else []
            if not paths: console.print("[yellow]git_add needs paths[/yellow]"); return
            proc = subprocess.run(["git","-C",str(ws),"add",*paths], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            console.print("[green]Added files[/green]" if proc.returncode==0 else Panel(proc.stderr or "git add failed", title="git", border_style="red"))
        elif t == "git_commit":
            import subprocess; msg = args.get("message") or args.get("m");
            if not msg: console.print("[yellow]git_commit needs message[/yellow]"); return
            proc = subprocess.run(["git","-C",str(ws),"commit","-m",msg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            console.print(proc.stdout or (Panel(proc.stderr or "git commit failed", title="git", border_style="red")))
        else:
            console.print(f"[yellow]Unknown tool from agent: {tool}[/yellow]")

    show_help()
    while True:
        try:
            line = input("dspy-coder> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break
        if not line:
            continue
        try:
            parts = shlex.split(line)
        except Exception:
            parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in {"help", "?"}:
            show_help()
        elif cmd == "ws":
            console.print(f"Workspace: {ws}")
        elif cmd == "cd" and args:
            new_ws = Path(" ".join(args)).expanduser()
            if new_ws.exists() and new_ws.is_dir():
                ws = new_ws
                # Only adjust logs if pointing at default
                if logs_path == ws / 'logs' or not logs_path.exists():
                    logs_path = ws / 'logs'
                console.print(f"[green]Workspace set to {ws}[/green]")
            else:
                console.print(f"[red]Not a directory: {new_ws}[/red]")
        elif cmd == "logs":
            if args:
                new_logs = Path(" ".join(args)).expanduser()
                logs_path = new_logs
                console.print(f"[green]Logs set to {logs_path}[/green]")
            else:
                console.print(f"Logs: {logs_path}")
        elif cmd == "ls":
            rel = args[0] if args else "."
            target = (ws / rel).resolve()
            if target.is_file():
                console.print(str(target))
            elif target.is_dir():
                try:
                    entries = sorted(target.iterdir())
                except Exception as e:
                    console.print(f"[red]Cannot list {target}: {e}[/red]")
                    continue
                lines = []
                for p in entries[:500]:
                    tag = "/" if p.is_dir() else ""
                    lines.append(p.name + tag)
                console.print(Panel("\n".join(lines) or "(empty)", title=str(target), border_style="blue"))
            else:
                console.print(f"[yellow]Not found: {target}[/yellow]")
        elif cmd == "tree":
            # Syntax: tree [PATH] [-d N]
            d = 2
            path_arg: Optional[Path] = None
            show_hidden = False
            i = 0
            while i < len(args):
                a = args[i]
                if a in ("-d", "--depth") and i + 1 < len(args):
                    try:
                        d = int(args[i + 1]); i += 1
                    except ValueError:
                        console.print("[yellow]Invalid depth[/yellow]")
                elif a in ("-a", "--hidden"):
                    show_hidden = True
                else:
                    path_arg = (ws / a).resolve()
                i += 1
            base = path_arg or ws
            console.print(_render_tree(base, max_depth=d, show_hidden=show_hidden))
        elif cmd == "ollama" and args:
            val = args[0].lower()
            provider_is_ollama = (val in {"on", "true", "1", "yes"})
            console.print(f"[green]Ollama {'enabled' if provider_is_ollama else 'disabled'}[/green]")
        elif cmd == "model" and args:
            model = " ".join(args)
            console.print(f"[green]Model set to {model}[/green]")
        elif cmd == "ctx":
            # Reuse logic from context command
            bundle, count = load_logs([logs_path])
            if not bundle:
                console.print("[yellow]No logs found.[/yellow]")
                continue
            _print_header("Log Key Events")
            key = extract_key_events(bundle)
            console.print(Panel.fit(key, title="Extracted Events", border_style="magenta"))
            lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key)
            if lm is None:
                continue
            _print_header("Enhanced Context (DSPy)")
            builder = ContextBuilder()
            pred = builder(task="Summarize logs for debugging", logs_preview=key)
            console.print(Panel.fit(pred.context, title="Context", border_style="cyan"))
            console.print(Panel.fit(pred.key_points, title="Key Points", border_style="green"))
        elif cmd == "codectx":
            path_arg = (ws / args[0]).resolve() if args else ws
            snap = build_code_snapshot(path_arg)
            _print_header("Code Snapshot")
            console.print(Panel.fit(snap[:8000] + ("\n..." if len(snap) > 8000 else ""), title=str(path_arg), border_style="magenta"))
            lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key)
            if lm:
                _print_header("Code Context (DSPy)")
                cc = CodeContext()
                out = cc(snapshot=snap, ask="Summarize key components, APIs, and likely modification points.")
                console.print(Panel.fit(out.summary, title="Summary", border_style="cyan"))
                console.print(Panel.fit(out.bullets, title="Bullets", border_style="green"))
        elif cmd == "plan" and args:
            task = " ".join(args)
            bundle, count = load_logs([logs_path])
            key = extract_key_events(bundle) if bundle else ""
            lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key)
            if lm is None:
                console.print("[yellow]No LM configured; cannot generate a plan.[/yellow]")
                continue
            builder = ContextBuilder()
            ctx = builder(task=task, logs_preview=key)
            _print_header("Agent Plan")
            agent = TaskAgent()
            out = agent(task=task, context=f"{ctx.context}\n\n{ctx.key_points}")
            console.print(Panel.fit(task, title="Task", border_style="white"))
            console.print(Panel.fit(ctx.context, title="Context", border_style="cyan"))
            console.print(Panel.fit(ctx.key_points, title="Key Points", border_style="green"))
            console.print(Panel.fit(out.plan, title="Proposed Plan", border_style="blue"))
            console.print(Panel.fit(out.commands or "(no commands)", title="Suggested Commands", border_style="yellow"))
        elif cmd == "grep":
            # Flags: -f fixed, -c N context, -g glob (repeat), -x exclude (repeat), -F file
            fixed = False
            ctx_n = 0
            globs: list[str] = []
            excls: list[str] = []
            file_arg: Optional[Path] = None
            pattern_parts: list[str] = []
            i = 0
            while i < len(args):
                a = args[i]
                if a in ("-f", "--fixed"):
                    fixed = True
                elif a in ("-c", "--context") and i + 1 < len(args):
                    try:
                        ctx_n = int(args[i + 1]); i += 1
                    except ValueError:
                        console.print("[yellow]Invalid context value[/yellow]"); return
                elif a in ("-g", "--glob") and i + 1 < len(args):
                    globs.append(args[i + 1]); i += 1
                elif a in ("-x", "--exclude") and i + 1 < len(args):
                    excls.append(args[i + 1]); i += 1
                elif a in ("-F", "--file") and i + 1 < len(args):
                    file_arg = (ws / args[i + 1]).resolve(); i += 1
                else:
                    pattern_parts.append(a)
                i += 1
            if not pattern_parts:
                console.print("[yellow]Usage: grep <PATTERN> [-f] [-c N] [-g GLOB]* [-x GLOB]* [-F FILE][/yellow]")
                return
            pattern = " ".join(pattern_parts)
            if file_arg:
                hits = file_search(file_arg, pattern, regex=not fixed)
            else:
                hits = search_text(ws, pattern, regex=not fixed, include_globs=globs or None, exclude_globs=excls or None)
            if not hits:
                console.print("[yellow]No matches found.[/yellow]")
                return
            for h in hits[:500]:
                try:
                    text = h.path.read_text(errors="ignore") if ctx_n else ""
                except Exception:
                    text = ""
                if ctx_n and text:
                    start, end, seg = extract_context(text, h.line_no, before=ctx_n, after=ctx_n)
                    title = f"{h.path} :{h.line_no} (lines {start}-{end})"
                    console.print(Panel(seg, title=title, border_style="blue"))
                else:
                    console.print(f"{h.path}:{h.line_no}: {h.line}")
        elif cmd == "extract":
            # Syntax: extract --file F [--symbol NAME | --re REGEX --before N --after N --nth K]
            file_arg: Optional[Path] = None
            symbol: Optional[str] = None
            regex: Optional[str] = None
            before = 3
            after = 3
            nth = 1
            i = 0
            while i < len(args):
                a = args[i]
                if a == "--file" and i + 1 < len(args):
                    file_arg = (ws / args[i + 1]).resolve(); i += 1
                elif a == "--symbol" and i + 1 < len(args):
                    symbol = args[i + 1]; i += 1
                elif a in ("--re", "--regex") and i + 1 < len(args):
                    regex = args[i + 1]; i += 1
                elif a == "--before" and i + 1 < len(args):
                    before = int(args[i + 1]); i += 1
                elif a == "--after" and i + 1 < len(args):
                    after = int(args[i + 1]); i += 1
                elif a == "--nth" and i + 1 < len(args):
                    nth = int(args[i + 1]); i += 1
                i += 1
            if not file_arg or not file_arg.exists():
                console.print("[yellow]extract --file <PATH> is required and must exist.[/yellow]")
                continue
            if symbol:
                if file_arg.suffix == ".py":
                    res = python_extract_symbol(file_arg, symbol)
                    if not res:
                        console.print("[yellow]Symbol not found.[/yellow]")
                        continue
                    start, end, seg = res
                    console.print(Panel(seg, title=f"{file_arg}::{symbol} (lines {start}-{end})", border_style="green"))
                    last_extract = seg
                else:
                    console.print("[yellow]--symbol currently supports Python files only.[/yellow]")
            elif regex:
                hits = file_search(file_arg, regex, regex=True)
                if not hits:
                    console.print("[yellow]No regex match in file.[/yellow]")
                    continue
                hit = hits[nth - 1] if len(hits) >= nth else hits[-1]
                text = file_arg.read_text(errors="ignore")
                start, end, seg = extract_context(text, hit.line_no, before=before, after=after)
                console.print(Panel(seg, title=f"{file_arg} lines {start}-{end}", border_style="green"))
                last_extract = seg
            else:
                console.print("[yellow]Provide --symbol or --regex for extract.[/yellow]")
        elif cmd == "index":
            _print_header("Building index")
            meta, items = build_index(ws)
            out_dir = save_index(ws, meta, items)
            console.print(f"[green]Indexed {len(items)} chunks. Saved to {out_dir}[/green]")
        elif cmd == "esearch":
            if not args:
                console.print("[yellow]Usage: esearch <QUERY>[/yellow]")
                continue
            query = " ".join(args)
            try:
                meta, items = load_index(ws)
            except FileNotFoundError:
                console.print("[yellow]No index found. Run 'index' first.[/yellow]")
                continue
            hits = semantic_search(query, meta, items, top_k=5)
            if not hits:
                console.print("[yellow]No results.[/yellow]")
                continue
            for score, it in hits:
                p = Path(it.path)
                try:
                    text = p.read_text(errors="ignore")
                    lines = text.splitlines()
                    start = max(1, it.start_line - 3)
                    end = min(len(lines), it.end_line + 3)
                    seg = "\n".join(lines[start - 1 : end])
                except Exception:
                    seg = "(unreadable)"
                    start = it.start_line
                    end = it.end_line
                title = f"{p}  score={score:.3f}  lines {start}-{end}"
                console.print(Panel(seg, title=title, border_style="blue"))
        elif cmd == "open":
            if not args:
                console.print("[yellow]Usage: open <PATH[:LINE[:COL]]>[/yellow]")
                continue
            target_spec = args[0]
            parts = target_spec.split(":")
            file_part = parts[0]
            line = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else None
            col = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
            target = (ws / file_part).resolve()
            editor = os.environ.get("EDITOR")
            try:
                if editor:
                    ed = os.path.basename(editor)
                    quoted = shlex.quote(str(target))
                    if line is not None and ("code" in ed):
                        loc = f"{quoted}:{line}:{col or 1}"
                        os.system(f"{editor} -g {loc}")
                    elif line is not None and ("subl" in ed or "sublime" in ed):
                        os.system(f"{editor} {quoted}:{line}:{col or 1}")
                    elif line is not None and ("vim" in ed or "nvim" in ed):
                        os.system(f"{editor} +{line} {quoted}")
                    elif line is not None and ("emacs" in ed):
                        if col is not None:
                            os.system(f"{editor} +{line}:{col} {quoted}")
                        else:
                            os.system(f"{editor} +{line} {quoted}")
                    elif line is not None and ("idea" in ed):
                        os.system(f"{editor} --line {line} {quoted}")
                    else:
                        os.system(f"{editor} {quoted}")
                else:
                    # macOS 'open', Linux 'xdg-open', Windows 'start'
                    if os.name == 'nt':
                        os.system(f'start "" {shlex.quote(str(target))}')
                    elif sys.platform == 'darwin':
                        os.system(f'open {shlex.quote(str(target))}')
                    else:
                        os.system(f'xdg-open {shlex.quote(str(target))}')
                console.print(f"[green]Opened {target}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to open: {e}[/red]")
        elif cmd == "diff":
            if not args:
                console.print("[yellow]Usage: diff <FILE>[/yellow]")
                continue
            if not last_extract:
                console.print("[yellow]No extract buffer to diff. Run 'extract' first.[/yellow]")
                continue
            file_arg = (ws / args[0]).resolve()
            patch_text = unified_diff_file_vs_text(file_arg, last_extract, n=3)
            console.print(patch_text or "(no differences)")
        elif cmd == "patch":
            if not args:
                console.print("[yellow]Usage: patch <patchfile>[/yellow]")
                continue
            patch_file = (ws / args[0]).resolve()
            if not patch_file.exists():
                console.print(f"[yellow]Patch file not found: {patch_file}[/yellow]")
                continue
            text = patch_file.read_text(errors="ignore")
            ok, msg = apply_unified_patch(text, ws)
            if ok:
                console.print(f"[green]{msg}[/green]")
            else:
                console.print(Panel(msg, title="patch failed", border_style="red"))
        elif cmd == "gstatus":
            try:
                import subprocess
                proc = subprocess.run(["git", "-C", str(ws), "status", "-s"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc.returncode == 0:
                    console.print(proc.stdout or "(clean)")
                else:
                    console.print(Panel(proc.stderr or "git status failed", title="git", border_style="red"))
            except Exception as e:
                console.print(f"[red]git error: {e}[/red]")
        elif cmd == "gadd":
            if not args:
                console.print("[yellow]Usage: gadd <PATHS...>[/yellow]")
                continue
            try:
                import subprocess
                proc = subprocess.run(["git", "-C", str(ws), "add", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc.returncode == 0:
                    console.print("[green]Added files.[/green]")
                else:
                    console.print(Panel(proc.stderr or "git add failed", title="git", border_style="red"))
            except Exception as e:
                console.print(f"[red]git error: {e}[/red]")
        elif cmd == "gcommit":
            # gcommit -m "message"
            msg = None
            i = 0
            while i < len(args):
                a = args[i]
                if a in ("-m", "--message") and i + 1 < len(args):
                    msg = args[i + 1]; i += 1
                i += 1
            if not msg:
                console.print("[yellow]Usage: gcommit -m \"message\"[/yellow]")
                continue
            try:
                import subprocess
                proc = subprocess.run(["git", "-C", str(ws), "commit", "-m", msg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc.returncode == 0:
                    console.print(proc.stdout or "[green]Committed.[/green]")
                else:
                    console.print(Panel(proc.stderr or "git commit failed", title="git", border_style="red"))
            except Exception as e:
                console.print(f"[red]git error: {e}[/red]")
        elif cmd == "write":
            if not args:
                console.print("[yellow]Usage: write <PATH>[/yellow]")
                continue
            if not last_extract:
                console.print("[yellow]No extract buffer to write. Run 'extract' first.[/yellow]")
                continue
            out_path = (ws / args[0]).resolve()
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(last_extract)
                console.print(f"[green]Wrote {len(last_extract)} bytes to {out_path}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to write: {e}[/red]")
        elif cmd == "sg":
            # ast-grep passthrough: -p PATTERN, -l LANG, -r RULE, --json
            exe = ast_grep_available()
            if not exe:
                console.print("[red]ast-grep not found. Install with brew or the official script.[/red]")
                continue
            pattern = None
            lang = None
            rule_file = None
            as_json = False
            i = 0
            while i < len(args):
                a = args[i]
                if a in ("-p", "--pattern") and i + 1 < len(args):
                    pattern = args[i + 1]; i += 1
                elif a in ("-l", "--lang") and i + 1 < len(args):
                    lang = args[i + 1]; i += 1
                elif a in ("-r", "--rule") and i + 1 < len(args):
                    p = (ws / args[i + 1]).resolve(); rule_file = p; i += 1
                elif a == "--json":
                    as_json = True
                i += 1
            code, out, err = run_ast_grep(root=ws, pattern=pattern, lang=lang, rule_file=rule_file, json=as_json)
            if err.strip():
                console.print(Panel(err, title="ast-grep stderr", border_style="red"))
            if out.strip():
                console.print(out)
        elif cmd == "watch":
            # watch [-n SECS] [-t LINES]
            interval = 2.0
            tail_lines = 0
            i = 0
            while i < len(args):
                a = args[i]
                if a in ("-n", "--interval") and i + 1 < len(args):
                    try:
                        interval = float(args[i + 1]); i += 1
                    except ValueError:
                        console.print("[yellow]Invalid interval[/yellow]")
                elif a in ("-t", "--tail") and i + 1 < len(args):
                    try:
                        tail_lines = int(args[i + 1]); i += 1
                    except ValueError:
                        console.print("[yellow]Invalid tail lines[/yellow]")
                i += 1
            _watch_logs(logs_path, tail_lines=tail_lines, interval=interval)
        elif cmd in {"exit", "quit"}:
            break
        else:
            # Natural instruction → multi-step auto orchestration
            orchestrate_chain(line)


@app.command()
def watch(
    logs: Optional[Path] = typer.Option(None, '--logs', file_okay=True, dir_okay=True, exists=True, help="Logs file or directory"),
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Default root if --logs not given"),
    interval: float = typer.Option(2.0, '--interval', help="Seconds between checks"),
    tail: int = typer.Option(0, '--tail', help="Tail N lines on changes"),
):
    target = logs or (workspace / 'logs')
    _watch_logs(target, tail_lines=tail, interval=interval)


@app.command()
def grep(
    query: str = typer.Argument(..., help="Regex or fixed string to search"),
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Root folder to search"),
    regex: bool = typer.Option(True, '--regex/--fixed', help="Interpret query as regex or fixed string"),
    glob: Optional[List[str]] = typer.Option(None, '--glob', help="Include glob (e.g., **/*.py). Can repeat."),
    exclude: Optional[List[str]] = typer.Option(None, '--exclude', help="Exclude glob. Can repeat."),
    file: Optional[Path] = typer.Option(None, '--file', exists=True, help="Limit search to a single file"),
    context: int = typer.Option(0, '--context', help="Show N context lines around each match"),
):
    """Search files in a workspace (fallback grep)."""
    if file:
        hits = file_search(file, query, regex=regex)
    else:
        inc = list(glob) if glob else None
        exc = list(exclude) if exclude else None
        hits = search_text(workspace, query, regex=regex, include_globs=inc, exclude_globs=exc)

    if not hits:
        console.print("[yellow]No matches found.[/yellow]")
        raise typer.Exit(0)

    for h in hits[:500]:  # soft cap to avoid flooding
        try:
            text = h.path.read_text(errors="ignore") if context else ""
        except Exception:
            text = ""
        if context and text:
            start, end, seg = extract_context(text, h.line_no, before=context, after=context)
            title = f"{h.path} :{h.line_no} (lines {start}-{end})"
            console.print(Panel(seg, title=title, border_style="blue"))
        else:
            console.print(f"{h.path}:{h.line_no}: {h.line}")


@app.command()
def extract(
    file: Path = typer.Option(..., '--file', exists=True, help="File to extract from"),
    symbol: Optional[str] = typer.Option(None, '--symbol', help="Python function/class name to extract"),
    regex: Optional[str] = typer.Option(None, '--regex', help="Regex to locate line; returns context range"),
    before: int = typer.Option(3, '--before', help="Context lines before match (regex mode)"),
    after: int = typer.Option(3, '--after', help="Context lines after match (regex mode)"),
    nth: int = typer.Option(1, '--nth', help="Which match to extract (regex mode)"),
    out: Optional[Path] = typer.Option(None, '--out', help="Write extracted segment to this file"),
):
    """Extract code segments from a file by symbol (Python) or regex context."""
    if symbol:
        if file.suffix == ".py":
            res = python_extract_symbol(file, symbol)
            if not res:
                console.print("[yellow]Symbol not found or file not parseable.[/yellow]")
                raise typer.Exit(1)
            start, end, seg = res
            console.print(Panel(seg, title=f"{file}::{symbol} (lines {start}-{end})", border_style="green"))
            if out:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(seg)
                console.print(f"[green]Wrote {len(seg)} bytes to {out}[/green]")
            raise typer.Exit(0)
        else:
            console.print("[yellow]--symbol currently supports Python files only.[/yellow]")
            raise typer.Exit(1)

    if regex:
        hits = file_search(file, regex, regex=True)
        if not hits:
            console.print("[yellow]No match for regex in file.[/yellow]")
            raise typer.Exit(1)
        hit = hits[nth - 1] if len(hits) >= nth else hits[-1]
        text = file.read_text(errors="ignore")
        start, end, seg = extract_context(text, hit.line_no, before=before, after=after)
        console.print(Panel(seg, title=f"{file} lines {start}-{end}", border_style="green"))
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(seg)
            console.print(f"[green]Wrote {len(seg)} bytes to {out}[/green]")
        raise typer.Exit(0)

    console.print("[red]Provide --symbol for Python or --regex for generic extraction.[/red]")
    raise typer.Exit(2)


@app.command()
def sg(
    pattern: Optional[str] = typer.Option(None, '--pattern', help="ast-grep pattern (-p)"),
    lang: Optional[str] = typer.Option(None, '--lang', help="Language for ast-grep (-l)"),
    rule: Optional[Path] = typer.Option(None, '--rule', exists=True, help="Path to ast-grep YAML rule (-r)"),
    root: Path = typer.Option(Path.cwd(), '--root', exists=True, dir_okay=True, help="Root directory to scan"),
    json: bool = typer.Option(False, '--json', help="Output JSON from ast-grep"),
):
    """Run ast-grep if available (sg/ast-grep must be on PATH)."""
    exe = ast_grep_available()
    if not exe:
        console.print("[red]ast-grep not found.[/red] Install:\n  - brew install ast-grep\n  - or: curl -fsSL https://raw.githubusercontent.com/ast-grep/ast-grep/main/install.sh | bash")
        raise typer.Exit(127)
    code, out, err = run_ast_grep(root=root, pattern=pattern, lang=lang, rule_file=rule, json=json)
    if err.strip():
        console.print(Panel(err, title="ast-grep stderr", border_style="red"))
    if out.strip():
        console.print(out)
    raise typer.Exit(code)


if __name__ == "__main__":
    app()
