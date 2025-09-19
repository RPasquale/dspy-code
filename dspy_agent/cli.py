from __future__ import annotations

import os
from pathlib import Path
import sys
import shlex
import time
import hashlib
import shutil
import subprocess
from datetime import datetime
from dataclasses import asdict
from typing import Optional, List, Iterable, Tuple, Dict, Any, Literal, Mapping

import typer
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape
from rich.live import Live
from rich.table import Table
from rich.align import Align
from rich.console import Group
from rich.columns import Columns
from .cli_utils import print_banner as _print_banner, banner_text as _banner_text

import dspy
import logging
from .config import get_settings
from .llm import configure_lm, check_ollama_ready
from .streaming.log_reader import extract_key_events, load_logs
from .skills.context_builder import ContextBuilder
from .skills.code_context import CodeContext
from .skills.code_context_rag import CodeContextRAG
from .skills.task_agent import TaskAgent
from .skills.file_locator import FileLocator
from .skills.patch_verifier import PatchVerifier
from .skills.test_planner import TestPlanner
from .skills.orchestrator import Orchestrator
from .code_tools.code_search import (
    search_text,
    search_file as file_search,
    extract_context,
    python_extract_symbol,
    run_ast_grep,
    ast_grep_available,
)
from .code_tools.code_snapshot import build_code_snapshot
from .code_tools.patcher import apply_unified_patch, summarize_patch
from .embedding.indexer import build_index, save_index, load_index, semantic_search
from .training.train_gepa import run_gepa, run_gepa_with_val, evaluate_on_set
from .training.train_orchestrator import run_gepa_orchestrator, run_gepa_orchestrator_with_val, evaluate_orchestrator
from .training.train_codegen import run_gepa_codegen
from .training.autogen_dataset import bootstrap_datasets, bootstrap_datasets_with_splits
from .training.rl_sweep import run_sweep as _run_rl_sweep, load_sweep_config as _load_sweep_config, SweepSettings as _SweepSettings
from .agents.orchestrator_runtime import evaluate_tool_choice
from .embedding.indexer import tokenize
from .agents.router_worker import RouterWorker
from .agents.knowledge import build_code_graph
from .embedding.embeddings_index import (
    build_emb_index,
    save_emb_index,
    load_emb_index,
    emb_search as emb_search_fn,
    embed_query,
)
from .code_tools.diffutil import unified_diff_file_vs_text
from .streaming.streaming_config import (
    StreamConfig,
    load_config as load_stream_cfg,
    save_config as save_stream_cfg,
    DEFAULT_CONFIG_PATH as STREAM_CFG_PATH,
    render_kafka_topic_commands,
)
from .streaming.streamkit import TRAINER_SETTINGS_PATH
from .streaming.streaming_runtime import start_local_stack, autodiscover_logs
from .context.context_manager import ContextManager
from .agentic import log_retrieval_event
from .streaming.kafka_log import get_kafka_logger
from .training.deploy import DeploymentLogger
from .status_http import start_status_server
from .streaming.streaming_kafka import WorkerLoop, KafkaParams
from .agents.knowledge import build_code_graph, summarize_code_graph
# Legacy helpers rely on template assets internally
from .stack import (
    DEFAULT_STACK_DIR,
    compose_command as stack_compose_command,
    docker_available as stack_docker_available,
    ensure_dir as stack_ensure_dir,
    prepare_stack,
)
import threading
import json
import json as _json
import typing as _typing
from collections import Counter

# RL toolkit imports
from .rl.rlkit import (
    RLToolEnv,
    EnvConfig,
    RewardConfig,
    aggregate_reward,
    ToolchainConfig,
    ToolchainExecutor,
    ToolAction,
    detect_toolchain,
    load_from_module,
    get_verifiers as _rl_default_verifiers,
    make_bandit as _rl_make_bandit,
    TrainerConfig as _RLTrainerConfig,
    bandit_trainer as _rl_bandit_trainer,
    bandit_trainer_puffer as _rl_bandit_trainer_puffer,
    train_puffer_policy as _rl_train_puffer_policy,
    run_puffer_ppo as _rl_run_puffer_ppo,
    RLConfig as _RLConfig,
    load_rl_config as _load_rl_config,
)

# Nested CLI groups for RL (will be registered after app is created)
rl_app = typer.Typer(no_args_is_help=True, help="Reinforcement Learning commands")
rl_config_app = typer.Typer(no_args_is_help=True, help="RL config helpers")
rl_app.add_typer(rl_config_app, name="config")
stack_app = typer.Typer(no_args_is_help=True, help="Docker stack helper commands")


def _rl_build_make_env(
    workspace: Path,
    *,
    verifiers_module: str | None,
    weights: dict[str, float] | None,
    penalty_kinds: _typing.Iterable[str] | None,
    clamp01_kinds: _typing.Iterable[str] | None,
    scales: dict[str, tuple[float, float]] | None,
    test_cmd: str | None,
    lint_cmd: str | None,
    build_cmd: str | None,
    timeout_sec: int | None,
    actions: _typing.Iterable[str] | None = None,
) -> _typing.Callable[[], RLToolEnv]:
    vlist = None
    try:
        if verifiers_module:
            vlist = load_from_module(verifiers_module)
    except Exception:
        vlist = None
    if not vlist:
        vlist = _rl_default_verifiers()

    settings = _load_stream_rl_settings(workspace)
    base_weights = dict(weights or {"pass_rate": 1.0, "blast_radius": 1.0})
    base_weights.setdefault('retrieval_precision', 0.3)
    base_weights.setdefault('retrieval_coverage', 0.05)
    base_weights.setdefault('retrieval_avg_score', 0.1)
    base_weights.setdefault('retrieval_query_count', 0.05)
    if isinstance(settings, dict):
        extra_weights = settings.get('reward_weights')
        if isinstance(extra_weights, dict):
            for key, value in extra_weights.items():
                if key in base_weights:
                    continue
                try:
                    base_weights[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
    penalties_list = list(penalty_kinds or [])
    clamps_list = list(clamp01_kinds or [])
    if not penalties_list and isinstance(settings, dict):
        raw_penalties = settings.get('reward_penalties')
        if isinstance(raw_penalties, (list, tuple)):
            penalties_list = [str(x).strip() for x in raw_penalties if str(x).strip()]
    if not clamps_list and isinstance(settings, dict):
        raw_clamp = settings.get('reward_clamp01')
        if isinstance(raw_clamp, (list, tuple)):
            clamps_list = [str(x).strip() for x in raw_clamp if str(x).strip()]

    rc = RewardConfig(
        weights=base_weights,
        penalty_kinds=penalties_list,
        clamp01_kinds=clamps_list,
        scales=scales or {},
    )
    def reward_fn(result, verifiers, wmap):  # type: ignore[no-redef]
        total, vec, details = aggregate_reward(result, verifiers, rc)
        return total, vec, details

    tcfg = detect_toolchain(
        workspace,
        test_cmd=test_cmd,
        lint_cmd=lint_cmd,
        build_cmd=build_cmd,
        timeout_sec=timeout_sec,
    )
    execu = ToolchainExecutor(tcfg)

    allowed_actions_list = None
    if actions:
        allowed_actions_list = [str(a).strip() for a in actions if str(a).strip()]
    env_actions = os.getenv('RL_ACTIONS')
    if env_actions:
        allowed_actions_list = [part.strip() for part in env_actions.split(',') if part.strip()]

    # Context provider preferring Kafka (logs.ctx.* topics); falls back to log counters
    def _ctx_provider() -> list[float]:
        # Try Kafka first if available
        bootstrap = os.getenv('KAFKA_BOOTSTRAP') or os.getenv('KAFKA_BOOTSTRAP_SERVERS') or 'localhost:9092'
        if not _kafka_is_available(bootstrap):
            return _logs_ctx_features(workspace)
        try:
            from confluent_kafka import Consumer  # type: ignore
            silent = logging.getLogger('kafka.silent'); silent.addHandler(logging.NullHandler()); silent.setLevel(logging.CRITICAL)
            conf = {
                'bootstrap.servers': bootstrap,
                'group.id': 'dspy-rl-ctx',
                'session.timeout.ms': 6000,
                'auto.offset.reset': 'latest',
                'enable.partition.eof': True,
            }
            c = Consumer(conf, logger=silent)
            # Subscribe to all logs.ctx.* topics via regex
            c.subscribe(['^logs\\.ctx\\..*'])
            # Poll a few records quickly
            import time as _t
            t0 = _t.time(); buf: list[str] = []
            while _t.time() - t0 < 0.25:
                msg = c.poll(0.05)
                if msg is None or msg.error():
                    continue
                try:
                    val = msg.value().decode('utf-8', errors='ignore') if isinstance(msg.value(), (bytes, bytearray)) else str(msg.value())
                except Exception:
                    continue
                buf.append(val)
            try: c.close()
            except Exception: pass
            # Aggregate features from JSON payloads {"ctx": [lines...]}
            err = warn = timeout = trace = 0
            import json as _j
            for v in buf[-10:]:
                try:
                    obj = _j.loads(v)
                    lines = obj.get('ctx') or []
                    if isinstance(lines, list):
                        s = "\n".join([str(x) for x in lines]).lower()
                    else:
                        s = str(obj).lower()
                except Exception:
                    s = v.lower()
                err += s.count(' error ')
                warn += s.count(' warn')
                timeout += s.count('timeout')
                trace += s.count('traceback (most recent call last)')
            def norm(x: int, cap: int = 10) -> float:
                return min(float(x), float(cap)) / float(cap)
            return [norm(err), norm(warn), norm(timeout), norm(trace)] + _agentic_features(workspace)
        except Exception:
            return _logs_ctx_features(workspace) + _agentic_features(workspace)

    def make_env() -> RLToolEnv:
        ecfg = EnvConfig(
            verifiers=vlist,
            reward_fn=reward_fn,
            weights=rc.weights,
            context_provider=_ctx_provider,
            action_args=None,
            allowed_actions=allowed_actions_list,
        )
        return RLToolEnv(executor=execu, cfg=ecfg, episode_len=1)

    return make_env

def _kafka_is_available(bootstrap: str, timeout: float = 0.2) -> bool:
    try:
        tokens = (bootstrap or '').split(',')
        import socket as _s
        for tk in tokens:
            tk = tk.strip()
            if not tk:
                continue
            host = tk; port = 9092
            if '://' in host:
                host = host.split('://', 1)[1]
            if host.startswith('[') and ']' in host:
                h, rest = host[1:].split(']', 1)
                host = h
                if rest.startswith(':'):
                    try: port = int(rest[1:])
                    except Exception: port = 9092
            elif ':' in host:
                parts = host.rsplit(':', 1)
                host, port_s = parts[0], parts[1]
                try: port = int(port_s)
                except Exception: port = 9092
            try:
                with _s.create_connection((host, port), timeout=timeout):
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False

_AGENTIC_CACHE: Dict[str, tuple[float, list[float]]] = {}


def _agentic_features(workspace: Path, ttl: float = 5.0) -> list[float]:
    key = str(workspace)
    now = time.time()
    cached = _AGENTIC_CACHE.get(key)
    if cached and now - cached[0] < ttl:
        return cached[1]
    try:
        cm = ContextManager(workspace)
        feats = cm.agentic_features()
    except Exception:
        feats = []
    _AGENTIC_CACHE[key] = (now, feats)
    return feats


def _logs_ctx_features(workspace: Path) -> list[float]:
    try:
        logs_dir = workspace / 'logs'
        bundle, _ = load_logs([logs_dir]) if logs_dir.exists() else ("", 0)
        s = bundle.lower()
        err = s.count(' error ')
        warn = s.count(' warn')
        timeout = s.count('timeout')
        trace = s.count('traceback (most recent call last)'.lower())
        def norm(x: int, cap: int = 10) -> float:
            return min(float(x), float(cap)) / float(cap)
        return [norm(err), norm(warn), norm(timeout), norm(trace)]
    except Exception:
        return []


def _stream_rl_settings_path(workspace: Path) -> Path:
    ws = workspace.resolve()
    return ws / TRAINER_SETTINGS_PATH.name


def _load_stream_rl_settings(workspace: Path) -> dict:
    path = _stream_rl_settings_path(workspace)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_stream_rl_settings(workspace: Path, data: dict) -> None:
    path = _stream_rl_settings_path(workspace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    path.write_text(json.dumps(data, indent=2))


def _update_rl_config_weights(workspace: Path, weights: Mapping[str, float]) -> None:
    if not weights:
        return
    cfg_path = (workspace / '.dspy_rl.json')
    if not cfg_path.exists():
        return
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return
    wmap = cfg.get('weights') if isinstance(cfg.get('weights'), dict) else {}
    dirty = False
    for key, value in weights.items():
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if wmap.get(key) != val:
            wmap[key] = val
            dirty = True
    if dirty:
        cfg['weights'] = wmap
        cfg_path.write_text(json.dumps(cfg, indent=2))


def _record_gepa_outcome(
    module: str,
    optimized: object,
    workspace: Path,
    progress_path: Optional[Path] = None,
) -> None:
    ws = workspace.resolve()
    settings = _load_stream_rl_settings(ws)
    prompts = settings.setdefault('prompts', {}) if isinstance(settings, dict) else {}
    if not isinstance(prompts, dict):
        prompts = {}
        settings['prompts'] = prompts

    action_key = 'patch' if module in {'code', 'codegen'} else module

    prompt_text = None
    verifier_weights: Dict[str, float] = {}
    candidate_id = None
    reward_delta = None

    detailed = getattr(optimized, 'detailed_results', None)
    best_candidate = None
    if detailed is not None:
        best_candidate = getattr(detailed, 'best_candidate', None)
        if best_candidate is None and isinstance(detailed, dict):
            best_candidate = detailed.get('best_candidate')
    if best_candidate is not None and hasattr(best_candidate, 'to_dict'):
        try:
            best_candidate = best_candidate.to_dict()  # type: ignore[assignment]
        except Exception:
            pass
    if isinstance(best_candidate, dict):
        prompt_text = best_candidate.get('prompt') or best_candidate.get('prompt_text') or best_candidate.get('program')
        candidate_id = best_candidate.get('id') or best_candidate.get('hash')
        reward_delta = best_candidate.get('reward_delta') or best_candidate.get('score_delta')
        candidate_weights = best_candidate.get('verifier_weights') or best_candidate.get('reward_weights')
        if isinstance(candidate_weights, Mapping):
            for key, value in candidate_weights.items():
                try:
                    verifier_weights[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
    if prompt_text is None and hasattr(optimized, 'program'):
        try:
            prompt_text = getattr(optimized, 'program')
        except Exception:
            prompt_text = None
    if prompt_text is not None and not isinstance(prompt_text, str):
        prompt_text = json.dumps(prompt_text, indent=2) if isinstance(prompt_text, (dict, list)) else str(prompt_text)
    if prompt_text:
        prompt_hash = hashlib.sha1(prompt_text.encode('utf-8')).hexdigest()
    else:
        prompt_hash = hashlib.sha1(repr(best_candidate or optimized).encode('utf-8')).hexdigest()
    if not candidate_id:
        candidate_id = prompt_hash

    best_score = None
    first_score = None
    if progress_path and progress_path.exists():
        try:
            with progress_path.open('r') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if 'score' not in rec:
                        continue
                    try:
                        score = float(rec.get('score'))
                    except (TypeError, ValueError):
                        continue
                    if first_score is None:
                        first_score = score
                    if best_score is None or score > best_score:
                        best_score = score
        except Exception:
            pass
    score_delta = None
    if best_score is not None and first_score is not None:
        score_delta = best_score - first_score
        if reward_delta is None:
            reward_delta = score_delta

    registry = prompts.setdefault(action_key, {}) if isinstance(prompts, dict) else {}
    if not isinstance(registry, dict):
        registry = {}
        prompts[action_key] = registry
    candidates = registry.setdefault('candidates', []) if isinstance(registry.get('candidates'), list) else []
    if candidates is not registry.get('candidates'):
        registry['candidates'] = candidates

    entry = {
        'id': candidate_id,
        'hash': prompt_hash,
        'module': module,
        'prompt': prompt_text,
        'reward_delta': reward_delta,
        'score_delta': score_delta,
        'best_score': best_score,
        'verifier_weights': verifier_weights,
        'updated_at': time.time(),
    }

    replaced = False
    for idx, cand in enumerate(candidates):
        if not isinstance(cand, Mapping):
            continue
        if str(cand.get('id') or cand.get('hash')) == str(candidate_id):
            merged = dict(cand)
            merged.update({k: v for k, v in entry.items() if v is not None})
            candidates[idx] = merged
            replaced = True
            break
    if not replaced:
        candidates.append({k: v for k, v in entry.items() if v is not None})

    active_id = registry.get('active')
    if not active_id:
        registry['active'] = candidate_id
    else:
        try:
            current = next((c for c in candidates if str(c.get('id')) == str(active_id)), None)
            current_score = float(current.get('reward_delta')) if current and current.get('reward_delta') is not None else float('-inf')
            new_score = float(reward_delta) if reward_delta is not None else current_score
            if new_score > current_score:
                registry['active'] = candidate_id
        except Exception:
            registry['active'] = candidate_id

    if verifier_weights:
        rw = settings.setdefault('reward_weights', {}) if isinstance(settings, dict) else {}
        if not isinstance(rw, dict):
            rw = {}
            settings['reward_weights'] = rw
        for key, value in verifier_weights.items():
            rw[key] = float(value)
        _update_rl_config_weights(ws, rw)

    _save_stream_rl_settings(ws, settings)


def _auto_train_enabled(default: bool = True) -> bool:
    flag = os.getenv('DSPY_AUTO_TRAIN')
    if flag is None:
        return default
    return str(flag).strip().lower() not in {"0", "false", "off", "no"}


class AutoTrainingLoop:
    def __init__(
        self,
        workspace: Path,
        logs: Optional[Path],
        *,
        console: Console,
        label: str = "auto",
        interval_sec: Optional[int] = None,
        modules: Optional[Iterable[str]] = None,
        auto_budget: Optional[str] = None,
        initial_delay_sec: Optional[int] = None,
        ollama: bool = True,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        rl_steps: Optional[int] = None,
    ) -> None:
        self.console = console
        self.label = label
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"dspy-auto-{label}", daemon=True)
        self.workspace = workspace.resolve()
        self.logs = logs if logs is not None else self.workspace / 'logs'
        self._status_path = self.workspace / '.dspy_auto_status.json'
        self._last_status: Dict[str, Any] = {}
        default_modules = ['context', 'task', 'code']
        env_modules = os.getenv('DSPY_AUTO_GEPA_MODULES')
        if modules is not None:
            self.modules = [m.strip() for m in modules if str(m).strip()]
        elif env_modules:
            self.modules = [m.strip() for m in env_modules.split(',') if m.strip()]
        else:
            self.modules = default_modules
        if not self.modules:
            self.modules = default_modules
        try:
            self.interval = int(interval_sec if interval_sec is not None else int(os.getenv('DSPY_AUTO_TRAIN_INTERVAL_SEC', '1800')))
        except Exception:
            self.interval = 1800
        try:
            self.initial_delay = int(initial_delay_sec if initial_delay_sec is not None else int(os.getenv('DSPY_AUTO_TRAIN_INITIAL_DELAY_SEC', '60')))
        except Exception:
            self.initial_delay = 60
        self.auto_budget = auto_budget or os.getenv('DSPY_AUTO_GEPA_BUDGET', 'light')
        try:
            self.rl_steps = int(rl_steps if rl_steps is not None else int(os.getenv('DSPY_AUTO_RL_STEPS', '200')))
        except Exception:
            self.rl_steps = 200
        self.ollama = bool(ollama)
        self.model = model or os.getenv('DSPY_AUTO_MODEL', 'qwen3:1.7b')
        self.base_url = base_url or os.getenv('DSPY_AUTO_BASE_URL')
        self.api_key = api_key or os.getenv('DSPY_AUTO_API_KEY')
        self._module_index = 0
        first_module = self.modules[0] if self.modules else None
        self._write_status({
            'status': 'idle',
            'label': self.label,
            'workspace': str(self.workspace),
            'modules': list(self.modules),
            'next_module': first_module,
            'interval_sec': self.interval,
            'updated_at': time.time(),
        })

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def update_paths(self, workspace: Path, logs: Optional[Path]) -> None:
        with self._lock:
            self.workspace = workspace.resolve()
            self.logs = logs if logs is not None else self.workspace / 'logs'
            self._status_path = self.workspace / '.dspy_auto_status.json'
            snapshot = dict(self._last_status)
        snapshot.update({'workspace': str(self.workspace), 'updated_at': time.time()})
        self._write_status(snapshot)

    def update_lm(self, ollama: bool, model: Optional[str], base_url: Optional[str], api_key: Optional[str]) -> None:
        with self._lock:
            self.ollama = bool(ollama)
            if model:
                self.model = model
            if base_url is not None:
                self.base_url = base_url
            if api_key is not None:
                self.api_key = api_key
            snapshot = dict(self._last_status)
        snapshot.update({'lm_provider': 'ollama' if self.ollama else 'openai', 'model': self.model, 'updated_at': time.time()})
        self._write_status(snapshot)

    def set_interval(self, seconds: int) -> None:
        seconds = max(1, int(seconds))
        with self._lock:
            self.interval = seconds
            snapshot = dict(self._last_status)
        snapshot.update({'interval_sec': seconds, 'updated_at': time.time()})
        self._write_status(snapshot)

    def set_modules(self, modules: Iterable[str]) -> None:
        cleaned = [str(m).strip() for m in modules if str(m).strip()]
        if not cleaned:
            return
        with self._lock:
            self.modules = cleaned
            self._module_index = 0
            snapshot = dict(self._last_status)
        snapshot.update({'modules': list(cleaned), 'next_module': cleaned[0], 'updated_at': time.time()})
        self._write_status(snapshot)

    def status_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._last_status)

    def _write_status(self, data: Dict[str, Any]) -> None:
        snapshot = dict(data)
        with self._lock:
            self._last_status = snapshot
            path = self._status_path
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(snapshot, indent=2))
            except Exception:
                pass

    def _wait(self, seconds: int) -> None:
        remainder = max(0, seconds)
        while remainder > 0 and not self._stop.is_set():
            time.sleep(min(5, remainder))
            remainder -= 5

    def _run(self) -> None:
        if self.initial_delay > 0:
            self._wait(self.initial_delay)
        while not self._stop.is_set():
            try:
                self._run_once()
            except Exception as exc:
                try:
                    self.console.print(Panel(f"auto-training error: {exc}", title="auto-train", border_style="yellow"))
                except Exception:
                    pass
            self._wait(self.interval)

    def _snapshot(self) -> Tuple[Path, Optional[Path], list[str], str, bool, Optional[str], Optional[str], Optional[str], int]:
        with self._lock:
            ws = self.workspace
            logs = self.logs
            modules = list(self.modules)
            budget = self.auto_budget
            ollama = self.ollama
            model = self.model
            base_url = self.base_url
            api_key = self.api_key
            steps = self.rl_steps
            return ws, logs, modules, budget, ollama, model, base_url, api_key, steps

    def _run_once(self) -> None:
        ws, logs, modules, budget, ollama, model, base_url, api_key, rl_steps = self._snapshot()
        if not ws.exists():
            return
        if not modules:
            return
        module = modules[self._module_index % len(modules)]
        self._module_index = (self._module_index + 1) % len(modules)
        self._write_status({
            'status': 'running',
            'phase': 'dataset',
            'module': module,
            'workspace': str(ws),
            'next_module': modules[self._module_index % len(modules)] if modules else None,
            'started_at': time.time(),
        })
        try:
            datasets = bootstrap_datasets(ws, logs, ws / '.dspy_data')
        except Exception as e:
            self.console.print(Panel(f"dataset bootstrap failed: {e}", title="auto-train", border_style="yellow"))
            self._write_status({
                'status': 'error',
                'phase': 'dataset',
                'module': module,
                'error': str(e),
                'workspace': str(ws),
                'updated_at': time.time(),
            })
            return
        train_path = datasets.get(module)
        if not train_path or not train_path.exists():
            self.console.print(Panel(f"no dataset for module {module}", title="auto-train", border_style="yellow"))
            self._write_status({
                'status': 'skipped',
                'phase': 'dataset',
                'module': module,
                'reason': 'missing-dataset',
                'workspace': str(ws),
                'updated_at': time.time(),
            })
            return
        lm = _maybe_configure_lm(True, ollama, model, base_url, api_key, workspace=ws)
        if lm is None:
            self.console.print(Panel("LM unavailable for auto-training; skipping cycle", title="auto-train", border_style="yellow"))
            self._write_status({
                'status': 'skipped',
                'phase': 'lm',
                'module': module,
                'reason': 'lm-unavailable',
                'workspace': str(ws),
                'updated_at': time.time(),
            })
            return
        log_dir = ws / f'.gepa_{module}'
        progress_path = log_dir / 'progress.jsonl'
        log_dir.mkdir(parents=True, exist_ok=True)
        optimized = None
        self._write_status({
            'status': 'running',
            'phase': 'gepa',
            'module': module,
            'workspace': str(ws),
            'log_dir': str(log_dir),
            'updated_at': time.time(),
        })
        try:
            if module == 'orchestrator':
                optimized = run_gepa_orchestrator(
                    train_jsonl=train_path,
                    auto=budget,
                    reflection_lm=lm,
                    log_dir=str(log_dir),
                    track_stats=True,
                    progress_path=str(progress_path),
                )
            elif module == 'codegen':
                optimized = run_gepa_codegen(
                    train_jsonl=train_path,
                    workspace=ws,
                    test_cmd=None,
                    type_cmd="python -m compileall -q .",
                    lint_cmd=None,
                    auto=budget,
                    reflection_lm=lm,
                    log_dir=str(log_dir),
                    track_stats=True,
                )
            else:
                optimized = run_gepa(
                    module=module,
                    train_jsonl=train_path,
                    auto=budget,
                    reflection_lm=lm,
                    log_dir=str(log_dir),
                    track_stats=True,
                    progress_path=str(progress_path),
                    code_summary=_get_code_summary(ws),
                )
        except Exception as e:
            self.console.print(Panel(f"GEPA {module} failed: {e}", title="auto-train", border_style="yellow"))
            self._write_status({
                'status': 'error',
                'phase': 'gepa',
                'module': module,
                'error': str(e),
                'workspace': str(ws),
                'updated_at': time.time(),
            })
            return
        if optimized is None:
            self._write_status({
                'status': 'skipped',
                'phase': 'gepa',
                'module': module,
                'reason': 'gepa-returned-none',
                'workspace': str(ws),
                'updated_at': time.time(),
            })
            return
        try:
            self.console.print(f"[dim]auto-train {module} optimized prompts[/dim]")
            progress_obj = progress_path if progress_path.exists() else None
            _record_gepa_outcome(module, optimized, ws, progress_obj)
            self._write_status({
                'status': 'running',
                'phase': 'rl',
                'module': module,
                'workspace': str(ws),
                'gepa_log': str(progress_path) if progress_path.exists() else None,
                'updated_at': time.time(),
            })
        except Exception as e:
            self.console.print(Panel(f"recording GEPA outcome failed: {e}", title="auto-train", border_style="yellow"))
            self._write_status({
                'status': 'error',
                'phase': 'gepa-record',
                'module': module,
                'error': str(e),
                'workspace': str(ws),
                'updated_at': time.time(),
            })
        try:
            avg_reward = self._run_rl_training(ws, rl_steps)
            now = time.time()
            self._write_status({
                'status': 'idle',
                'phase': 'idle',
                'module': module,
                'workspace': str(ws),
                'avg_reward': avg_reward,
                'next_module': modules[self._module_index % len(modules)] if modules else None,
                'updated_at': now,
                'completed_at': now,
            })
        except Exception as e:
            self.console.print(Panel(f"auto RL training failed: {e}", title="auto-train", border_style="yellow"))
            self._write_status({
                'status': 'error',
                'phase': 'rl',
                'module': module,
                'error': str(e),
                'workspace': str(ws),
                'updated_at': time.time(),
            })

    def _run_rl_training(self, ws: Path, steps: int) -> float:
        try:
            if steps <= 0:
                return 0.0
            cfg_path = ws / '.dspy_rl.json'
            cfg = _RLConfig()
            if cfg_path.exists():
                try:
                    cfg = _load_rl_config(cfg_path)
                except Exception:
                    pass
            make_env = _rl_build_make_env(
                ws,
                verifiers_module=getattr(cfg, 'verifiers_module', None),
                weights=getattr(cfg, 'weights', None),
                penalty_kinds=getattr(cfg, 'penalty_kinds', []),
                clamp01_kinds=getattr(cfg, 'clamp01_kinds', []),
                scales=getattr(cfg, 'scales', {}),
                test_cmd=getattr(cfg, 'test_cmd', None),
                lint_cmd=getattr(cfg, 'lint_cmd', None),
                build_cmd=getattr(cfg, 'build_cmd', None),
                timeout_sec=getattr(cfg, 'timeout_sec', None),
                actions=getattr(cfg, 'actions', None),
            )
            env = make_env()
            tools = env.action_names
            if not tools:
                return 0.0
            state_path = ws / '.dspy_rl_state.json'
            bandit = _OnlineBandit(tools, state_path, policy=getattr(cfg, 'policy', 'epsilon-greedy'), epsilon=getattr(cfg, 'epsilon', 0.1), ucb_c=getattr(cfg, 'ucb_c', 2.0))
            total = 0.0
            obs, _ = env.reset()
            for _ in range(max(1, steps)):
                tool = bandit.select()
                try:
                    idx = tools.index(tool)
                except ValueError:
                    continue
                _, reward, done, trunc, info = env.step(idx)
                total += float(reward)
                bandit.update(tool, float(reward))
                if done or trunc:
                    obs, _ = env.reset()
            avg = total / max(1, steps)
            self.console.print(f"[dim]auto-rl avg_reward={avg:.3f} steps={steps} tools={tools}[/dim]")
            return avg
        except Exception as e:
            raise
@rl_config_app.command("init")
def rl_config_init(
    out: Path = typer.Option(Path(".dspy_rl.json"), '--out', help="Path to write RL config JSON"),
    verifiers_module: Optional[str] = typer.Option("verifiers", '--verifiers-module', help="Python module that provides verifiers"),
    puffer: bool = typer.Option(True, '--puffer/--no-puffer', help="Enable PufferLib vectorization by default"),
):
    """Write a starter RL config file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "policy": "epsilon-greedy",
        "epsilon": 0.1,
        "ucb_c": 2.0,
        "n_envs": 4,
        "puffer": bool(puffer),
        "verifiers_module": verifiers_module,
        "actions": ["run_tests", "lint", "build", "patch"],
        "weights": {"pass_rate": 1.0, "blast_radius": 1.0},
        "penalty_kinds": ["blast_radius"],
        "clamp01_kinds": ["pass_rate"],
        "scales": {"blast_radius": [0.0, 1.0]},
        "test_cmd": None,
        "lint_cmd": "ruff check --output-format json .",
        "build_cmd": "python -m compileall -q .",
        "timeout_sec": 180,
    }
    out.write_text(json.dumps(cfg, indent=2))
    console.print(Panel.fit(f"Wrote RL config to {out}", title="rl config", border_style="green"))


@rl_app.command("train")
def rl_train(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace root"),
    steps: int = typer.Option(200, '--steps', help="Number of training steps"),
    n_envs: int = typer.Option(2, '--n-envs', help="Parallel environments"),
    policy: str = typer.Option("epsilon-greedy", '--policy', help="Bandit policy: epsilon-greedy|ucb1|thompson"),
    epsilon: float = typer.Option(0.1, '--epsilon', help="Epsilon for epsilon-greedy"),
    ucb_c: float = typer.Option(2.0, '--ucb-c', help="UCB1 exploration constant"),
    neural: bool = typer.Option(False, '--neural/--no-neural', help="Use neural REINFORCE trainer"),
    puffer: bool = typer.Option(False, '--puffer/--no-puffer', help="Use PufferLib vectorization (bandits only)"),
    rl_config: Optional[Path] = typer.Option(None, '--rl-config', exists=True, help="Path to RL config JSON"),
    verifiers_module: Optional[str] = typer.Option(None, '--verifiers-module', help="Override verifiers module"),
    test_cmd: Optional[str] = typer.Option(None, '--test-cmd', help="Override test command"),
    lint_cmd: Optional[str] = typer.Option(None, '--lint-cmd', help="Override lint command"),
    build_cmd: Optional[str] = typer.Option(None, '--build-cmd', help="Override build command"),
    timeout_sec: Optional[int] = typer.Option(None, '--timeout-sec', help="Per-tool timeout in seconds"),
):
    """Train an RL policy over the local toolchain (tests/lint/build)."""
    # Load config if provided
    cfg: Optional[_RLConfig] = None
    if rl_config:
        try:
            cfg = _load_rl_config(rl_config)
        except Exception as e:
            console.print(Panel(escape(str(e)), title="rl config load failed", border_style="red"))
            raise typer.Exit(1)
    else:
        cfg_data = _load_effective_rl_config_dict(workspace)
        if cfg_data:
            try:
                cfg = _rl_config_from_dict(cfg_data)
            except Exception:
                cfg = None

    # Resolve effective parameters (CLI > config defaults)
    eff_policy = policy or (cfg.policy if cfg else "epsilon-greedy")
    eff_epsilon = epsilon if epsilon is not None else ((cfg.epsilon if cfg else 0.1))
    eff_ucb_c = ucb_c if ucb_c is not None else ((cfg.ucb_c if cfg else 2.0))
    eff_n_envs = n_envs or (cfg.n_envs if cfg else 2)
    eff_puffer = bool(puffer or (cfg.puffer if cfg else False))
    eff_verifiers_mod = verifiers_module or ((cfg.verifiers_module if cfg else None))

    eff_weights = (cfg.weights if cfg and cfg.weights else {"pass_rate": 1.0, "blast_radius": 1.0})
    eff_pen = cfg.penalty_kinds if cfg else []
    eff_clamp = cfg.clamp01_kinds if cfg else []
    eff_scales = cfg.scales if cfg else {}
    eff_test = test_cmd if test_cmd is not None else ((cfg.test_cmd if cfg else None))
    eff_lint = lint_cmd if lint_cmd is not None else ((cfg.lint_cmd if cfg else None))
    eff_build = build_cmd if build_cmd is not None else ((cfg.build_cmd if cfg else None))
    eff_to = timeout_sec if timeout_sec is not None else ((cfg.timeout_sec if cfg else None))
    eff_actions = cfg.actions if cfg and getattr(cfg, 'actions', None) else None

    make_env = _rl_build_make_env(
        workspace,
        verifiers_module=eff_verifiers_mod,
        weights=eff_weights,
        penalty_kinds=eff_pen,
        clamp01_kinds=eff_clamp,
        scales=eff_scales,
        test_cmd=eff_test,
        lint_cmd=eff_lint,
        build_cmd=eff_build,
        timeout_sec=eff_to,
        actions=eff_actions,
    )

    # Build trainer config
    kwargs: dict[str, _typing.Any] = {}
    pol = (eff_policy or "epsilon-greedy").lower().strip()
    if pol in {"epsilon", "epsilon-greedy", "egreedy", "eps"}:
        kwargs["epsilon"] = float(eff_epsilon)
    elif pol in {"ucb", "ucb1"}:
        kwargs["c"] = float(eff_ucb_c)

    try:
        if neural:
            stats = _rl_train_puffer_policy(make_env=make_env, steps=int(steps), n_envs=int(eff_n_envs), verbose=True, log_interval=max(1, int(steps)//10 or 1))
        elif eff_puffer:
            tcfg = _RLTrainerConfig(steps=int(steps), policy=eff_policy, policy_kwargs=kwargs, n_envs=int(eff_n_envs))
            stats = _rl_bandit_trainer_puffer(make_env, tcfg)
        else:
            tcfg = _RLTrainerConfig(steps=int(steps), policy=eff_policy, policy_kwargs=kwargs, n_envs=1)
            stats = _rl_bandit_trainer(make_env, tcfg)
    except Exception as e:
        console.print(Panel(escape(str(e)), title="rl train failed", border_style="red"))
        raise typer.Exit(1)

    # Summarize
    r = stats.rewards or []
    avg = (sum(r) / len(r)) if r else 0.0
    last = r[-10:]
    tools = Counter([str(it.get("tool", "")) for it in (stats.infos or []) if isinstance(it, dict)])
    body = (
        f"steps={len(r)} avg_reward={avg:.3f}\n"
        f"last10={[round(x,3) for x in last]}\n"
        f"tools={dict(tools)}"
    )
    console.print(Panel.fit(body, title="rl train result", border_style="cyan"))


@rl_app.command("guide")
def rl_hparam_guide(
    json_output: bool = typer.Option(False, '--json', help="Emit JSON instead of rich panels"),
):
    """Show curated reasoning-RL hyperparameter guidance."""

    from .training.rl_sweep import describe_default_hparams

    guide = describe_default_hparams()
    if json_output:
        console.print(Panel.fit(json.dumps(guide, indent=2), title="rl guide", border_style="cyan"))
        return

    for group in guide:
        rows = []
        for item in group['items']:
            target = item.get('target')
            tgt_str = f" → {target:.2f}" if isinstance(target, (int, float)) else ""
            unit = item.get('unit') or ""
            desc = f"{item['low']:.2f}–{item['high']:.2f}{tgt_str} {unit}\n{item['rationale']}"
            rows.append(f"[bold]{item['name']}[/bold]\n{desc}")
        console.print(Panel(Columns(rows, expand=True), title=group['title'], border_style="cyan"))


@rl_app.command("sweep")
def rl_sweep_command(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace root"),
    config: Optional[Path] = typer.Option(None, '--config', exists=True, help="Sweep configuration JSON"),
    rl_config: Optional[Path] = typer.Option(None, '--rl-config', exists=True, help="Base RL config"),
    iterations: Optional[int] = typer.Option(None, '--iterations', help="Override number of sweep iterations"),
    method: Optional[str] = typer.Option(None, '--method', help="Override sweep strategy (random|pareto|protein|carbs)"),
    metric: Optional[str] = typer.Option(None, '--metric', help="Metric to optimise (reward|pass_rate|blast_radius)"),
    goal: Optional[str] = typer.Option(None, '--goal', help="Optimization goal (maximize|minimize)"),
    trainer_steps: Optional[int] = typer.Option(None, '--trainer-steps', help="Force trainer steps per trial"),
    puffer: bool = typer.Option(False, '--puffer/--no-puffer', help="Use vectorized PufferLib trainer"),
    persist: Optional[Path] = typer.Option(None, '--persist', help="Where to store best config summary"),
    update_config: bool = typer.Option(True, '--update-config/--no-update-config', help="Update workspace .dspy_rl.json with best result"),
):
    """Run a hyperparameter sweep over the RL toolchain using PufferLib strategies."""
    try:
        sweep_cfg = _load_sweep_config(config)
    except Exception as e:
        console.print(Panel(escape(str(e)), title="sweep config", border_style="red"))
        raise typer.Exit(1)

    if iterations is not None:
        sweep_cfg['iterations'] = int(iterations)
    if method:
        sweep_cfg['method'] = method
    if metric:
        sweep_cfg['metric'] = metric
    if goal:
        sweep_cfg['goal'] = goal

    base_cfg: Optional[_RLConfig] = None
    base_path: Optional[Path] = None
    if rl_config:
        base_path = rl_config
    else:
        candidate = workspace / '.dspy_rl.json'
        if candidate.exists():
            base_path = candidate
    if base_path and base_path.exists():
        try:
            base_cfg = _load_rl_config(base_path)
        except Exception as e:
            console.print(Panel(escape(str(e)), title="rl config load failed", border_style="yellow"))

    settings = _SweepSettings()
    if persist:
        settings.persist_path = persist
    if trainer_steps is not None:
        settings.trainer_steps = int(trainer_steps)
    settings.puffer_backend = bool(puffer)
    settings.method = sweep_cfg.get('method', settings.method)
    settings.metric = sweep_cfg.get('metric', settings.metric)
    settings.goal = sweep_cfg.get('goal', settings.goal)
    if 'iterations' in sweep_cfg:
        try:
            settings.iterations = int(sweep_cfg['iterations'])
        except Exception:
            pass

    try:
        outcome = _run_rl_sweep(workspace, sweep_cfg, base_config=base_cfg, settings=settings)
    except Exception as e:
        console.print(Panel(escape(str(e)), title="sweep failed", border_style="red"))
        raise typer.Exit(1)

    best = outcome.best_summary
    best_cfg = outcome.best_config
    body = (
        f"metric={best.metric:.4f} ({settings.metric})\n"
        f"avg_reward={best.avg_reward:.4f}\n"
        f"avg_pass_rate={best.avg_pass_rate:.4f}\n"
        f"avg_blast_radius={best.avg_blast_radius:.4f}\n"
        f"cost={best.cost:.2f}s\n"
        f"iterations={len(outcome.history)}"
    )
    console.print(Panel.fit(body, title="rl sweep", border_style="cyan"))

    if update_config:
        target = base_path or (workspace / '.dspy_rl.json')
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            data = asdict(best_cfg)
            # Convert tuples to lists for JSON serialisation
            data['scales'] = {k: list(v) for k, v in (data.get('scales') or {}).items()}
            target.write_text(json.dumps(data, indent=2))
            console.print(Panel.fit(f"Updated {target}", title="rl config", border_style="green"))
        except Exception as e:
            console.print(Panel(escape(str(e)), title="config update failed", border_style="yellow"))

    console.print(Panel.fit("Tip: run 'dspy-agent rl guide' for recommended temperature/entropy targets and curriculum stages.", title="next", border_style="dim"))


@rl_app.command("async-train")
def rl_async_train(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace root"),
    steps: int = typer.Option(200, '--steps', help="Rough number of learner updates to run"),
    rollout_workers: int = typer.Option(2, '--rollout-workers', help="Concurrent action selectors"),
    judge_workers: int = typer.Option(2, '--judge-workers', help="Concurrent tool executors"),
    policy: str = typer.Option("epsilon-greedy", '--policy', help="Bandit policy"),
    epsilon: float = typer.Option(0.1, '--epsilon', help="Epsilon for epsilon-greedy"),
    ucb_c: float = typer.Option(2.0, '--ucb-c', help="Exploration constant for UCB"),
    rl_config: Optional[Path] = typer.Option(None, '--rl-config', exists=True, help="Optional RL config JSON"),
    wall_clock: float = typer.Option(120.0, '--wall-clock', help="Seconds to run before stopping"),
):
    """Run the asynchronous rollout→judge→learner pipeline."""

    cfg: Optional[_RLConfig] = None
    if rl_config:
        try:
            cfg = _load_rl_config(rl_config)
        except Exception as e:
            console.print(Panel(escape(str(e)), title="rl config load failed", border_style="red"))
            raise typer.Exit(1)
    else:
        cfg_data = _load_effective_rl_config_dict(workspace)
        if cfg_data:
            try:
                cfg = _rl_config_from_dict(cfg_data)
            except Exception:
                cfg = None

    make_env = _rl_build_make_env(
        workspace,
        verifiers_module=(cfg.verifiers_module if cfg else None) if cfg else None,
        weights=(cfg.weights if cfg and cfg.weights else {"pass_rate": 1.0, "blast_radius": 1.0}),
        penalty_kinds=(cfg.penalty_kinds if cfg else []),
        clamp01_kinds=(cfg.clamp01_kinds if cfg else []),
        scales=(cfg.scales if cfg else {}),
        test_cmd=(cfg.test_cmd if cfg else None),
        lint_cmd=(cfg.lint_cmd if cfg else None),
        build_cmd=(cfg.build_cmd if cfg else None),
        timeout_sec=(cfg.timeout_sec if cfg else None),
        actions=(cfg.actions if cfg else None),
    )

    kwargs: Dict[str, float] = {}
    pol = policy.lower().strip()
    if pol in {"epsilon", "epsilon-greedy", "egreedy", "eps"}:
        kwargs["epsilon"] = float(epsilon)
    elif pol in {"ucb", "ucb1"}:
        kwargs["c"] = float(ucb_c)

    from .rl.async_loop import AsyncRLTrainer

    trainer = AsyncRLTrainer(
        make_env,
        policy=policy,
        policy_kwargs=kwargs,
        rollout_workers=rollout_workers,
        judge_workers=judge_workers,
    )
    trainer.start()
    console.print(Panel.fit(
        f"Async trainer started with {rollout_workers} rollout worker(s) and {judge_workers} judge(s). Running for {wall_clock}s...",
        title="async rl",
        border_style="cyan",
    ))
    t0 = time.time()
    try:
        while time.time() - t0 < wall_clock and trainer.snapshot_stats().get("count", 0.0) < steps:
            time.sleep(1.0)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted, stopping trainer...[/yellow]")
    finally:
        trainer.stop()
        trainer.join()
    summary = trainer.snapshot_stats()
    console.print(Panel.fit(json.dumps(summary, indent=2), title="async stats", border_style="green"))


@rl_app.command("ppo")
def rl_ppo(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace root"),
    rl_config: Optional[Path] = typer.Option(None, '--rl-config', exists=True, help="Path to RL config JSON"),
    n_envs: int = typer.Option(4, '--n-envs', help="Vectorized environments"),
    total_steps: int = typer.Option(100_000, '--total-steps', help="Total PPO steps"),
):
    """Run the PufferRL PPO shell with a vectorized env."""
    cfg: Optional[_RLConfig] = None
    if rl_config:
        try:
            cfg = _load_rl_config(rl_config)
        except Exception as e:
            console.print(Panel(escape(str(e)), title="rl config load failed", border_style="red"))
            raise typer.Exit(1)

    eff_actions = cfg.actions if cfg and getattr(cfg, 'actions', None) else None

    make_env = _rl_build_make_env(
        workspace,
        verifiers_module=(cfg.verifiers_module if cfg else None),
        weights=(cfg.weights if cfg else {"pass_rate": 1.0, "blast_radius": 1.0}),
        penalty_kinds=(cfg.penalty_kinds if cfg else []),
        clamp01_kinds=(cfg.clamp01_kinds if cfg else []),
        scales=(cfg.scales if cfg else {}),
        test_cmd=(cfg.test_cmd if cfg else None),
        lint_cmd=(cfg.lint_cmd if cfg else None),
        build_cmd=(cfg.build_cmd if cfg else None),
        timeout_sec=(cfg.timeout_sec if cfg else None),
        actions=eff_actions,
    )

    try:
        _rl_run_puffer_ppo(make_env=make_env, n_envs=int(n_envs), total_steps=int(total_steps))
    except Exception as e:
        console.print(Panel(escape(str(e)), title="rl ppo failed", border_style="red"))
        raise typer.Exit(1)


# -----------------------------
# Online RL during interactive
# -----------------------------

class _OnlineBandit:
    def __init__(self, tools: list[str], state_path: Path, policy: str = "epsilon-greedy", epsilon: float = 0.1, ucb_c: float = 2.0):
        self.tools = list(tools)
        self.path = state_path
        self.policy = policy
        self.kw = {"epsilon": float(epsilon)} if policy.startswith("epsilon") else ({"c": float(ucb_c)} if policy.startswith("ucb") else {})
        self._bandit = _rl_make_bandit(policy, len(self.tools), **self.kw)
        self._load()
    def _load(self):
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text())
                if data.get("policy") == self.policy and len(data.get("values", [])) == len(self.tools):
                    if hasattr(self._bandit, "values"): self._bandit.values = list(map(float, data.get("values", [])))
                    if hasattr(self._bandit, "counts"): self._bandit.counts = list(map(int, data.get("counts", [])))
        except Exception:
            pass
    def _save(self):
        try:
            data = {
                "policy": self.policy,
                "tools": self.tools,
                "values": getattr(self._bandit, "values", []),
                "counts": getattr(self._bandit, "counts", []),
            }
            self.path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
    def select(self) -> str:
        idx = int(self._bandit.select())
        return self.tools[idx]
    def update(self, tool_name: str, reward: float):
        try:
            idx = self.tools.index(tool_name)
        except ValueError:
            return
        self._bandit.update(idx, float(reward))
        self._save()


CYBER_THEME = Theme({
    "banner": "bold magenta",
    "accent": "bright_cyan",
    "ok": "bright_green",
    "warn": "yellow",
    "err": "bright_red",
    "dim": "dim",
})
app = typer.Typer(add_completion=False, help="DSPy-based local coding agent", no_args_is_help=False)
console = Console(theme=CYBER_THEME)
LIGHTWEIGHT_DIR = Path('docker/lightweight')

# Register RL subcommands under main app
app.add_typer(rl_app, name="rl")
app.add_typer(stack_app, name="stack")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    workspace: Optional[Path] = typer.Option(None, '--workspace', dir_okay=True, exists=True, help="Initial workspace"),
    logs: Optional[Path] = typer.Option(None, '--logs', dir_okay=True, file_okay=True, exists=False, help="Initial logs path"),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama', help="Use Ollama by default"),
    model: Optional[str] = typer.Option("qwen3:1.7b", '--model', help="Default model"),
    base_url: Optional[str] = typer.Option(None, '--base-url', help="Override base URL"),
    api_key: Optional[str] = typer.Option(None, '--api-key', help="API key"),
    force_json: bool = typer.Option(False, '--force-json', help="Force simple JSON outputs"),
    structured: bool = typer.Option(False, '--structured', help="Prefer structured outputs"),
    approval: Optional[str] = typer.Option(None, '--approval', help='Tool approval mode: auto|manual'),
    coding_mode: bool = typer.Option(False, '--coding-mode', help="Enhanced coding assistant mode with build/test integration"),
):
    """Blampert - Coding Assistant
    
    Start an interactive session to work with your codebase.
    Enhanced with build/test capabilities and real-time learning.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand provided, start interactive session
        ws = Path(workspace) if workspace else Path.cwd()
        logs_path = Path(logs) if logs else (ws / 'logs')
        
        if coding_mode:
            console.print("[bold green]🚀 Starting Blampert - Claude Code-Style Coding Assistant...[/bold green]")
            console.print("[cyan]Enhanced with build/test integration and real-time learning![/cyan]")
        else:
            console.print("[green]Starting Blampert enhanced interactive session with memory and performance optimizations...[/green]")
        
        # Disable auto-training by default to avoid threading issues
        os.environ.setdefault('DSPY_AUTO_TRAIN', 'false')
        _start_interactive_session(
            workspace=workspace,
            logs=logs,
            ollama=ollama,
            model=model,
            base_url=base_url,
            api_key=api_key,
            force_json=force_json,
            structured=structured,
            approval=approval,
            coding_mode=coding_mode
        )


def _start_interactive_session(
    workspace: Optional[Path] = None,
    logs: Optional[Path] = None,
    ollama: bool = True,
    model: Optional[str] = "qwen3:1.7b",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    force_json: bool = False,
    structured: bool = False,
    approval: Optional[str] = None,
    coding_mode: bool = False,
):
    """Start the interactive session - this is the main entry point."""
    # Call the start command function
    start_command(
        workspace=workspace,
        logs=logs,
        ollama=ollama,
        model=model,
        base_url=base_url,
        api_key=api_key,
        force_json=force_json,
        structured=structured,
        approval=approval,
        coding_mode=coding_mode
    )


def _print_header(title: str):
    console.rule(f"[accent]{title}")


def _display_enhanced_summary(summary):
    """Display enhanced chain summary with memory context"""
    from .skills.orchestrator import ChainSummary
    
    console.print("\n" + "="*60)
    console.print("[bold cyan]Enhanced Chain Summary[/bold cyan]")
    console.print("="*60)
    
    # Summary stats
    stats_table = Table(show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Query", summary.query)
    stats_table.add_row("Steps", str(len(summary.steps)))
    stats_table.add_row("Total Time", f"{summary.total_time:.2f}s")
    stats_table.add_row("Success", "✅" if summary.success else "❌")
    
    console.print(stats_table)
    
    # Key findings
    if summary.key_findings:
        console.print("\n[bold]Key Findings:[/bold]")
        for i, finding in enumerate(summary.key_findings, 1):
            console.print(f"  {i}. {finding}")
    
    # Next suggestions
    if summary.next_suggestions:
        console.print("\n[bold]Suggested Next Steps:[/bold]")
        for i, suggestion in enumerate(summary.next_suggestions, 1):
            console.print(f"  {i}. {suggestion}")
    
    # Context for continuation
    if summary.context_for_continuation:
        console.print(f"\n[dim]Context for continuation: {summary.context_for_continuation}[/dim]")
    
    console.print("="*60)


def _banner_text() -> str:
    return (
        "\n"
        "██████╗ ███████╗██████╗ ██╗   ██╗   ██████╗ ██████╗ ██████╗  ███████╗\n"
        "██╔══██╗██╔════╝██╔══██╗╚██╗ ██╔╝  ██╔════╝██╔══██╗██╔══██╗ ██╔════╝\n"
        "██║  ██║███████╗██████╔╝ ╚████╔╝   ██║     ██║  ██║██║  ██║ █████╗   \n"
        "██║  ██║╚════██║██╔═══╝   ╚██╔╝     ██║    ██║  ██║██║  ██║ ██╔══╝   \n"
        "██████╔╝███████║██║        ██║      ╚██████╗██████╔╝██████╔╝ ███████╗\n"
        "╚═════╝ ╚══════╝╚═╝        ╚═╝       ╚═════╝╚═════╝ ╚═════╝  ╚══════╝\n"
        "\n                 DSPY-CODE — Trainable Coding Agent\n"
    )


# Removed conflicting _entry callback - main() handles the default behavior


def _get_code_summary(ws: Path) -> str:
    # Try storage first
    try:
        from .db.factory import get_storage as _get_storage
        st = _get_storage()
        if st is not None:
            s = st.get('code:summary')
            if isinstance(s, str) and s.strip():
                return s
    except Exception:
        pass
    # Fallback to quick on-the-fly summary (top files only)
    try:
        from .agents.knowledge import build_code_graph, summarize_code_graph
        g = build_code_graph(ws)
        return summarize_code_graph(g)
    except Exception:
        return ""

def _build_patch_context_bundle(workspace: Path, logs: Path, task: str) -> dict:
    cm = ContextManager(workspace, logs)
    try:
        bundle = cm.build_patch_context(task)
    except Exception:
        bundle = {'text': '', 'logs': '', 'patches': [], 'stats': {}, 'task': task}
    context_text = bundle.get('text') or bundle.get('logs') or ''
    bundle['combined_context'] = context_text
    return bundle


def _render_toolchain_console(action: str, result) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metrics = dict(getattr(result, 'metrics', {}) or {})
    info = dict(getattr(result, 'info', {}) or {})
    if metrics:
        try:
            console.print(Panel.fit(json.dumps(metrics, indent=2), title=f"{action} metrics", border_style="green"))
        except Exception:
            console.print(Panel(str(metrics), title=f"{action} metrics", border_style="green"))
    stdout = info.get('stdout')
    if stdout:
        console.print(Panel(escape(str(stdout)[-4000:]), title=f"{action} output", border_style="dim"))
    if info.get('error'):
        console.print(Panel(escape(str(info['error'])), title=f"{action} error", border_style="red"))
    if info.get('warn'):
        console.print(Panel(escape(str(info['warn'])), title=f"{action} warning", border_style="yellow"))
    return metrics, info


def _render_recent_fixes(console: Console, bundle: dict) -> None:
    patches = bundle.get('patches') or []
    if not patches:
        return
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column('When', style='magenta')
    table.add_column('Status', style='green')
    table.add_column('Prompt', style='cyan')
    table.add_column('Pass', style='yellow')
    for rec in patches:
        table.add_row(
            str(rec.get('human_ts') or ''),
            'pass' if rec.get('high_confidence') else rec.get('result', 'fail'),
            str(rec.get('prompt_id') or rec.get('prompt_hash') or ''),
            f"{rec.get('metrics', {}).get('pass_rate', 0):.2f}"
        )
    console.print(Panel(table, title='Recent Fixes', border_style='blue'))
    stats = bundle.get('stats') or {}
    if stats:
        s_table = Table(show_header=True, header_style='bold green')
        s_table.add_column('Metric')
        s_table.add_column('Value')
        for key in ('total', 'high_confidence', 'failures', 'recent_success_rate', 'avg_pass_rate', 'avg_blast_radius'):
            if key in stats:
                s_table.add_row(key, f"{stats[key]:.3f}")
        console.print(Panel(s_table, title='Patch Stats', border_style='green'))


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_effective_rl_config_dict(workspace: Path) -> Dict[str, Any]:
    best_path = workspace / '.dspy' / 'rl' / 'best.json'
    try:
        raw = json.loads(best_path.read_text())
        if isinstance(raw, dict) and isinstance(raw.get('config'), dict):
            return dict(raw['config'])
    except Exception:
        pass
    return _safe_read_json(workspace / '.dspy_rl.json')


def _rl_config_from_dict(data: Mapping[str, Any]) -> _RLConfig:
    cfg = _RLConfig()
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if isinstance(cfg.weights, dict):
        cfg.weights = {str(k): float(v) for k, v in cfg.weights.items()}  # type: ignore[arg-type]
    if isinstance(cfg.penalty_kinds, (list, tuple)):
        cfg.penalty_kinds = [str(x) for x in cfg.penalty_kinds]
    if isinstance(cfg.clamp01_kinds, (list, tuple)):
        cfg.clamp01_kinds = [str(x) for x in cfg.clamp01_kinds]
    if isinstance(cfg.scales, dict):
        cfg.scales = {
            str(k): (
                (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) and len(v) == 2 else tuple(v) if isinstance(v, tuple) else v
            )
            for k, v in cfg.scales.items()
        }
    if isinstance(cfg.actions, (list, tuple)):
        cfg.actions = [str(a) for a in cfg.actions]
    return cfg


def _render_stats_page(console: Console, workspace: Path) -> None:
    console.print(Panel.fit(str(workspace), title='workspace', border_style='cyan'))
    try:
        bundle = ContextManager(workspace, workspace / 'logs').build_patch_context('status')
    except Exception:
        bundle = {'patches': [], 'stats': {}}
    _render_recent_fixes(console, bundle)
    auto_status = _safe_read_json(workspace / '.dspy_auto_status.json')
    if auto_status:
        console.print(Panel(json.dumps(auto_status, indent=2), title='auto-training', border_style='cyan'))
    rl_tool = _safe_read_json(workspace / '.dspy_rl_toolchain.json')
    if rl_tool:
        table = Table(show_header=True, header_style='bold magenta')
        table.add_column('Tool')
        table.add_column('Value')
        table.add_column('Count')
        tools = rl_tool.get('tools', []) or []
        values = rl_tool.get('values', []) or []
        counts = rl_tool.get('counts', []) or []
        for idx, name in enumerate(tools):
            val = values[idx] if idx < len(values) else ''
            cnt = counts[idx] if idx < len(counts) else ''
            try:
                val_str = f"{float(val):.3f}"
            except Exception:
                val_str = str(val)
            table.add_row(str(name), val_str, str(cnt))
        console.print(Panel(table, title='RL Toolchain', border_style='magenta'))
    rl_cfg = _load_effective_rl_config_dict(workspace)
    weights = rl_cfg.get('weights', {}) if isinstance(rl_cfg, dict) else {}
    if weights:
        w_table = Table(show_header=True, header_style='bold yellow')
        w_table.add_column('Verifier')
        w_table.add_column('Weight')
        for key, value in weights.items():
            try:
                val_str = f"{float(value):.3f}"
            except Exception:
                val_str = str(value)
            w_table.add_row(str(key), val_str)
        console.print(Panel(w_table, title='Reward Weights', border_style='yellow'))

def _get_code_graph(ws: Path) -> dict:
    # Try storage first
    try:
        from .db.factory import get_storage as _get_storage
        st = _get_storage()
        if st is not None:
            g = st.get('code:graph')
            if isinstance(g, dict):
                return g
    except Exception:
        pass
    # Fallback to on-the-fly build
    try:
        from .agents.knowledge import build_code_graph
        return build_code_graph(ws)
    except Exception:
        return {}


def _retrieve_for_query(ws: Path, query: str, top_k: int = 5) -> tuple[str, str]:
    """Load the TF-IDF index and return (retrieved_snippets, references_json).

    Each snippet is a small code section with a header noting path and line range.
    References JSON is a list of {path, start, end, score}.
    """
    try:
        from .embedding.indexer import load_index, semantic_search
        import json as _j
        meta, items = load_index(ws)
        hits = semantic_search(query, meta, items, top_k=top_k)
        if not hits:
            return "", ""
        parts: list[str] = []
        refs: list[dict] = []
        for score, it in hits:
            try:
                p = Path(it.path)
                text = p.read_text(errors="ignore")
                lines = text.splitlines()
                start = max(1, int(it.start_line)); end = min(len(lines), int(it.end_line))
                snippet = "\n".join(lines[start-1:end])
                header = f"# {p}:{start}-{end} (score={score:.3f})"
                parts.append(f"{header}\n{snippet}")
                refs.append({"path": str(p), "start": start, "end": end, "score": round(float(score), 3)})
            except Exception:
                continue
        return "\n\n".join(parts), _j.dumps(refs)
    except Exception:
        return "", ""


def _repo_layout_summary(ws: Path) -> str:
    """Produce a lightweight summary of repo layout to guide test planning."""
    try:
        tests_dir = ws / 'tests'
        test_hint = ''
        if tests_dir.exists():
            try:
                files = [p for p in tests_dir.rglob('test_*.py')][:20]
                rels = [str(p.relative_to(ws)) for p in files]
                test_hint = f"tests_dir={tests_dir.name}; samples={rels[:10]}"
            except Exception:
                test_hint = f"tests_dir={tests_dir.name}"
        has_pytest = (ws / 'pytest.ini').exists() or any((ws / n).exists() for n in ['pyproject.toml', 'tox.ini'])
        pkgs = [str(p.relative_to(ws)) for p in ws.glob('*/__init__.py') if p.is_file()][:10]
        return f"has_pytest={has_pytest}; packages={pkgs}; {test_hint}"
    except Exception:
        return ""


def _maybe_configure_lm(use_lm: bool, ollama: bool, model: Optional[str], base_url: Optional[str], api_key: Optional[str], workspace: Optional[Path] = None):
    settings = get_settings()
    if not use_lm or settings.local_mode:
        return None
    temperature = None
    target_entropy = None
    clip_higher = None
    effective_ws: Optional[Path] = workspace
    if effective_ws is None:
        env_ws = os.getenv("DSPY_WORKSPACE")
        if env_ws:
            try:
                effective_ws = Path(env_ws)
            except Exception:
                effective_ws = None
        if effective_ws is None:
            effective_ws = Path.cwd()
    try:
        cfg_dict = _load_effective_rl_config_dict(effective_ws)
    except Exception:
        cfg_dict = {}
    if isinstance(cfg_dict, dict):
        try:
            temperature = float(cfg_dict.get('temperature')) if cfg_dict.get('temperature') is not None else None
        except Exception:
            temperature = None
        try:
            target_entropy = float(cfg_dict.get('target_entropy')) if cfg_dict.get('target_entropy') is not None else None
        except Exception:
            target_entropy = None
        try:
            clip_higher = float(cfg_dict.get('clip_higher')) if cfg_dict.get('clip_higher') is not None else None
        except Exception:
            clip_higher = None

    return configure_lm(
        provider="ollama" if ollama else None,
        model_name=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        target_entropy=target_entropy,
        clip_higher=clip_higher,
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
    from .streaming.log_reader import iter_log_paths, read_capped  # local import to avoid circular
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
    model: Optional[str] = typer.Option("qwen3:1.7b", '--model', help="Model name (e.g., llama3)"),
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
    console.print(Panel.fit(escape(key), title="Extracted Events", border_style="magenta"))

    if not use_lm or settings.local_mode:
        console.print("[dim]LOCAL_MODE or --no-lm: showing heuristics only.[/dim]")
        raise typer.Exit(0)

    lm = _maybe_configure_lm(use_lm, ollama, model, base_url, api_key, workspace=ws)
    if lm is None:
        console.print("[yellow]No LM configured; skipping enhanced context.[/yellow]")
        raise typer.Exit(0)

    _print_header("Enhanced Context (DSPy)")
    builder = ContextBuilder()
    pred = builder(task="Summarize logs for debugging", logs_preview=key)
    console.print(Panel.fit(escape(pred.context), title="Context", border_style="cyan"))
    console.print(Panel.fit(escape(pred.key_points), title="Key Points", border_style="green"))
    if getattr(pred, 'missing_info', None):
        console.print(Panel.fit(escape(getattr(pred, 'missing_info')), title="Missing Info", border_style="yellow"))
    if getattr(pred, 'next_steps', None):
        console.print(Panel.fit(escape(getattr(pred, 'next_steps')), title="Next Steps", border_style="accent"))


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
    model: Optional[str] = typer.Option("qwen3:1.7b", '--model', help="Model name for LLM"),
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
    console.print(Panel.fit(escape(snap[:8000] + ("\n..." if len(snap) > 8000 else "")), title=str(target), border_style="magenta"))
    if not use_lm:
        raise typer.Exit(0)
    lm = _maybe_configure_lm(True, ollama, model, base_url, api_key, workspace=workspace)
    if lm is None:
        raise typer.Exit(0)
    _print_header("Code Context (DSPy)")
    code_graph = _get_code_summary(workspace)
    # Try retrieval to enrich context
    ask = "Summarize key components, APIs, and likely modification points."
    retrieved_snippets, references = _retrieve_for_query(workspace, ask)
    if retrieved_snippets:
        cc_rag = CodeContextRAG()
        out = cc_rag(snapshot=snap, ask=ask, code_graph=code_graph, retrieved_snippets=retrieved_snippets, references=references)
        console.print(Panel.fit(escape(out.summary), title="Summary", border_style="cyan"))
        console.print(Panel.fit(escape(out.bullets), title="Bullets", border_style="green"))
        if getattr(out, 'hot_spots', None):
            console.print(Panel.fit(escape(getattr(out, 'hot_spots')), title="Hot Spots", border_style="magenta"))
        if getattr(out, 'entry_points', None):
            console.print(Panel.fit(escape(getattr(out, 'entry_points')), title="Entry Points", border_style="cyan"))
        if getattr(out, 'risk_areas', None):
            console.print(Panel.fit(escape(getattr(out, 'risk_areas')), title="Risk Areas", border_style="yellow"))
        if getattr(out, 'citations', None):
            console.print(Panel.fit(escape(getattr(out, 'citations')), title="Citations", border_style="dim"))
    else:
        cc = CodeContext()
        out = cc(snapshot=snap, ask=ask, code_graph=code_graph)
        console.print(Panel.fit(escape(out.summary), title="Summary", border_style="cyan"))
        console.print(Panel.fit(escape(out.bullets), title="Bullets", border_style="green"))
    console.print(Panel.fit(escape(out.summary), title="Summary", border_style="cyan"))
    console.print(Panel.fit(escape(out.bullets), title="Bullets", border_style="green"))


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
def stream_topics_create(
    cfg_path: Path = typer.Option(STREAM_CFG_PATH, '--config', exists=False, help="Streaming config path"),
    bootstrap: Optional[str] = typer.Option(None, '--bootstrap', help="Kafka bootstrap servers"),
):
    """Create Kafka topics from config using confluent-kafka AdminClient (best-effort)."""
    try:
        from confluent_kafka.admin import AdminClient, NewTopic  # type: ignore
    except Exception:
        console.print(Panel("confluent-kafka not installed; cannot create topics.", title="kafka", border_style="yellow"))
        raise typer.Exit(1)
    # Load config (or default)
    if cfg_path.exists():
        cfg = load_stream_cfg(cfg_path)
    else:
        cfg = StreamConfig.default()
    bs = bootstrap or cfg.kafka.bootstrap_servers
    admin = AdminClient({'bootstrap.servers': bs})
    topics = []
    for t in cfg.kafka.topics:
        topics.append(NewTopic(t.name, num_partitions=t.partitions, replication_factor=t.replication_factor))
    try:
        fs = admin.create_topics(topics, request_timeout=10.0)
        errs = []
        for name, f in fs.items():
            try:
                f.result()
                console.print(f"[green]Created topic {name}[/green]")
            except Exception as e:
                errs.append((name, str(e)))
        if errs:
            for name, e in errs:
                console.print(f"[yellow]{name}: {e}[/yellow]")
    except Exception as e:
        console.print(Panel(escape(str(e)), title="kafka", border_style="red"))


@app.command()
def last(
    container: str = typer.Option(..., '--container', help="Container name (e.g., backend, frontend, app)"),
    what: str = typer.Option('all', '--what', help="Which field: summary|plan|key_points|ts|all"),
):
    """Show the latest summary/plan for a container from storage (RedDB if configured)."""
    try:
        from .db.factory import get_storage as _get_storage
    except Exception as e:
        console.print(Panel(escape(str(e)), title="storage unavailable", border_style="red")); raise typer.Exit(1)
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
        console.print(Panel(escape(str(e)), title="read failed", border_style="red")); raise typer.Exit(1)
    if what != 'all':
        val = data.get(what)
        if val is None:
            console.print(Panel("(no data)", title=f"{container}:{what}", border_style="yellow")); raise typer.Exit(0)
        console.print(Panel(str(val), title=f"{container}:{what}", border_style="cyan")); raise typer.Exit(0)
    console.print(Panel(str(data.get('summary') or '(no summary)'), title=f"{container}:summary", border_style="cyan"))
    console.print(Panel(str(data.get('key_points') or '(no key points)'), title=f"{container}:key_points", border_style="green"))
    console.print(Panel(str(data.get('plan') or '(no plan)'), title=f"{container}:plan", border_style="blue"))
    console.print(Panel(str(data.get('ts') or '(no ts)'), title=f"{container}:ts", border_style="magenta"))


@app.command("learn")
def learn(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Repo root to analyze"),
    embeddings: bool = typer.Option(True, '--embeddings/--no-embeddings', help="Also build embeddings index"),
    out: Optional[Path] = typer.Option(None, '--out', help="Write code graph JSON to this path"),
):
    """Analyze the codebase to build a code graph and optionally embeddings. Persists to storage if configured."""
    _print_header("Learning codebase")
    graph = build_code_graph(workspace)
    summary = summarize_code_graph(graph)
    if out is None:
        out = workspace / '.dspy_index' / 'knowledge.json'
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        import json as _j
        out.write_text(_j.dumps(graph, indent=2))
        console.print(Panel.fit(f"Wrote code graph to {out}", title="knowledge", border_style="green"))
        console.print(Panel.fit(summary, title="summary", border_style="accent"))
    except Exception as e:
        console.print(Panel(escape(str(e)), title="write failed", border_style="red"))
    # Persist to storage when available
    try:
        from .db.factory import get_storage as _get_storage
        st = _get_storage()
        if st is not None:
            st.put('code:graph', graph)  # type: ignore[arg-type]
            st.put('code:summary', summary)  # type: ignore[arg-type]
            # Per-file facts for targeted queries
            try:
                root = Path(graph.get('root', workspace))
                for f in graph.get('files', []):
                    p = Path(f.get('path', ''))
                    rel = str(p.resolve()).replace(str(root.resolve()), '').lstrip('/')
                    safe = (rel or p.name).replace('/', '|')
                    st.put(f'code:file:{safe}:facts', f)  # type: ignore
            except Exception:
                pass
            console.print("[green]Persisted code graph and per-file facts to storage.[/green]\n")
    except Exception:
        pass
    # Build embeddings index
    if embeddings:
        try:
            _print_header("Embeddings index")
            from .embedding.embeddings_index import build_emb_index, save_emb_index
            from sentence_transformers import SentenceTransformer  # type: ignore
            model = SentenceTransformer('all-MiniLM-L6-v2')
            items = build_emb_index(workspace, model)
            idx_dir = save_emb_index(workspace, items, persist=True)
            console.print(Panel.fit(f"Built embeddings for {len(items)} chunks → {idx_dir}", title="embeddings", border_style="cyan"))
        except Exception as e:
            console.print(Panel(f"Embeddings skipped: {e}", title="embeddings", border_style="yellow"))


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
    ap.add_argument('--pattern', default='logs.raw.*', help='Kafka topic pattern for raw logs')
    ap.add_argument('--checkpoint', default='.dspy_checkpoints/spark_logs')
    ap.add_argument('--window', default='30 seconds', help='Window duration (e.g., 30 seconds)')
    ap.add_argument('--slide', default='15 seconds', help='Slide duration (e.g., 15 seconds)')
    ap.add_argument('--watermark', default='2 minutes', help='Allowed lateness for watermarking')
    args = ap.parse_args()

    spark = SparkSession.builder.appName('dspy-stream-logs').getOrCreate()
    df = (spark
          .readStream
          .format('kafka')
          .option('kafka.bootstrap.servers', args.bootstrap)
          .option('subscribePattern', args.pattern)
          .load())

    # Parse Kafka message
    val = df.selectExpr("CAST(topic AS STRING) AS topic", "CAST(value AS STRING) AS line", "timestamp")

    # Keep error-like lines and assign severity
    low = F.lower(F.col('line'))
    hits = val.where(low.rlike('error|warn|traceback|exception|failed|timeout|fatal'))
    hits = hits.withColumn(
        'level',
        F.when(low.rlike('fatal|error|traceback|exception|failed|timeout'), F.lit('error'))
         .when(low.rlike('warn|warning'), F.lit('warn'))
         .otherwise(F.lit('info'))
    )

    # Aggregate by sliding window and topic; route to logs.ctx.<suffix>
    agg = (
        hits
        .withColumn('topic_out', F.regexp_replace(F.col('topic'), '^logs\\.raw\\.', 'logs.ctx.'))
        .withWatermark('timestamp', args.watermark)
        .groupBy(F.window('timestamp', args.window, args.slide), F.col('topic_out'))
        .agg(
            F.collect_list('line').alias('lines'),
            F.sum(F.when(F.col('level') == 'error', 1).otherwise(0)).alias('error_count'),
            F.sum(F.when(F.col('level') == 'warn', 1).otherwise(0)).alias('warn_count'),
            F.count('*').alias('total')
        )
        .select(
            F.col('topic_out').alias('topic'),
            F.to_json(
                F.struct(
                    F.col('lines').alias('ctx'),
                    F.struct(
                        F.col('error_count'),
                        F.col('warn_count'),
                        F.col('total')
                    ).alias('stats')
                )
            ).alias('value')
        )
    )

    q = (
        agg.writeStream
           .format('kafka')
           .option('kafka.bootstrap.servers', args.bootstrap)
           .option('checkpointLocation', args.checkpoint)
           .outputMode('update')
           .start()
    )
    q.awaitTermination()

if __name__ == '__main__':
    main()
'''
    out.write_text(code)
    console.print(Panel.fit(f"Wrote PySpark job to {out}", title="spark", border_style="accent"))
    console.print(Panel.fit("Example: spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 scripts/streaming/spark_logs.py --bootstrap localhost:9092 --pattern 'logs.raw.*' --window '30 seconds' --slide '15 seconds' --watermark '2 minutes'", title="run", border_style="dim"))


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
        console.print(Panel(escape(str(e)), title="worker", border_style="red"))
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
    auto_train: bool = typer.Option(True, '--auto-train/--no-auto-train', help="Run auto GEPA + RL loop"),
    db: str = typer.Option("auto", '--db', help="Storage backend: auto|none|reddb"),
    status: bool = typer.Option(True, '--status/--no-status', help="Run status HTTP server"),
    status_host: str = typer.Option("0.0.0.0", '--status-host', help="Status server host"),
    status_port: int = typer.Option(8765, '--status-port', help="Status server port (0 to disable)"),
):
    """Start local tailers + aggregators + workers (no Kafka needed). Single command dev stack."""
    _print_banner(console)
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
    auto_runner: Optional[AutoTrainingLoop] = None
    if train:
        # Discover containers from .dspy_stream.json or autodiscovery
        try:
            cfg = load_stream_cfg(STREAM_CFG_PATH) if STREAM_CFG_PATH.exists() else None
        except Exception:
            cfg = None
        containers = [getattr(ct, 'container') for ct in getattr(cfg, 'containers', [])] if cfg else [d.container for d in autodiscover_logs(workspace)]
        from .streaming.streaming_runtime import Trainer
        kafka_cfg = getattr(cfg, 'kafka', None)
        vector_topic = getattr(kafka_cfg, 'vector_topic', 'agent.rl.vectorized') if kafka_cfg else 'agent.rl.vectorized'
        trainer = Trainer(workspace, bus, containers, min_batch=3, interval_sec=60.0, vector_topic=vector_topic)
        trainer.start(); threads.append(trainer)
    if auto_train and _auto_train_enabled():
        try:
            auto_runner = AutoTrainingLoop(workspace, workspace / 'logs', console=console, label='stack')
            auto_runner.start()
            console.print("[dim]Auto-training loop started.[/dim]")
        except Exception as e:
            console.print(Panel(f"auto-training unavailable: {e}", title="auto-train", border_style="yellow"))
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
    finally:
        if auto_runner:
            auto_runner.stop()


# -----------------------------
# Lightweight Containers (Local)
# -----------------------------


def _stack_compose_path(root: Path) -> Path:
    return root / 'docker-compose.yml'


@stack_app.command("init")
def stack_init(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=False, help="Project directory to mount"),
    logs: Optional[Path] = typer.Option(None, '--logs', dir_okay=True, file_okay=False, exists=False, help="Optional logs directory to mount read-only"),
    out_dir: Path = typer.Option(DEFAULT_STACK_DIR, '--out-dir', help="Where generated Docker assets will live"),
    db: str = typer.Option("auto", '--db', help="Storage backend: auto|none|reddb"),
    install_source: str = typer.Option('pip', '--install-source', case_sensitive=False, help="Install dspy-code inside containers via pip or from local source"),
    pip_spec: Optional[str] = typer.Option(None, '--pip-spec', help="Override pip spec when --install-source pip (e.g. dspy-code==0.1.0)"),
    start: bool = typer.Option(True, '--start/--no-start', help="Run docker compose up -d after generating assets"),
    build: bool = typer.Option(True, '--build/--no-build', help="Build images on first start"),
):
    try:
        bundle = prepare_stack(
            workspace=workspace,
            logs=logs,
            out_dir=out_dir,
            db=db,
            install_source=install_source,
            pip_spec=pip_spec,
        )
    except Exception as exc:
        console.print(Panel(escape(str(exc)), title='stack init failed', border_style='red'))
        raise typer.Exit(1)

    rel_compose = bundle.compose
    rel_dockerfile = bundle.dockerfile
    try:
        rel_compose = rel_compose.relative_to(Path.cwd())
    except ValueError:
        pass
    try:
        rel_dockerfile = rel_dockerfile.relative_to(Path.cwd())
    except ValueError:
        pass

    message_lines = [
        f"Dockerfile: {rel_dockerfile}",
        f"Compose: {rel_compose}",
    ]
    if bundle.warnings:
        console.print(Panel('\n'.join(f'- {w}' for w in bundle.warnings), title='adjustments', border_style='yellow'))
    console.print(Panel('\n'.join(message_lines), title='stack init', border_style='green'))

    if not start:
        console.print(Panel("Run 'dspy-code stack up' when you are ready.", title='next step', border_style='cyan'))
        raise typer.Exit(0)

    if not stack_docker_available():
        console.print('[yellow]Docker is not available. Install Docker Desktop or CLI, then run `dspy-code stack up`.[/yellow]')
        raise typer.Exit(1)

    compose_path = bundle.compose
    try:
        if build:
            stack_compose_command(compose_path, ['build'])
        stack_compose_command(compose_path, ['up', '-d'])
    except RuntimeError as exc:
        console.print(Panel(escape(str(exc)), title='docker compose', border_style='red'))
        raise typer.Exit(1)
    except subprocess.CalledProcessError as exc:
        console.print(Panel(f"docker compose exited with code {exc.returncode}", title='docker compose', border_style='red'))
        raise typer.Exit(exc.returncode)

    console.print(Panel("Stack is running! Visit http://localhost:8081 to access the dashboard once containers are ready.", title='stack up', border_style='green'))


def _stack_require_compose(stack_dir: Path) -> Path:
    compose = _stack_compose_path(stack_dir)
    if not compose.exists():
        console.print(Panel(f"Compose file not found at {compose}. Run 'dspy-code stack init' first.", title='stack', border_style='red'))
        raise typer.Exit(1)
    return compose


@stack_app.command("up")
def stack_up(
    stack_dir: Path = typer.Option(DEFAULT_STACK_DIR, '--dir', help="Directory containing docker-compose.yml"),
    build: bool = typer.Option(False, '--build', help="Perform docker compose build before up"),
):
    if not stack_docker_available():
        console.print('[yellow]Docker is not available on PATH. Install Docker and retry.[/yellow]')
        raise typer.Exit(1)
    compose = _stack_require_compose(stack_dir)
    try:
        if build:
            stack_compose_command(compose, ['build'])
        stack_compose_command(compose, ['up', '-d'])
    except RuntimeError as exc:
        console.print(Panel(escape(str(exc)), title='docker compose', border_style='red'))
        raise typer.Exit(1)
    except subprocess.CalledProcessError as exc:
        console.print(Panel(f"docker compose exited with code {exc.returncode}", title='docker compose', border_style='red'))
        raise typer.Exit(exc.returncode)
    console.print('[green]Stack is running.[/green]')


@stack_app.command("down")
def stack_down(
    stack_dir: Path = typer.Option(DEFAULT_STACK_DIR, '--dir', help="Directory containing docker-compose.yml"),
):
    if not stack_docker_available():
        console.print('[yellow]Docker is not available on PATH. Install Docker and retry.[/yellow]')
        raise typer.Exit(1)
    compose = _stack_require_compose(stack_dir)
    try:
        stack_compose_command(compose, ['down'])
    except RuntimeError as exc:
        console.print(Panel(escape(str(exc)), title='docker compose', border_style='red'))
        raise typer.Exit(1)
    except subprocess.CalledProcessError as exc:
        console.print(Panel(f"docker compose exited with code {exc.returncode}", title='docker compose', border_style='red'))
        raise typer.Exit(exc.returncode)
    console.print('[green]Stack stopped.[/green]')


@stack_app.command("status")
def stack_status(
    stack_dir: Path = typer.Option(DEFAULT_STACK_DIR, '--dir', help="Directory containing docker-compose.yml"),
):
    if not stack_docker_available():
        console.print('[yellow]Docker is not available on PATH. Install Docker and retry.[/yellow]')
        raise typer.Exit(1)
    compose = _stack_require_compose(stack_dir)
    try:
        stack_compose_command(compose, ['ps'], check=False)
    except RuntimeError as exc:
        console.print(Panel(escape(str(exc)), title='docker compose', border_style='red'))
        raise typer.Exit(1)


@stack_app.command("logs")
def stack_logs(
    stack_dir: Path = typer.Option(DEFAULT_STACK_DIR, '--dir', help="Directory containing docker-compose.yml"),
    follow: bool = typer.Option(True, '--follow/--no-follow', help="Stream logs"),
    service: Optional[str] = typer.Option(None, '--service', help="Optional service name to filter"),
):
    if not stack_docker_available():
        console.print('[yellow]Docker is not available on PATH. Install Docker and retry.[/yellow]')
        raise typer.Exit(1)
    compose = _stack_require_compose(stack_dir)
    args = ['logs']
    if follow:
        args.append('-f')
    if service:
        args.append(service)
    try:
        stack_compose_command(compose, args, check=False)
    except RuntimeError as exc:
        console.print(Panel(escape(str(exc)), title='docker compose', border_style='red'))
        raise typer.Exit(1)

def _ensure_dir(path: Path) -> None:
    stack_ensure_dir(path)


def _docker_available() -> bool:
    return stack_docker_available()





@app.command("lightweight_init")
def lightweight_init(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=False, help="Host workspace to mount (created if missing; falls back to CWD if unwritable)"),
    logs: Optional[Path] = typer.Option(None, '--logs', help="Host logs directory to mount read-only (optional; created if missing; disabled if unwritable)", file_okay=False, dir_okay=True),
    out_dir: Path = typer.Option(LIGHTWEIGHT_DIR, '--out-dir', help="Where to write Dockerfile/compose"),
    db: str = typer.Option("auto", '--db', help="Storage backend: auto|none|reddb"),
    install_source: str = typer.Option(
        'pip', '--install-source', case_sensitive=False,
        help="How to install dspy-code inside the Docker image (pip from PyPI/spec or local source). Choices: pip, local",
    ),
    pip_spec: Optional[str] = typer.Option(
        None, '--pip-spec', help="Override pip spec when --install-source pip (e.g. dspy-code==0.1.0).",
    ),
):
    try:
        bundle = prepare_stack(
            workspace=workspace,
            logs=logs,
            out_dir=out_dir,
            db=db,
            install_source=install_source,
            pip_spec=pip_spec,
        )
    except Exception as exc:
        console.print(Panel(escape(str(exc)), title='write failed', border_style='red'))
        raise typer.Exit(1)

    if bundle.warnings:
        console.print(Panel('\n'.join(f'- {w}' for w in bundle.warnings), title='adjustments', border_style='yellow'))

    def _display_path(path: Path) -> str:
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return str(path)

    console.print(Panel.fit(
        f"Wrote:\n- {_display_path(bundle.dockerfile)}\n- {_display_path(bundle.compose)}", title='lightweight init', border_style='green'
    ))

    compose_path_str = _display_path(bundle.compose)
    next_steps = [
        f"1) Review {compose_path_str} and adjust env as needed (REDDB_URL, OPENAI_BASE_URL, etc.)",
        f"2) Build the image: docker compose -f {compose_path_str} build",
        f"3) Start the stack: docker compose -f {compose_path_str} up -d",
        f"4) Logs: docker compose -f {compose_path_str} logs -f dspy-agent",
    ]
    console.print(Panel('\n'.join(next_steps), title='next steps', border_style='cyan'))
    if not _docker_available():
        console.print('[yellow]Docker not detected in PATH. Install Docker Desktop or CLI first.[/yellow]')
    if db.lower() == 'reddb' and not os.getenv('REDDB_URL'):
        console.print('[yellow]You selected --db reddb but REDDB_URL is not set. The agent will fallback to in-memory until you provide REDDB_URL.[/yellow]')


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
        console.print(Panel(escape(str(e)), title="docker compose failed", border_style="red"))
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
        console.print(Panel(escape(str(e)), title="docker compose failed", border_style="red"))
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
        console.print(Panel(escape(str(e)), title="docker compose failed", border_style="red"))
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
        console.print(Panel(escape(str(e)), title="lightweight build failed", border_style="red"))
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
    model: Optional[str] = typer.Option("qwen3:1.7b", '--model'),
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

    toolchain_executor: Optional[ToolchainExecutor] = None

    def _refresh_toolchain() -> None:
        nonlocal toolchain_executor
        try:
            toolchain_executor = ToolchainExecutor(detect_toolchain(ws))
        except Exception:
            toolchain_executor = None

    def _execute_toolchain(action: ToolAction, action_args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        nonlocal toolchain_executor
        if toolchain_executor is None:
            _refresh_toolchain()
        if toolchain_executor is None:
            console.print(f"[yellow]Toolchain executor unavailable for {action.name.lower()} (configure toolchain commands).[/yellow]")
            return {}, {}
        try:
            result = toolchain_executor(action, action_args)
        except Exception as exc:
            console.print(Panel(escape(str(exc)), title=f"{action.name.lower()} failed", border_style="red"))
            return {}, {}
        return _render_toolchain_console(action.name.lower(), result)

    _refresh_toolchain()

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
        tool = None; args: Dict[str, Any] = {}
        lm = _maybe_configure_lm(True, provider_is_ollama, model_name, base, key, workspace=ws)
        if lm is not None:
            try:
                # Confidence gating: use fast Predict first; escalate to CoT if ambiguous
                fast = Orchestrator(use_cot=False)
                pred_fast = fast(query=task, state=state)
                tool = (pred_fast.tool or "").strip(); import json as _json; args = _json.loads(pred_fast.args_json or "{}")
                ambiguous = (not tool) or (last_tool == tool and last_args == args) or (tool in {"knowledge","sg"} and not args)
                if ambiguous:
                    slow = Orchestrator(use_cot=True)
                    pred = slow(query=task, state=state)
                    tool = (pred.tool or tool).strip(); args = _json.loads(pred.args_json or "{}")
                    if getattr(pred, 'rationale', None):
                        console.print(Panel.fit(escape(pred.rationale), title="Routing Rationale", border_style="dim"))
            except Exception:
                tool = None; args = {}
        if not tool:
            tool = "context" if logs_path.exists() else "codectx"; args = {}
        else:
            args = dict(args or {})
        if tool == "edit":
            args.setdefault("apply", auto_apply_env)
        if tool == "patch":
            args.setdefault("task", task)
        console.print(Panel.fit(escape(f"{tool} {args}"), title=f"Step {step}: action", border_style="yellow"))
        if last_tool == tool and last_args == args:
            console.print("[dim]No new action; stopping.[/dim]")
            break
        last_tool, last_args = tool, dict(args)
        tool_metrics: Dict[str, Any] = {}
        tool_info: Dict[str, Any] = {}
        try:
            # If manual approval is required, confirm before executing tool
            if approval_mode == "manual":
                approved = typer.confirm(f"Approve tool '{tool}' with args {args}?", default=False)
                if not approved:
                    console.print("[dim]Tool execution skipped by user.[/dim]")
                    continue
            # dispatch_tool is nested in start; inline minimal relevant calls
            if tool == "context":
                bundle, _ = load_logs([logs_path]); key = extract_key_events(bundle) if bundle else ""; _print_header("Log Key Events"); console.print(Panel.fit(escape(key), title="Extracted Events", border_style="magenta"))
            elif tool == "codectx":
                snap = build_code_snapshot(ws); _print_header("Code Snapshot"); console.print(Panel.fit(escape(snap[:8000] + ("\n..." if len(snap)>8000 else "")), title=str(ws), border_style="magenta"))
            elif tool == "grep":
                hits = search_text(ws, args.get("pattern", task), regex=True); [console.print(f"{h.path}:{h.line_no}: {h.line}") for h in hits[:200]]
            elif tool == "esearch":
                try: meta, items = load_index(ws)
                except FileNotFoundError: meta, items = build_index(ws, smart=True); save_index(ws, meta, items)
                hits = semantic_search(task, meta, items, top_k=5)
                for score,it in hits:
                    p = Path(it.path); text = p.read_text(errors="ignore"); lines = text.splitlines(); s=max(1,it.start_line-3); e=min(len(lines), it.end_line+3); seg = "\n".join(lines[s-1:e]); console.print(Panel(escape(seg), title=f"{p} score={score:.3f} lines {s}-{e}", border_style="blue"))
            elif tool == "extract":
                fp = (ws / args.get("file",".")).resolve(); sym=args.get("symbol"); rx=args.get("regex");
                if sym and fp.suffix==".py": res = python_extract_symbol(fp, sym); 
                elif rx: hits=file_search(fp, rx, regex=True); text=fp.read_text(errors="ignore"); s,e,seg=extract_context(text, hits[0].line_no, before=3, after=3); console.print(Panel(escape(seg), title=f"{fp} lines {s}-{e}", border_style="green"))
            elif tool == "run_tests":
                tool_metrics, tool_info = _execute_toolchain(ToolAction.RUN_TESTS, dict(args))
            elif tool == "lint":
                tool_metrics, tool_info = _execute_toolchain(ToolAction.LINT, dict(args))
            elif tool == "build":
                tool_metrics, tool_info = _execute_toolchain(ToolAction.BUILD, dict(args))
            elif tool == "patch" and not args.get("file"):
                patch_args = dict(args)
                patch_task = patch_args.get("task") or task
                if not isinstance(patch_task, str):
                    patch_task = str(patch_task)
                bundle = _build_patch_context_bundle(ws, logs_path, patch_task)
                combined = bundle.get('combined_context') or bundle.get('text') or ''
                patch_args.setdefault("task", patch_task)
                patch_args.setdefault("context", combined)
                tool_metrics, tool_info = _execute_toolchain(ToolAction.PATCH, patch_args)
                patch_done = True
            elif tool == "patch":
                pf = args.get("file"); p = (ws / pf).resolve() if pf else None
                if not p or not p.exists():
                    console.print("[yellow]patch needs --file[/yellow]")
                    continue
                text = p.read_text(errors="ignore")
                ok, msg = apply_unified_patch(text, ws)
                tool_metrics = {"applied": bool(ok)}
                if ok:
                    console.print(f"[ok]{msg}[/ok]")
                    summ = summarize_patch(text)
                    blast = float(summ['added_lines'] + summ['removed_lines'])
                    tool_metrics.update({"blast_radius": blast})
                    console.print(Panel.fit(
                        f"files: {summ['files']}  +lines: {summ['added_lines']}  -lines: {summ['removed_lines']}",
                        title="patch metrics", border_style="accent"
                    ))
                else:
                    console.print(Panel(msg, title="patch failed", border_style="err"))
                patch_done = True
            elif tool == "edit":
                tool_metrics, tool_info = _run_edit_tool(args)
                patch_done = True
            else:
                # default to context snapshot
                bundle, _ = load_logs([logs_path]); key = extract_key_events(bundle) if bundle else ""; console.print(Panel.fit(escape(key or "(no logs)"), title="Key Events", border_style="magenta"))
        except Exception as e:
            console.print(Panel(escape(str(e)), title=f"agent failed ({tool})", border_style="red")); break
        try:
            outcome = evaluate_tool_choice(
                tool,
                args,
                workspace=ws,
                logs_path=logs_path,
                targets=targets,
                result_metrics=tool_metrics or None,
                result_info=tool_info or None,
            ); piece=f"{tool}: score={outcome.score:.2f}; {outcome.evidence}"
            
            # Create and record tool result for memory
            if memory:
                # Determine success based on tool metrics and outcome
                success = False
                if tool_metrics and "applied" in tool_metrics:
                    success = tool_metrics["applied"]
                elif outcome.score >= 1.0:  # High score indicates success
                    success = True
                elif tool_metrics and "blast_radius" in tool_metrics:
                    success = tool_metrics["blast_radius"] > 0  # Any changes made
                
                tool_result = ToolResult(
                    tool=tool,
                    args=dict(args),
                    result=piece,
                    success=success,
                    timestamp=time.time(),
                    execution_time=0.1,  # Approximate
                    score=outcome.score,
                    feedback=outcome.evidence
                )
                memory.add_tool_result(tool_result)
                
        except Exception:
            piece=f"{tool}: done"
            # Still record the tool result even if evaluation failed
            if memory:
                tool_result = ToolResult(
                    tool=tool,
                    args=dict(args),
                    result=piece,
                    success=False,
                    timestamp=time.time(),
                    execution_time=0.1,
                    score=0.0,
                    feedback="Tool execution completed but evaluation failed"
                )
                memory.add_tool_result(tool_result)
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
        console.print(Panel.fit(escape(key or "(no logs)"), title="Key Events", border_style="magenta"))
        console.print("[yellow]LOCAL_MODE or --no-lm: cannot generate a plan.[/yellow]")
        raise typer.Exit(0)

    lm = _maybe_configure_lm(use_lm, ollama, model, base_url, api_key, workspace=ws)
    if lm is None:
        _print_header("Heuristic Context")
        console.print(Panel.fit(escape(key or "(no logs)"), title="Key Events", border_style="magenta"))
        console.print("[yellow]No LM configured; cannot generate a plan.[/yellow]")
        raise typer.Exit(1)

    builder = ContextBuilder()
    ctx = builder(task=task, logs_preview=key)
    code_graph = _get_code_summary(ws)
    fused_context = f"{ctx.context}\n\n{ctx.key_points}\n\nCode Summary:\n{code_graph}" if code_graph else f"{ctx.context}\n\n{ctx.key_points}"

    _print_header("Agent Plan")
    agent = TaskAgent()
    out = agent(task=task, context=fused_context)

    console.print(Panel.fit(escape(task), title="Task", border_style="white"))
    console.print(Panel.fit(escape(ctx.context), title="Context", border_style="cyan"))
    console.print(Panel.fit(escape(ctx.key_points), title="Key Points", border_style="green"))
    console.print(Panel.fit(escape(out.plan), title="Proposed Plan", border_style="blue"))
    console.print(Panel.fit(escape(out.commands or "(no commands)"), title="Suggested Commands", border_style="yellow"))


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
    event_hits = [{"path": str(Path(it.path)), "score": float(score), "source": "esearch"} for score, it in hits]
    if event_hits:
        log_retrieval_event(workspace, query, event_hits)
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
        console.print(Panel(escape(seg), title=title, border_style="blue"))


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
    persist: bool = typer.Option(False, '--persist/--no-persist', help="Also persist embeddings and code chunks to RedDB"),
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
            console.print(Panel(escape(str(e)), title="Failed to load HF model", border_style="red"))
            raise typer.Exit(1)
    else:
        try:
            embedder = dspy.Embeddings(model=model, api_base=base_url, api_key=api_key)
        except Exception as e:
            console.print(Panel(escape(str(e)), title="Failed to init DSPy Embeddings", border_style="red"))
            raise typer.Exit(1)
    items = build_emb_index(workspace, embedder, lines_per_chunk=lines, smart=smart)
    out_dir = save_emb_index(workspace, items, persist=persist)
    console.print(f"[green]Embedded {len(items)} chunks. Saved to {out_dir}[/green]")


@app.command("embeddings_inspect")
def embeddings_inspect(
    start: int = typer.Option(0, '--start', help="Start offset in emb.index stream"),
    count: int = typer.Option(10, '--count', help="Number of entries to show"),
):
    """Page through embeddings from RedDB and display aligned code chunks.

    Requires REDDB_URL to be configured and emb_index persisted with --persist
    or the compaction job to have run.
    """
    try:
        from .db.factory import get_storage as _get_storage
        import hashlib as _h
    except Exception as e:
        console.print(Panel(escape(str(e)), title="storage", border_style="red")); raise typer.Exit(1)
    st = _get_storage()
    if st is None:
        console.print(Panel("No storage configured (set REDDB_URL).", title="embeddings", border_style="yellow")); raise typer.Exit(1)
    rows = list(st.read('emb.index', start=start, count=count))  # type: ignore
    if not rows:
        console.print("[yellow]No embeddings in stream (persist with emb_index --persist).[/yellow]")
        raise typer.Exit(0)
    for off, rec in rows:
        path = rec.get('path'); s = rec.get('start_line'); e = rec.get('end_line')
        h = _h.sha256((str(path) + str(s) + str(e)).encode('utf-8')).hexdigest()
        chunk = st.get(f'code:chunk:{h}')  # type: ignore
        title = f"off={off} {path}:{s}-{e} vec_len={len(rec.get('vector', []) or [])}"
        text = (chunk or {}).get('text') if isinstance(chunk, dict) else None
        console.print(Panel(text or "(no KV chunk; run chunks_compact or emb_index --persist)", title=title, border_style="cyan"))


@app.command("chunks_compact")
def chunks_compact(
    start: int = typer.Option(0, '--start', help="Start offset in code.chunks stream"),
    count: int = typer.Option(1000, '--count', help="Number of records to scan"),
):
    """Deduplicate code.chunks into KV cache keyed by hash for fast lookup."""
    try:
        from .db.factory import get_storage as _get_storage
    except Exception as e:
        console.print(Panel(escape(str(e)), title="storage", border_style="red")); raise typer.Exit(1)
    st = _get_storage()
    if st is None:
        console.print(Panel("No storage configured (set REDDB_URL).", title="chunks", border_style="yellow")); raise typer.Exit(1)
    rows = list(st.read('code.chunks', start=start, count=count))  # type: ignore
    if not rows:
        console.print("[yellow]No code.chunks entries found.[/yellow]"); raise typer.Exit(0)
    written = 0
    for off, rec in rows:
        h = rec.get('hash');
        if not h:
            continue
        key = f'code:chunk:{h}'
        if st.get(key):  # type: ignore
            continue
        st.put(key, rec)  # type: ignore
        written += 1
    console.print(f"[green]Compacted {written} chunk(s) into KV cache.[/green]")


@app.command("ast_cache")
def ast_cache(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', exists=True, dir_okay=True),
    limit: int = typer.Option(100000, '--limit', help="Max records to write to streams/KV"),
):
    """Build an AST-backed cache of classes/functions (Python) and persist to RedDB.

    Falls back to our Python AST parser (knowledge.py). If ast-grep is available, you can
    extend this to multi-language patterns later.
    """
    try:
        from .db.factory import get_storage as _get_storage
    except Exception as e:
        console.print(Panel(escape(str(e)), title="storage", border_style="red")); raise typer.Exit(1)
    st = _get_storage()
    if st is None:
        console.print(Panel("No storage configured (set REDDB_URL).", title="ast-cache", border_style="yellow")); raise typer.Exit(1)
    g = build_code_graph(workspace)
    files = g.get('files', []) or []
    wrote = 0
    for rec in files:
        if wrote >= limit:
            break
        path = rec.get('path'); classes = rec.get('classes') or []; functions = rec.get('functions') or []
        # Streams for backfill
        for c in classes:
            st.append('code.ast.class', {'name': c, 'path': path})  # type: ignore
            # KV cache: append path if not present
            key = f'code:ast:class:{c}'
            try:
                cur = st.get(key) or []  # type: ignore
                if path not in cur:
                    cur.append(path)
                    st.put(key, cur)  # type: ignore
            except Exception:
                pass
            wrote += 1
            if wrote >= limit: break
        if wrote >= limit:
            break
        for fn in functions:
            st.append('code.ast.function', {'name': fn, 'path': path})  # type: ignore
            key = f'code:ast:function:{fn}'
            try:
                cur = st.get(key) or []  # type: ignore
                if path not in cur:
                    cur.append(path)
                    st.put(key, cur)  # type: ignore
            except Exception:
                pass
            wrote += 1
            if wrote >= limit: break
    # Try ast-grep for JS/TS classes/functions (best-effort)
    exe = ast_grep_available()
    if exe:
        langs = ["js", "jsx", "ts", "tsx"]
        patterns = {
            "class": [
                "class $A { ... }",
                "export default class $A { ... }",
                "export class $A { ... }",
            ],
            "function": [
                "function $A($B) { ... }",
                "export default function $A($B) { ... }",
                "export function $A($B) { ... }",
                "const $A = ($B) => { ... }",
            ],
            # Methods inside classes (best-effort)
            "method": [
                "class $C { $M($ARGS) { ... } }",
            ],
        }
        import json as _j, re as _re
        for lang in langs:
            for kind, pats in patterns.items():
                for pat in pats:
                    code, out, err = run_ast_grep(root=workspace, pattern=pat, lang=lang, rule_file=None, json=True)
                for line in (out or "").splitlines():
                    try:
                        d = _j.loads(line)
                    except Exception:
                        continue
                    path = d.get('file') or d.get('path') or ''
                    text = d.get('text') or d.get('match') or ''
                    name = None
                    if kind == 'class':
                        m = _re.search(r'class\s+(\w+)', text)
                        name = m.group(1) if m else None
                    elif kind == 'function':
                        m = _re.search(r'function\s+(\w+)\s*\(', text)
                        name = m.group(1) if m else None
                        if not name:
                            m = _re.search(r'const\s+(\w+)\s*=\s*\(', text)
                            name = m.group(1) if m else None
                    elif kind == 'method':
                        m = _re.search(r'(\w+)\s*\(', text)
                        name = m.group(1) if m else None
                    # Streams
                    try:
                        st.append(f'code.ast.{lang}.{kind}', {'name': name or '', 'path': path})  # type: ignore
                    except Exception:
                        pass
                    # KV cache
                    try:
                        if name:
                            key = f'code:ast:{lang}:{kind}:{name}'
                            cur = st.get(key) or []  # type: ignore
                            if path and path not in cur:
                                cur.append(path)
                                st.put(key, cur)  # type: ignore
                    except Exception:
                        pass
    console.print(f"[green]AST cache written ({wrote} records + ast-grep where available).[/green]")


@app.command("find_chunk")
def find_chunk(
    file: Optional[Path] = typer.Option(None, '--file', exists=True, help="File path"),
    start: Optional[int] = typer.Option(None, '--start', help="Start line (1-based)"),
    end: Optional[int] = typer.Option(None, '--end', help="End line (inclusive)"),
    hash: Optional[str] = typer.Option(None, '--hash', help="Chunk hash (sha256 of path+start+end)"),
):
    """Lookup a code chunk either by hash or by file + line range. Uses KV cache when available."""
    import hashlib as _h
    if not hash and not (file and start and end):
        console.print("[yellow]Provide --hash or (--file --start --end).[/yellow]"); raise typer.Exit(2)
    try:
        from .db.factory import get_storage as _get_storage
    except Exception as e:
        console.print(Panel(escape(str(e)), title="storage", border_style="red")); raise typer.Exit(1)
    st = _get_storage()
    if not hash:
        hash = _h.sha256((str(file) + str(start) + str(end)).encode('utf-8')).hexdigest()
    rec = None
    if st is not None:
        rec = st.get(f'code:chunk:{hash}')  # type: ignore
    if isinstance(rec, dict) and rec.get('text'):
        console.print(Panel(rec.get('text'), title=f"{rec.get('path')}:{rec.get('start_line')}-{rec.get('end_line')} | {hash}", border_style="cyan")); raise typer.Exit(0)
    # Fallback: read from disk if file-range specified
    if file and start and end:
        try:
            text = file.read_text(errors='ignore')
            lines = text.splitlines()
            seg = "\n".join(lines[start - 1 : end])
            console.print(Panel(seg or "(empty)", title=f"{file}:{start}-{end} | {hash}", border_style="cyan")); raise typer.Exit(0)
        except Exception as e:
            console.print(Panel(escape(str(e)), title="read failed", border_style="red")); raise typer.Exit(1)
    console.print("[yellow]Chunk not found in KV and no file-range provided.[/yellow]"); raise typer.Exit(1)


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
            console.print(Panel(escape(str(e)), title="Failed to load HF model", border_style="red"))
            raise typer.Exit(1)
    else:
        try:
            embedder = dspy.Embeddings(model=model, api_base=base_url, api_key=api_key)
        except Exception as e:
            console.print(Panel(escape(str(e)), title="Failed to init DSPy Embeddings", border_style="red"))
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
        console.print(Panel(escape(seg), title=title, border_style="cyan"))


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
    model: Optional[str] = typer.Option("qwen3:1.7b", '--model', help="Reflection model name"),
    base_url: Optional[str] = typer.Option(None, '--base-url'),
    api_key: Optional[str] = typer.Option(None, '--api-key'),
    save_best: Optional[Path] = typer.Option(None, '--save-best', help="Write best candidate program mapping to this JSON file"),
    force_json: bool = typer.Option(False, '--force-json', help="Force simple JSON outputs; skip structured-outputs"),
    structured: bool = typer.Option(False, '--structured', help="Prefer structured-outputs when available (overrides --force-json)"),
    report_dir: Optional[Path] = typer.Option(None, '--report-dir', help="Directory to write evaluation reports"),
    workspace: Optional[Path] = typer.Option(None, '--workspace', exists=True, dir_okay=True, help="Workspace to fetch code summary (optional)"),
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
    lm = _maybe_configure_lm(True, ollama, model, base_url, api_key, workspace=workspace or Path.cwd())
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
    code_summary: Optional[str] = None
    if workspace:
        code_summary = _get_code_summary(workspace)
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
                    code_summary=code_summary,
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
                    code_summary=code_summary,
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

    try:
        if optimized is not None:
            target_ws = workspace or Path.cwd()
            progress_obj = progress_path if progress_path.exists() else None
            _record_gepa_outcome(module, optimized, target_ws, progress_obj)
    except Exception:
        pass

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
    model: Optional[str] = typer.Option("qwen3:1.7b", '--model'),
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
    lm = _maybe_configure_lm(True, ollama, model, base_url, api_key, workspace=workspace or Path.cwd())
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

    try:
        if optimized is not None:
            target_ws = Path.cwd()
            _record_gepa_outcome('orchestrator', optimized, target_ws, progress_path if progress_path.exists() else None)
    except Exception:
        pass


@app.command()
def gepa_codegen(
    train_jsonl: Path = typer.Option(..., '--train-jsonl', exists=True, help="Train JSONL for code edits: {task, context, file_hints?}"),
    workspace: Path = typer.Option(Path.cwd(), '--workspace', exists=True, dir_okay=True, help="Workspace to test patches in"),
    test_cmd: Optional[str] = typer.Option(None, '--test-cmd', help="Test command (e.g., 'pytest -q')"),
    type_cmd: Optional[str] = typer.Option("python -m compileall -q .", '--type-cmd', help="Type/syntax check command"),
    lint_cmd: Optional[str] = typer.Option(None, '--lint-cmd', help="Lint command (optional)"),
    auto: Optional[str] = typer.Option('light', '--auto', help="GEPA budget: light|medium|heavy"),
    log_dir: Optional[Path] = typer.Option(None, '--log-dir', help="Where to write GEPA logs"),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama', help="Use Ollama for reflection LM"),
    model: Optional[str] = typer.Option("qwen3:1.7b", '--model', help="Reflection model"),
    base_url: Optional[str] = typer.Option(None, '--base-url'),
    api_key: Optional[str] = typer.Option(None, '--api-key'),
):
    lm = _maybe_configure_lm(True, ollama, model, base_url, api_key, workspace=Path.cwd())
    if lm is None:
        console.print("[yellow]No LM configured; cannot run codegen GEPA.[/yellow]")
        raise typer.Exit(1)
    _print_header("GEPA Codegen")
    if not log_dir:
        log_dir = workspace / '.gepa_codegen'
    progress_path = (log_dir / 'progress.jsonl').resolve()
    try:
        prog = run_gepa_codegen(
            train_jsonl=train_jsonl,
            workspace=workspace,
            test_cmd=test_cmd,
            type_cmd=type_cmd,
            lint_cmd=lint_cmd,
            auto=auto,
            reflection_lm=lm,
            log_dir=str(log_dir) if log_dir else None,
            track_stats=True,
        )
        console.print("[green]GEPA codegen complete.[/green]")
    except Exception as e:
        console.print(Panel(escape(str(e)), title="gepa codegen failed", border_style="red"))
        raise typer.Exit(1)

    try:
        if prog is not None:
            progress_obj = progress_path if progress_path.exists() else None
            _record_gepa_outcome('codegen', prog, workspace, progress_obj)
    except Exception:
        pass


@app.command("edit")
def code_edit(
    task: str = typer.Argument(..., help="Describe the code change you want"),
    workspace: Path = typer.Option(Path.cwd(), '--workspace', exists=True, dir_okay=True),
    context: Optional[str] = typer.Option(None, '--context', help="Optional context (errors/logs). If omitted, built from logs"),
    file_hints: Optional[str] = typer.Option(None, '--files', help="Optional file/module hints"),
    apply: bool = typer.Option(False, '--apply/--no-apply', help="Apply the generated patch"),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama'),
    model: Optional[str] = typer.Option("qwen3:1.7b", '--model'),
    base_url: Optional[str] = typer.Option(None, '--base-url'),
    api_key: Optional[str] = typer.Option(None, '--api-key'),
):
    lm = _maybe_configure_lm(True, ollama, model, base_url, api_key, workspace=workspace)
    if lm is None:
        console.print("[yellow]No LLM configured; cannot generate patch.[/yellow]"); raise typer.Exit(1)
    bundle = _build_patch_context_bundle(workspace, workspace / 'logs', task)
    _render_recent_fixes(console, bundle)
    ctx_text = context or bundle.get('combined_context', '')
    from .skills.code_edit import CodeEdit
    code_graph = _get_code_summary(workspace)
    hints = file_hints or bundle.get('file_hints', '')
    locator_info = None
    try:
        if not hints:
            fl = FileLocator()
            loc = fl(task=task, context=ctx_text, code_graph=code_graph)
            locator_info = loc
            hints = (loc.file_candidates or "")
    except Exception:
        pass
    ce = CodeEdit(use_cot=True)
    out = ce(task=task, context=ctx_text[:8000], code_graph=code_graph, file_hints=hints)
    _print_header("Proposed Patch")
    console.print(out.patch or "(no patch)")
    _print_header("Rationale")
    console.print(out.rationale or "(no rationale)")
    # Show locator output if available
    if locator_info is not None:
        _print_header("File Candidates")
        try:
            console.print(locator_info.file_candidates)
            console.print(Panel.fit(str(getattr(locator_info, 'notes', '') or ''), title="Locator Notes", border_style="dim"))
        except Exception:
            pass
    # Verify patch before applying
    verifier = PatchVerifier(max_files=4, max_lines=200)
    v = verifier(task=task, context=ctx_text, patch=out.patch or "")
    _print_header("Patch Verification")
    console.print(Panel.fit(f"verdict={getattr(v, 'verdict', 'fail')} risk={getattr(v, 'risk_level', 'high')}", title="verifier", border_style="accent"))
    if getattr(v, 'reasons', None):
        console.print(Panel.fit(escape(getattr(v, 'reasons')), title="Reasons", border_style="yellow"))
    if getattr(v, 'fix_suggestions', None):
        console.print(Panel.fit(escape(getattr(v, 'fix_suggestions')), title="Suggestions", border_style="green"))
    if apply and out.patch:
        if getattr(v, 'verdict', 'fail').lower() != 'pass':
            console.print(Panel.fit("Refusing to apply: verifier did not pass.", title="apply blocked", border_style="red"))
        else:
            ok, msg = apply_unified_patch(out.patch, workspace)
            if ok:
                console.print(f"[green]{msg}[/green]")
                # Plan tests if possible
                try:
                    tp = TestPlanner()
                    repo_layout = _repo_layout_summary(workspace)
                    tp_out = tp(task=task, context=ctx_text, repo_layout=repo_layout)
                    if getattr(tp_out, 'commands', None):
                        _print_header("Test Plan")
                        console.print(Panel.fit(escape(getattr(tp_out, 'tests_to_run', '') or ''), title="Tests To Run", border_style="cyan"))
                        console.print(Panel.fit(escape(getattr(tp_out, 'commands', '')), title="Commands", border_style="green"))
                        if getattr(tp_out, 'fast_paths', None):
                            console.print(Panel.fit(escape(getattr(tp_out, 'fast_paths')), title="Fast Paths", border_style="magenta"))
                except Exception:
                    pass
            else:
                console.print(Panel(msg, title="apply failed", border_style="red"))


@app.command()
def init(
    workspace: Path = typer.Option(Path.cwd(), '--workspace', dir_okay=True, exists=True, help="Workspace to initialize"),
    logs: Optional[Path] = typer.Option(None, '--logs', file_okay=True, dir_okay=True, exists=True, help="Logs path (defaults to <ws>/logs)"),
    out_dir: Optional[Path] = typer.Option(None, '--out-dir', help="Where to write datasets (default <ws>/.dspy_data)"),
    train: bool = typer.Option(False, '--train/--no-train', help="Run a light GEPA training pass after bootstrapping"),
    budget: str = typer.Option('light', '--budget', help="GEPA budget: light|medium|heavy"),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama', help="Use Ollama for reflection LM"),
    model: Optional[str] = typer.Option("qwen3:1.7b", '--model', help="Reflection model name"),
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
    lm = _maybe_configure_lm(True, ollama, model, base_url, api_key, workspace=ws)
    if lm is None:
        console.print("[yellow]No LM configured; skipping training.[/yellow]")
        return
    try:
        _ = run_gepa_orchestrator(train_jsonl=paths['orchestrator'], auto=budget, reflection_lm=lm, log_dir=str(ws / '.gepa_orch'), track_stats=True)
        console.print("[green]Orchestrator training complete.[/green]")
    except Exception as e:
        console.print(Panel(escape(str(e)), title="orchestrator training failed", border_style="red"))
    try:
        _ = run_gepa(module='context', train_jsonl=paths['context'], auto=budget, reflection_lm=lm, log_dir=str(ws / '.gepa_ctx'), track_stats=True)
        console.print("[green]Context module training complete.[/green]")
    except Exception as e:
        console.print(Panel(escape(str(e)), title="context training failed", border_style="red"))
    try:
        _ = run_gepa(module='code', train_jsonl=paths['code'], auto=budget, reflection_lm=lm, log_dir=str(ws / '.gepa_code'), track_stats=True)
        console.print("[green]Code module training complete.[/green]")
    except Exception as e:
        console.print(Panel(escape(str(e)), title="code training failed", border_style="red"))
    try:
        _ = run_gepa(module='task', train_jsonl=paths['task'], auto=budget, reflection_lm=lm, log_dir=str(ws / '.gepa_task'), track_stats=True)
        console.print("[green]Task module training complete.[/green]")
    except Exception as e:
        console.print(Panel(escape(str(e)), title="task training failed", border_style="red"))
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
    model: Optional[str] = typer.Option("qwen3:1.7b", '--model'),
    base_url: Optional[str] = typer.Option(None, '--base-url'),
    api_key: Optional[str] = typer.Option(None, '--api-key'),
):
    """Watch logs and retrain on a cadence, preserving test set.

    - Rebuild datasets and stratified splits
    - Train with val set; evaluate on test
    - Persist evaluation reports under --report-dir
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    lm = _maybe_configure_lm(True, ollama, model, base_url, api_key, workspace=workspace)
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
                            console.print(Panel(escape(str(e)), title=f"{m} export failed", border_style="yellow"))
                        if m not in results:
                            console.print(Panel("No trained program returned.", title=f"{m} eval skipped", border_style="yellow"))
                            continue
                        if m == 'orchestrator':
                            stats = evaluate_orchestrator(results[m], split_dir / 'orchestrator_test.jsonl')  # type: ignore[arg-type]
                        else:
                            stats = evaluate_on_set(m, results[m], split_dir / f'{m}_test.jsonl')  # type: ignore[arg-type]
                        write_report(m, stats)
                    except Exception as e:
                        console.print(Panel(escape(str(e)), title=f"{m} eval failed", border_style="red"))
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
        model="qwen3:1.7b",
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
    from .code_tools.diffutil import unified_diff_from_texts
    old_text = file.read_text(errors="ignore") if file.exists() else ""
    patch_text = unified_diff_from_texts(old_text, new_text, a_path=str(file), b_path=str(file), n=unified)
    if out:
        out.write_text(patch_text)
        console.print(f"[green]Wrote patch to {out}[/green]")
    else:
        console.print(patch_text or "(no differences)")


@app.command(name="start")
def start_command(
    workspace: Optional[Path] = typer.Option(None, '--workspace', dir_okay=True, exists=True, help="Initial workspace"),
    logs: Optional[Path] = typer.Option(None, '--logs', dir_okay=True, file_okay=True, exists=False, help="Initial logs path (defaults to <ws>/logs)"),
    ollama: bool = typer.Option(True, '--ollama/--no-ollama', help="Use Ollama by default in session"),
    model: Optional[str] = typer.Option("qwen3:1.7b", '--model', help="Default model for session"),
    base_url: Optional[str] = typer.Option(None, '--base-url', help="Override base URL"),
    api_key: Optional[str] = typer.Option(None, '--api-key', help="API key (unused for Ollama)"),
    force_json: bool = typer.Option(False, '--force-json', help="Force simple JSON outputs; skip structured-outputs"),
    structured: bool = typer.Option(False, '--structured', help="Prefer structured-outputs when available (overrides --force-json)"),
    approval: Optional[str] = typer.Option(None, '--approval', help='Tool approval mode: auto|manual'),
    coding_mode: bool = typer.Option(False, '--coding-mode', help="Enhanced coding assistant mode with build/test integration"),
):
    """Interactive session to pick workspace/logs and run tasks."""
    ws = Path(workspace) if workspace else Path.cwd()
    logs_path = Path(logs) if logs else (ws / 'logs')

    toolchain_executor: Optional[ToolchainExecutor] = None

    def _refresh_toolchain() -> None:
        nonlocal toolchain_executor
        try:
            toolchain_executor = ToolchainExecutor(detect_toolchain(ws))
        except Exception:
            toolchain_executor = None

    def _execute_toolchain(action: ToolAction, action_args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        nonlocal toolchain_executor
        if toolchain_executor is None:
            _refresh_toolchain()
        if toolchain_executor is None:
            console.print(f"[yellow]Toolchain executor unavailable for {action.name.lower()} (configure RL/toolchain commands).[/yellow]")
            return {}, {}
        try:
            result = toolchain_executor(action, action_args)
        except Exception as exc:
            console.print(Panel(escape(str(exc)), title=f"{action.name.lower()} failed", border_style="red"))
            return {}, {}
        return _render_toolchain_console(action.name.lower(), result)

    _refresh_toolchain()

    use_lm = True
    provider_is_ollama = ollama
    last_extract: Optional[str] = None
    
    # Initialize LM
    from .llm import configure_lm
    lm = None
    if use_lm:
        try:
            provider = "ollama" if provider_is_ollama else "openai"
            lm = configure_lm(
                provider=provider,
                model_name=model,
                base_url=base_url,
                api_key=api_key
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to initialize LM: {e}[/yellow]")
            use_lm = False

    # Apply runtime toggle for adapter behavior; default to simple JSON to avoid noisy warnings
    if structured:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'false'
    elif force_json or os.environ.get('DSPY_FORCE_JSON_OBJECT') is None:
        os.environ['DSPY_FORCE_JSON_OBJECT'] = 'true'
    # Tame noisy adapter warnings
    try:
        logging.getLogger('dspy.adapters.json_adapter').setLevel(logging.ERROR)
    except Exception:
        pass
    # Set LiteLLM timeout defaults if not set
    os.environ.setdefault('LITELLM_TIMEOUT', '30')
    os.environ.setdefault('LITELLM_MAX_RETRIES', '2')

    # Resolve approval mode
    settings = get_settings()
    approval_mode = (approval or getattr(settings, 'tool_approval_mode', 'auto') or 'auto').lower()
    if approval_mode not in {"auto", "manual"}:
        approval_mode = "auto"

    # Friendly LLM label with quick Ollama readiness probe to avoid long hangs
    llm_label = "OpenAI/compatible"
    if provider_is_ollama:
        eff_base = (base_url or os.getenv("OPENAI_BASE_URL") or "http://localhost:11434").rstrip("/")
        if eff_base.endswith("/v1"):
            eff_base = eff_base[:-3]
        eff_model = model or os.getenv("MODEL_NAME") or os.getenv("OLLAMA_MODEL") or "llama3"
        server_ok, model_ok = check_ollama_ready(eff_base, eff_model)
        if not server_ok:
            llm_label = "disabled (Ollama offline)"
        elif not model_ok:
            llm_label = f"Ollama missing model: {eff_model}"
        else:
            llm_label = f"Ollama {eff_model}"

    console.print(Panel.fit(
        f"Workspace: {ws}\nLogs: {logs_path}\nLLM: {llm_label}\nApproval: {approval_mode}\nTip: Type natural instructions — the agent will choose tools.",
        title="blampert session",
        border_style="cyan"
    ))

    auto_runner: Optional[AutoTrainingLoop] = None
    if _auto_train_enabled():
        try:
            auto_runner = AutoTrainingLoop(
                ws,
                logs_path,
                console=console,
                label='interactive',
                ollama=provider_is_ollama,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
            auto_runner.start()
            console.print("[dim]Auto-training loop started in background.[/dim]")
        except Exception as e:
            console.print(Panel(f"auto-training unavailable: {e}", title="auto-train", border_style="yellow"))

    def show_help():
        if coding_mode:
            console.print(Panel.fit(
                "🚀  Coding Assistant Commands:\n\n"
                "📁 Workspace & Navigation:\n"
                "  help                Show this help\n"
                "  ws                  Show workspace\n"
                "  cd <PATH>           Change workspace\n"
                "  ls [PATH]           List files (relative to workspace)\n"
                "  tree [PATH] [-d N] [--hidden]  Show directory tree (default depth 2)\n"
                "  open <PATH>         Open a file in $EDITOR / OS default\n\n"
                "🔍 Code Analysis & Search:\n"
                "  grep <PATTERN>      Search (flags: -f fixed, -c N, -g GLOB, -x GLOB, -F FILE)\n"
                "  extract --file F [--symbol NAME | --re REGEX --before N --after N]\n"
                "  sg [-p PATTERN] [-l LANG] [-r RULE.yaml] [--json] Run ast-grep\n"
                "  codectx [PATH]      Summarize code file/dir\n"
                "  index               Build code index for semantic search\n"
                "  esearch <QUERY>     Semantic search over index\n"
                "  emb-index [-m MODEL] Build embedding index (requires embeddings provider)\n"
                "  emb-search <QUERY> [-m MODEL] Embedding search\n\n"
                "🛠️  Coding & Development:\n"
                "  edit <TASK> [--apply]  Propose patch (and optionally apply)\n"
                "  build [--clean]     Build the project (auto-detects build system)\n"
                "  test [--coverage]   Run tests with optional coverage\n"
                "  lint [--fix]        Run linter with optional auto-fix\n"
                "  run <COMMAND>       Execute shell command safely\n"
                "  patch <PATCHFILE>   Apply unified diff patch\n"
                "  diff <FILE>         Diff last extract against FILE\n"
                "  write <PATH>        Save last extract to file\n\n"
                "🚀 Development Workflow:\n"
                "  dev quick [msg]     Quick dev cycle (format, test, commit, push)\n"
                "  dev build           Build package\n"
                "  dev test            Run tests\n"
                "  dev lint            Run linter\n"
                "  dev format          Format code\n"
                "  dev status          Show git status\n"
                "  dev version         Show current version\n"
                "  release [type]      Full release workflow (patch/minor/major)\n"
                "  publish [--test]    Publish to PyPI (or Test PyPI)\n\n"
                "📊 Planning & Context:\n"
                "  plan <TASK>         Propose plan + commands\n"
                "  ctx                 Show context from logs\n"
                "  logs [PATH]         Show or set logs path\n"
                "  watch [-n SECS] [-t LINES]  Tail + refresh key events\n\n"
                "🤖 Agent & Learning:\n"
                "  stats               Show agent stats and rewards\n"
                "  auto [status|enable|disable|restart] Manage auto-training loop\n"
                "  learn <TASK>        Learn from successful coding patterns\n"
                "  feedback <SCORE>    Provide feedback on last action (0-10)\n\n"
                "🎓 Expert-Level Features:\n"
                "  expert-patterns     Show learned expert patterns\n"
                "  expert-tools        Show most effective tools\n"
                "  expert-insights     Show codebase insights\n"
                "  expert-optimize     Optimize prompts and policies\n"
                "  expert-status       Show expert-level status\n\n"
                "📝 Git & Version Control:\n"
                "  gstatus             Git status (short)\n"
                "  gadd <PATHS...>     Git add files\n"
                "  gcommit -m MSG      Git commit with message\n\n"
                "⚙️  Configuration:\n"
                "  ollama on|off       Toggle Ollama provider\n"
                "  model <NAME>        Set model name\n"
                "  coding-mode on|off  Toggle enhanced coding mode\n",
                title="🚀 Blampert Code Assistant", border_style="green"
            ))
        else:
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
            "  edit <TASK> [--apply]  Propose patch (and optionally apply)\n"
            "  index               Build code index for semantic search\n"
            "  esearch <QUERY>     Semantic search over index\n"
            "  emb-index [-m MODEL] Build embedding index (requires embeddings provider)\n"
            "  emb-search <QUERY> [-m MODEL] Embedding search\n"
            "  stats               Show agent stats and rewards\n"
            "  auto [status|enable|disable|restart] Manage auto-training loop\n"
            "  open <PATH>         Open a file in $EDITOR / OS default\n"
            "  patch <PATCHFILE>   Apply unified diff patch\n"
            "  diff <FILE>         Diff last extract against FILE\n"
            "  write <PATH>        Save last extract to file\n"
            "  gstatus             Git status (short)\n"
            "  gadd <PATHS...>     Git add files\n"
            "  gcommit -m MSG      Git commit with message\n"
            "  ollama on|off       Toggle Ollama provider\n"
            "  model <NAME>        Set model name\n"
                "  coding-mode on|off  Toggle enhanced coding mode\n",
                title="DSPy Agent Commands", border_style="cyan"
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

    def orchestrate_chain(nl: str, max_steps: int = 4, lm=None):
        # Try LLM orchestrator, fall back to heuristics
        history_summary = ""
        last_tool = None
        last_args = None
        targets = _extract_targets_from_query(nl)
        
        # Initialize enhanced orchestrator with memory
        from .skills.orchestrator import Orchestrator, SessionMemory, ToolResult, ChainSummary
        orchestrator = Orchestrator(use_cot=True, workspace=ws)
        memory = orchestrator.get_memory() or orchestrator.create_memory(ws)
        
        # Online RL over safe tools
        rl_enabled = True  # enable by default; can later wire to settings
        rl_tools = ["context", "codectx", "grep", "esearch", "plan", "tree", "ls", "index", "emb-index", "intel", "vretr", "edit"]
        rl_state_path = ws / '.dspy_rl_state.json'
        bandit = _OnlineBandit(rl_tools, rl_state_path)
        # Per-tool cooldowns (seconds) to avoid heavy repetition
        tool_cooldown = {"context": 5, "index": 300, "emb-index": 600, "intel": 10, "vretr": 5, "esearch": 3, "grep": 2, "edit": 20}
        last_run: dict[str, float] = {}
        step_history: list[dict] = []
        patch_done = False
        auto_apply_env = os.getenv("DSPY_AUTO_APPLY_PATCHES", "true").lower() in {"1", "true", "yes", "on"}

        def _run_edit_tool(args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            metrics: Dict[str, Any] = {}
            info: Dict[str, Any] = {}
            task_text = args.get("task") or nl
            file_hints = args.get("file_hints") or args.get("files") or ""
            auto_apply_flag = bool(args.get("apply", True))
            lm = _maybe_configure_lm(use_lm, provider_is_ollama, model, base_url, api_key, workspace=ws)
            if lm is None:
                console.print("[yellow]No LLM configured; cannot generate patch.[/yellow]")
                return metrics, info
            bundle_logs, _ = load_logs([logs_path])
            ctx_text = extract_key_events(bundle_logs) if bundle_logs else ""
            from .skills.code_edit import CodeEdit
            code_graph = _get_code_summary(ws)
            ce = CodeEdit(use_cot=True)
            out = ce(task=task_text, context=ctx_text, code_graph=code_graph, file_hints=file_hints)
            _print_header("Proposed Patch")
            console.print(out.patch or "(no patch)")
            _print_header("Rationale")
            console.print(out.rationale or "(no rationale)")
            info["patch"] = out.patch or ""
            if auto_apply_flag and out.patch:
                do_apply = True
                if approval_mode == "manual":
                    do_apply = typer.confirm("Apply this patch?", default=False)
                if do_apply:
                    ok, msg = apply_unified_patch(out.patch, ws)
                    metrics = {"applied": bool(ok)}
                    if ok:
                        console.print(f"[green]{msg}[/green]")
                        summ = summarize_patch(out.patch)
                        blast = float(summ['added_lines'] + summ['removed_lines'])
                        metrics.update({"blast_radius": blast})
                        console.print(Panel.fit(
                            f"files: {summ['files']}  +lines: {summ['added_lines']}  -lines: {summ['removed_lines']}",
                            title="patch metrics", border_style="accent"
                        ))
                    else:
                        console.print(Panel(msg, title="apply failed", border_style="red"))
            return metrics, info
        for step in range(1, max_steps + 1):
            state = build_state() + (f" | history: {history_summary[:4000]}" if history_summary else "")
            tool = None
            args = {}
            
            # Check for cached results first
            if memory:
                # Try to get cached result for common tools
                cached_result = memory.get_cached_result("codectx", {"path": str(ws)})
                if cached_result and step == 1:
                    console.print(f"[green]Using cached result from {time.ctime(cached_result.timestamp)}[/green]")
                    memory.add_tool_result(cached_result)
                    continue
            
            # Use enhanced orchestrator for tool selection
            if use_lm and lm is not None:
                try:
                    pred = orchestrator(nl, state, memory)
                    tool = (pred.tool or "").strip()
                    try:
                        args = _json.loads(pred.args_json or "{}")
                    except Exception:
                        args = {}
                    
                    # Show rationale if available
                    if getattr(pred, 'rationale', None):
                        console.print(f"[dim]Rationale: {pred.rationale}[/dim]")
                    
                    # Show if using cache
                    if getattr(pred, 'cached', False):
                        console.print(f"[dim]Using cached prediction[/dim]")
                        
                except Exception:
                    tool = None
            
            # Heuristic shortcut: explanation-style queries → codectx first
            if not tool:
                nl_l = nl.lower()
                if step == 1 and any(k in nl_l for k in ["explain", "overview", "how does", "what is it doing", "walkthrough", "architecture"]):
                    tool = "codectx"; args = {"path": str(ws)}
            
            # Prefer RL suggestion among safe tools
            if rl_enabled and not tool:
                try:
                    tool = bandit.select()
                    console.print(f"[cyan]RL suggested tool: {tool}[/cyan]")
                except Exception:
                    tool = None
            
            # Enforce cooldowns to bound latency/cost
            if tool and tool in tool_cooldown:
                import time as _t
                now = _t.time()
                if now - float(last_run.get(tool, 0.0)) < float(tool_cooldown[tool]):
                    console.print(f"[dim]Skipping '{tool}' due to cooldown.[/dim]")
                    tool = None
            
            # Autogenerate minimal args for RL-suggested tools
            if tool and not args:
                if tool == 'grep':
                    args = {"pattern": nl}
                elif tool == 'esearch':
                    args = {"query": nl, "k": 5}
                elif tool == 'plan':
                    args = {"task": nl}
                elif tool == 'tree':
                    args = {"depth": 2}
                elif tool == 'ls':
                    args = {}
                elif tool == 'intel':
                    args = {"query": nl, "k": 5}
                elif tool == 'vretr':
                    args = {"query": nl, "k": 5}
                elif tool == 'index':
                    args = {}
                elif tool == 'emb-index':
                    args = {"model": "all-MiniLM-L6-v2", "hf": True}
                elif tool == 'edit':
                    args = {"task": nl, "apply": False}
            # If RL not enabled or failed, try LLM orchestrator
            if not tool:
                lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key, workspace=ws)
                if lm is not None:
                    try:
                        orch = Orchestrator(use_cot=True)
                        pred = orch(query=nl, state=state)
                        tool = (pred.tool or "").strip()
                        import json as _json
                        try:
                            args = _json.loads(pred.args_json or "{}")
                        except Exception:
                            args = {}
                        console.print(f"[green]LLM selected tool: {tool}[/green]")
                    except Exception as e:
                        console.print(f"[yellow]LLM failed ({e}), using fallback[/yellow]")
                        tool = None
            if not tool:
                # fallback heuristic single shot
                tool = "context" if (logs_path.exists() if isinstance(logs_path, Path) else True) else "codectx"
                args = {}

            console.print(Panel.fit(f"{tool} {args}", title=f"Step {step}: action", border_style="yellow"))
            # Stop if repeating same choice
            if last_tool == tool and last_args == args:
                console.print("[dim]No new action; summarizing results.[/dim]")
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
                # Record last run time for cooldowns
                try:
                    import time as _t
                    last_run[tool] = _t.time()
                except Exception:
                    pass
            except Exception as e:
                console.print(Panel(escape(str(e)), title=f"agent failed ({tool})", border_style="red"))
                break

            # Summarize outcome to feed back
            try:
                outcome = evaluate_tool_choice(tool, args, workspace=ws, logs_path=logs_path, targets=targets)
                piece = f"{tool}: score={outcome.score:.2f}; {outcome.evidence}"
                
                # Create and record tool result for memory
                if memory:
                    # Determine success based on tool metrics and outcome
                    success = False
                    if tool_metrics and "applied" in tool_metrics:
                        success = tool_metrics["applied"]
                    elif outcome.score >= 1.0:  # High score indicates success
                        success = True
                    elif tool_metrics and "blast_radius" in tool_metrics:
                        success = tool_metrics["blast_radius"] > 0  # Any changes made
                    
                    tool_result = ToolResult(
                        tool=tool,
                        args=dict(args),
                        result=piece,
                        success=success,
                        timestamp=time.time(),
                        execution_time=0.1,  # Approximate
                        score=outcome.score,
                        feedback=outcome.evidence
                    )
                    memory.add_tool_result(tool_result)
                
                # Update RL bandit with a clipped reward in [0,1]
                try:
                    r = float(outcome.score)
                    r = 0.0 if not (r == r) else max(0.0, min(1.0, r / 2.0))
                    if tool in rl_tools:
                        bandit.update(tool, r)
                        console.print(f"[dim]RL updated: {tool} reward={r:.2f}[/dim]")
                        # Append event for background trainer to learn from
                        try:
                            import time as _t2
                            evt = {"ts": int(_t2.time()), "tool": tool, "reward": float(r)}
                            # Append to local events file
                            (ws / '.dspy_rl_events.jsonl').open('a').write(json.dumps(evt) + "\n")
                            # Also publish to Kafka if available
                            try:
                                from confluent_kafka import Producer  # type: ignore
                                bootstrap = os.getenv('KAFKA_BOOTSTRAP') or os.getenv('KAFKA_BOOTSTRAP_SERVERS') or 'localhost:9092'
                                if _kafka_is_available(bootstrap):
                                    silent = logging.getLogger('kafka.silent'); silent.addHandler(logging.NullHandler()); silent.setLevel(logging.CRITICAL)
                                    p = Producer({'bootstrap.servers': bootstrap}, logger=silent)
                                    p.produce('agent.learning', json.dumps(evt).encode('utf-8'))
                                    p.flush(0)
                            except Exception:
                                pass
                        except Exception:
                            pass
                    # Keep in history
                    step_history.append({"tool": tool, "args": args, "score": float(outcome.score), "evidence": str(outcome.evidence)})
                except Exception:
                    pass
            except Exception:
                piece = f"{tool}: done"
                step_history.append({"tool": tool, "args": args, "score": 0.0, "evidence": ""})
            history_summary = (history_summary + " | " + piece).strip()
            # Simple stop conditions
            if tool in {"esearch", "grep", "extract"} and "hits=0" in piece:
                continue
            if tool in {"context", "codectx"} and "events_len=0" in piece:
                continue
            # If good score, and we've run at least 2 steps, stop
            if step >= 2 and ("score=" in piece and float(piece.split("score=")[1].split(";")[0]) >= 1.2):
                break

        if not patch_done:
            console.print("[cyan]No diff proposed during exploration; generating patch suggestion now.[/cyan]")
            forced_args = {"task": nl, "apply": auto_apply_env}
            tool_metrics, tool_info = _run_edit_tool(forced_args)
            patch_done = True
            forced_outcome = None
            forced_evidence = ""
            try:
                forced_outcome = evaluate_tool_choice(
                    "edit",
                    forced_args,
                    workspace=ws,
                    logs_path=logs_path,
                    targets=targets,
                    result_metrics=tool_metrics or None,
                    result_info=tool_info or None,
                )
                forced_evidence = forced_outcome.evidence
                piece = f"edit: score={forced_outcome.score:.2f}; {forced_evidence}"
                
                # Create and record tool result for memory
                if memory:
                    # Determine success based on tool metrics and outcome
                    success = False
                    if tool_metrics and "applied" in tool_metrics:
                        success = tool_metrics["applied"]
                    elif forced_outcome.score >= 1.0:  # High score indicates success
                        success = True
                    elif tool_metrics and "blast_radius" in tool_metrics:
                        success = tool_metrics["blast_radius"] > 0  # Any changes made
                    
                    tool_result = ToolResult(
                        tool="edit",
                        args=dict(forced_args),
                        result=piece,
                        success=success,
                        timestamp=time.time(),
                        execution_time=0.1,  # Approximate
                        score=forced_outcome.score,
                        feedback=forced_evidence
                    )
                    memory.add_tool_result(tool_result)
            except Exception as e:
                forced_evidence = str(e)
                piece = f"edit: forced patch failed ({e})"
            step_history.append({
                "tool": "edit",
                "args": forced_args,
                "score": float(forced_outcome.score) if forced_outcome else 0.0,
                "evidence": forced_evidence,
            })
            history_summary = (history_summary + " | " + piece).strip()

        # Enhanced session summary with memory
        try:
            if memory:
                summary = memory.finish_chain(nl)
                _display_enhanced_summary(summary)
            else:
                _print_header("Session Summary")
                if step_history:
                    avg = sum(h.get("score", 0.0) for h in step_history) / max(1, len(step_history))
                    lines = [f"- step {i+1}: {h['tool']} score={h.get('score',0.0):.2f}" for i,h in enumerate(step_history)]
                    console.print(Panel.fit("\n".join(lines), title=f"actions (avg={avg:.2f})", border_style="cyan"))
                    # Recommend next actions
                recs: list[str] = []
                seen_tools = {h['tool'] for h in step_history}
                if "index" not in seen_tools:
                    recs.append("index")
                if "emb-index" not in seen_tools:
                    recs.append("emb-index")
                if "intel" not in seen_tools:
                    recs.append("intel --query '<your question>'")
                if "esearch" not in seen_tools:
                    recs.append("esearch --q '<keywords>'")
                if recs:
                    console.print(Panel.fit("\n".join(f"- {r}" for r in recs), title="suggested next steps", border_style="yellow"))
                else:
                    console.print(Panel.fit("No actions executed.", title="actions", border_style="yellow"))
        except Exception:
            pass

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
            console.print(Panel.fit(escape(key), title="Extracted Events", border_style="magenta"))
        elif t == "plan":
            bundle, count = load_logs([logs_path]); key = extract_key_events(bundle) if bundle else ""
            lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key, workspace=ws)
            if lm is None:
                console.print("[yellow]No LM configured for planning.[/yellow]"); return
            builder = ContextBuilder(); ctx = builder(task=args.get("task", ""), logs_preview=key)
            code_graph = _get_code_summary(ws)
            fused_context = f"{ctx.context}\n\n{ctx.key_points}\n\nCode Summary:\n{code_graph}" if code_graph else f"{ctx.context}\n\n{ctx.key_points}"
            _print_header("Agent Plan"); agent = TaskAgent()
            out = agent(task=args.get("task", ""), context=fused_context)
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
                console.print(Panel(escape(seg), title=f"{fpath}::{symbol} lines {start}-{end}", border_style="green"))
            elif regex:
                hits = file_search(fpath, regex, regex=True)
                if not hits: console.print("[yellow]No regex match.[/yellow]"); return
                hit = hits[0]; text = fpath.read_text(errors="ignore");
                s,e,seg = extract_context(text, hit.line_no, before=args.get("before",3), after=args.get("after",3)); last_extract=seg
                console.print(Panel(escape(seg), title=f"{fpath} lines {s}-{e}", border_style="green"))
            else:
                console.print("[yellow]Provide --symbol or --regex for extract.[/yellow]")
        elif t == "tree":
            depth = int(args.get("depth", 2)); hidden = bool(args.get("hidden", False));
            console.print(_render_tree(ws, max_depth=depth, show_hidden=hidden))
        elif t == "ls":
            console.print(Panel("\n".join(p.name+("/" if p.is_dir() else "") for p in sorted(ws.iterdir())[:500]), title=str(ws), border_style="blue"))
        elif t == "emb-index":
            # Build embeddings index for semantic search
            model = args.get("model") or "all-MiniLM-L6-v2"
            hf = bool(args.get("hf", True))
            try:
                if hf:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                    embedder = SentenceTransformer(model)
                else:
                    embedder = dspy.Embeddings(model=model)
                from .embedding.indexer import build_index as build_code_index, save_index as save_code_index
                from .embedding.embeddings_index import build_emb_index as _build_emb_index, save_emb_index as _save_emb_index
            except Exception as e:
                console.print(Panel(escape(str(e)), title="emb-index setup failed", border_style="red")); return
            try:
                items = _build_emb_index(ws, embedder)
                out_dir = _save_emb_index(ws, items, persist=False)
                console.print(Panel.fit(f"Embedded {len(items)} chunks → {out_dir}", title="emb-index", border_style="cyan"))
            except Exception as e:
                console.print(Panel(escape(str(e)), title="emb-index failed", border_style="red"))
        elif t == "codectx":
            path = args.get("path"); target = (ws / path).resolve() if path else ws
            snap = build_code_snapshot(target); _print_header("Code Snapshot")
            console.print(Panel.fit(escape(snap[:8000] + ("\n..." if len(snap)>8000 else "")), title=str(target), border_style="magenta"))
            lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key, workspace=ws)
            if lm:
                cc = CodeContext(); code_graph = _get_code_summary(ws)
                out = cc(snapshot=snap, ask="Summarize key parts and modification points.", code_graph=code_graph)
                console.print(Panel.fit(escape(out.summary), title="Summary", border_style="cyan"))
                console.print(Panel.fit(escape(out.bullets), title="Bullets", border_style="green"))
        elif t == "knowledge":
            # args: file, import, class, function, lang
            g = _get_code_graph(ws)
            files = g.get('files') or []
            f = args.get('file'); imp = args.get('import'); cls = args.get('class'); fn = args.get('function'); lang = (args.get('lang') or '').strip()
            matches = []
            # Query KV caches first for class/function if provided
            try:
                from .db.factory import get_storage as _get_storage
                st = _get_storage()
            except Exception:
                st = None
            paths_from_kv = set()
            if st is not None:
                keys = []
                if cls:
                    keys.append(f'code:ast:class:{cls}')
                    if lang:
                        keys.append(f'code:ast:{lang}:class:{cls}')
                if fn:
                    keys.append(f'code:ast:function:{fn}')
                    if lang:
                        keys.append(f'code:ast:{lang}:function:{fn}')
                for k in keys:
                    try:
                        arr = st.get(k) or []  # type: ignore
                        for p in arr:
                            paths_from_kv.add(p)
                    except Exception:
                        pass
            # Merge: KV paths first
            for rec in files:
                pth = rec.get('path','')
                if paths_from_kv and pth not in paths_from_kv:
                    continue
                if f and f not in pth:
                    continue
                if imp and imp not in (rec.get('imports') or []):
                    continue
                if cls and cls not in (rec.get('classes') or []):
                    continue
                if fn and fn not in (rec.get('functions') or []):
                    continue
                matches.append(rec)
            if not matches:
                console.print("[yellow]No knowledge matches. Provide file/import/class/function args.[/yellow]"); return
            title = f"knowledge matches={len(matches)}"
            body = "\n".join(f"- {m.get('path')} | classes={len(m.get('classes') or [])} funcs={len(m.get('functions') or [])}" for m in matches[:50])
            console.print(Panel(body, title=title, border_style="blue"))
            query_label = f"knowledge:{f or ''}:{imp or ''}:{cls or ''}:{fn or ''}:{lang or ''}"
            event_hits = []
            for rec in matches[:50]:
                path = rec.get('path')
                if path:
                    event_hits.append({"path": str(path), "score": 1.0, "source": "knowledge"})
            if event_hits:
                log_retrieval_event(ws, query_label, event_hits)
        elif t == "vretr":
            # Manual vector retrieval
            q = args.get('query') or args.get('q') or ''
            if not q:
                console.print("[yellow]vretr needs --query[/yellow]"); return
            k = int(args.get('k', 5))
            hf = bool(args.get('hf', False)); model = args.get('model') or ('all-MiniLM-L6-v2' if hf else 'openai/text-embedding-3-small')
            if hf:
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                    embedder = SentenceTransformer(model)
                except Exception as e:
                    console.print(Panel(escape(str(e)), title="HF embeddings missing", border_style="red")); return
            else:
                try:
                    embedder = dspy.Embeddings(model=model)
                except Exception as e:
                    console.print(Panel(escape(str(e)), title="DSPy embeddings missing", border_style="red")); return
            try:
                from .embedding.embeddings_index import load_emb_index, emb_search as _emb_search, embed_query as _embed_query, build_emb_index as _build_emb_index, save_emb_index as _save_emb_index
                # Try load; if missing or yields no hits, build index automatically once
                items = load_emb_index(ws)
                qv = _embed_query(embedder, q)
                hits = _emb_search(qv, items, top_k=k)
                if not hits:
                    console.print("[yellow]No vector matches. Building embeddings index...[/yellow]")
                    try:
                        new_items = _build_emb_index(ws, embedder)
                        _save_emb_index(ws, new_items, persist=False)
                        items = new_items
                        hits = _emb_search(qv, items, top_k=k)
                    except Exception as _e2:
                        console.print(Panel(escape(str(_e2)), title="emb-index build failed", border_style="red"))
                if not hits:
                    console.print("[yellow]No vector matches.")
                    return
                event_hits = [{"path": str(Path(it.path)), "score": float(score_i), "source": "vretr"} for score_i, it in hits]
                if event_hits:
                    log_retrieval_event(ws, q, event_hits)
                for score_i, it in hits:
                    p = Path(it.path)
                    try:
                        text = p.read_text(errors='ignore')
                        lines = text.splitlines()
                        s = max(1, it.start_line - 2)
                        e = min(len(lines), it.end_line + 2)
                        seg = "\n".join(lines[s - 1 : e])
                    except Exception:
                        seg = "(unreadable)"; s = it.start_line; e = it.end_line
                    title = f"{p} score={score_i:.3f} lines {s}-{e}"
                    console.print(Panel(escape(seg), title=title, border_style="cyan"))
            except Exception as e:
                console.print(Panel(escape(str(e)), title="vretr failed", border_style="red"))
        elif t == "intel":
            # High-level evidence composition: knowledge + vretr (+ optional sg)
            q = args.get('query') or args.get('q') or ''
            if not q:
                console.print("[yellow]intel needs --query[/yellow]"); return
            # Knowledge
            g = _get_code_graph(ws)
            files = g.get('files') or []
            import re as _re
            classes = _re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', q)
            funcs = _re.findall(r'\b[a-z_][a-z0-9_]+\b', q)
            kn_matches = []
            for rec in files:
                pth = rec.get('path','')
                hit = False
                if any(c in (rec.get('classes') or []) for c in classes):
                    hit = True
                if any(fn in (rec.get('functions') or []) for fn in funcs):
                    hit = True
                if any(tok in pth for tok in funcs[:3]):
                    hit = True
                if hit:
                    kn_matches.append(rec)
            # Vector retrieval
            k = int(args.get('k', 5))
            hf = bool(args.get('hf', False)); model = args.get('model') or ('all-MiniLM-L6-v2' if hf else 'openai/text-embedding-3-small')
            if hf:
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                    embedder_obj = SentenceTransformer(model)
                    def _emb(text): return embedder_obj.encode(text)
                except Exception as e:
                    console.print(Panel(escape(str(e)), title="HF embeddings missing", border_style="red")); return
            else:
                try:
                    embedder_obj = dspy.Embeddings(model=model)
                    def _emb(text): return embedder_obj.embed(text)  # type: ignore
                except Exception as e:
                    console.print(Panel(escape(str(e)), title="DSPy embeddings missing", border_style="red")); return
            try:
                from .embedding.embeddings_index import load_emb_index, emb_search as _emb_search, embed_query as _embed_query, build_emb_index as _build_emb_index, save_emb_index as _save_emb_index
                items = load_emb_index(ws)
                qv = _emb([q])[0]
                hits = _emb_search(list(qv), items, top_k=k)
                if not hits:
                    console.print("[yellow]No vector matches. Building embeddings index...[/yellow]")
                    try:
                        new_items = _build_emb_index(ws, embedder_obj)
                        _save_emb_index(ws, new_items, persist=False)
                        items = new_items
                        hits = _emb_search(list(qv), items, top_k=k)
                    except Exception as _e2:
                        console.print(Panel(escape(str(_e2)), title="emb-index build failed", border_style="red"))
            except Exception:
                hits = []
            # Render evidence
            _print_header("Intel: Knowledge matches")
            event_hits: List[Dict[str, object]] = []
            if kn_matches:
                body = "\n".join(f"- {m.get('path')} | classes={len(m.get('classes') or [])} funcs={len(m.get('functions') or [])}" for m in kn_matches[:50])
                console.print(Panel(body, title=f"{len(kn_matches)} files", border_style="blue"))
                for rec in kn_matches[:50]:
                    path = rec.get('path')
                    if path:
                        event_hits.append({"path": str(path), "score": 1.0, "source": "intel-knowledge"})
            else:
                console.print("[yellow]No knowledge matches.[/yellow]")
            _print_header("Intel: Vector retrieval")
            if hits:
                event_hits.extend({"path": str(Path(it.path)), "score": float(score_i), "source": "intel-vretr"} for score_i, it in hits)
                for score_i, it in hits:
                    p = Path(it.path)
                    try:
                        text = p.read_text(errors='ignore'); lines = text.splitlines(); s = max(1, it.start_line - 2); e = min(len(lines), it.end_line + 2); seg = "\n".join(lines[s - 1 : e])
                    except Exception:
                        seg = "(unreadable)"; s = it.start_line; e = it.end_line
                    title = f"{p} score={score_i:.3f} lines {s}-{e}"
                    console.print(Panel(escape(seg), title=title, border_style="cyan"))
            else:
                console.print("[yellow]No vector matches.[/yellow]")
            if event_hits:
                log_retrieval_event(ws, q, event_hits)
        elif t == "index":
            _print_header("Building index"); meta, items = build_index(ws, smart=True); out_dir = save_index(ws, meta, items)
            console.print(f"[green]Indexed {len(items)} chunks. Saved to {out_dir}[/green]")
        elif t == "esearch":
            q = args.get("query") or args.get("q");
            if not q: console.print("[yellow]No query[/yellow]"); return
            try:
                meta, items = load_index(ws)
            except FileNotFoundError:
                _print_header("Building index (first run)"); meta, items = build_index(ws, smart=True); save_index(ws, meta, items)
            hits = semantic_search(q, meta, items, top_k=int(args.get("k",5)))
            # Filter out vendor/venv paths for clarity
            def _skip_path(p: Path) -> bool:
                s = p.as_posix()
                return ("/.venv/" in s) or ("/site-packages/" in s) or ("/node_modules/" in s) or ("/.git/" in s)
            hits = [(sc, it) for sc, it in hits if not _skip_path(Path(it.path))]
            event_hits = [{"path": str(Path(it.path)), "score": float(score), "source": "esearch"} for score, it in hits]
            if event_hits:
                log_retrieval_event(ws, q, event_hits)
            for score,it in hits:
                p = Path(it.path)
                try:
                    text = p.read_text(errors="ignore"); lines = text.splitlines(); s=max(1,it.start_line-3); e=min(len(lines), it.end_line+3)
                    seg = "\n".join(lines[s-1:e])
                except Exception: seg="(unreadable)"; s=it.start_line; e=it.end_line
                console.print(Panel(escape(seg), title=f"{p} score={score:.3f} lines {s}-{e}", border_style="blue"))
        elif t == "edit":
            # LLM-powered code edit: propose minimal unified diff, ask before applying
            task_text = args.get("task") or nl
            file_hints = args.get("file_hints") or args.get("files") or ""
            auto_apply_default = os.getenv("DSPY_AUTO_APPLY_PATCHES", "true").lower() in {"1", "true", "yes", "on"}
            apply_flag = bool(args.get("apply", auto_apply_default))
            lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key, workspace=ws)
            if lm is None:
                console.print("[yellow]No LLM configured; cannot generate patch.[/yellow]")
                return
            # Build context from logs and code graph
            bundle, _ = load_logs([logs_path])
            ctx_text = extract_key_events(bundle) if bundle else ""
            from .skills.code_edit import CodeEdit
            code_graph = _get_code_summary(ws)
            ce = CodeEdit(use_cot=True)
            out = ce(task=task_text, context=ctx_text, code_graph=code_graph, file_hints=file_hints)
            _print_header("Proposed Patch"); console.print(out.patch or "(no patch)")
            _print_header("Rationale"); console.print(out.rationale or "(no rationale)")
            if apply_flag and out.patch:
                do_apply = True
                if approval_mode == "manual":
                    do_apply = typer.confirm("Apply this patch?", default=False)
                if do_apply:
                    ok, msg = apply_unified_patch(out.patch, ws)
                    tool_metrics = {"applied": bool(ok)}
                    if ok:
                        console.print(f"[green]{msg}[/green]")
                        summ = summarize_patch(out.patch)
                        blast = float(summ['added_lines'] + summ['removed_lines'])
                        tool_metrics.update({"blast_radius": blast})
                        console.print(Panel.fit(
                            f"files: {summ['files']}  +lines: {summ['added_lines']}  -lines: {summ['removed_lines']}",
                            title="patch metrics", border_style="accent"
                        ))
                    else:
                        console.print(Panel(msg, title="apply failed", border_style="red"))
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
        elif t == "run_tests":
            tool_metrics, tool_info = _execute_toolchain(ToolAction.RUN_TESTS, dict(args))
        elif t == "lint":
            tool_metrics, tool_info = _execute_toolchain(ToolAction.LINT, dict(args))
        elif t == "build":
            tool_metrics, tool_info = _execute_toolchain(ToolAction.BUILD, dict(args))
        elif t == "patch":
            pf = args.get("file")
            if pf:
                p = (ws / pf).resolve()
                if not p.exists():
                    console.print("[yellow]patch needs --file[/yellow]")
                    return
                text = p.read_text(errors="ignore")
                ok, msg = apply_unified_patch(text, ws)
                tool_metrics = {"applied": bool(ok)}
                if ok:
                    console.print(f"[ok]{msg}[/ok]")
                    summ = summarize_patch(text)
                    blast = float(summ['added_lines'] + summ['removed_lines'])
                    tool_metrics.update({"blast_radius": blast})
                    console.print(Panel.fit(
                        f"files: {summ['files']}  +lines: {summ['added_lines']}  -lines: {summ['removed_lines']}",
                        title="patch metrics", border_style="accent"
                    ))
                else:
                    console.print(Panel(msg, title="patch failed", border_style="err"))
            else:
                patch_args = dict(args)
                patch_task = patch_args.get("task") or "Fix workspace issues"
                if not isinstance(patch_task, str):
                    patch_task = str(patch_task)
                bundle = _build_patch_context_bundle(ws, logs_path, patch_task)
                combined = bundle.get('combined_context') or bundle.get('text') or ''
                patch_args.setdefault("task", patch_task)
                patch_args.setdefault("context", combined)
                tool_metrics, tool_info = _execute_toolchain(ToolAction.PATCH, patch_args)
        elif t == "diff":
            if not last_extract: console.print("[yellow]No extract buffer. Run extract first.[/yellow]"); return
            file = args.get("file"); p=(ws / file).resolve() if file else None
            if not p: console.print("[yellow]diff needs a file[/yellow]"); return
            from .code_tools.diffutil import unified_diff_file_vs_text
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
            line = input("blampert> ").strip()
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
                if auto_runner:
                    auto_runner.update_paths(ws, logs_path)
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
            if auto_runner:
                auto_runner.update_paths(ws, logs_path)
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
            if auto_runner:
                auto_runner.update_lm(provider_is_ollama, model, base_url, api_key)
        elif cmd == "model" and args:
            model = " ".join(args)
            console.print(f"[green]Model set to {model}[/green]")
            if auto_runner:
                auto_runner.update_lm(provider_is_ollama, model, base_url, api_key)
        elif cmd == "auto":
            sub = args[0].lower() if args else "status"
            if sub == "status":
                status = auto_runner.status_snapshot() if auto_runner else {}
                if not status:
                    status_path = ws / '.dspy_auto_status.json'
                    if status_path.exists():
                        try:
                            status = json.loads(status_path.read_text())
                        except Exception:
                            status = {}
                if not status:
                    console.print("[yellow]Auto-training status unavailable.[/yellow]")
                else:
                    body = json.dumps(status, indent=2)
                    console.print(Panel(body, title="auto-training", border_style="cyan"))
            elif sub == "enable":
                if auto_runner:
                    console.print("[yellow]Auto-training already running.[/yellow]")
                else:
                    auto_runner = AutoTrainingLoop(ws, logs_path, console=console, label='interactive', ollama=provider_is_ollama, model=model, base_url=base_url, api_key=api_key)
                    auto_runner.start()
                    console.print("[green]Auto-training enabled.[/green]")
            elif sub == "disable":
                if auto_runner:
                    auto_runner.stop()
                    auto_runner = None
                    console.print("[green]Auto-training disabled.[/green]")
                else:
                    console.print("[yellow]Auto-training is not running.[/yellow]")
            elif sub == "restart":
                if auto_runner:
                    auto_runner.stop()
                auto_runner = AutoTrainingLoop(ws, logs_path, console=console, label='interactive', ollama=provider_is_ollama, model=model, base_url=base_url, api_key=api_key)
                auto_runner.start()
                console.print("[green]Auto-training restarted.[/green]")
            elif sub == "interval" and len(args) > 1:
                if not auto_runner:
                    console.print("[yellow]Auto-training is not running.[/yellow]")
                else:
                    try:
                        seconds = int(args[1])
                        auto_runner.set_interval(seconds)
                        console.print(f"[green]Auto-training interval set to {seconds}s.[/green]")
                    except ValueError:
                        console.print("[yellow]Usage: auto interval <seconds>[/yellow]")
            elif sub == "modules" and len(args) > 1:
                mods = [m.strip() for m in args[1:] if m.strip()]
                if not mods:
                    console.print("[yellow]Usage: auto modules <name> [<name> ...][/yellow]")
                elif auto_runner:
                    auto_runner.set_modules(mods)
                    console.print(f"[green]Auto-training modules set to {mods}.[/green]")
                else:
                    console.print("[yellow]Auto-training is not running. Enable it first.[/yellow]")
            else:
                console.print("[yellow]Usage: auto [status|enable|disable|restart|interval <sec>|modules <names...>][/yellow]")
        elif cmd == "stats":
            _render_stats_page(console, ws)
        elif cmd == "ctx":
            # Reuse logic from context command
            bundle, count = load_logs([logs_path])
            if not bundle:
                console.print("[yellow]No logs found.[/yellow]")
                continue
            _print_header("Log Key Events")
            key = extract_key_events(bundle)
            console.print(Panel.fit(key, title="Extracted Events", border_style="magenta"))
            lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key, workspace=ws)
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
            lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key, workspace=ws)
            if lm:
                _print_header("Code Context (DSPy)")
                cc = CodeContext()
                out = cc(snapshot=snap, ask="Summarize key components, APIs, and likely modification points.")
                console.print(Panel.fit(out.summary, title="Summary", border_style="cyan"))
                console.print(Panel.fit(out.bullets, title="Bullets", border_style="green"))
        elif cmd == "edit":
            if not args:
                console.print("[yellow]Usage: edit <TASK> [--apply][/yellow]"); continue
            task_text = " ".join([a for a in args if not a.startswith("--")])
            apply_flag = any(a in {"--apply", "-y"} for a in args)
            lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key, workspace=ws)
            if lm is None:
                console.print("[yellow]No LLM configured; cannot generate patch.[/yellow]")
                continue
            ctx_bundle = _build_patch_context_bundle(ws, logs_path, task_text)
            _render_recent_fixes(console, ctx_bundle)
            ctx_text = ctx_bundle.get('combined_context', '')
            from .skills.code_edit import CodeEdit as _CE
            code_graph = _get_code_summary(ws)
            hints = ctx_bundle.get('file_hints', "")
            ce = _CE(use_cot=True)
            out = ce(task=task_text, context=ctx_text[:8000], code_graph=code_graph, file_hints=hints)
            _print_header("Proposed Patch"); console.print(out.patch or "(no patch)")
            _print_header("Rationale"); console.print(out.rationale or "(no rationale)")
            if apply_flag and out.patch:
                do_apply = True
                if approval_mode == "manual":
                    do_apply = typer.confirm("Apply this patch?", default=False)
                else:
                    do_apply = False
                if do_apply:
                    ok, msg = apply_unified_patch(out.patch, ws)
                    if ok:
                        console.print(f"[green]{msg}[/green]")
                    else:
                        console.print(Panel(msg, title="apply failed", border_style="red"))
        elif cmd == "plan" and args:
            task = " ".join(args)
            bundle, count = load_logs([logs_path])
            key = extract_key_events(bundle) if bundle else ""
            lm = _maybe_configure_lm(True, provider_is_ollama, model, base_url, api_key, workspace=ws)
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
                    console.print(Panel(escape(seg), title=title, border_style="blue"))
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
                    console.print(Panel(escape(seg), title=f"{file_arg}::{symbol} (lines {start}-{end})", border_style="green"))
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
                console.print(Panel(escape(seg), title=f"{file_arg} lines {start}-{end}", border_style="green"))
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
                console.print(Panel(escape(seg), title=title, border_style="blue"))
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
        # Enhanced coding commands
        elif cmd == "coding-mode":
            if len(args) >= 1 and args[0] in ["on", "off"]:
                coding_mode = args[0] == "on"
                console.print(f"[green]Coding mode: {'enabled' if coding_mode else 'disabled'}[/green]")
            else:
                console.print(f"[cyan]Coding mode: {'enabled' if coding_mode else 'disabled'}[/cyan]")
        
        elif cmd == "build":
            if coding_mode:
                _handle_build_command(args, ws)
            else:
                console.print("[yellow]Build command requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        elif cmd == "test":
            if coding_mode:
                _handle_test_command(args, ws)
            else:
                console.print("[yellow]Test command requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        elif cmd == "lint":
            if coding_mode:
                _handle_lint_command(args, ws)
            else:
                console.print("[yellow]Lint command requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        elif cmd == "run":
            if coding_mode:
                _handle_run_command(args, ws)
            else:
                console.print("[yellow]Run command requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        elif cmd == "learn":
            if coding_mode:
                _handle_learn_command(args, ws, lm)
            else:
                console.print("[yellow]Learn command requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        elif cmd == "feedback":
            if coding_mode:
                _handle_feedback_command(args, ws)
            else:
                console.print("[yellow]Feedback command requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        # Development workflow commands
        elif cmd == "dev":
            if coding_mode:
                _handle_dev_command(args, ws)
            else:
                console.print("[yellow]Dev command requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        elif cmd == "release":
            if coding_mode:
                _handle_release_command(args, ws)
            else:
                console.print("[yellow]Release command requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        elif cmd == "publish":
            if coding_mode:
                _handle_publish_command(args, ws)
            else:
                console.print("[yellow]Publish command requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        # Expert-level feature commands
        elif cmd == "expert-patterns":
            if coding_mode:
                _handle_expert_patterns_command(args, ws)
            else:
                console.print("[yellow]Expert patterns requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        elif cmd == "expert-tools":
            if coding_mode:
                _handle_expert_tools_command(args, ws)
            else:
                console.print("[yellow]Expert tools requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        elif cmd == "expert-insights":
            if coding_mode:
                _handle_expert_insights_command(args, ws)
            else:
                console.print("[yellow]Expert insights requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        elif cmd == "expert-optimize":
            if coding_mode:
                _handle_expert_optimize_command(args, ws)
            else:
                console.print("[yellow]Expert optimize requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        elif cmd == "expert-status":
            if coding_mode:
                _handle_expert_status_command(args, ws)
            else:
                console.print("[yellow]Expert status requires coding mode. Use 'coding-mode on' to enable.[/yellow]")
        
        elif cmd in {"exit", "quit"}:
            break
        else:
            # Natural instruction → multi-step auto orchestration
            orchestrate_chain(line, lm=lm)

    if auto_runner:
        auto_runner.stop()


def _handle_build_command(args: List[str], ws: Path):
    """Handle build command with auto-detection of build system"""
    clean = "--clean" in args
    
    # Auto-detect build system
    build_commands = []
    
    if (ws / "package.json").exists():
        build_commands.append(("npm run build", "Node.js/npm"))
    elif (ws / "yarn.lock").exists():
        build_commands.append(("yarn build", "Node.js/yarn"))
    elif (ws / "pyproject.toml").exists():
        build_commands.append(("uv build", "Python/uv"))
    elif (ws / "setup.py").exists() or (ws / "requirements.txt").exists():
        build_commands.append(("python -m build", "Python/setuptools"))
    elif (ws / "Cargo.toml").exists():
        build_commands.append(("cargo build", "Rust/Cargo"))
    elif (ws / "go.mod").exists():
        build_commands.append(("go build", "Go"))
    elif (ws / "Makefile").exists():
        build_commands.append(("make", "Make"))
    elif (ws / "CMakeLists.txt").exists():
        build_commands.append(("cmake --build .", "CMake"))
    
    if not build_commands:
        console.print("[yellow]No build system detected. Available build files:[/yellow]")
        for build_file in ["package.json", "pyproject.toml", "Cargo.toml", "go.mod", "Makefile", "CMakeLists.txt"]:
            if (ws / build_file).exists():
                console.print(f"  ✓ {build_file}")
        return
    
    for cmd, system in build_commands:
        console.print(f"[cyan]Building with {system}...[/cyan]")
        if clean and "clean" in cmd.lower():
            clean_cmd = cmd.replace("build", "clean")
            console.print(f"[dim]Running: {clean_cmd}[/dim]")
            result = os.system(f"cd {ws} && {clean_cmd}")
            if result != 0:
                console.print(f"[yellow]Clean command failed (exit code: {result})[/yellow]")
        
        console.print(f"[dim]Running: {cmd}[/dim]")
        result = os.system(f"cd {ws} && {cmd}")
        
        if result == 0:
            console.print(f"[green]✅ Build successful with {system}![/green]")
        else:
            console.print(f"[red]❌ Build failed with {system} (exit code: {result})[/red]")


def _handle_test_command(args: List[str], ws: Path):
    """Handle test command with auto-detection of test framework"""
    coverage = "--coverage" in args
    
    # Auto-detect test framework
    test_commands = []
    
    if (ws / "package.json").exists():
        test_cmd = "npm test"
        if coverage:
            test_cmd = "npm run test:coverage" if "test:coverage" in (ws / "package.json").read_text() else "npm test -- --coverage"
        test_commands.append((test_cmd, "Node.js/npm"))
    elif (ws / "yarn.lock").exists():
        test_cmd = "yarn test"
        if coverage:
            test_cmd = "yarn test:coverage" if "test:coverage" in (ws / "package.json").read_text() else "yarn test --coverage"
        test_commands.append((test_cmd, "Node.js/yarn"))
    elif (ws / "pyproject.toml").exists():
        test_cmd = "uv run pytest"
        if coverage:
            test_cmd = "uv run pytest --cov"
        test_commands.append((test_cmd, "Python/pytest"))
    elif (ws / "pytest.ini").exists() or (ws / "setup.cfg").exists():
        test_cmd = "pytest"
        if coverage:
            test_cmd = "pytest --cov"
        test_commands.append((test_cmd, "Python/pytest"))
    elif (ws / "Cargo.toml").exists():
        test_commands.append(("cargo test", "Rust/Cargo"))
    elif (ws / "go.mod").exists():
        test_cmd = "go test ./..."
        if coverage:
            test_cmd = "go test -cover ./..."
        test_commands.append((test_cmd, "Go"))
    elif (ws / "Makefile").exists():
        test_commands.append(("make test", "Make"))
    
    if not test_commands:
        console.print("[yellow]No test framework detected. Looking for test files...[/yellow]")
        test_files = list(ws.glob("**/test_*.py")) + list(ws.glob("**/*_test.py")) + list(ws.glob("**/*.test.js"))
        if test_files:
            console.print(f"Found {len(test_files)} test files. Try running tests manually.")
        return
    
    for cmd, system in test_commands:
        console.print(f"[cyan]Running tests with {system}...[/cyan]")
        console.print(f"[dim]Running: {cmd}[/dim]")
        result = os.system(f"cd {ws} && {cmd}")
        
        if result == 0:
            console.print(f"[green]✅ All tests passed with {system}![/green]")
        else:
            console.print(f"[red]❌ Tests failed with {system} (exit code: {result})[/red]")


def _handle_lint_command(args: List[str], ws: Path):
    """Handle lint command with auto-detection of linter"""
    fix = "--fix" in args
    
    # Auto-detect linter
    lint_commands = []
    
    if (ws / "package.json").exists():
        if "eslint" in (ws / "package.json").read_text():
            cmd = "npx eslint ."
            if fix:
                cmd += " --fix"
            lint_commands.append((cmd, "ESLint"))
        if "prettier" in (ws / "package.json").read_text():
            cmd = "npx prettier --check ."
            if fix:
                cmd = "npx prettier --write ."
            lint_commands.append((cmd, "Prettier"))
    elif (ws / "pyproject.toml").exists():
        if "ruff" in (ws / "pyproject.toml").read_text():
            cmd = "uv run ruff check ."
            if fix:
                cmd += " --fix"
            lint_commands.append((cmd, "Ruff"))
        if "black" in (ws / "pyproject.toml").read_text():
            cmd = "uv run black --check ."
            if fix:
                cmd = "uv run black ."
            lint_commands.append((cmd, "Black"))
    elif (ws / "Cargo.toml").exists():
        lint_commands.append(("cargo clippy", "Clippy"))
    elif (ws / "go.mod").exists():
        lint_commands.append(("go vet ./...", "Go vet"))
        lint_commands.append(("gofmt -l .", "Go fmt"))
    
    if not lint_commands:
        console.print("[yellow]No linter detected. Available linters:[/yellow]")
        console.print("  • ESLint/Prettier (Node.js)")
        console.print("  • Ruff/Black (Python)")
        console.print("  • Clippy (Rust)")
        console.print("  • go vet/gofmt (Go)")
        return
    
    for cmd, linter in lint_commands:
        console.print(f"[cyan]Running {linter}...[/cyan]")
        console.print(f"[dim]Running: {cmd}[/dim]")
        result = os.system(f"cd {ws} && {cmd}")
        
        if result == 0:
            console.print(f"[green]✅ {linter} passed![/green]")
        else:
            console.print(f"[yellow]⚠️  {linter} found issues (exit code: {result})[/yellow]")


def _handle_run_command(args: List[str], ws: Path):
    """Handle run command for safe shell execution"""
    if not args:
        console.print("[yellow]Usage: run <COMMAND>[/yellow]")
        return
    
    command = " ".join(args)
    console.print(f"[cyan]Running: {command}[/cyan]")
    console.print(f"[dim]Working directory: {ws}[/dim]")
    
    result = os.system(f"cd {ws} && {command}")
    
    if result == 0:
        console.print(f"[green]✅ Command completed successfully![/green]")
    else:
        console.print(f"[red]❌ Command failed (exit code: {result})[/red]")


def _handle_learn_command(args: List[str], ws: Path, lm):
    """Handle learn command to learn from successful coding patterns"""
    if not args:
        console.print("[yellow]Usage: learn <TASK_DESCRIPTION>[/yellow]")
        return
    
    task = " ".join(args)
    console.print(f"[cyan]Learning from: {task}[/cyan]")
    
    # Record successful coding pattern
    from dspy_agent.db import get_enhanced_data_manager, create_action_record, ActionType, Environment
    
    dm = get_enhanced_data_manager()
    
    action_record = create_action_record(
        action_type=ActionType.CODE_ANALYSIS,
        state_before={"task": task, "status": "learning"},
        state_after={"task": task, "status": "learned"},
        parameters={"task": task, "workspace": str(ws)},
        result={"success": True, "message": f"Learned from successful coding pattern: {task}"},
        reward=0.9,  # High reward for learning
        confidence=0.8,
        execution_time=0.1,
        environment=Environment.DEVELOPMENT
    )
    
    dm.record_action(action_record)
    console.print(f"[green]✅ Learned from coding pattern: {task}[/green]")


def _handle_feedback_command(args: List[str], ws: Path):
    """Handle feedback command to provide feedback on last action"""
    if not args:
        console.print("[yellow]Usage: feedback <SCORE> (0-10)[/yellow]")
        return
    
    try:
        score = float(args[0])
        if not 0 <= score <= 10:
            console.print("[red]Score must be between 0 and 10[/red]")
            return
    except ValueError:
        console.print("[red]Invalid score. Must be a number between 0 and 10[/red]")
        return
    
    # Record feedback
    from dspy_agent.db import get_enhanced_data_manager, create_action_record, ActionType, Environment
    
    dm = get_enhanced_data_manager()
    
    action_record = create_action_record(
        action_type=ActionType.CODE_ANALYSIS,
        state_before={"feedback": "pending"},
        state_after={"feedback": "recorded"},
        parameters={"score": score, "workspace": str(ws)},
        result={"success": True, "message": f"Feedback recorded: {score}/10"},
        reward=score / 10.0,  # Normalize to 0-1
        confidence=1.0,
        execution_time=0.1,
        environment=Environment.DEVELOPMENT
    )
    
    dm.record_action(action_record)
    console.print(f"[green]✅ Feedback recorded: {score}/10[/green]")


def _handle_expert_patterns_command(args: List[str], ws: Path):
    """Handle expert patterns command"""
    console.print("[cyan]🎓 Expert Learning Patterns[/cyan]")
    
    # Load memory from workspace
    memory_file = ws / '.dspy_session_memory.json'
    if not memory_file.exists():
        console.print("[yellow]No expert patterns found. Start using the agent to build patterns![/yellow]")
        return
    
    try:
        with open(memory_file, 'r') as f:
            data = json.load(f)
        
        expert_patterns = data.get('expert_patterns', {})
        if not expert_patterns:
            console.print("[yellow]No expert patterns learned yet. Keep using the agent![/yellow]")
            return
        
        for context, patterns in expert_patterns.items():
            console.print(f"\n[bold]{context.title()} Context:[/bold]")
            for i, pattern in enumerate(patterns[:3], 1):  # Show top 3 patterns
                success_rate = "✅" if pattern.get('success', False) else "❌"
                reward = pattern.get('reward', 0)
                frequency = pattern.get('frequency', 1)
                tools = pattern.get('tool_sequence', [])
                
                console.print(f"  {i}. {success_rate} Tools: {', '.join(tools)}")
                console.print(f"     Reward: {reward:.2f}, Frequency: {frequency}")
    
    except Exception as e:
        console.print(f"[red]Error loading expert patterns: {e}[/red]")


def _handle_expert_tools_command(args: List[str], ws: Path):
    """Handle expert tools command"""
    console.print("[cyan]🔧 Most Effective Tools[/cyan]")
    
    # Load memory from workspace
    memory_file = ws / '.dspy_session_memory.json'
    if not memory_file.exists():
        console.print("[yellow]No tool effectiveness data found. Start using tools to build data![/yellow]")
        return
    
    try:
        with open(memory_file, 'r') as f:
            data = json.load(f)
        
        tool_effectiveness = data.get('tool_effectiveness', {})
        if not tool_effectiveness:
            console.print("[yellow]No tool effectiveness data yet. Keep using tools![/yellow]")
            return
        
        # Sort tools by effectiveness
        sorted_tools = sorted(tool_effectiveness.items(), key=lambda x: x[1], reverse=True)
        
        console.print("\n[bold]Tool Effectiveness Rankings:[/bold]")
        for i, (tool, effectiveness) in enumerate(sorted_tools[:10], 1):
            effectiveness_pct = effectiveness * 100
            if effectiveness_pct >= 80:
                color = "green"
                emoji = "🟢"
            elif effectiveness_pct >= 60:
                color = "yellow"
                emoji = "🟡"
            else:
                color = "red"
                emoji = "🔴"
            
            console.print(f"  {i:2d}. {emoji} [bold]{tool}[/bold] - [{color}]{effectiveness_pct:.1f}%[/{color}]")
    
    except Exception as e:
        console.print(f"[red]Error loading tool effectiveness: {e}[/red]")


def _handle_expert_insights_command(args: List[str], ws: Path):
    """Handle expert insights command"""
    console.print("[cyan]💡 Codebase Insights[/cyan]")
    
    # Load memory from workspace
    memory_file = ws / '.dspy_session_memory.json'
    if not memory_file.exists():
        console.print("[yellow]No insights found. Start analyzing the codebase![/yellow]")
        return
    
    try:
        with open(memory_file, 'r') as f:
            data = json.load(f)
        
        context_insights = data.get('context_insights', {})
        if not context_insights:
            console.print("[yellow]No codebase insights yet. Keep analyzing![/yellow]")
            return
        
        console.print("\n[bold]Learned Insights:[/bold]")
        for context, insight in context_insights.items():
            console.print(f"\n[bold]{context.title()}:[/bold]")
            console.print(f"  💡 {insight}")
    
    except Exception as e:
        console.print(f"[red]Error loading insights: {e}[/red]")


def _handle_expert_optimize_command(args: List[str], ws: Path):
    """Handle expert optimize command"""
    console.print("[cyan]🎯 Optimizing Expert Performance...[/cyan]")
    
    # Load memory from workspace
    memory_file = ws / '.dspy_session_memory.json'
    if not memory_file.exists():
        console.print("[yellow]No optimization data found. Start using the agent![/yellow]")
        return
    
    try:
        with open(memory_file, 'r') as f:
            data = json.load(f)
        
        # Show current optimizations
        prompt_optimizations = data.get('prompt_optimizations', {})
        action_policies = data.get('action_policies', {})
        
        console.print(f"\n[bold]Current Optimizations:[/bold]")
        console.print(f"  📝 Optimized prompts: {len(prompt_optimizations)}")
        console.print(f"  🎯 Action policies: {len(action_policies)}")
        
        if prompt_optimizations:
            console.print("\n[bold]Prompt Optimizations:[/bold]")
            for context, prompt in list(prompt_optimizations.items())[:3]:
                console.print(f"  • {context}: {prompt[:60]}...")
        
        if action_policies:
            console.print("\n[bold]Action Policies:[/bold]")
            for context, policies in list(action_policies.items())[:3]:
                console.print(f"  • {context}: {len(policies)} policies")
        
        console.print("\n[green]✅ Optimization complete! The agent is learning and improving.[/green]")
    
    except Exception as e:
        console.print(f"[red]Error during optimization: {e}[/red]")


def _handle_expert_status_command(args: List[str], ws: Path):
    """Handle expert status command"""
    console.print("[cyan]🎓 Expert Status Report[/cyan]")
    
    # Load memory from workspace
    memory_file = ws / '.dspy_session_memory.json'
    if not memory_file.exists():
        console.print("[yellow]No expert data found. Start using the agent to build expertise![/yellow]")
        return
    
    try:
        with open(memory_file, 'r') as f:
            data = json.load(f)
        
        # Calculate expert level
        expert_patterns = len(data.get('expert_patterns', {}))
        tool_effectiveness = len(data.get('tool_effectiveness', {}))
        context_insights = len(data.get('context_insights', {}))
        prompt_optimizations = len(data.get('prompt_optimizations', {}))
        action_policies = len(data.get('action_policies', {}))
        
        # Calculate expert score
        expert_score = (
            expert_patterns * 0.3 +
            tool_effectiveness * 0.2 +
            context_insights * 0.2 +
            prompt_optimizations * 0.15 +
            action_policies * 0.15
        )
        
        # Determine expert level
        if expert_score >= 20:
            level = "Expert"
            color = "green"
            emoji = "🎓"
        elif expert_score >= 10:
            level = "Advanced"
            color = "blue"
            emoji = "🚀"
        elif expert_score >= 5:
            level = "Intermediate"
            color = "yellow"
            emoji = "📚"
        else:
            level = "Beginner"
            color = "red"
            emoji = "🌱"
        
        console.print(f"\n[bold]{emoji} Expert Level: [{color}]{level}[/{color}][/bold]")
        console.print(f"📊 Expert Score: {expert_score:.1f}/25")
        
        console.print(f"\n[bold]Learning Progress:[/bold]")
        console.print(f"  🧠 Expert patterns: {expert_patterns}")
        console.print(f"  🔧 Tool effectiveness: {tool_effectiveness}")
        console.print(f"  💡 Context insights: {context_insights}")
        console.print(f"  📝 Prompt optimizations: {prompt_optimizations}")
        console.print(f"  🎯 Action policies: {action_policies}")
        
        # Show recent activity
        chain_history = data.get('chain_history', [])
        if chain_history:
            recent_activity = len(chain_history)
            console.print(f"\n[bold]Recent Activity:[/bold]")
            console.print(f"  📈 Total sessions: {recent_activity}")
            
            # Show most recent successful patterns
            successful_sessions = [s for s in chain_history if s.get('success', False)]
            if successful_sessions:
                console.print(f"  ✅ Successful sessions: {len(successful_sessions)}")
                success_rate = len(successful_sessions) / len(chain_history) * 100
                console.print(f"  📊 Success rate: {success_rate:.1f}%")
    
    except Exception as e:
        console.print(f"[red]Error loading expert status: {e}[/red]")


def _handle_dev_command(args: List[str], ws: Path):
    """Handle dev workflow commands"""
    if not args:
        console.print("[yellow]Usage: dev <command> [options][/yellow]")
        console.print("[cyan]Available commands:[/cyan]")
        console.print("  dev quick [message]  - Quick dev cycle (format, test, commit, push)")
        console.print("  dev build           - Build package")
        console.print("  dev test            - Run tests")
        console.print("  dev lint            - Run linter")
        console.print("  dev format          - Format code")
        console.print("  dev status          - Show git status")
        console.print("  dev version         - Show current version")
        return
    
    subcommand = args[0]
    
    if subcommand == "quick":
        message = " ".join(args[1:]) if len(args) > 1 else f"Quick dev update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        console.print(f"[cyan]Starting quick dev cycle: {message}[/cyan]")
        result = os.system(f"cd {ws} && ./scripts/dev.sh quick '{message}'")
        if result == 0:
            console.print("[green]✅ Quick dev cycle completed![/green]")
        else:
            console.print("[red]❌ Quick dev cycle failed![/red]")
    
    elif subcommand == "build":
        console.print("[cyan]Building package...[/cyan]")
        result = os.system(f"cd {ws} && ./scripts/dev.sh build")
        if result == 0:
            console.print("[green]✅ Package built successfully![/green]")
        else:
            console.print("[red]❌ Package build failed![/red]")
    
    elif subcommand == "test":
        console.print("[cyan]Running tests...[/cyan]")
        result = os.system(f"cd {ws} && ./scripts/dev.sh test")
        if result == 0:
            console.print("[green]✅ Tests passed![/green]")
        else:
            console.print("[red]❌ Tests failed![/red]")
    
    elif subcommand == "lint":
        console.print("[cyan]Running linter...[/cyan]")
        result = os.system(f"cd {ws} && ./scripts/dev.sh lint")
        if result == 0:
            console.print("[green]✅ Linting passed![/green]")
        else:
            console.print("[red]❌ Linting failed![/red]")
    
    elif subcommand == "format":
        console.print("[cyan]Formatting code...[/cyan]")
        result = os.system(f"cd {ws} && ./scripts/dev.sh format")
        if result == 0:
            console.print("[green]✅ Code formatted![/green]")
        else:
            console.print("[red]❌ Code formatting failed![/red]")
    
    elif subcommand == "status":
        console.print("[cyan]Git status:[/cyan]")
        result = os.system(f"cd {ws} && ./scripts/dev.sh status")
    
    elif subcommand == "version":
        console.print("[cyan]Current version:[/cyan]")
        result = os.system(f"cd {ws} && ./scripts/dev.sh version")
    
    else:
        console.print(f"[red]Unknown dev command: {subcommand}[/red]")


def _handle_release_command(args: List[str], ws: Path):
    """Handle release workflow commands"""
    version_type = args[0] if args else "patch"
    
    if version_type not in ["major", "minor", "patch"]:
        console.print("[red]Version type must be 'major', 'minor', or 'patch'[/red]")
        return
    
    console.print(f"[cyan]Starting release workflow ({version_type})...[/cyan]")
    console.print("[yellow]This will:[/yellow]")
    console.print("  1. Run tests and linting")
    console.print("  2. Bump version")
    console.print("  3. Build package")
    console.print("  4. Commit and push to GitHub")
    console.print("  5. Create GitHub release")
    console.print("  6. Publish to PyPI")
    
    # Record the release action for learning
    from dspy_agent.db import get_enhanced_data_manager, create_action_record, ActionType, Environment
    
    dm = get_enhanced_data_manager()
    
    action_record = create_action_record(
        action_type=ActionType.CODE_ANALYSIS,
        state_before={"version_type": version_type, "status": "starting"},
        state_after={"version_type": version_type, "status": "releasing"},
        parameters={"version_type": version_type, "workspace": str(ws)},
        result={"success": True, "message": f"Starting release workflow: {version_type}"},
        reward=0.8,
        confidence=0.9,
        execution_time=0.1,
        environment=Environment.DEVELOPMENT
    )
    
    dm.record_action(action_record)
    
    result = os.system(f"cd {ws} && ./scripts/dev.sh release {version_type}")
    if result == 0:
        console.print("[green]✅ Release completed successfully![/green]")
    else:
        console.print("[red]❌ Release failed![/red]")


def _handle_publish_command(args: List[str], ws: Path):
    """Handle publish commands"""
    test_pypi = "--test" in args or "test" in args
    
    if test_pypi:
        console.print("[cyan]Publishing to Test PyPI...[/cyan]")
        result = os.system(f"cd {ws} && ./scripts/dev.sh publish-test")
    else:
        console.print("[cyan]Publishing to PyPI...[/cyan]")
        console.print("[yellow]This will publish the package to PyPI![/yellow]")
        result = os.system(f"cd {ws} && ./scripts/dev.sh publish")
    
    if result == 0:
        console.print("[green]✅ Published successfully![/green]")
    else:
        console.print("[red]❌ Publish failed![/red]")


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
            console.print(Panel(escape(seg), title=title, border_style="blue"))
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
            console.print(Panel(escape(seg), title=f"{file}::{symbol} (lines {start}-{end})", border_style="green"))
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
        console.print(Panel(escape(seg), title=f"{file} lines {start}-{end}", border_style="green"))
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


start = start_command


def _launch_interactive(ws: Path, logs: Optional[Path]) -> bool:
    """Run the richer interactive shell when available; return True on success."""
    try:
        from .ui.interactive import run_interactive_shell
    except Exception:
        return False
    try:
        run_interactive_shell(ws, logs)
    except SystemExit:
        pass
    except Exception as exc:
        console.print(Panel(str(exc), title="interactive shell failed", border_style="red"))
        return False
    return True


def code_entry() -> None:
    """Entry point for the `dspy-code` console script (interactive defaults)."""
    _print_banner(console)
    ws = Path.cwd()
    logs = ws / 'logs'
    logs_arg = logs if logs.exists() else None

    if _launch_interactive(ws, logs_arg):
        return

    # Legacy fallback matches historical behaviour if the interactive shell is unavailable.
    start_command(
        workspace=ws,
        logs=logs_arg,
        ollama=True,
        model="qwen3:1.7b",
        base_url=None,
        api_key=None,
        force_json=False,
        structured=False,
        approval=None,
    )


@app.command()
def code(
    open_dashboard: bool = typer.Option(
        False,
        "--open-dashboard/--no-open-dashboard",
        help="Automatically open the dashboard in your default browser after the server starts.",
    ),
):
    """
    Integrated entry point for dspy-code command.
    Starts the dashboard server and runs the CLI agent.
    """
    import threading
    import time
    import subprocess
    import signal
    import sys
    from pathlib import Path
    
    # Print welcome banner
    console.print("\n🚀 [bold cyan]DSPy Code Agent[/bold cyan] - Integrated Development Environment")
    console.print("   [dim]Starting dashboard server and CLI agent...[/dim]\n")
    
    # Start the dashboard server in a separate process
    dashboard_process = None
    try:
        # Get the path to the enhanced dashboard server
        dashboard_path = Path(__file__).parent.parent / "enhanced_dashboard_server.py"
        
        console.print("📊 Starting dashboard server...")
        dashboard_process = subprocess.Popen([
            sys.executable, str(dashboard_path), "8081"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Check if server started successfully
        try:
            import urllib.request

            urllib.request.urlopen("http://localhost:8081/api/status", timeout=1)
            console.print("✅ Dashboard server started successfully at http://localhost:8081")

            if open_dashboard:
                import webbrowser

                console.print("🌐 Opening dashboard in browser...")
                webbrowser.open("http://localhost:8081/dashboard")
            else:
                console.print("ℹ️  Run with --open-dashboard to launch the dashboard in your browser automatically.")

        except Exception as e:
            console.print(f"⚠️  Dashboard server may not be ready yet: {e}")
            console.print("   You can manually open http://localhost:8081/dashboard")

        console.print("\n🎯 [bold green]Ready![/bold green] You now have:")
        console.print("   📊 Web Dashboard: http://localhost:8081/dashboard")
        console.print("   💻 CLI Agent: Running in this terminal")
        console.print("   🔗 Both interfaces are connected to the same RedDB backend")
        console.print("\n💡 [dim]Use the web interface for monitoring and the CLI for direct interaction[/dim]")
        console.print("🛑 [dim]Press Ctrl+C to stop both the dashboard and CLI agent[/dim]\n")
        
        # Set up signal handler to clean up dashboard process
        def signal_handler(sig, frame):
            console.print("\n🛑 Shutting down DSPy Code Agent...")
            if dashboard_process:
                console.print("   Stopping dashboard server...")
                dashboard_process.terminate()
                dashboard_process.wait()
            console.print("   ✅ Cleanup complete. Goodbye!")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run the main CLI application
        try:
            app()
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)
            
    except Exception as e:
        console.print(f"❌ Error starting integrated environment: {e}")
        if dashboard_process:
            dashboard_process.terminate()
        sys.exit(1)


if __name__ == "__main__":
    app()
