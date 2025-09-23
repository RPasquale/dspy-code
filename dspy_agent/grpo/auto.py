from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .trainer import GRPOConfig, GRPOTrainer
from .mining import mine_reddb_to_grpo, mine_reddb_hierarchical
from .policy_nudges import compute_policy_nudges
from ..policy import update_policy_with_feedback
try:
    from ..system.metrics import guard_resources
except Exception:
    guard_resources = None  # type: ignore


class GrpoAutoOrchestrator:
    """Automates dataset mining + GRPO training on a cadence.

    Modes:
      - single: mine a single dataset and train repeatedly
      - hierarchical: mine multiple level-specific datasets and iterate
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self.state: Dict[str, Any] = {
            'running': False,
            'mode': 'single',
            'iterations': 0,
            'error': None,
            'last_cycle': None,
            'last_datasets': {},
            'config': {},
            'policy_updates': [],
        }

    def start(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        if self._thread and self._thread.is_alive():
            return {'ok': False, 'error': 'auto mode already running'}
        self._stop = False
        self.state.update({'running': True, 'error': None, 'config': dict(cfg or {})})

        def run():
            try:
                self._loop(cfg)
            except Exception as e:
                self.state['error'] = str(e)
            finally:
                self.state['running'] = False

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return {'ok': True}

    def stop(self) -> Dict[str, Any]:
        self._stop = True
        return {'ok': True}

    def status(self) -> Dict[str, Any]:
        s = dict(self.state)
        s['timestamp'] = time.time()
        return s

    # Internal loop ----------------------------------------------------------
    def _loop(self, cfg: Dict[str, Any]) -> None:
        period = float(cfg.get('period_sec', 300))
        hours = int(cfg.get('hours', 24))
        min_groups = int(cfg.get('min_groups', 20))
        mode = str(cfg.get('mode', 'single')).strip()
        steps = int(cfg.get('steps', 400))
        out_dir = Path(cfg.get('out_dir', '.grpo/auto'))
        out_dir.mkdir(parents=True, exist_ok=True)
        model_name = cfg.get('model_name') or None
        device = cfg.get('device') or None
        levels = cfg.get('levels') or ['global', 'signature', 'patch']
        apply_policy = bool(cfg.get('apply_policy', False))
        workspace = Path(cfg.get('workspace', Path.cwd()))
        top_k = int(cfg.get('policy_top_k', 3))
        bottom_k = int(cfg.get('policy_bottom_k', 1))
        min_avg_for_deny = float(cfg.get('policy_min_avg_for_deny', 0.05))

        min_free_gb = float(cfg.get('min_free_gb', 2.0))
        min_ram_gb = float(cfg.get('min_ram_gb', 1.0))
        min_vram_mb = float(cfg.get('min_vram_mb', 0.0))

        while not self._stop:
            self.state['last_cycle'] = time.time()
            # Resource guard: do not proceed if thresholds not met
            try:
                if guard_resources:
                    guard = guard_resources(workspace=workspace, min_free_gb=min_free_gb, min_ram_gb=min_ram_gb, min_vram_mb=min_vram_mb)
                else:
                    # Minimal guard: only disk
                    from shutil import disk_usage  # type: ignore
                    total, used, free = disk_usage(str(workspace))
                    free_gb = free / (1024 ** 3)
                    guard = {'ok': free_gb >= min_free_gb, 'disk_ok': free_gb >= min_free_gb, 'ram_ok': True, 'gpu_ok': True}
                self.state['guard'] = guard
                if not guard.get('ok', True):
                    self.state['error'] = 'insufficient resources; waiting for capacity'
                    # Sleep until next cycle; do not mine/train
                    t0 = time.time()
                    while not self._stop and time.time() - t0 < period:
                        time.sleep(1.0)
                    continue
            except Exception as e:
                self.state['error'] = f'guard: {e}'

            datasets: Dict[str, Path] = {}
            if mode == 'hierarchical':
                datasets = mine_reddb_hierarchical(out_dir, hours=hours)
            else:
                datasets = {'global': mine_reddb_to_grpo(out_dir / 'global.jsonl', hours=hours)}
            self.state['last_datasets'] = {k: str(v) for k, v in datasets.items()}

            # Train on each dataset if it meets min_groups
            for level in levels:
                p = datasets.get(level)
                if not p or not Path(p).exists():
                    continue
                try:
                    n = sum(1 for _ in Path(p).open('r', encoding='utf-8'))
                except Exception:
                    n = 0
                if n < min_groups:
                    continue
                cfg_tr = GRPOConfig(
                    dataset_path=p,
                    model_name=model_name,
                    reference_model_name=model_name,
                    device=device,
                    max_steps=steps,
                    out_dir=out_dir / f'ckpts_{level}',
                    lr_step_size=cfg.get('lr_step_size'),
                    lr_gamma=cfg.get('lr_gamma'),
                    kl_warmup_steps=cfg.get('kl_warmup_steps'),
                    kl_target=cfg.get('kl_target'),
                )
                try:
                    trainer = GRPOTrainer(cfg_tr)
                    trainer.train(max_steps=steps)
                except Exception as e:
                    # Continue to next level, record error
                    self.state['error'] = f"{level}: {e}"
            self.state['iterations'] = int(self.state.get('iterations', 0) or 0) + 1
            # Apply policy nudges when requested
            if apply_policy:
                try:
                    nudges = compute_policy_nudges(hours=hours, top_k=top_k, bottom_k=bottom_k, min_avg_for_deny=min_avg_for_deny)
                    updates = []
                    for it in nudges.get('prefer', []):
                        update_policy_with_feedback(workspace, prefer_tool=it.get('prefer_tool'), regex=it.get('regex'))
                        updates.append({ 'prefer': it.get('prefer_tool'), 'regex': it.get('regex') })
                    for it in nudges.get('deny', []):
                        update_policy_with_feedback(workspace, deny_tool=it.get('deny_tool'), regex=it.get('regex'))
                        updates.append({ 'deny': it.get('deny_tool'), 'regex': it.get('regex') })
                    self.state['policy_updates'] = updates[-20:]
                except Exception as e:
                    self.state['error'] = f"policy: {e}"
            # Sleep until next cycle
            t0 = time.time()
            while not self._stop and time.time() - t0 < period:
                time.sleep(1.0)
