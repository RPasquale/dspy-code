from __future__ import annotations

import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch  # noqa: F401
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

from .trainer import GRPOConfig, GRPOTrainer, GRPOMetrics
from .auto import GrpoAutoOrchestrator


class GRPOService:
    """Background GRPO trainer service with metrics buffering and simple lifecycle."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = [False]
        self._status: Dict[str, Any] = {
            'running': False,
            'error': None,
            'step': 0,
            'started_at': None,
            'finished_at': None,
            'config': None,
        }
        self._metrics_lock = threading.Lock()
        self._recent: List[Dict[str, Any]] = []
        # Auto orchestrator
        self._auto = GrpoAutoOrchestrator()

    def start(self, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self._thread and self._thread.is_alive():
            return {'ok': False, 'error': 'GRPO already running'}
        if not _TORCH_OK:
            return {'ok': False, 'error': 'torch is not available in this environment'}
        try:
            cfg = GRPOConfig(
                dataset_path=Path(cfg_dict.get('dataset_path', 'grpo.jsonl')),
                model_name=cfg_dict.get('model_name') or None,
                reference_model_name=cfg_dict.get('reference_model_name') or None,
                device=cfg_dict.get('device') or None,
                batch_groups=int(cfg_dict.get('batch_groups', 8)),
                lr=float(cfg_dict.get('lr', 1e-5)),
                max_steps=int(cfg_dict.get('max_steps', 1000)),
                log_interval=int(cfg_dict.get('log_interval', 20)),
                ckpt_interval=int(cfg_dict.get('ckpt_interval', 200)),
                out_dir=Path(cfg_dict.get('out_dir', '.grpo')),
                adv_clip=float(cfg_dict.get('adv_clip', 5.0)),
                kl_coeff=float(cfg_dict.get('kl_coeff', 0.02)),
                seed=int(cfg_dict.get('seed', 42)),
            )
        except Exception as e:
            return {'ok': False, 'error': f'Invalid config: {e}'}

        self._stop_flag[0] = False
        self._status.update({'running': True, 'error': None, 'step': 0, 'started_at': time.time(), 'finished_at': None, 'config': cfg_dict})
        self._recent.clear()

        def on_metrics(m: GRPOMetrics):
            with self._metrics_lock:
                self._recent.append(asdict(m))
                # Keep bounded history in memory
                if len(self._recent) > 500:
                    self._recent = self._recent[-500:]
            self._status['step'] = m.step

        def run():
            try:
                trainer = GRPOTrainer(cfg)
                trainer.train(max_steps=cfg.max_steps, stop_flag=self._stop_flag, on_metrics=on_metrics)
            except Exception as e:
                self._status['error'] = str(e)
            finally:
                self._status['running'] = False
                self._status['finished_at'] = time.time()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return {'ok': True}

    def stop(self) -> Dict[str, Any]:
        if not self._thread:
            return {'ok': True}
        self._stop_flag[0] = True
        return {'ok': True}

    def status(self) -> Dict[str, Any]:
        out = dict(self._status)
        out['timestamp'] = time.time()
        try:
            auto = self._auto.status()
            out['auto'] = auto
        except Exception:
            out['auto'] = {'running': False}
        return out

    def metrics(self, limit: int = 200) -> Dict[str, Any]:
        with self._metrics_lock:
            data = list(self._recent[-limit:])
        return {
            'metrics': data,
            'count': len(data),
            'timestamp': time.time(),
        }

    # Auto controls ----------------------------------------------------------
    def auto_start(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return self._auto.start(cfg)

    def auto_stop(self) -> Dict[str, Any]:
        return self._auto.stop()


# Global singleton used by the HTTP server
GlobalGrpoService = GRPOService()
