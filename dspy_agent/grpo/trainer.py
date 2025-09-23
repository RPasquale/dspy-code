from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time
import json
import math

try:
    import torch
except Exception:  # pragma: no cover - runtime guard
    torch = None  # type: ignore

from .dataset import GroupPreferenceDataset, GroupSample
from .model import PolicyModel, HFPolicy, BowPolicy, PolicyConfig


@dataclass
class GRPOConfig:
    dataset_path: Path
    model_name: Optional[str] = None
    reference_model_name: Optional[str] = None
    device: Optional[str] = None
    batch_groups: int = 8
    lr: float = 1e-5
    max_steps: int = 1000
    log_interval: int = 20
    ckpt_interval: int = 200
    out_dir: Path = Path('.grpo')
    adv_clip: float = 5.0
    kl_coeff: float = 0.02
    seed: int = 42
    # Optional training improvements
    lr_step_size: Optional[int] = None  # decay LR every N steps
    lr_gamma: Optional[float] = None    # multiplicative LR decay
    kl_warmup_steps: Optional[int] = None  # linearly anneal KL from 0→kl_coeff over N steps
    kl_target: Optional[float] = None   # final KL coeff (defaults to kl_coeff)


@dataclass
class GRPOMetrics:
    step: int
    loss: float
    kl: float
    adv_mean: float
    adv_std: float
    lr: float
    timestamp: float


class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        if torch is None:
            raise RuntimeError("PyTorch not available; install torch to use GRPO")
        self.cfg = cfg
        self.rng = torch.Generator()
        self.rng.manual_seed(cfg.seed)

        self.ds = GroupPreferenceDataset(cfg.dataset_path)
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Model + reference
        pol_cfg = PolicyConfig(model_name=cfg.model_name, device=self.device)
        ref_cfg = PolicyConfig(model_name=cfg.reference_model_name or cfg.model_name, device=self.device)
        self.policy = self._make_policy(pol_cfg)
        self.reference = self._make_policy(ref_cfg)
        for p in self.reference.parameters():
            try:
                p.requires_grad_(False)
            except Exception:
                pass

        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=cfg.lr)
        self.scheduler = None
        if cfg.lr_step_size and cfg.lr_gamma:
            try:
                from torch.optim.lr_scheduler import StepLR
                self.scheduler = StepLR(self.opt, step_size=int(cfg.lr_step_size), gamma=float(cfg.lr_gamma))
            except Exception:
                self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.startswith('cuda')))

        self.out_dir = cfg.out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / 'metrics.jsonl'
        self._metrics_f = self.metrics_path.open('a', encoding='utf-8')

        self.step = 0

    def _make_policy(self, cfg: PolicyConfig) -> PolicyModel:
        # Attempt HF first, fallback to BoW
        try:
            return HFPolicy(cfg)
        except Exception:
            return BowPolicy(cfg)

    def _sample_batch(self, n_groups: int) -> List[GroupSample]:
        idxs = torch.randint(0, len(self.ds), (n_groups,), generator=self.rng).tolist()
        return [self.ds[i] for i in idxs]

    def _compute_kl(self, prompts: List[str], responses: List[str]):
        # KL approx via difference in log-prob on selected sequences
        with torch.no_grad():
            lp_ref = self.reference.log_probs(prompts, responses)
        lp_pol = self.policy.log_probs(prompts, responses)
        # D_KL(p||q) ~ E_x [ log p(x) - log q(x) ], here swap roles for penalty
        kl = (lp_pol - lp_ref).mean()
        return kl, lp_pol

    def _grpo_loss(self, groups: List[GroupSample]) -> Tuple[Any, Dict[str, float]]:
        import torch
        prompts: List[str] = []
        responses: List[str] = []
        rewards: List[float] = []
        group_slices: List[Tuple[int, int]] = []

        # Flatten groups
        cur = 0
        for g in groups:
            k = len(g.candidates)
            sl = (cur, cur + k)
            group_slices.append(sl)
            for c in g.candidates:
                prompts.append(g.prompt)
                responses.append(c.text)
                rewards.append(float(c.reward))
            cur += k

        # Compute log-probs and KL
        kl, lp_pol = self._compute_kl(prompts, responses)

        # Compute group-wise standardized advantages
        adv = torch.zeros_like(lp_pol)
        r = torch.tensor(rewards, dtype=lp_pol.dtype, device=lp_pol.device)
        for (a, b) in group_slices:
            r_g = r[a:b]
            mu = r_g.mean()
            sd = r_g.std(unbiased=False).clamp_min(1e-6)
            adv[a:b] = (r_g - mu) / sd
        adv = adv.clamp(-self.cfg.adv_clip, self.cfg.adv_clip)

        # Policy loss: negative weighted log-likelihood
        loss_pi = -(adv.detach() * lp_pol).mean()
        # KL coefficient (optionally annealed)
        kl_coeff = float(self.cfg.kl_coeff)
        if self.cfg.kl_warmup_steps and self.cfg.kl_warmup_steps > 0:
            t = min(1.0, float(max(0, self.step)) / float(self.cfg.kl_warmup_steps))
            target = float(self.cfg.kl_target if self.cfg.kl_target is not None else self.cfg.kl_coeff)
            kl_coeff = t * target
        loss = loss_pi + kl_coeff * kl
        stats = {
            'kl': float(kl.detach().cpu().item()),
            'adv_mean': float(adv.detach().cpu().mean().item()),
            'adv_std': float(adv.detach().cpu().std(unbiased=False).item()),
        }
        return loss, stats

    def train(self, *, max_steps: Optional[int] = None, stop_flag: Optional[List[bool]] = None, on_metrics: Optional[Any] = None) -> None:
        import torch
        self.policy.train()
        self.reference.eval()
        steps = int(max_steps or self.cfg.max_steps)
        last_log = time.time()

        for s in range(steps):
            if stop_flag is not None and stop_flag and stop_flag[0]:
                break

            groups = self._sample_batch(self.cfg.batch_groups)
            self.opt.zero_grad(set_to_none=True)
            loss, stats = self._grpo_loss(groups)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    pass
            self.step += 1

            if (self.step % self.cfg.log_interval) == 0 or s == steps - 1:
                m = GRPOMetrics(
                    step=self.step,
                    loss=float(loss.detach().cpu().item()),
                    kl=float(stats['kl']),
                    adv_mean=float(stats['adv_mean']),
                    adv_std=float(stats['adv_std']),
                    lr=float(self.opt.param_groups[0]['lr']),
                    timestamp=time.time(),
                )
                self._emit_metrics(m)
                if on_metrics:
                    try:
                        on_metrics(m)
                    except Exception:
                        pass

            if (self.step % self.cfg.ckpt_interval) == 0:
                self._save_checkpoint()

        # Final checkpoint
        self._save_checkpoint()

    def _emit_metrics(self, m: GRPOMetrics) -> None:
        rec = asdict(m)
        self._metrics_f.write(json.dumps(rec) + "\n")
        self._metrics_f.flush()

    def _save_checkpoint(self) -> None:
        try:
            # Save minimal state dict
            path = self.out_dir / 'checkpoints' / f"policy_step{self.step}.pt"
            state = { 'step': self.step }
            # Try fetching model state dict when available
            try:
                import torch
                # Heuristic: policy may be HF backed or linear
                mod = getattr(self.policy, 'model', None)
                lin = getattr(self.policy, 'linear', None)
                if mod is not None and hasattr(mod, 'state_dict'):
                    state['model'] = mod.state_dict()
                elif lin is not None and hasattr(lin, 'state_dict'):
                    state['model'] = lin.state_dict()
            except Exception:
                pass
            import torch
            torch.save(state, path)
        except Exception:
            pass
