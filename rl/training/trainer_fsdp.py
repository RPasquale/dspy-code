from __future__ import annotations

import json
import os
from dataclasses import dataclass


# NOTE: Define as simple class to ensure import via
# importlib.util.spec_from_file_location works reliably in some environments.
class Config:
    def __init__(
        self,
        data_root: str = "/data/train",
        export_dir: str = "/models/export",
        batch_size: int = 4,
        epochs: int = 1,
        lr: float = 2e-4,
    ) -> None:
        self.data_root = data_root
        self.export_dir = export_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def run_training(cfg: Config) -> None:
    if not _has_torch():
        print("[trainer] torch not available; dry-run only")
        return
    import torch
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
        from torch.distributed.fsdp.wrap import always_wrap_policy  # type: ignore
    except Exception:
        FSDP = None  # type: ignore

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Minimal model
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16))
    if torch.cuda.is_available():
        model = model.cuda()

    if FSDP is not None and torch.cuda.is_available():
        model = FSDP(model, auto_wrap_policy=always_wrap_policy)
    else:
        model = DDP(model) if torch.cuda.is_available() else model

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs):
        x = torch.randn(cfg.batch_size, 16)
        if torch.cuda.is_available():
            x = x.cuda()
        y = model(x).sum()
        y.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    print("[trainer] completed")


def main() -> None:
    cfg_path = os.environ.get("CFG")
    cfg = Config()
    if cfg_path and os.path.exists(cfg_path):
        try:
            cfg = Config(**json.loads(open(cfg_path).read()))
        except Exception:
            pass
    run_training(cfg)


if __name__ == "__main__":
    main()
