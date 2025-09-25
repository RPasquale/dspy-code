from __future__ import annotations

"""
Minimal GRPO tool-policy training stub.

Reads dataset shards written by streaming/topics/tool_batch_export.py:
  datasets/grpo_tool_batches/manifest.json

Each row has:
  - aggregate_tokens: List[int]
  - advantage: float

This stub pads/truncates to max_len, builds a tiny LM-like model, and trains a
policy by maximizing advantage-weighted log-prob on next-token prediction.
This is illustrative; plug in your real tokenizer/model for production.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class TrainConfig:
    manifest: Path
    epochs: int = 1
    batch_size: int = 16
    max_len: int = 512
    vocab_size: int = 32000  # demo placeholder
    lr: float = 1e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    bf16: bool = True


class ToolShardDataset(Dataset):
    def __init__(self, manifest: Path, max_len: int = 512):
        self.max_len = int(max_len)
        items: List[Tuple[List[int], float]] = []
        data = json.loads(Path(manifest).read_text())
        for row in data:
            p = Path(row['path'])
            import pyarrow.parquet as pq  # type: ignore
            tbl = pq.read_table(p)
            toks = tbl.column('aggregate_tokens').to_pylist()
            adv = tbl.column('advantage').to_pylist()
            for ts, a in zip(toks, adv):
                if not isinstance(ts, list) or len(ts) < 2:
                    continue
                items.append((ts, float(a)))
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        tokens, adv = self.items[idx]
        # Pad/truncate
        ids = tokens[: self.max_len]
        attn = [1] * len(ids)
        if len(ids) < self.max_len:
            pad = self.max_len - len(ids)
            ids = ids + [0] * pad
            attn = attn + [0] * pad
        # Next-token labels (shifted)
        labels = ids[1:] + [0]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(adv, dtype=torch.float),
        )


class TinyPolicy(nn.Module):
    def __init__(self, vocab_size: int = 32000, dim: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.lm = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        # Masking is ignored in this tiny stub; for production, apply properly.
        y, _ = self.lm(x)
        logits = self.head(y)
        return logits


def train(cfg: TrainConfig) -> None:
    ds = ToolShardDataset(cfg.manifest, max_len=cfg.max_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, drop_last=True)
    model = TinyPolicy(cfg.vocab_size, dim=256).to(cfg.device)
    if cfg.bf16 and cfg.device.startswith('cuda'):
        model = model.to(dtype=torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    total_steps = cfg.epochs * math.ceil(len(ds) / cfg.batch_size)
    step = 0
    for ep in range(cfg.epochs):
        model.train()
        for batch in dl:
            input_ids, attn_mask, labels, adv = [b.to(cfg.device) for b in batch]
            logits = model(input_ids, attn_mask)
            # Compute per-token CE
            vocab = logits.size(-1)
            loss_tok = loss_fn(logits.view(-1, vocab), labels.view(-1))
            loss_tok = loss_tok.view(input_ids.size(0), input_ids.size(1))
            # Mask out pads
            mask = attn_mask.float()
            tok_mean = (loss_tok * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            # Advantage-weighted policy gradient surrogate: -A * logp ≈ A * CE
            loss = (adv * tok_mean).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 10 == 0:
                print(f"ep={ep} step={step}/{total_steps} loss={loss.item():.4f} bs={cfg.batch_size}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, help='Path to datasets/grpo_tool_batches/manifest.json')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-len', type=int, default=512)
    ap.add_argument('--vocab-size', type=int, default=32000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--no-bf16', action='store_true')
    args = ap.parse_args()
    cfg = TrainConfig(
        manifest=Path(args.manifest),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        vocab_size=args.vocab_size,
        lr=args.lr,
        bf16=not args.no_bf16,
    )
    train(cfg)

