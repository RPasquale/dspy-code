from __future__ import annotations

"""
Cleaner GRPO Tool-Policy Trainer (HuggingFace)

Reads datasets/grpo_tool_batches/manifest.json (Parquet shards) with columns:
  - aggregate_tokens (List[int])
  - advantage (float)

Uses a HuggingFace tokenizer/model for next-token prediction and optimizes the
advantage-weighted policy surrogate: mean(advantage * CE(logits, labels)).

For large models, launch with accelerate/deepspeed/torchrun outside this script.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as e:  # pragma: no cover
    raise RuntimeError("transformers is required for train_grpo_tool_hf.py") from e


@dataclass
class HFTrainConfig:
    manifest: Path
    model_name: str
    epochs: int = 1
    batch_size: int = 4
    max_len: int = 1024
    lr: float = 1e-5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    bf16: bool = True
    gradient_checkpointing: bool = True


class HFToolDataset(Dataset):
    def __init__(self, manifest: Path, tokenizer: AutoTokenizer, max_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        rows: List[Tuple[List[int], float]] = []
        items = json.loads(Path(manifest).read_text())
        import pyarrow.parquet as pq  # type: ignore
        for row in items:
            tbl = pq.read_table(row['path'])
            toks = tbl.column('aggregate_tokens').to_pylist()
            adv = tbl.column('advantage').to_pylist()
            for ts, a in zip(toks, adv):
                if not isinstance(ts, list) or len(ts) < 2:
                    continue
                rows.append((ts, float(a)))
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        tokens, adv = self.rows[idx]
        ids = tokens[: self.max_len]
        attn = [1] * len(ids)
        if len(ids) < self.max_len:
            pad = self.max_len - len(ids)
            ids = ids + [self.tokenizer.pad_token_id] * pad
            attn = attn + [0] * pad
        labels = ids[1:] + [self.tokenizer.pad_token_id]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(adv, dtype=torch.float),
        )


def train(cfg: HFTrainConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or '<|pad|>'
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    if cfg.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    model.to(cfg.device)
    if cfg.bf16 and cfg.device.startswith('cuda'):
        model = model.to(dtype=torch.bfloat16)

    ds = HFToolDataset(cfg.manifest, tokenizer, max_len=cfg.max_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    total_steps = cfg.epochs * math.ceil(len(ds) / cfg.batch_size)
    step = 0
    model.train()
    for ep in range(cfg.epochs):
        for batch in dl:
            input_ids, attn_mask, labels, adv = [b.to(cfg.device) for b in batch]
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = out.logits
            vocab = logits.size(-1)
            loss_tok = loss_fn(logits.view(-1, vocab), labels.view(-1))
            loss_tok = loss_tok.view(input_ids.size(0), input_ids.size(1))
            mask = attn_mask.float()
            tok_mean = (loss_tok * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            loss = (adv * tok_mean).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 10 == 0:
                print(f"ep={ep} step={step}/{total_steps} loss={loss.item():.4f}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--model', required=True, help='HF model id (e.g., mistralai/Mistral-7B-Instruct)')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-len', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--no-bf16', action='store_true')
    ap.add_argument('--no-grad-checkpoint', action='store_true')
    args = ap.parse_args()
    cfg = HFTrainConfig(
        manifest=Path(args.manifest),
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        lr=args.lr,
        bf16=not args.no_bf16,
        gradient_checkpointing=not args.no_grad_checkpoint,
    )
    train(cfg)

