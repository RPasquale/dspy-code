from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pq = None  # type: ignore


@dataclass
class TrainCfg:
    dataset_dir: Path
    epochs: int = 1
    batch_size: int = 8
    max_code: int = 4000
    max_log: int = 256
    lr: float = 3e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class CodeLogDataset(Dataset):
    def __init__(self, dataset_dir: Path, max_code: int, max_log: int):
        if pq is None:
            raise RuntimeError("pyarrow is required to read parquet datasets")
        files = list(dataset_dir.rglob('*.parquet'))
        self.items: List[Tuple[str,str]] = []
        for f in files:
            try:
                tbl = pq.read_table(f)
                code = tbl.column('code_text').to_pylist()
                logt = tbl.column('log_text').to_pylist()
                for c, l in zip(code, logt):
                    if isinstance(c, str) and isinstance(l, str):
                        self.items.append((c[:max_code], l[:max_log]))
            except Exception:
                continue
        # Build a simple character vocabulary
        vocab = set()
        for c,l in self.items:
            for ch in (c + l):
                vocab.add(ch)
        self.itos = ['<pad>', '<bos>', '<eos>'] + sorted(vocab)
        self.stoi = {ch:i for i,ch in enumerate(self.itos)}
        self.max_code = max_code
        self.max_log = max_log

    def __len__(self) -> int:
        return len(self.items)

    def encode(self, s: str, max_len: int) -> List[int]:
        ids = [1] + [self.stoi.get(ch, 0) for ch in s][:max_len-2] + [2]
        if len(ids) < max_len:
            ids = ids + [0]*(max_len - len(ids))
        return ids

    def __getitem__(self, idx: int):
        c, l = self.items[idx]
        x = torch.tensor(self.encode(c, self.max_code), dtype=torch.long)
        y = torch.tensor(self.encode(l, self.max_log), dtype=torch.long)
        return x, y


class CodeToLog(nn.Module):
    def __init__(self, vocab_size: int, d: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.enc = nn.GRU(d, d, batch_first=True)
        self.dec = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, code_ids: torch.Tensor, log_ids: torch.Tensor):
        enc = self.emb(code_ids)
        _, h = self.enc(enc)
        dec_in = self.emb(log_ids[:, :-1])
        y, _ = self.dec(dec_in, h)
        logits = self.head(y)
        return logits


def train(cfg: TrainCfg) -> None:
    ds = CodeLogDataset(cfg.dataset_dir, cfg.max_code, cfg.max_log)
    if len(ds) == 0:
        print("no code-log samples found; skipping")
        return
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    model = CodeToLog(vocab_size=len(ds.itos), d=256).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(cfg.epochs):
        for i, (code_ids, log_ids) in enumerate(dl):
            code_ids = code_ids.to(cfg.device)
            log_ids = log_ids.to(cfg.device)
            logits = model(code_ids, log_ids)
            # Teacher forcing loss: predict tokens 1..T from 0..T-1
            vocab = logits.size(-1)
            tgt = log_ids[:, 1:]
            loss = loss_fn(logits.contiguous().view(-1, vocab), tgt.contiguous().view(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if i % 10 == 0:
                print(f"ep={ep} step={i} loss={loss.item():.4f}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-code', type=int, default=4000)
    ap.add_argument('--max-log', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    args = ap.parse_args()
    cfg = TrainCfg(dataset_dir=Path(args.dataset_dir), epochs=args.epochs, batch_size=args.batch_size, max_code=args.max_code, max_log=args.max_log, lr=args.lr)
    train(cfg)

