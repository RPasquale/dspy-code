from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pq = None  # type: ignore

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments  # type: ignore
except Exception as e:
    raise RuntimeError("transformers is required for HF training")


@dataclass
class HFConfig:
    dataset_dir: Path
    model_id: str = os.getenv('CODELOG_MODEL', 'Salesforce/codet5p-220m')
    epochs: int = int(os.getenv('CODELOG_EPOCHS', '1'))
    batch_size: int = int(os.getenv('CODELOG_BATCH', '4'))
    lr: float = float(os.getenv('CODELOG_LR', '2e-4'))
    max_code: int = int(os.getenv('CODELOG_MAX_CODE', '1024'))
    max_log: int = int(os.getenv('CODELOG_MAX_LOG', '256'))
    out_dir: Path = Path(os.getenv('CODELOG_OUT', '/warehouse/models/code_log_hf'))


class CodeLogHFDataset(Dataset):
    def __init__(self, dataset_dir: Path, tok, max_code: int, max_log: int):
        if pq is None:
            raise RuntimeError("pyarrow is required to read parquet datasets")
        self.tok = tok
        self.max_code = max_code
        self.max_log = max_log
        files = list(dataset_dir.rglob('*.parquet'))
        items = []
        for f in files:
            try:
                tbl = pq.read_table(f)
                code = tbl.column('code_text').to_pylist()
                logt = tbl.column('log_text').to_pylist()
                for c, l in zip(code, logt):
                    if isinstance(c, str) and isinstance(l, str):
                        items.append((c, l))
            except Exception:
                continue
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        c, l = self.items[idx]
        enc = self.tok(
            c,
            truncation=True,
            max_length=self.max_code,
            return_tensors='pt'
        )
        with self.tok.as_target_tokenizer():
            lab = self.tok(
                l,
                truncation=True,
                max_length=self.max_log,
                return_tensors='pt'
            )
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item['labels'] = lab['input_ids'].squeeze(0)
        return item


def train(cfg: HFConfig) -> None:
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_id)
    ds = CodeLogHFDataset(cfg.dataset_dir, tok, cfg.max_code, cfg.max_log)
    if len(ds) == 0:
        print('no code-log samples; skip')
        return
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(cfg.out_dir),
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to=[]
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tok)
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--model', default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--max-code', type=int, default=None)
    ap.add_argument('--max-log', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    args = ap.parse_args()
    cfg = HFConfig(dataset_dir=Path(args.dataset_dir))
    if args.model: cfg.model_id = args.model
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.batch is not None: cfg.batch_size = args.batch
    if args.max_code is not None: cfg.max_code = args.max_code
    if args.max_log is not None: cfg.max_log = args.max_log
    if args.lr is not None: cfg.lr = args.lr
    train(cfg)

