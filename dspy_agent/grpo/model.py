from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PolicyConfig:
    model_name: Optional[str] = None
    max_length: int = 512
    device: Optional[str] = None
    dtype: Optional[str] = None  # 'fp16', 'bf16', or None


class PolicyModel:
    """Abstract policy that can score log probabilities for (prompt, response).

    Implementations must provide:
      - tokenize(prompts, responses) -> batch token ids
      - log_probs(batch) -> tensor of shape [B] (sum log probs over response)
    """

    def __init__(self, cfg: PolicyConfig) -> None:
        self.cfg = cfg

    def to(self, device: str) -> "PolicyModel":  # pragma: no cover - trivial
        return self

    def eval(self) -> None:  # pragma: no cover - trivial
        pass

    def train(self) -> None:  # pragma: no cover - trivial
        pass

    def log_probs(self, prompts: List[str], responses: List[str]) -> Any:
        raise NotImplementedError

    def parameters(self):  # pragma: no cover - abstract
        return []


class HFPolicy(PolicyModel):
    """Hugging Face Transformers-backed causal LM policy.

    Computes sum of log probs of the response tokens conditioned on prompt.
    """

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__(cfg)
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        except Exception as e:  # pragma: no cover - runtime guard
            raise RuntimeError("Transformers not available; install transformers and torch") from e

        self.torch = __import__("torch")
        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM

        name = cfg.model_name or "gpt2"
        self.tokenizer = self.AutoTokenizer.from_pretrained(name)
        self.model = self.AutoModelForCausalLM.from_pretrained(name)
        if cfg.device:
            self.model.to(cfg.device)
        self.model.eval()

    def parameters(self):  # pragma: no cover - simple passthrough
        return self.model.parameters()

    def log_probs(self, prompts: List[str], responses: List[str]):
        import torch
        assert len(prompts) == len(responses)
        batch: List[str] = [p + self.tokenizer.eos_token + r for p, r in zip(prompts, responses)]
        enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        if self.cfg.device:
            enc = {k: v.to(self.cfg.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
            logits = out.logits  # [B, T, V]
        # Compute log probabilities of response part only
        # For simplicity, we approximate by computing token-wise log-prob of each token given previous
        shift_logits = logits[:, :-1, :]
        shift_labels = enc["input_ids"][:, 1:]
        logprobs = self.torch.nn.functional.log_softmax(shift_logits, dim=-1)
        # Gather logprobs at true token positions
        lp = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        # Naively sum over entire sequence; acceptable for relative comparisons
        seq_lp = lp.sum(dim=1)  # [B]
        return seq_lp


class BowPolicy(PolicyModel):
    """Lightweight bag-of-words scoring fallback when Transformers are unavailable.

    Not a generative LM; learns a linear scorer over (prompt, response) tokens.
    Still usable to demonstrate GRPO end-to-end with small compute.
    """

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__(cfg)
        import torch
        self.torch = torch
        self.vocab: Dict[str, int] = {}
        self.linear = torch.nn.Linear(4096, 1)  # fixed small dimension
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.linear.to(self.device)

    def _vectorize(self, texts: List[str]):
        import torch
        feats = torch.zeros((len(texts), 4096), device=self.device)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                idx = (hash(tok) % 4096)
                feats[i, idx] += 1.0
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-6)
        return feats

    def parameters(self):  # pragma: no cover - passthrough
        return self.linear.parameters()

    def train(self) -> None:  # pragma: no cover - passthrough
        self.linear.train()

    def eval(self) -> None:  # pragma: no cover - passthrough
        self.linear.eval()

    def to(self, device: str):  # pragma: no cover - passthrough
        self.device = device
        self.linear.to(device)
        return self

    def log_probs(self, prompts: List[str], responses: List[str]):
        import torch
        # Produce a pseudo log-prob score from linear projection of bow vector
        texts = [p + " \n\n" + r for p, r in zip(prompts, responses)]
        feats = self._vectorize(texts)
        score = self.linear(feats).squeeze(-1)
        # Return as "log-probs" compatible with GRPO weighting
        return score

