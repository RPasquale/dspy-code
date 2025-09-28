#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

_LOG_FORMAT = os.getenv("DSPY_EMBED_LOG_FORMAT", "%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dspy-embedder")
logging.basicConfig(level=os.getenv("DSPY_EMBED_LOG_LEVEL", "INFO"), format=_LOG_FORMAT)


def _log_event(level: int, event: str, *, exc_info: Any = None, **fields: Any) -> None:
    payload = {"event": event, **fields}
    try:
        message = json.dumps(payload, sort_keys=True, default=str)
    except TypeError:
        message = f"{event} | " + " ".join(f"{k}={v!r}" for k, v in sorted(fields.items()))
    logger.log(level, message, exc_info=exc_info)

try:
    import dspy
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )
except ImportError as exc:  # pragma: no cover - should not happen in container
    raise RuntimeError("DSPy must be installed for the embedder service") from exc

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore

try:
    from fastembed import TextEmbedding  # type: ignore
except ImportError:
    TextEmbedding = None  # type: ignore

try:
    from dspy import Example  # type: ignore
except Exception:  # pragma: no cover
    Example = None  # type: ignore

HAS_GEPA = hasattr(dspy, "GEPA") and Example is not None


class EmbedRequest(BaseModel):
    model: Optional[str] = None
    inputs: List[str]


# ---------------------------------------------------------------------------
# DSPy signature + GEPA helpers
# ---------------------------------------------------------------------------


class DefaultEmbeddingSignature(dspy.Signature):  # type: ignore[misc]
    """Signature used to normalize text prior to embedding."""

    raw_text = dspy.InputField(desc="Raw text provided by the client")  # type: ignore
    normalized_text = dspy.OutputField(  # type: ignore
        desc="Normalized text string suitable for embedding"
    )


def _resolve_signature() -> type:
    sig_path = os.getenv("DSPY_EMBED_SIGNATURE")
    if not sig_path:
        return DefaultEmbeddingSignature
    module_name, _, class_name = sig_path.rpartition(":")
    if not module_name:
        raise RuntimeError(
            "DSPY_EMBED_SIGNATURE must be in the form 'module:ClassName'"
        )
    module = __import__(module_name, fromlist=[class_name])
    signature = getattr(module, class_name)
    return signature


class EmbeddingNormalizer(dspy.Module):  # type: ignore[misc]
    def __init__(self, signature: type):
        super().__init__()
        self._predictor = dspy.ChainOfThought(signature)  # type: ignore[attr-defined]

    def forward(self, raw_text: str):  # type: ignore[override]
        return self._predictor(raw_text=raw_text)


def _configure_lm() -> bool:
    model = os.getenv("DSPY_LM_MODEL")
    if not model:
        _log_event(
            logging.INFO,
            "gepa_lm_skipped",
            reason="dspy_lm_model_missing",
        )
        return False

    provider = os.getenv("DSPY_LM_PROVIDER", "openai").lower()
    kwargs: Dict[str, Any] = {}
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required when provider=openai")
        kwargs["api_key"] = api_key
        api_base = os.getenv("OPENAI_BASE_URL")
        if api_base:
            kwargs["api_base"] = api_base
    else:
        _log_event(logging.INFO, "gepa_lm_provider", provider=provider)

    lm = dspy.LM(model, **kwargs)  # type: ignore[call-arg]
    dspy.configure(lm=lm)
    _log_event(logging.INFO, "gepa_lm_configured", model=model)
    return True


def _load_gepa_dataset(path: Path) -> List[Any]:
    if Example is None:  # pragma: no cover
        return []
    dataset: List[Any] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                _log_event(logging.WARNING, "gepa_dataset_invalid_json", path=str(path))
                continue
            raw_text = payload.get("text") or payload.get("raw_text") or payload.get("input")
            normalized = (
                payload.get("normalized_text")
                or payload.get("normalized")
                or payload.get("output")
                or payload.get("target")
            )
            if not raw_text or not normalized:
                continue
            ex = Example(raw_text=raw_text, normalized_text=normalized)  # type: ignore[call-arg]
            ex = ex.with_inputs("raw_text")  # type: ignore[attr-defined]
            dataset.append(ex)
    return dataset


def _gepa_metric():
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):  # type: ignore[override]
        expected = (getattr(gold, "normalized_text", "") or "").lower()
        actual = (getattr(pred, "normalized_text", "") or "").lower()
        if not expected:
            return {"score": 1.0, "feedback": "No reference provided."}
        if not actual:
            return {"score": 0.0, "feedback": "Empty output."}

        exp_tokens = set(expected.split())
        act_tokens = set(actual.split())
        overlap = exp_tokens & act_tokens
        precision = len(overlap) / max(1, len(act_tokens))
        recall = len(overlap) / max(1, len(exp_tokens))
        if precision + recall == 0:
            score = 0.0
        else:
            score = (2 * precision * recall) / (precision + recall)

        missing = exp_tokens - overlap
        feedback = f"precision={precision:.2f}, recall={recall:.2f}"
        if missing:
            feedback += " | missing: " + ", ".join(sorted(missing))
        return {"score": float(score), "feedback": feedback}

    return metric


def _run_gepa(signature: type) -> Optional[Any]:
    dataset_path = os.getenv("DSPY_GEPA_DATASET")
    if not dataset_path:
        _log_event(logging.INFO, "gepa_dataset_not_configured")
        return None

    if not HAS_GEPA:
        _log_event(logging.WARNING, "gepa_not_available")
        return None

    path = Path(dataset_path)
    if not path.exists():
        _log_event(logging.WARNING, "gepa_dataset_missing_path", path=str(path))
        return None

    try:
        lm_ready = _configure_lm()
    except Exception as exc:
        _log_event(logging.WARNING, "gepa_lm_configure_failed", error=repr(exc), exc_info=exc)
        lm_ready = False

    if not lm_ready:
        _log_event(logging.INFO, "gepa_skipped_lm_not_ready")
        return None

    dataset = _load_gepa_dataset(path)
    if not dataset:
        _log_event(logging.WARNING, "gepa_dataset_empty", path=str(path))
        return None

    auto_mode = os.getenv("DSPY_GEPA_AUTO", "light")
    log_dir = os.getenv("DSPY_GEPA_LOG_DIR")

    try:
        metric = _gepa_metric()
        gepa = dspy.GEPA(metric=metric, auto=auto_mode, log_dir=log_dir, track_stats=False)  # type: ignore[attr-defined]
        base_module = EmbeddingNormalizer(signature)
        _log_event(
            logging.INFO,
            "gepa_run",
            path=str(path),
            examples=len(dataset),
            auto=auto_mode,
            log_dir=log_dir,
        )
        optimized = gepa.compile(base_module, trainset=dataset, valset=dataset)
        return optimized
    except Exception as exc:  # pragma: no cover - GEPA internals
        _log_event(
            logging.WARNING,
            "gepa_optimization_failed",
            error=repr(exc),
            exc_info=exc,
        )
        return None


class TextPreprocessor:
    def __init__(self, module: Optional[Any]) -> None:
        self._module = module

    def __call__(self, text: str) -> str:
        if not text:
            return text
        cleaned = text.strip()
        if not self._module:
            return cleaned
        try:
            result = self._module(raw_text=cleaned)  # type: ignore[operator]
            normalized = getattr(result, "normalized_text", None) or getattr(result, "normalized", None)
            if isinstance(normalized, str) and normalized.strip():
                return normalized.strip()
        except Exception as exc:  # pragma: no cover - defensive
            _log_event(logging.DEBUG, "preprocessor_failure", error=repr(exc))
        return cleaned.lower()


# ---------------------------------------------------------------------------
# Embedding backend (Sentence Transformers / FastEmbed)
# ---------------------------------------------------------------------------


class EmbeddingBackend:
    def __init__(self, normalize: bool = False) -> None:
        self._kind: Optional[str] = None
        self._model_id: Optional[str] = None
        self._model: Any = None
        self._normalize = normalize

    def load(self, model_id: str) -> None:
        if self._model is not None and self._model_id == model_id:
            return

        errors: List[str] = []

        if SentenceTransformer is not None:
            try:
                _log_event(logging.INFO, "embed_backend_load", kind="sentence-transformers", model=model_id)
                self._model = SentenceTransformer(model_id)
                self._kind = "sentence-transformers"
                self._model_id = model_id
                return
            except Exception as exc:
                errors.append(f"sentence-transformers: {exc}")

        if TextEmbedding is not None:
            try:
                _log_event(logging.INFO, "embed_backend_load", kind="fastembed", model=model_id)
                self._model = TextEmbedding(model_name=model_id)
                self._kind = "fastembed"
                self._model_id = model_id
                return
            except Exception as exc:
                errors.append(f"fastembed: {exc}")

        raise RuntimeError(
            f"Failed to load embedding model '{model_id}'. Errors: {' | '.join(errors)}"
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            raise RuntimeError("Embedding model not loaded")

        if self._kind == "sentence-transformers":
            arr = self._model.encode(texts, normalize_embeddings=self._normalize)
            try:
                return [list(map(float, vec)) for vec in arr]
            except Exception:
                import numpy as _np
                return [list(map(float, _np.asarray(vec).tolist())) for vec in arr]

        if self._kind == "fastembed":
            vectors = []
            for vec in self._model.embed(texts):
                vectors.append(list(map(float, vec)))
            return vectors

        raise RuntimeError(f"Unknown embedding backend kind: {self._kind}")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="DSPy Embedder Service", version="1.0")

REGISTRY = CollectorRegistry()
EMBED_REQUESTS = Counter(
    "dspy_embed_requests_total",
    "Total embed requests handled by DSPy embedder service",
    ["status"],
    registry=REGISTRY,
)
EMBED_LATENCY = Histogram(
    "dspy_embed_request_latency_seconds",
    "Latency of embed requests",
    registry=REGISTRY,
)

_DEFAULT_MODEL = os.getenv("DSPY_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_NORMALIZE = os.getenv("DSPY_EMBED_NORMALIZE", "0").lower() in {"1", "true", "yes", "on"}
_EMBED_BACKEND = EmbeddingBackend(normalize=_NORMALIZE)
_SIGNATURE = _resolve_signature()
_PREPROCESSOR: TextPreprocessor


@app.on_event("startup")
def _startup() -> None:
    global _PREPROCESSOR
    gepa_module = _run_gepa(_SIGNATURE)
    _PREPROCESSOR = TextPreprocessor(gepa_module)

    try:
        _EMBED_BACKEND.load(_DEFAULT_MODEL)
    except Exception as exc:
        _log_event(
            logging.ERROR,
            "embed_default_model_failed",
            model=_DEFAULT_MODEL,
            error=repr(exc),
            exc_info=exc,
        )
        raise

    _log_event(
        logging.INFO,
        "embedder_ready",
        model=_DEFAULT_MODEL,
        normalize=_NORMALIZE,
        gepa_enabled=bool(gepa_module),
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": _DEFAULT_MODEL,
        "normalize": _NORMALIZE,
        "gepa_enabled": HAS_GEPA and bool(os.getenv("DSPY_GEPA_DATASET")),
        "timestamp": time.time(),
    }


@app.post("/embed")
def embed(request: EmbedRequest) -> Dict[str, Any]:
    status_label = "success"
    start_time = time.perf_counter()
    try:
        if not isinstance(request.inputs, list) or not all(isinstance(x, str) for x in request.inputs):
            status_label = "bad_request"
            _log_event(logging.WARNING, "embed_bad_request", reason="non_str_inputs")
            raise HTTPException(status_code=400, detail="inputs must be a list[str]")

        model_id = request.model or _DEFAULT_MODEL
        input_count = len(request.inputs)
        _log_event(logging.DEBUG, "embed_request", model=model_id, inputs=input_count)
        try:
            _EMBED_BACKEND.load(model_id)
        except Exception as exc:
            status_label = "model_load_error"
            _log_event(
                logging.ERROR,
                "embed_model_load_failed",
                model=model_id,
                error=repr(exc),
                exc_info=exc,
            )
            raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

        processed = [_PREPROCESSOR(text) for text in request.inputs]

        try:
            vectors = _EMBED_BACKEND.embed(processed)
        except Exception as exc:
            status_label = "embedding_error"
            _log_event(
                logging.ERROR,
                "embed_backend_failed",
                model=model_id,
                error=repr(exc),
                exc_info=exc,
            )
            raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc

        dimension = len(vectors[0]) if vectors and vectors[0] else 0
        duration = time.perf_counter() - start_time
        _log_event(
            logging.INFO,
            "embed_success",
            model=model_id,
            inputs=input_count,
            dimension=dimension,
            backend=getattr(_EMBED_BACKEND, "_kind", None),
            duration_seconds=round(duration, 6),
        )

        return {
            "model": model_id,
            "dimension": dimension,
            "vectors": vectors,
            "processed_inputs": processed,
        }
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive catch
        status_label = "unexpected_error"
        _log_event(logging.ERROR, "embed_unexpected_error", error=repr(exc), exc_info=exc)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc
    finally:
        EMBED_REQUESTS.labels(status=status_label).inc()
        EMBED_LATENCY.observe(time.perf_counter() - start_time)


@app.get("/metrics")
def metrics() -> Response:
    payload = generate_latest(REGISTRY)
    return Response(content=payload, media_type="text/plain; version=0.0.4")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("dspy_agent.embedding.dspy_embedder_service:app", host="0.0.0.0", port=8080)
