#!/usr/bin/env python3
from __future__ import annotations
import os
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="InferMesh (CPU) Compatible Server")


class EmbedRequest(BaseModel):
    model: Optional[str] = None
    inputs: List[str]


class _EmbedBackend:
    def __init__(self) -> None:
        self._kind = None
        self._model_id = None
        self._model = None

    def load(self, model_id: str) -> None:
        if self._model is not None and self._model_id == model_id:
            return
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=model_id)
            self._kind = 'fastembed'
            self._model_id = model_id
            return
        except Exception:
            pass
        # Fallback: sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_id)
            self._kind = 'st'
            self._model_id = model_id
            return
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_id}': {e}")

    def embed(self, texts: List[str], normalize: bool = False) -> List[List[float]]:
        if self._model is None:
            raise RuntimeError("model not loaded")
        if self._kind == 'fastembed':
            gen = self._model.embed(texts)
            return [list(map(float, v)) for v in gen]
        else:
            arr = self._model.encode(texts, normalize_embeddings=normalize)
            try:
                return [list(map(float, v)) for v in arr]
            except Exception:
                import numpy as _np
                return [list(map(float, _np.asarray(v).tolist())) for v in arr]


_BACKEND = _EmbedBackend()
_DEFAULT_MODEL = os.getenv('MODEL_ID', 'BAAI/bge-small-en-v1.5')
_NORMALIZE = os.getenv('EMBED_NORMALIZE', '0').lower() in ('1','true','yes','on')


@app.on_event('startup')
def _on_startup():
    _BACKEND.load(_DEFAULT_MODEL)


@app.get('/health')
def health():
    return {'status': 'ok', 'model': _DEFAULT_MODEL, 'ts': time.time()}


@app.post('/embed')
def embed(req: EmbedRequest):
    model_id = req.model or _DEFAULT_MODEL
    if model_id != _DEFAULT_MODEL:
        _BACKEND.load(model_id)
    if not isinstance(req.inputs, list) or not all(isinstance(x, str) for x in req.inputs):
        raise HTTPException(400, detail='inputs must be a list[str]')
    vecs = _BACKEND.embed(req.inputs, normalize=_NORMALIZE)
    return {'vectors': vecs}

