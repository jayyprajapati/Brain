"""Local embedding model (fastembed / ONNX). Loaded once, reused everywhere."""
from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

from .config import settings

_model: TextEmbedding | None = None
_dim: int | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=settings.embed_model)
    return _model


def embed_documents(texts: list[str]) -> list[np.ndarray]:
    if not texts:
        return []
    return list(_get_model().embed(texts))


def embed_query(text: str) -> np.ndarray:
    # fastembed exposes query_embed for asymmetric models; falls back to embed.
    model = _get_model()
    query_embed = getattr(model, "query_embed", None)
    if callable(query_embed):
        return next(iter(query_embed(text)))
    return next(iter(model.embed([text])))


def vector_size() -> int:
    """Embed a probe string once to learn the model's output dimension."""
    global _dim
    if _dim is None:
        _dim = int(embed_query("probe").shape[0])
    return _dim


def warmup() -> None:
    """Force model download/load at startup so the first request isn't slow."""
    vector_size()
