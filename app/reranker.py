"""Local cross-encoder reranker (fastembed / ONNX). Loaded once, reused everywhere."""
from __future__ import annotations

from fastembed.rerank.cross_encoder import TextCrossEncoder

from .config import settings

_model: TextCrossEncoder | None = None


def _get_model() -> TextCrossEncoder:
    global _model
    if _model is None:
        _model = TextCrossEncoder(model_name=settings.rerank_model)
    return _model


def rerank(query: str, passages: list[str]) -> list[float]:
    """Return a relevance score per passage (higher = more relevant)."""
    if not passages:
        return []
    return list(_get_model().rerank(query, passages))


def warmup() -> None:
    _get_model()
