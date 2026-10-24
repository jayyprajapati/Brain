"""Local cross-encoder reranker (fastembed / ONNX). Loaded once, reused everywhere."""
from __future__ import annotations

from fastembed.rerank.cross_encoder import TextCrossEncoder

from .config import settings

# fastembed's default batch is 64 passages. A multi-query chat reranks ~160
# candidates, so every batch is a full 64 x 512-token pass, and each attention
# layer briefly needs hundreds of MB per worker. On the shared 8 GB box that got
# brain-backend OOM-killed mid-chat. 16 keeps the peak small at no real speed cost.
RERANK_BATCH_SIZE = 16

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
    return list(_get_model().rerank(query, passages, batch_size=RERANK_BATCH_SIZE))


def warmup() -> None:
    _get_model()
