"""Orchestration: ingest, retrieve, query-contextualization, and chat."""
from __future__ import annotations

from typing import AsyncIterator

import numpy as np

from . import embeddings, llm, reranker, vectorstore
from .chunking import chunk_text
from .config import settings
from .prompts import QUERY_REWRITE_PROMPT, build_chat_system
from .schemas import Message

# Payload keys Brain owns — caller metadata may never override these.
_RESERVED_KEYS = {"text", "heading", "urls", "chunk_index", "doc_id", "namespace"}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def ingest(
    app_name: str,
    doc_id: str,
    text: str,
    namespace: str | None = None,
    dedup: bool = False,
    metadata: dict | None = None,
) -> dict:
    """Replace a document: ensure the app's collection, wipe this doc's old chunks,
    then chunk → embed → (optionally dedup) → upsert.

    With ``dedup`` + ``namespace``, a candidate chunk is skipped when it is a near
    duplicate (cosine ≥ ``DEDUP_THRESHOLD``) of an already-stored chunk in the
    same namespace or of an earlier chunk accepted in this same batch — so
    overlapping resume versions don't pile up redundant vectors."""
    vectorstore.ensure_collection(app_name, embeddings.vector_size())
    vectorstore.delete(app_name, doc_id=doc_id)
    chunks = chunk_text(text)
    if not chunks:
        return {"chunk_count": 0, "skipped_duplicates": 0}

    vectors = embeddings.embed_documents([c.text for c in chunks])
    extra = {k: v for k, v in (metadata or {}).items() if k not in _RESERVED_KEYS}

    kept_vectors: list = []
    kept_payloads: list[dict] = []
    accepted: list[np.ndarray] = []  # in-batch dedup
    skipped = 0

    for chunk, vec in zip(chunks, vectors):
        if dedup and namespace:
            v = np.asarray(vec, dtype=float)
            if vectorstore.max_similarity(app_name, vec, namespace) >= settings.dedup_threshold:
                skipped += 1
                continue
            if any(_cosine(v, a) >= settings.dedup_threshold for a in accepted):
                skipped += 1
                continue
            accepted.append(v)
        kept_vectors.append(vec)
        kept_payloads.append(
            {**extra, "text": chunk.text, "heading": chunk.heading, "urls": chunk.urls, "chunk_index": chunk.chunk_index}
        )

    if kept_vectors:
        vectorstore.upsert(app_name, doc_id, kept_vectors, kept_payloads, namespace=namespace)
    return {"chunk_count": len(kept_vectors), "skipped_duplicates": skipped}


def _last_user_message(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content.strip():
            return m.content.strip()
    return ""


async def contextualize(messages: list[Message], model: str | None = None) -> str:
    """Rewrite the latest user message into a standalone search query.

    With no prior turns there's nothing to resolve, so we pass it through."""
    latest = _last_user_message(messages)
    prior = [m for m in messages if m.content.strip()][:-1]
    if not latest or not prior:
        return latest
    history = "\n".join(f"{m.role}: {m.content}" for m in prior[-6:])
    user = f"Conversation so far:\n{history}\n\nLatest message: {latest}"
    try:
        rewritten = (await llm.generate(QUERY_REWRITE_PROMPT, user, model=model)).strip()
        return rewritten or latest
    except Exception:
        # Never let query rewriting break a chat — fall back to the raw message.
        return latest


def retrieve(
    app_name: str,
    search_query: str,
    doc_ids: list[str] | None,
    namespace: str | None = None,
    top_k: int | None = None,
) -> list:
    """Embed → vector search (top-K) → cross-encoder rerank → top-N."""
    if not search_query:
        return []
    query_vec = embeddings.embed_query(search_query)
    limit = top_k or settings.retrieve_top_k
    hits = vectorstore.search(app_name, query_vec, doc_ids, limit, namespace=namespace)
    if not hits:
        return []
    scores = reranker.rerank(search_query, [h.text for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
    return [hit for hit, _ in ranked[: settings.rerank_top_n]]


async def chat(
    app_name: str,
    messages: list[Message],
    client_prompt: str,
    doc_ids: list[str] | None,
    model: str | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """Yield (event, data) tuples: token* → sources → done."""
    search_query = await contextualize(messages, model=model)
    chunks = retrieve(app_name, search_query, doc_ids)
    system = build_chat_system(client_prompt, chunks)
    turns = [{"role": m.role, "content": m.content} for m in messages if m.content.strip()]

    async for delta in llm.chat_stream(system, turns, model=model):
        yield "token", {"text": delta}

    sources = [
        {"doc_id": c.doc_id, "heading": c.heading, "score": c.score, "urls": c.urls}
        for c in chunks
    ]
    yield "sources", {"sources": sources}
    yield "done", {"finished": True}
