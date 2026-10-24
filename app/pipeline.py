"""Orchestration: ingest, retrieve, query-contextualization, and chat."""
from __future__ import annotations

import logging
import re
from typing import AsyncIterator

import numpy as np

from . import embeddings, llm, reranker, vectorstore
from .chunking import chunk_text
from .config import settings
from .prompts import QUERY_REWRITE_PROMPT, QUERY_SPLIT_PROMPT, build_chat_system, build_raw_system
from .schemas import Message

logger = logging.getLogger("brain")

# Payload keys Brain owns — caller metadata may never override these.
_RESERVED_KEYS = {"text", "heading", "urls", "chunk_index", "doc_id", "namespace", "title"}


def _context_header(title: str, heading: str) -> str:
    """"Doc title > section heading", the breadcrumb a chunk is embedded and
    reranked with. A bare chunk like "p50 636 ms" says nothing about which
    project it belongs to; the breadcrumb does."""
    parts = [p for p in (title.strip(), heading.strip()) if p]
    # Skip the heading when it just repeats the title (a doc's H1).
    if len(parts) == 2 and parts[1].lower() in parts[0].lower():
        parts = parts[:1]
    return " > ".join(parts)


def _with_context(title: str, heading: str, text: str) -> str:
    header = _context_header(title, heading)
    return f"{header}\n{text}" if header else text


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
    title: str | None = None,
) -> dict:
    """Replace a document: ensure the app's collection, wipe this doc's old chunks,
    then chunk → embed → (optionally dedup) → upsert.

    With ``dedup`` + ``namespace``, a candidate chunk is skipped when it is a near
    duplicate (cosine ≥ ``DEDUP_THRESHOLD``) of an already-stored chunk in the
    same namespace or of an earlier chunk accepted in this same batch — so
    overlapping resume versions don't pile up redundant vectors.

    With ``title``, every chunk is embedded as "title > heading" + text
    (contextual chunk headers) and the title is stored on the payload so
    retrieval can rerank and label it the same way. Without it, behaviour is
    unchanged."""
    vectorstore.ensure_collection(app_name, embeddings.vector_size())
    vectorstore.delete(app_name, doc_id=doc_id)
    chunks = chunk_text(text)
    if not chunks:
        return {"chunk_count": 0, "skipped_duplicates": 0}

    title = (title or "").strip()
    if title:
        vectors = embeddings.embed_documents([_with_context(title, c.heading, c.text) for c in chunks])
    else:
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
        payload = {**extra, "text": chunk.text, "heading": chunk.heading, "urls": chunk.urls, "chunk_index": chunk.chunk_index}
        if title:
            payload["title"] = title
        kept_payloads.append(payload)

    if kept_vectors:
        vectorstore.upsert(app_name, doc_id, kept_vectors, kept_payloads, namespace=namespace)
    return {"chunk_count": len(kept_vectors), "skipped_duplicates": skipped}


def _last_user_message(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content.strip():
            return m.content.strip()
    return ""


async def contextualize(
    messages: list[Message], model: str | None = None, query_context: str | None = None
) -> str:
    queries = await contextualize_many(messages, model=model, query_context=query_context, split=False)
    return queries[0] if queries else ""


async def contextualize_many(
    messages: list[Message], model: str | None = None, query_context: str | None = None, split: bool = True
) -> list[str]:
    """Rewrite the latest user message into a standalone search query.

    With no prior turns there's nothing to resolve, so we pass it through,
    unless the caller supplied ``query_context`` (e.g. "he/him means Jay
    Prajapati, the portfolio owner"): then even a first message is rewritten so
    its pronouns resolve to the right subject before retrieval."""
    latest = _last_user_message(messages)
    prior = [m for m in messages if m.content.strip()][:-1]
    if not latest or (not prior and not query_context):
        return [latest] if latest else []
    history = "\n".join(f"{m.role}: {m.content}" for m in prior[-6:]) or "(none)"
    user = f"Conversation so far:\n{history}\n\nLatest message: {latest}"
    if query_context:
        user = f"Context: {query_context.strip()}\n\n{user}"
    try:
        # The budget must cover hidden reasoning too: gpt-oss spends ~100-300
        # tokens thinking before it writes, so a 64-token cap returned an empty
        # string every time and retrieval silently ran on the raw follow-up.
        rewritten = (
            await llm.generate(
                QUERY_SPLIT_PROMPT if split else QUERY_REWRITE_PROMPT,
                user,
                model=settings.query_rewrite_model,
                max_tokens=768,
                temperature=0.0,
                think=settings.query_rewrite_think or None,
            )
        ).strip()
        # Keep only the queries themselves if the model added anything around them.
        lines = [_clean_query(line) for line in rewritten.splitlines()]
        queries = list(dict.fromkeys(q for q in lines if q))[: (MAX_SUBQUERIES if split else 1)]
        if not queries:
            logger.warning("query rewrite returned nothing; using the raw message")
        return queries or [latest]
    except Exception:
        # Never let query rewriting break a chat — fall back to the raw message.
        return [latest]


MAX_SUBQUERIES = 4


def _clean_query(line: str) -> str:
    q = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line)
    return q.strip().strip('"\'`').strip()[:300]


def retrieve(
    app_name: str,
    search_query: str,
    doc_ids: list[str] | None,
    namespace: str | None = None,
    top_k: int | None = None,
    top_n: int | None = None,
    min_score: float | None = None,
    score_margin: float | None = None,
    min_n: int = 0,
    max_per_doc: int | None = None,
) -> list:
    """Embed → vector search (top-K) → cross-encoder rerank → top-N.

    Dynamic K (opt-in): with ``min_score`` and/or ``score_margin`` set, the
    reranked list is cut where relevance falls off instead of always returning
    exactly N. A chunk is kept only if its cross-encoder score is at least
    ``min_score`` AND within ``score_margin`` of the best hit, so a pointed
    question gets one or two chunks, a broad one gets up to N, and an
    off-topic one gets none. ``min_n`` guarantees a floor regardless.

    ``max_per_doc`` (opt-in) caps how many chunks one document may contribute,
    so a question spanning two documents ("compare X and Y") isn't answered
    from eight chunks of X."""
    if not search_query:
        return []
    query_vec = embeddings.embed_query(search_query)
    keep = top_n or settings.rerank_top_n
    # Always over-fetch relative to what we keep so the reranker has room to work.
    limit = max(top_k or settings.retrieve_top_k, keep)
    hits = vectorstore.search(app_name, query_vec, doc_ids, limit, namespace=namespace)
    if not hits:
        return []
    try:
        scores = reranker.rerank(
            search_query, [_with_context(h.title, h.heading, h.text) if h.title else h.text for h in hits]
        )
        ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
        hits = []
        for hit, score in ranked:
            hit.rerank_score = float(score)
            hits.append(hit)
    except Exception:  # noqa: BLE001 — reranker unavailable (e.g. model not yet downloaded)
        # Fall back to vector-search order rather than failing the request; Qdrant
        # already returns hits by descending similarity. Dynamic K needs rerank
        # scores, so it is skipped and the fixed top-N applies.
        logger.warning("reranker unavailable; falling back to vector-search order", exc_info=True)
        return hits[:keep]

    if max_per_doc:
        per_doc: dict[str, int] = {}
        diverse = []
        for h in hits:
            if per_doc.get(h.doc_id, 0) < max_per_doc:
                per_doc[h.doc_id] = per_doc.get(h.doc_id, 0) + 1
                diverse.append(h)
        hits = diverse
    if min_score is None and score_margin is None:
        return hits[:keep]
    best = hits[0].rerank_score
    kept = [
        h for h in hits[:keep]
        if (min_score is None or h.rerank_score >= min_score)
        and (score_margin is None or h.rerank_score >= best - score_margin)
    ]
    if len(kept) < min_n:
        kept = hits[: min(min_n, keep)]
    return kept


def retrieve_many(app_name: str, queries: list[str], doc_ids, top_n: int | None = None, **kw) -> list:
    """Retrieve per sub-query, then merge round-robin so every intent in a
    multi-part question is represented. Each sub-query keeps its own dynamic
    cutoff, so an off-topic part ("and the capital of France?") adds nothing."""
    if len(queries) <= 1:
        return retrieve(app_name, queries[0] if queries else "", doc_ids, top_n=top_n, **kw)
    keep = top_n or settings.rerank_top_n
    per_query = [retrieve(app_name, q, doc_ids, top_n=keep, **kw) for q in queries]
    merged, seen = [], set()
    for rank in range(keep):
        for hits in per_query:
            if rank < len(hits):
                h = hits[rank]
                key = (h.doc_id, h.heading, h.text[:80])
                if key not in seen:
                    seen.add(key)
                    merged.append(h)
    # Room for a little more than one question's worth, never unbounded.
    return merged[: keep + 2 * (len(queries) - 1)]


async def chat(
    app_name: str,
    messages: list[Message],
    client_prompt: str,
    doc_ids: list[str] | None,
    model: str | None = None,
    raw_system: bool = False,
    top_n: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    score_margin: float | None = None,
    min_n: int = 0,
    max_per_doc: int | None = None,
    query_context: str | None = None,
    reasoning: str | None = None,
    multi_query: bool = False,
) -> AsyncIterator[tuple[str, dict]]:
    """Yield (event, data) tuples: token* → sources → done."""
    queries = await contextualize_many(messages, model=model, query_context=query_context, split=multi_query)
    search_query = " | ".join(queries)
    chunks = retrieve_many(
        app_name, queries, doc_ids, top_k=top_k, top_n=top_n,
        min_score=min_score, score_margin=score_margin, min_n=min_n, max_per_doc=max_per_doc,
    )
    if raw_system:
        system = build_raw_system(client_prompt, chunks)
    else:
        system = build_chat_system(client_prompt, chunks)
    turns = [{"role": m.role, "content": m.content} for m in messages if m.content.strip()]

    async for delta in llm.chat_stream(system, turns, model=model, temperature=temperature, think=reasoning):
        yield "token", {"text": delta}

    sources = [
        {
            "doc_id": c.doc_id, "title": c.title, "heading": c.heading, "score": c.score,
            "rerank_score": c.rerank_score, "urls": c.urls,
        }
        for c in chunks
    ]
    yield "sources", {"sources": sources, "query": search_query, "queries": queries}
    yield "done", {"finished": True}
