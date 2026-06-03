"""Orchestration: ingest, retrieve, query-contextualization, and chat."""
from __future__ import annotations

from typing import AsyncIterator

from . import embeddings, llm, reranker, vectorstore
from .chunking import chunk_text
from .config import settings
from .prompts import QUERY_REWRITE_PROMPT, build_chat_system
from .schemas import Message


def ingest(app_name: str, doc_id: str, text: str) -> int:
    """Replace a document: ensure the app's collection, wipe its old chunks, then
    chunk → embed → upsert the new ones."""
    vectorstore.ensure_collection(app_name, embeddings.vector_size())
    vectorstore.delete(app_name, doc_id)
    chunks = chunk_text(text)
    if not chunks:
        return 0
    vectors = embeddings.embed_documents([c.text for c in chunks])
    payloads = [
        {"text": c.text, "heading": c.heading, "urls": c.urls, "chunk_index": c.chunk_index}
        for c in chunks
    ]
    vectorstore.upsert(app_name, doc_id, vectors, payloads)
    return len(chunks)


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


def retrieve(app_name: str, search_query: str, doc_ids: list[str] | None) -> list:
    """Embed → vector search (top-K) → cross-encoder rerank → top-N."""
    if not search_query:
        return []
    query_vec = embeddings.embed_query(search_query)
    hits = vectorstore.search(app_name, query_vec, doc_ids, settings.retrieve_top_k)
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
