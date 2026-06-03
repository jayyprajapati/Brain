# Brain

A small, client-agnostic RAG service. It ingests text, chunks it semantically,
embeds it locally with [fastembed](https://github.com/qdrant/fastembed), stores
vectors in **Qdrant Cloud**, and answers questions by retrieving + reranking
context and streaming a reply from **Ollama Cloud**.

Brain knows nothing about any particular app. The **personality / identity** of
the answering voice is supplied by the caller on every chat request as
`client_prompt` — Brain only owns the generic conversation mechanics (first
person, concise, grounded, ends with a follow-up question). There is **no
default personality**: if `client_prompt` is missing, `/v1/chat` returns `400`.

## Stack

- **FastAPI** + uvicorn
- **fastembed** for embeddings (`BAAI/bge-base-en-v1.5`) and cross-encoder reranking
- **Qdrant Cloud** vector store (one collection per app, selected by `app_name`)
- **Ollama Cloud** for the LLM (`gpt-oss:20b` by default)

No Docker, no local LLM, no torch.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the keys
```

Required in `.env`: `BRAIN_API_KEY`, `OLLAMA_API_KEY`, `QDRANT_URL`,
`QDRANT_API_KEY`. Optional: `QDRANT_COLLECTION_PREFIX` (namespaces every app's
collection). Confirm your chat model exists on the cloud:

```bash
python scripts/check_ollama.py
```

Run:

```bash
uvicorn app.main:app --port 8000 --reload
```

The first run downloads the embedding + reranker ONNX models (a few hundred MB).

## API

All routes except `GET /health` require `Authorization: Bearer <BRAIN_API_KEY>`.

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `{status, chat_model, embed_model}` |
| POST | `/v1/generate` | `{app_name?, prompt, data}` | `{text}` |
| POST | `/v1/ingest` | `{app_name, doc_id, text}` | `{doc_id, chunk_count}` |
| POST | `/v1/delete` | `{app_name, doc_id}` | `{ok, deleted}` |
| POST | `/v1/chat` | `{app_name, messages, client_prompt, doc_ids?, model?}` | SSE |

`app_name` selects the app's **dedicated Qdrant collection**, so each app's
vectors stay isolated (e.g. `app_name="portfolio"` → collection `portfolio`,
optionally namespaced by `QDRANT_COLLECTION_PREFIX`). Collections are created
lazily on first ingest. Within a collection, retrieval is scoped by `doc_id`.

`/v1/generate` is pure LLM (no retrieval): it turns structured `data` into prose
using `prompt` (`app_name` is accepted but unused). `/v1/ingest` replaces a
document (ensures the collection, then deletes the old `doc_id` chunks before
re-inserting). `/v1/chat` rewrites follow-ups into a standalone query, then
retrieves, reranks, and streams the answer.

**Chat SSE events:** `token` (`{text}`) repeated, then `sources`
(`{sources:[{doc_id,heading,score,urls}]}`), then `done` (`{finished:true}`).
Errors arrive as `error` (`{message}`).

Conversation context is **stateless**: send the recent `messages` on every turn.
