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
| GET | `/health` | — | `{status, chat_model, embed_model, providers}` |
| POST | `/v1/generate` | `{prompt, system?, data?, llm?, response_format?, max_tokens?, temperature?}` | `{text, json?}` |
| POST | `/v1/llm/ping` | `{llm}` | `{ok, provider, model}` |
| POST | `/v1/extract` | multipart `file` + `{app_name?, doc_id?, namespace?, ingest?, dedup?}` | `{text, char_count, doc_id?, chunk_count, skipped_duplicates, ingested}` |
| POST | `/v1/ingest` | `{app_name, doc_id, text, namespace?, dedup?, metadata?}` | `{doc_id, chunk_count, skipped_duplicates}` |
| POST | `/v1/retrieve` | `{app_name, query, doc_ids?, namespace?, top_k?}` | `{chunks:[…]}` |
| POST | `/v1/delete` | `{app_name, doc_id?, namespace?}` | `{ok, deleted}` |
| POST | `/v1/chat` | `{app_name, messages, client_prompt, doc_ids?, model?}` | SSE |

`app_name` selects the app's **dedicated Qdrant collection**, so each app's
vectors stay isolated (e.g. `app_name="portfolio"` → collection `portfolio`,
optionally namespaced by `QDRANT_COLLECTION_PREFIX`). Collections are created
lazily on first ingest. Within a collection, retrieval/delete are scoped by
`doc_id` (one document) and/or `namespace` (a tenant/user grouping).

**BYOK** — every LLM route (`/v1/generate`, `/v1/llm/ping`; `/v1/chat`
optionally) accepts an `llm` override `{provider, api_key?, model?, base_url?}`
with `provider ∈ {openai, anthropic, ollama_cloud, ollama_local}`. Omit it to use
Brain's own Ollama Cloud default. Brain never reads provider keys from its env —
they arrive per request.

`/v1/generate` is pure LLM (no retrieval): `system` + `prompt` (+ optional
`data`) → text, or set `response_format:"json"` to get parsed `json`.
`/v1/extract` pulls plain text from an uploaded PDF/DOCX and can optionally
ingest it (with cross-version `dedup` within a `namespace`). `/v1/ingest`
replaces a document (deletes the old `doc_id` chunks, then chunks/embeds/upserts,
skipping near-duplicates when `dedup` + `namespace` are set). `/v1/retrieve`
exposes the embed→search→rerank pipeline as a primitive. `/v1/chat` rewrites
follow-ups into a standalone query, then retrieves, reranks, and streams.

**Chat SSE events:** `token` (`{text}`) repeated, then `sources`
(`{sources:[{doc_id,heading,score,urls}]}`), then `done` (`{finished:true}`).
Errors arrive as `error` (`{message}`).

Conversation context is **stateless**: send the recent `messages` on every turn.
