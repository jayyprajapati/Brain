# Brain — Product Requirements Document

**Version:** 2.0.0
**Status:** Live in production (Hetzner)
**Owner:** Jay Prajapati
**Last updated:** 2026-07-25

---

## 1. Executive Summary

Brain is a small, **client-agnostic RAG (Retrieval-Augmented Generation) + LLM microservice**. It gives any number of unrelated applications (a portfolio site, a document-analysis product, an internal tool, etc.) a shared backend for three things:

1. **Turning documents into searchable knowledge** — extract text from files, chunk it semantically, embed it, and store it in an isolated per-app vector collection.
2. **Answering questions grounded in that knowledge** — retrieve the most relevant chunks, rerank them, and stream a persona-driven chat reply from an LLM.
3. **Making raw LLM calls** — a generic generate/JSON-mode endpoint for any caller that just needs a prompt answered, with or without retrieval.

Brain itself has **no opinion about what any given app is** — no hardcoded personas, no per-app business logic, no knowledge of "resumes" vs. "contracts" vs. "product docs." Every app that calls Brain supplies its own identity (`client_prompt`), its own documents, and optionally its own LLM credentials (BYOK). Brain only owns the generic mechanics: extraction, chunking, embedding, storage, retrieval, reranking, and streaming.

It currently powers **Portfolio** (chatbot answering as "Jay" from ingested profile notes) and **DocLens** (document extraction/RAG). Both talk to the same Brain instance from separate, fully isolated vector collections.

---

## 2. Problem Statement

Every new application that wants "chat with your documents" or "ask an LLM something" functionality ends up re-solving the same set of problems: parsing PDFs/DOCX, splitting text into embeddable chunks without breaking sentences or losing headings, choosing and running an embedding model, standing up a vector database, deduplicating near-identical content across re-uploads, reranking retrieved results, prompting an LLM to stay grounded and in-voice, and streaming the response back to a client.

Building this once *per app* means:
- Duplicated infrastructure (each app running its own embedding model, its own vector DB).
- Inconsistent RAG quality — chunking and prompting logic drifts between apps.
- Wasted compute — embedding models (ONNX, a few hundred MB) get loaded once per app instead of shared.
- No single place to reason about ingestion/retrieval correctness, dedup, or prompt-injection-style grounding failures.

**Brain exists to be that shared layer once**, well-built, so every new app that needs RAG or LLM access just calls an HTTP API instead of re-implementing the pipeline.

---

## 3. Goals & Non-Goals

### Goals
- **Client-agnostic**: zero hardcoded knowledge of any specific app's domain, persona, or document types.
- **Multi-tenant by construction**: every app's vectors live in a dedicated, isolated Qdrant collection keyed by `app_name`; further scoping within a collection by `doc_id` and `namespace`.
- **Stateless**: no server-side conversation memory — every chat call carries its own message history. No app-specific database rows.
- **BYOK-friendly**: callers can use Brain's default LLM (Ollama Cloud) or bring their own OpenAI/Anthropic/local-Ollama credentials per request.
- **Good-enough RAG out of the box**: structure-aware semantic chunking, cross-encoder reranking, near-duplicate suppression, URL preservation — without every caller having to tune it.
- **Small operational footprint**: one FastAPI container, no local LLM, no GPU requirement, deployable behind shared infra.

### Non-Goals
- Not a general-purpose document management system (no versioning UI, no document storage/retrieval of original files — only extracted text and vectors).
- Not a conversation-history store — callers own their own chat persistence if they need it.
- Not a multi-user auth/identity system — a single shared bearer API key gates the whole service; per-app/per-user access control is the caller's responsibility.
- Not a fine-tuned or hosted-model provider — Brain proxies to third-party LLM providers rather than serving its own model weights.

---

## 4. Target Consumers

Brain is consumed by other backend services, not directly by end users. Current consumers:

| App | Use of Brain |
|---|---|
| **Portfolio** (`api.jayyprajapati.com`) | Ingests profile/identity notes into the `portfolio` collection; `/v1/chat` answers site visitors in first person as "Jay," grounded in those notes. |
| **DocLens** | Uses `/v1/extract` to pull text from uploaded PDFs/DOCX and `/v1/ingest` + `/v1/retrieve` for document RAG, isolated in its own `doclens` collection. |

Any future app follows the same pattern: pick a unique `app_name`, ingest its content, supply its own `client_prompt` persona at chat time.

---

## 5. Approach / Design Philosophy

1. **`app_name` is the tenancy boundary.** It maps deterministically to a dedicated Qdrant collection (optionally namespaced by `QDRANT_COLLECTION_PREFIX`). Collections are created lazily on first ingest — no provisioning step for a new app.
2. **`doc_id` and `namespace` are the finer-grained boundaries inside a collection** — `doc_id` scopes one logical document (ingest replaces all of that doc's old chunks), `namespace` scopes a tenant/user grouping used for retrieval filtering and duplicate-detection scope.
3. **No default persona.** `/v1/chat` requires the caller to pass `client_prompt` on every request — Brain refuses (`400`) without it. This keeps identity entirely in the caller's admin layer, not baked into Brain's code or config.
4. **BYOK is opt-in, not required.** Every LLM-touching route accepts an optional `llm: {provider, api_key?, model?, base_url?}` override. Omit it and the request uses Brain's own Ollama Cloud key — this is what lets Brain be a zero-config default for simple callers while still letting sophisticated callers use their own OpenAI/Anthropic keys per request without Brain ever persisting them.
5. **Local, not hosted, embeddings/reranking.** fastembed (ONNX runtime) runs the embedding and cross-encoder reranker models in-process — no external embedding API, no per-token embedding cost, and no network dependency for that stage. Models are baked into the Docker image at build time so cold start is fast.
6. **Everything degrades gracefully.** Model warmup failures at startup don't crash the service (lazy retry). Reranker failures fall back to raw vector-search order rather than failing the request. Query-rewrite failures fall back to the raw user message. This reflects a "retrieval is best-effort, don't let it become a hard outage" stance.
7. **Statelessness over convenience.** Brain stores no conversation history — the caller resends the relevant message window on every `/v1/chat` call. This keeps Brain horizontally scalable (no session affinity needed) and keeps the caller in full control of context-window/history trimming decisions.

---

## 6. Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| **Language / runtime** | Python 3.12 | slim Docker base image |
| **Web framework** | FastAPI + Uvicorn | `--workers 1`, `--proxy-headers` (behind Caddy) |
| **Validation / config** | Pydantic v2 + `pydantic-settings` | `.env`-driven `Settings` singleton (`app/config.py`) |
| **HTTP client** | `httpx` (async) | Used for all outbound LLM provider calls, including streaming |
| **Embeddings** | `fastembed` — `BAAI/bge-base-en-v1.5` (ONNX, local, CPU) | Symmetric/asymmetric embed via `embed`/`query_embed` |
| **Reranking** | `fastembed` cross-encoder — `Xenova/ms-marco-MiniLM-L-6-v2` (ONNX, local, CPU) | |
| **Vector store** | Qdrant (self-hosted on Hetzner shared infra; Qdrant Cloud in the legacy DO deployment) | Cosine distance; one collection per `app_name`; payload-indexed on `doc_id`/`namespace` |
| **LLM providers** | Ollama Cloud (default, `gpt-oss:120b`), OpenAI, Anthropic, self-hosted Ollama — selectable per request | Raw REST calls per provider, no SDKs |
| **Document parsing** | `pdfplumber` (primary, layout + table-aware) → `pypdf` (fallback) for PDF; `python-docx` for DOCX; UTF-8 passthrough for `.md`/`.txt` | Legacy `.doc` explicitly unsupported |
| **Numerics** | `numpy` | Cosine similarity for semantic chunk grouping and dedup |
| **Containerization** | Docker (`python:3.12-slim`), non-root `appuser`, ONNX models baked in at build time | |
| **CI/CD** | GitHub Actions → GHCR (`Publish Backend Image`), manual/one-click `Deploy Backend` via SSH | Server only pulls images, never compiles |
| **Reverse proxy / TLS** | Caddy (shared infra, auto-HTTPS) — current; Nginx + Certbot — legacy DO deployment | |
| **Process manager** | Docker Compose (`restart: unless-stopped`) — current; systemd — legacy | |

No GPU, no torch, no local LLM weights — the only "ML" running in-process is the two small ONNX models (embedding + reranker); the chat/generation model itself is always a remote API call.

---

## 7. System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              Caller apps                 │
                        │  (Portfolio, DocLens, future apps …)      │
                        └───────────────┬───────────────────────────┘
                                        │  HTTPS + Bearer BRAIN_API_KEY
                                        ▼
                        ┌─────────────────────────────────────────┐
                        │        Caddy (shared reverse proxy)       │
                        └───────────────┬───────────────────────────┘
                                        ▼
                        ┌─────────────────────────────────────────┐
                        │      Brain (FastAPI, single container)    │
                        │                                            │
                        │  auth.py    → bearer key check             │
                        │  main.py    → routes                       │
                        │  pipeline.py→ orchestration                │
                        │  chunking.py→ semantic chunker              │
                        │  embeddings.py → fastembed (local ONNX)    │
                        │  reranker.py   → cross-encoder (local ONNX)│
                        │  extract.py    → PDF/DOCX → text            │
                        │  llm.py        → multi-provider LLM client │
                        │  vectorstore.py→ Qdrant wrapper             │
                        └──────┬───────────────────────┬─────────────┘
                               │                        │
                    (data net) ▼                        ▼ (outbound internet)
                    ┌────────────────┐      ┌─────────────────────────────┐
                    │  Qdrant         │      │ Ollama Cloud / OpenAI /      │
                    │  (self-hosted,  │      │ Anthropic / self-hosted      │
                    │  shared infra)  │      │ Ollama — selected per request │
                    │  1 collection   │      └─────────────────────────────┘
                    │  per app_name   │
                    └────────────────┘
```

Brain is entirely stateless on disk — the only persistent state is inside Qdrant. This means Brain containers can be freely restarted, redeployed, or scaled horizontally without any data-migration concern.

---

## 8. Core Features

### 8.1 Document text extraction (`/v1/extract`)
Accepts a multipart file upload (PDF, DOCX, MD, TXT) and returns clean plain text.
- PDF: `pdfplumber` first (preserves reading order, detects tables and renders them as Markdown so structure survives into chunking/embedding; table regions are excluded from the prose pass to avoid duplicate content), falling back to `pypdf` if pdfplumber is unavailable or yields nothing.
- DOCX: paragraph text + tables rendered as Markdown, via `python-docx`.
- `.doc` (legacy binary Word) is explicitly rejected with a clear error.
- MD/TXT/unknown-but-UTF-8: decoded directly.
- Optional one-shot ingest: pass `ingest=true` (+`app_name`, optional `doc_id`/`namespace`/`dedup`) to extract *and* immediately chunk/embed/store in the same call.

### 8.2 Semantic chunking (`app/chunking.py`)
A structure-aware pipeline, not naive fixed-size splitting:
1. Split on markdown headings and blank lines; track the current heading so it travels with each chunk as metadata.
2. Recognize and strip non-semantic noise: horizontal rules and `===== file.md =====`-style document-separator delimiters (used when callers concatenate many source files into one ingest payload) — these reset the "current heading" so a headingless section never inherits the wrong heading.
3. Sentence-split each block.
4. Group consecutive sentences into chunks while they remain semantically coherent (cosine similarity of the running centroid ≥ `SEMANTIC_THRESHOLD`) and under a token budget (`CHUNK_MAX_TOKENS`, floor `CHUNK_MIN_TOKENS`).
5. Merge runt trailing fragments into the previous chunk; carry a 1-sentence overlap between neighboring chunks so context isn't severed mid-thought.
6. Drop chunks with no real semantic content (below `CHUNK_MIN_CONTENT_WORDS`, unless they contain a URL) and collapse exact duplicates.
7. Extract URLs into structured metadata (`urls: []`) while keeping them inline in the chunk text, so "give me the link" style questions can be answered exactly.

### 8.3 Embedding + storage (`/v1/ingest`, `/v1/extract?ingest=true`)
- Ensures the app's Qdrant collection exists (created lazily, cosine distance, dimension inferred from a probe embedding).
- **Replace semantics**: ingesting a `doc_id` first deletes that document's existing chunks, then re-chunks/embeds/upserts — so re-ingesting an updated document doesn't leave stale chunks behind.
- **Cross-version dedup** (opt-in via `dedup=true` + `namespace`): a candidate chunk is skipped if its cosine similarity to an already-stored chunk in the same namespace, or to an already-accepted chunk earlier in the same batch, meets/exceeds `DEDUP_THRESHOLD` (default 0.92). This prevents near-identical content across overlapping document versions (e.g. resume revisions) from piling up redundant vectors.
- Caller-supplied `metadata` is merged onto every chunk's payload, except reserved keys Brain owns (`text`, `heading`, `urls`, `chunk_index`, `doc_id`, `namespace`).

### 8.4 Retrieval (`/v1/retrieve`, used internally by `/v1/chat`)
Embed → Qdrant vector search (top-`RETRIEVE_TOP_K`, optionally filtered by `doc_ids`/`namespace`) → cross-encoder rerank → top-`RERANK_TOP_N`. If the reranker is unavailable, retrieval degrades to raw vector-search order instead of failing.

### 8.5 Conversational RAG chat (`/v1/chat`)
- **Query contextualization**: the latest user message is rewritten into a standalone search query using conversation history (LLM call), so follow-ups like "tell me more about that" resolve correctly before retrieval. Falls back to the raw message on any failure.
- **Retrieval + reranking** as above, scoped optionally to specific `doc_ids`.
- **Prompt assembly** (`app/prompts.py`): Brain's generic conversational rules (first-person voice, grounded-only claims, no data-dumping, always end with a follow-up question) are combined with the caller's mandatory `client_prompt` (persona/voice/length, which always wins on tone/length conflicts) and the retrieved chunks rendered as "background notes." A reminder is restated *after* the context block — deliberately placed last so a smaller model doesn't drift into "summarize what I retrieved" instead of staying in character.
- **Streaming**: response tokens stream from the LLM as Server-Sent Events (`token` events), followed by a `sources` event (which chunks/doc_ids/URLs were used) and a `done` event. Errors surface as an `error` SSE event rather than breaking the HTTP connection.
- `client_prompt` is **mandatory** — there is no default persona; omitting it is a `400`.

### 8.6 Generic LLM access (`/v1/generate`)
A retrieval-free, single-shot LLM call: `system` + `prompt` (+ optional `data`, stringified if it's structured) → `text`, or `response_format: "json"` for a best-effort-parsed JSON reply (tolerates markdown code fences and leading/trailing prose by extracting the outermost balanced `{...}`/`[...]` span). Supports a backwards-compatible mode (used by Portfolio) where omitting `system` treats `prompt` itself as the system instruction.

### 8.7 Provider connectivity check (`/v1/llm/ping`)
A cheap single-token generation used to validate that a given provider/credential/model combination is reachable and correctly configured, without doing a full generation.

### 8.8 Deletion (`/v1/delete`)
Removes chunks by `doc_id` and/or `namespace` within an app's collection. At least one selector is required — Brain refuses to accept a request that would wipe an entire collection unscoped.

### 8.9 Multi-provider LLM abstraction with BYOK (`app/llm.py`)
A single internal interface (`generate`, `chat_stream`, `ping`) dispatches to provider-specific request builders:

| Provider | Auth | Default model | Notes |
|---|---|---|---|
| `ollama_cloud` | Brain's own key by default, BYOK override allowed | `gpt-oss:120b` (`CHAT_MODEL`) | Brain's zero-config default |
| `openai` | Caller-supplied `api_key` required | `gpt-4o-mini` | Chat Completions API |
| `anthropic` | Caller-supplied `api_key` required | `claude-sonnet-4-6` | Messages API |
| `ollama_local` | None (network-reachable `base_url`) | `llama3.1` | Self-hosted Ollama, e.g. on the caller's own infra |

Provider keys are **never read from Brain's environment** except the built-in Ollama Cloud default — every other provider's credentials arrive per-request and are never persisted.

### 8.10 Health check (`GET /health`, unauthenticated)
Returns `{status, chat_model, embed_model, providers}` — intended as an uptime-monitoring and smoke-test target.

---

## 9. API Reference

All routes except `GET /health` require `Authorization: Bearer <BRAIN_API_KEY>` (constant-time comparison via `secrets.compare_digest`).

| Method | Path | Request body | Response | Purpose |
|---|---|---|---|---|
| GET | `/health` | — | `{status, chat_model, embed_model, providers}` | Liveness / uptime monitoring |
| POST | `/v1/generate` | `{app_name?, system?, prompt, data?, llm?, response_format?, max_tokens?, temperature?}` | `{text, json?}` | Retrieval-free LLM call |
| POST | `/v1/llm/ping` | `{llm?}` | `{ok, provider, model}` | Verify provider/credentials reachability |
| POST | `/v1/extract` | multipart `file` + `{app_name?, doc_id?, namespace?, ingest?, dedup?}` | `{text, char_count, doc_id?, chunk_count, skipped_duplicates, ingested}` | PDF/DOCX/MD/TXT → text (+ optional ingest) |
| POST | `/v1/ingest` | `{app_name, doc_id, text, namespace?, dedup?, metadata?}` | `{doc_id, chunk_count, skipped_duplicates}` | Chunk/embed/store a document (replaces existing `doc_id`) |
| POST | `/v1/retrieve` | `{app_name, query, doc_ids?, namespace?, top_k?}` | `{chunks: [{text, heading, score, doc_id, urls}]}` | Raw retrieval primitive (embed→search→rerank) |
| POST | `/v1/delete` | `{app_name, doc_id?, namespace?}` | `{ok, deleted}` | Remove chunks by doc_id/namespace |
| POST | `/v1/chat` | `{app_name, messages[], client_prompt, doc_ids?, model?}` | SSE stream | Grounded, persona-driven streaming chat |

**Chat SSE event sequence:** `token` (`{text}`) × N → `sources` (`{sources: [{doc_id, heading, score, urls}]}`) → `done` (`{finished: true}`). Errors arrive as an `error` event (`{message}`) instead of breaking the stream.

`app_name` selects the app's dedicated Qdrant collection (optionally namespaced by `QDRANT_COLLECTION_PREFIX`); it is created lazily on first ingest. `doc_id` scopes a single logical document; `namespace` scopes a tenant/user grouping used for retrieval filtering and dedup.

---

## 10. End-to-End Flows

### 10.1 Ingest flow
```
Caller → POST /v1/extract (file, app_name, ingest=true, dedup=true, namespace)
  → extract.extract_text()               # PDF/DOCX/MD/TXT → clean plain text
  → pipeline.ingest(app_name, doc_id, text, namespace, dedup)
      → vectorstore.ensure_collection()    # lazy create, cosine distance
      → vectorstore.delete(doc_id)         # wipe this doc's prior chunks
      → chunking.chunk_text(text)          # structure-aware semantic chunks
      → embeddings.embed_documents()       # local ONNX, one vector per chunk
      → [optional] per-chunk dedup check   # vectorstore.max_similarity() vs namespace
      → vectorstore.upsert()               # store surviving chunks + payload
  ← {doc_id, chunk_count, skipped_duplicates}
```

### 10.2 Chat flow
```
Caller → POST /v1/chat (app_name, messages[], client_prompt, doc_ids?)
  → pipeline.contextualize(messages)       # LLM rewrites latest msg into standalone query
  → pipeline.retrieve(app_name, query)     # embed → Qdrant search (top-K) → rerank → top-N
  → prompts.build_chat_system(client_prompt, chunks)
        # BASE_CHAT_PROMPT + caller's persona + retrieved chunks as "background notes"
        # + RESPONSE_REMINDER (restated last, anti-drift for small models)
  → llm.chat_stream(system, messages)      # streams from selected provider (default Ollama Cloud)
  ← SSE: token* → sources → done
```

### 10.3 Generic generation flow
```
Caller → POST /v1/generate (system?, prompt, data?, llm?, response_format?)
  → llm.generate(system, user, ...)        # single-shot call to selected provider
  → [if response_format="json"] llm.parse_json()  # tolerant JSON extraction
  ← {text, json?}
```

### 10.4 Retrieval-only flow (primitive, e.g. for a caller building its own prompt)
```
Caller → POST /v1/retrieve (app_name, query, doc_ids?, namespace?, top_k?)
  → pipeline.retrieve()                     # embed → search → rerank
  ← {chunks: [...]}
```

---

## 11. Data Model

Brain has no relational database — all persistent state lives in Qdrant.

**Collection naming:** `{QDRANT_COLLECTION_PREFIX}{slugified(app_name)}` — one collection per app, cosine-distance vectors, dimension inferred at runtime from the embedding model.

**Point payload** (per chunk):
| Field | Type | Description |
|---|---|---|
| `doc_id` | string | Logical document this chunk belongs to (Brain-owned, always set) |
| `namespace` | string (optional) | Tenant/user grouping (Brain-owned when present) |
| `text` | string | The chunk's text content |
| `heading` | string | Nearest markdown heading above this chunk, or `""` |
| `urls` | string[] | URLs found inside the chunk |
| `chunk_index` | int | Contiguous position within the document's surviving chunks |
| `...metadata` | any | Caller-supplied extra fields (any key not in the reserved set above) |

**Indexes:** `doc_id` and `namespace` are payload-indexed (keyword) on every collection for fast filtered search/delete.

---

## 12. Multi-Tenancy & Isolation Model

Three nested scopes, each serving a distinct purpose:

1. **`app_name` → Qdrant collection.** Hard isolation — one app can never retrieve or accidentally delete another app's vectors, since they live in physically separate collections.
2. **`doc_id` → one logical document within a collection.** Ingesting a `doc_id` again fully replaces its prior chunks (delete-then-reinsert), so document updates don't accumulate stale vectors.
3. **`namespace` → a tenant/user grouping within a collection.** Used to scope both retrieval filtering (only search a given namespace) and dedup detection (only compare against chunks in the same namespace) — e.g. isolating multiple end-users' documents within a single app's collection.

---

## 13. Security

- **Authentication:** single shared bearer token (`BRAIN_API_KEY`) required on every route except `/health`, checked via constant-time comparison (`secrets.compare_digest`) to avoid timing side-channels.
- **Credential handling (BYOK):** third-party LLM provider keys (OpenAI, Anthropic, external Ollama) arrive per-request in the `llm` override and are **never persisted or logged** — Brain's own environment only holds its default Ollama Cloud key.
- **No secrets in the image:** production secrets live in a gitignored `brain.env` (`chmod 600`) on the server, injected via Docker Compose `env_file`; nothing is baked into the image.
- **Non-root container:** the Docker image runs as an unprivileged `appuser` (uid 10001).
- **No public network exposure of the vector store:** Qdrant on the shared Hetzner infra has no API key and is reachable only over the internal `data` Docker network — it is never published to the internet. TLS termination and all public exposure happen at Caddy.
- **Input validation:** `app_name`/`prompt`/`client_prompt`/message presence are validated per-route with explicit `400`s rather than silently proceeding with empty/defaulted values (notably: `/v1/chat` has no default persona and will refuse without `client_prompt`).

---

## 14. Deployment & Infrastructure

**Current: Hetzner shared-infra stack (Docker).**
- Brain runs as a single container (`brain-backend`) at `/srv/brain` on a shared Hetzner box that also hosts Caddy (reverse proxy/TLS), Postgres, Redis, Mongo, and Qdrant — one instance of each, shared across multiple projects (Portfolio, Admin, DocLens, Brain itself).
- Brain joins two external Docker networks: `web` (so the shared Caddy — and DocLens, in-cluster — can reach it) and `data` (so it can reach the shared Qdrant). It publishes no ports directly.
- **Build/deploy split:** GitHub Actions (`Publish Backend Image`) builds the Docker image on push to `main`/`master` (paths: `app/**`, `scripts/**`, `requirements.txt`, `Dockerfile`) and pushes to GHCR (`ghcr.io/jayyprajapati/brain-backend`), tagged `latest` and by commit SHA. The Hetzner server **never compiles** — a second workflow (`Deploy Backend`, manual `workflow_dispatch`) SSHes in and runs `docker compose pull && docker compose up -d`.
- ONNX embedding/reranker models are baked into the image at build time (as `appuser`) so container startup doesn't need a cold ~500 MB download.
- Healthcheck: Docker probes `GET /health` via Python's stdlib (slim image has no curl/wget), with a 90s `start_period` to allow model warmup.
- DNS: `brain.jayprajapati.dev` → Hetzner IP, Cloudflare DNS-only (grey cloud) so Caddy can complete the Let's Encrypt HTTP-01 challenge.

**Legacy: DigitalOcean droplet** (kept in `DEPLOY.md` for reference/disaster recovery) — Ubuntu 24.04, Python venv, systemd unit, Nginx + Certbot for TLS, Qdrant **Cloud** instead of self-hosted. Superseded by the Hetzner flow but documents the same underlying pipeline running without Docker.

**Statelessness / scaling:** Brain keeps no state on local disk beyond the baked model cache — all durable state is in Qdrant. This means it can be scaled horizontally (multiple containers/droplets behind a load balancer) with no session affinity required; the constraint to watch is RAM (fastembed is memory-bound), not CPU.

---

## 15. Configuration Reference

All settings load from environment / `.env` via `app/config.py` (`pydantic-settings`).

| Variable | Default | Purpose |
|---|---|---|
| `BRAIN_API_KEY` | `change-me` | Shared bearer secret every caller must send |
| `OLLAMA_API_KEY` | `""` | Brain's own Ollama Cloud key (default LLM provider) |
| `OLLAMA_BASE_URL` | `https://ollama.com` | Ollama Cloud endpoint |
| `CHAT_MODEL` | `gpt-oss:120b` | Default chat model when caller doesn't override |
| `QDRANT_URL` | `""` | Qdrant endpoint (self-hosted `http://qdrant:6333` or Qdrant Cloud URL) |
| `QDRANT_API_KEY` | `""` | Qdrant API key (empty for self-hosted, unpublished instance) |
| `QDRANT_COLLECTION_PREFIX` | `""` | Optional namespace prefix applied to every app's collection name |
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | fastembed embedding model |
| `RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | fastembed cross-encoder reranker model |
| `RETRIEVE_TOP_K` | `25` | Vector-search candidate count before reranking |
| `RERANK_TOP_N` | `5` | Final chunk count returned after reranking |
| `DEDUP_THRESHOLD` | `0.92` | Cosine similarity ≥ this ⇒ treated as a duplicate chunk at ingest |
| `CHUNK_MAX_TOKENS` | `320` | Upper bound on estimated tokens per chunk |
| `CHUNK_MIN_TOKENS` | `80` | Lower bound before a semantic/size break is allowed to close a chunk |
| `SEMANTIC_THRESHOLD` | `0.5` | Cosine similarity below which a new chunk starts |
| `CHUNK_MIN_CONTENT_WORDS` | `3` | Minimum real word tokens for a chunk to be kept (URLs always kept) |

---

## 16. Non-Functional Characteristics

- **Statelessness:** no conversation memory, no server-side sessions — every `/v1/chat` call is self-contained given the `messages` array supplied.
- **Resilience / graceful degradation:** model-warmup failures at startup don't crash the service (lazy retry on first use); reranker failures fall back to raw vector-search ordering; query-rewrite failures fall back to the raw user message — retrieval quality degrades before availability does.
- **Performance profile:** memory-bound, not compute-bound (ONNX embedding/reranking models resident in process memory); LLM generation latency is dominated by the upstream provider, not Brain itself. Long timeouts (300s) are used for LLM calls to tolerate slow cloud models.
- **Streaming-first chat:** SSE with `Cache-Control: no-cache` and `X-Accel-Buffering: no`; the legacy Nginx config explicitly documents that `proxy_buffering off` + long `proxy_read_timeout` are required, or `/v1/chat` will hang/close early.
- **No API docs exposed in production:** `docs_url`, `redoc_url`, and `openapi_url` are all disabled on the FastAPI app.

---

## 17. Current Limitations / Explicitly Out of Scope

- No per-caller/per-app authentication — one shared API key gates all callers; distinguishing/rate-limiting individual apps is not implemented at the Brain layer.
- No admin UI — all operations are HTTP API calls made by caller backends.
- No original-file storage — only extracted text and its vector chunks persist; the source PDF/DOCX itself is not retained.
- No built-in conversation history persistence — entirely the caller's responsibility.
- Legacy `.doc` (binary Word) format is unsupported by design.
- Single-region, single-instance Qdrant — no built-in vector-store replication/HA at the Brain layer (inherited from whatever the shared infra provides).
