# Testing Brain

How to stand up a safe local test environment, what to run, and what "safe"
means when a route calls a paid third-party API. This is the methodology doc —
for the latest actual numbers, see `TEST_RESULTS.md`.

---

## 1. Local self-hosted Qdrant (required before any retrieval/chat testing)

Brain never talks to Qdrant Cloud or any other cloud vector store in local
dev — only the self-hosted engine, exactly like production. Bring up a
**persistent** local Qdrant (named container + named volume, survives
restarts — not a throwaway):

```bash
cd /path/to/Brain
docker compose -f docker-compose.dev.yml up -d
```

This starts `brain-dev-qdrant` on `localhost:6333` (REST) / `6334` (gRPC),
backed by the named volume `brain-dev-qdrant`. Confirm it's up:

```bash
curl -s localhost:6333/collections
# {"result":{"collections":[]},"status":"ok","time":...}
```

Point Brain's local `.env` at it:

```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

(Local self-hosted Qdrant has no auth — leave the key blank. Production's
Qdrant, also self-hosted, **does** require a key over the internal `data`
network; see `DEPLOY.md`.)

Run Brain against it:

```bash
.venv/bin/uvicorn app.main:app --port 8000 --reload
```

Sanity check end-to-end (not just `/health` — `/health` doesn't touch
Qdrant, so it can report `ok` even when the vector store is unreachable):

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/v1/ingest \
  -H "Authorization: Bearer $BRAIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"app_name":"smoketest","doc_id":"doc-1","text":"some sample text to embed"}'

curl -s -X POST localhost:8000/v1/retrieve \
  -H "Authorization: Bearer $BRAIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"app_name":"smoketest","query":"sample text"}'

# clean up what you just ingested
curl -s -X POST localhost:8000/v1/delete \
  -H "Authorization: Bearer $BRAIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"app_name":"smoketest","doc_id":"doc-1"}'
curl -s -X DELETE localhost:6333/collections/smoketest   # drop the now-empty collection
```

A 500 from `/v1/retrieve` or `/v1/chat` with `/health` reporting `ok` is the
signature of a Qdrant misconfiguration (wrong URL, dead cluster, wrong key) —
`pipeline.retrieve()` only guards the reranker call, not `vectorstore.search()`,
so a Qdrant failure propagates as an unhandled exception on both routes.

**Never point `QDRANT_URL` at a `*.cloud.qdrant.io` host, or any other
managed/cloud vector store, in local `.env`.** If you find one there, it's
stale — repoint it at `http://localhost:6333` per above.

---

## 2. Golden-set retrieval + faithfulness eval (`scripts/eval.py`)

For **regression testing** a chunking/reranking/prompt/model change against a
per-app golden set of known-answer queries:

```bash
python scripts/eval.py --app portfolio                # one app
python scripts/eval.py                                  # every golden file present
python scripts/eval.py --skip-faithfulness               # retrieval-only, fast & free
python scripts/eval.py --min-recall 0.8 --min-faithfulness 0.9   # gate mode (non-zero exit on fail)
```

Runs **in-process** (imports `app.pipeline`/`app.llm`/`app.prompts` directly,
no HTTP) against whatever `.env` Brain itself uses — so step 1 above must be
done first, and a real `OLLAMA_API_KEY` is needed for `check_faithfulness`
cases (each one is a real, billable Ollama Cloud call: one answer-generation
call + one judge call). `--skip-faithfulness` avoids that spend entirely.

Golden sets live in `eval/goldens/<app_name>.json` — see
`eval/goldens/README.md` for the format. Only `portfolio.example.json` ships
today as a template; real per-app golden files with pinned `doc_id`s still
need to be built (see `PRD.md` §8.11 for why they aren't wired into CI yet).

### Ad-hoc reranked-vs-unreranked comparison (accuracy methodology)

`scripts/eval.py` scores the *normal* (reranked) path only. To specifically
measure what the cross-encoder reranker is buying — hit-rate@5, recall@5, and
MRR reranked vs. left in raw vector-search order — there is no permanent
script for this (it's a periodic health-check, not a CI gate), but the
pattern used for the numbers in `TEST_RESULTS.md` is:

1. Pick ~10 real, distinct project documents (not fabricated filler — real
   prose gives genuinely specific, checkable facts). Ingest each via
   `POST /v1/ingest` into a scratch `app_name` (e.g.
   `brain_local_eval_<date>`).
2. Write 5 hand-written fact-lookup queries per document, each with one known
   correct `doc_id`.
3. For each query: embed it (`app.embeddings.embed_query`), do one
   `vectorstore.search()` for the top-`RETRIEVE_TOP_K` candidate pool (the
   real value in `app/config.py`, not whatever a stale doc claims it is).
   **Unreranked** = that pool's top 5, left in raw cosine order. **Reranked**
   = the same pool run through `app.reranker.rerank()`, top 5. Only the
   ordering step differs — this is the apples-to-apples comparison that
   isolates the reranker's actual contribution.
4. Aggregate hit-rate@5 / recall@5 / mean MRR for both, plus reranker latency
   (mean/p50/max) from the `rerank()` calls themselves.
5. **Clean up**: `POST /v1/delete` every ingested `doc_id`, then
   `DELETE /collections/<app_name>` on Qdrant directly once it's empty (an
   empty collection still shows up in `GET /collections` otherwise). Verify
   the collection is gone and/or the point count is back to whatever it was
   before you started — never leave scratch data behind in the persistent
   local Qdrant.

This is intentionally cheap (no LLM calls, pure retrieval) so it can be rerun
freely against local Qdrant. Do **not** run this ingestion pattern against
production Qdrant — it's reachable read-only for this kind of testing only
via the read-only pattern in §4 below.

---

## 3. Load-testing `/v1/retrieve` and `/v1/chat`

### `/v1/retrieve` — safe to push hard locally

Pure retrieval (embed → Qdrant search → rerank) touches no paid API once the
docs are ingested — the embedding and reranker models are local ONNX
(fastembed). Against the **local, self-hosted Qdrant from §1**, concurrency
ramps of 10 / 50 / 100 are fine and cost nothing. A simple pattern:

```python
import asyncio, httpx, time

async def one(client, payload):
    t0 = time.perf_counter()
    r = await client.post("/v1/retrieve", json=payload, headers=HEADERS)
    return time.perf_counter() - t0, r.status_code

async def burst(concurrency, total):
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=60) as client:
        results = await asyncio.gather(*[one(client, PAYLOAD) for _ in range(total)])
    # compute p50/p95/p99 from the first element of each tuple; check status_code==200
```

Report p50/p95/p99 latency, throughput (requests/sec = total / wall time),
and error rate at each concurrency level. Watch for the throughput *ceiling*
— with `--workers 1`, embed+rerank inference is CPU-bound and serializes on
one event loop, so past a certain offered concurrency, added load only adds
queueing latency, not completed work (see `TEST_RESULTS.md` §Throughput and
the worker-count discussion in §5 below).

### `/v1/chat` — real, paid Ollama Cloud; stay conservative

Every `/v1/chat` call that doesn't override `llm` bills against the real
`OLLAMA_API_KEY` in `.env`. **Do not blind-ramp this the way `/v1/retrieve`
gets ramped.** Escalate slowly and stop as soon as you have a data point,
rather than chasing a breaking point:

- Sequential (n≈3) → then 2 → 3 → 5 concurrent. Stop there unless there's a
  specific reason to go further (e.g. chasing an actual error, not just more
  data).
- Measure via the SSE stream itself: time-to-first-token = first `token`
  event; time-to-done = the `done` event.
- If you need higher concurrency for a specific question, pass an `llm`
  override pointing at a free/local provider (`ollama_local` against a
  locally-running Ollama, or a `openai`/`anthropic` key on a low-cost model)
  instead of hammering the default paid Ollama Cloud path.

### `/v1/generate` and faithfulness-judge calls

Same caution as `/v1/chat` — both are real LLM calls unless `llm` is
overridden. `scripts/eval.py --skip-faithfulness` avoids the judge calls
entirely when you only want retrieval numbers.

---

## 4. Checking collection health / point counts

**Local** (self-hosted Qdrant from §1, no auth):

```bash
curl -s localhost:6333/collections
curl -s localhost:6333/collections/<name>   # points_count, status, vector config
```

**Production** — Qdrant is on the internal `data` Docker network only, never
published to the internet, and requires an API key. Reach it **read-only**
from inside the already-running `brain-backend` container over SSH, using
that container's own `QDRANT_API_KEY` from its environment — never open a
port, never write:

```bash
ssh deploy@<hetzner-ip>
docker exec brain-backend python3 -c "
import urllib.request, json, os
key = os.environ['QDRANT_API_KEY']
req = urllib.request.Request('http://qdrant:6333/collections', headers={'api-key': key})
print(json.dumps(json.loads(urllib.request.urlopen(req).read()), indent=2))
"
```

Then per collection: `GET http://qdrant:6333/collections/<name>` (same
header) for `points_count`. Never run `/v1/ingest`, `/v1/delete`, or any
Qdrant write call against production data as part of a test — see §2's
warning.

To confirm an app's production wiring (which collection it actually writes
to) without touching its data, check its own container's env, read-only:

```bash
docker exec <app>-backend printenv | grep -i -E 'BRAIN|QDRANT'
```

---

## 5. Worker-count tuning rationale

`Dockerfile`'s `CMD` sets `uvicorn --workers N`. Each worker is a full
separate process that loads its **own copy** of both ONNX models (embedder +
reranker) — fastembed is memory-bound, not compute-bound, so the real
constraint on worker count is RAM, not CPU alone.

To tune it for real instead of guessing:

1. **Measure actual per-worker RSS**, warmed up (not cold-start size):
   ```bash
   .venv/bin/uvicorn app.main:app --port 8000 &
   # hit /v1/ingest + /v1/retrieve at least once to force both models to load
   ps -o pid,ppid,rss,comm -p <worker-pid>   # RSS in KB
   ```
   Or in prod: `docker stats <container> --no-stream` after the container has
   served real traffic (startup already bakes/warms the models per the
   Dockerfile, so this is close to steady-state from boot).
2. **Check the host's real headroom**, not just its total spec — on a shared
   box, `free -h` and `docker stats` across *every* container tell you what's
   actually available, not what's nominally installed:
   ```bash
   free -h
   docker stats --no-stream
   ```
3. **Apply the rule of thumb literally only on a dedicated box**: "1 worker
   per GB of RAM, capped at vCPU count" (from this doc's original systemd
   section) assumes Brain owns the whole machine. On a **shared** host running
   several other projects' containers, budget only the RAM Brain can
   reasonably claim without starving its neighbors — see `TEST_RESULTS.md`
   for the current production numbers and the specific worker count chosen
   from them.
4. **Re-run the §3 `/v1/retrieve` throughput ramp** after any worker-count
   change to confirm the ceiling actually moved — a worker-count change is a
   claim about concurrency headroom, and the only way to know it worked is to
   re-measure, not assume.

---

## 6. Cleanup checklist (every testing session)

- [ ] Every `doc_id` ingested into a scratch `app_name` deleted via
      `POST /v1/delete`.
- [ ] The scratch collection itself removed (`DELETE /collections/<name>`) if
      it's now empty — Qdrant doesn't auto-drop empty collections.
- [ ] Point count verified back to whatever it was before the session
      started (`GET /collections/<name>`, or absence from
      `GET /collections` entirely).
- [ ] No test traffic sent to production Qdrant beyond read-only `GET`s.
- [ ] Any process/container started only for testing (a second Brain
      instance, an ephemeral Qdrant) is stopped and removed — unless it's the
      persistent local Qdrant from §1, which should be **left running** for
      next time.
