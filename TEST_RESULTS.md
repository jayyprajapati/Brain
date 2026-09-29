# Brain — Test Results

Living results doc. Factual findings only, each dated — see `TESTING.md` for
methodology. Full narrative report with charts:
`https://claude.ai/code/artifact/95074a1a-6e0a-4301-bcb4-b3c29c8ba292`.

---

## 2026-08-15 (pass 2) — local Qdrant fixed persistently, production usage audit, worker tuning

### What changed since pass 1

Local Brain's `.env` had `QDRANT_URL` pointing at a dead Qdrant Cloud cluster
(every request returned a generic Go-server `404`, DNS/LB resolved but the
cluster mapping was gone — a stale credential from before the self-hosted
migration; `.env` was last touched 2026-06-03). Pass 1 worked around this with
a fully throwaway/ephemeral Qdrant container. That's now fixed for real:

- Added `docker-compose.dev.yml` at the repo root: a **named, persistent**
  `brain-dev-qdrant` container (`qdrant/qdrant:v1.12.4`) on `localhost:6333`,
  backed by a named Docker volume (`brain-dev-qdrant`) — survives restarts,
  intended to stay up as the standing local dev Qdrant, not a one-off.
- Local `.env` repointed: `QDRANT_URL=http://localhost:6333`,
  `QDRANT_API_KEY=` (blank — local self-hosted has no auth).
- Local Brain process restarted against the corrected config.
- Verified end-to-end: `GET /health` → `200 ok`; `POST /v1/ingest` →
  `chunk_count:1`; `POST /v1/retrieve` → returned the ingested chunk with a
  real similarity score (`0.57`). No more 404s/500s. Test doc cleaned up via
  `POST /v1/delete`; collection count confirmed back to 0.
- Production's Qdrant pointing was **not** touched — a read-only check
  (`docker exec brain-backend printenv`) confirmed prod already has
  `QDRANT_URL=http://qdrant:6333` (correct, self-hosted, internal-only). This
  was always a local-dev-only problem.

### Retrieval accuracy — reranked vs. unreranked, re-run against real local Qdrant

Same methodology as pass 1 (§TESTING.md §2): 10 real project documents
(Brain's own README + PRD, 4 Offrloop `CLAUDE.md`/`ARCHITECTURE.md`
references, 4 Offrloop feature specs), 5 hand-written fact-lookup queries per
document (50 total), ingested into a scratch `app_name` on the now-persistent
local Qdrant. 225 chunks ingested.

| Metric | Reranked | Unreranked | Delta |
|---|---:|---:|---:|
| Hit-rate@5 | 94.0% | 94.0% | 0.0 pt |
| Recall@5 | 94.0% | 94.0% | 0.0 pt |
| Mean MRR | 0.9150 | 0.8247 | **+0.0903** |
| Reranker latency (mean / p50 / max) | 123.4 ms / 126.6 ms / 238.0 ms | — | — |

n = 50 queries, 10 docs, 1 expected `doc_id` per query. 1 reranker regression
(hit→miss: "How many background workers start after Mongo connects in the
Offrloop backend?" — correct doc dropped out of top-5 after rerank), 1
improvement (miss→hit: "How many contacts maximum are allowed per group in
Offrloop?" — correct doc pulled into top-5 by rerank). Net: hit-rate/recall
identical (the one regression and one improvement cancel out), but the
reranker's **ordering** contribution is clearly positive and larger than pass
1 measured (+0.0903 MRR here vs. +0.016 in pass 1) — consistent with pass 1's
own read that a reranker's value is in ranking quality, not necessarily
rescuing misses, and that this shows up more or less depending on which
specific queries land in a given run's golden set. Reranker latency
(123.4/126.6/238.0 ms) is within noise of pass 1's isolated-container numbers
(122.7/124.9/235.0 ms) — the persistent local Qdrant behaves the same as the
throwaway one did, as expected since the reranker itself is unrelated to
which Qdrant instance backs it.

**Cleanup verified:** all 10 docs' chunks deleted via `POST /v1/delete`; the
scratch collection dropped via `DELETE /collections/<name>`; confirmed via
`GET /collections` — empty, matching the pre-test baseline (local Qdrant had
0 collections before this run, 0 after).

### Production Qdrant collection / usage audit (read-only)

Pulled read-only over SSH, `docker exec brain-backend` reaching
`http://qdrant:6333` with that container's own `QDRANT_API_KEY` (Qdrant
requires the key even over the internal network — not keyless as some infra
notes assumed; worth a docs correction elsewhere, not urgent).

| Collection | Points | Notes |
|---|---:|---|
| `portfolio` | 58 | Portfolio's real chatbot content — unchanged since pass 1 |
| `deploy_smoketest` | 0 | Empty — a deploy-time smoke-test artifact, not real app data |
| `offrloop_resumes` | **does not exist** | Offrloop's configured `BRAIN_APP_NAME` — see below |
| `doclens` | **does not exist** | Confirms pass 1's finding |

**Why usage looks low — checked, not guessed:** all three apps' production
containers were checked directly (`docker exec <app>-backend printenv`) for
their actual Brain wiring:

- **Offrloop**: `BRAIN_BASE_URL=http://brain-backend:8000`,
  `BRAIN_APP_NAME=offrloop_resumes`, and its `BRAIN_API_KEY` was confirmed to
  **match** brain-backend's own key byte-for-byte (compared via a shell
  equality check, values never printed together). Wiring is correct. Zero
  ingests in production means the AI features that call Brain — Resume Lab
  upload/analyze, DSA Lab, Compose rewrite (all BYOK-gated per
  `backend/src/CLAUDE.md`) — have genuinely not been exercised by real
  production users yet, most plausibly because they require the user to
  configure their own LLM key first (BYOK) before any of those routes will
  even attempt a Brain call.
- **DocLens**: `BRAIN_BASE_URL=http://brain-backend:8000`,
  `BRAIN_APP_NAME=doclens`. Also correctly wired. Same conclusion as pass 1:
  genuinely zero production RAG usage so far, not a broken integration.
- **Portfolio**: the one app with real usage (58 points) — makes sense, it's
  the one Brain consumer with an always-on feature (the chatbot) that isn't
  gated behind a user bringing their own key.

**Verdict:** low measured usage is real and explained by the data, not a bug.
Two of three consumers are correctly wired but not yet organically used
(BYOK-gated features with presumably few users who've completed BYOK setup);
the third (Portfolio) is small in absolute terms because it's a single
person's profile content, not a multi-tenant corpus. Nothing here points to
ingestion failures, auth mismatches, or misrouted `app_name`s — every wiring
check came back correct.

### Worker-count tuning — measured, prepared, not deployed

Load test from pass 1 found `--workers 1` caps `/v1/retrieve` throughput at
~7 rps and drives p50 to 17.1s at 100 concurrent, because embed+rerank
inference is CPU-bound and serializes on one process's event loop.

**Measured, not assumed:**
- Production `brain-backend` (1 worker, warmed, serving real traffic for 2+
  weeks): **1.217 GiB RSS** (`docker stats`, single sample).
- Local Brain (1 worker, warmed via one ingest + one retrieve call): **~911
  MiB RSS** (`ps -o rss` on the actual uvicorn worker process, not the
  `--reload` watcher parent). Consistent with the production number given
  platform/build differences — corroborates "~1 GB per warmed worker" as a
  real, not guessed, figure.
- Production host: **8 GiB RAM / 4 vCPU total**, shared by **12 containers
  across 7 projects** (`brain`, `offrloop`, `portfolio`, `doclens`, `settl`,
  `codehive`, `admin`, plus `postgres`/`redis`/`mongo`/`qdrant`/`caddy`).
  `free -h` at measurement time: 3.9 GiB used, 652 MiB free, 3.5 GiB
  reclaimable buff/cache (3.7 GiB "available"). Per-container snapshot
  (`docker stats --no-stream`): brain-backend is already the single largest
  consumer on the box at 1.217 GiB (16.1% of total), ahead of
  settl-backend (832 MiB) and offrloop-backend (456 MiB); everything else is
  well under 300 MiB.

**Decision: `--workers 2`** (changed in `Dockerfile`, **not yet deployed**).
Rationale:
- Doubling to 2 workers directly targets the measured bottleneck (CPU-bound
  inference serialized on 1 process) and should roughly double
  `/v1/retrieve` throughput headroom before the same queueing behavior
  reappears — worth re-confirming with the §3 throughput ramp after deploy.
- Cost: another ~1.0–1.2 GiB RSS, taking Brain to ~2.2–2.4 GiB (~30% of the
  box) — the largest consumer by a wide margin, but the box's ~3.5–3.7 GiB
  reclaimable/available headroom comfortably absorbs it today.
- **Did not** use the `DEPLOY.md` "1 worker per GB, capped at vCPU count"
  rule literally — that assumes Brain owns the whole machine. It doesn't:
  this is an 8GB/4vCPU box shared with 6 other projects' containers, and
  Brain is already the biggest single memory consumer on it. Taking it to 4
  workers (the vCPU cap) would mean ~4.5–4.9 GiB for Brain alone — more than
  half the box — which risks starving the other 6 projects under any
  simultaneous load spike. 2 workers is a deliberately conservative middle
  ground: real throughput gain, bounded blast radius.
- Changed in `Dockerfile`'s `CMD` (`--workers 1` → `--workers 2`) with an
  inline comment recording this rationale and pointing back here.
  **This change is uncommitted in the working tree, pending the user's own
  review/deploy schedule — it has not been pushed, and the live production
  container is still running `--workers 1`.**
- Local dev's run command (`uvicorn ... --reload`) is unchanged and should
  stay at 1 worker — `--reload` doesn't meaningfully support multiple
  workers, and dev doesn't need the throughput.

---

## 2026-08-15 (pass 1) — original stress test findings

Full detail, methodology, and charts in the published report (see link at
top). Summary of the numbers that still stand:

- **Retrieval accuracy** (isolated throwaway Qdrant, 50 queries / 10 docs /
  204 chunks): hit-rate@5 98.0% reranked vs. 100.0% unreranked (−2.0pt, one
  single-query regression drove the entire delta); mean MRR 0.9267 reranked
  vs. 0.9107 unreranked (+0.0160). Reranker latency 122.7/124.9/235.0 ms
  (mean/p50/max).
- **`/v1/retrieve` latency/throughput** (isolated Qdrant, reranked path): p50
  1.5s / 7.1s / 17.1s at concurrency 10/50/100; throughput peaks ~7.07 rps at
  c=50 and falls to 5.53 rps at c=100 (classic single-worker saturation — see
  worker-tuning above); **zero errors** at any level tested (up to 800
  sequential-queued requests at c=100).
- **`/v1/chat` latency** (real, paid Ollama Cloud, conservative escalation
  1→2→3→5 concurrent): TTFT p50 1.77s→3.60s, total p50 2.36s→3.88s as
  concurrency rose 1→5; zero errors.
- **Cost avoided** (real production Qdrant `portfolio` collection, 58 points,
  17,273 measured characters ≈ 4,318 tokens at the 4-chars-per-token rule of
  thumb): ≈ $0.00009 vs. OpenAI `text-embedding-3-small` at $0.02/1M tokens
  for re-embedding the current snapshot — genuinely tiny in absolute terms
  today, but zero marginal cost regardless of ingestion volume growth, and
  understated because it doesn't count the per-query embedding cost every
  `/v1/retrieve`/`/v1/chat` call would incur on a paid API.
- **Root cause found**: local `.env`'s `QDRANT_URL` pointed at a dead Qdrant
  Cloud cluster (fixed for real in pass 2, above).

---

## Open items / not yet done

- Per-app golden sets (`eval/goldens/<app>.json`) still need real, pinned
  `doc_id`s for `portfolio` and `doclens` — only the `.example.json` template
  exists. Blocks wiring `scripts/eval.py` into CI (see `PRD.md` §8.11).
- `pipeline.retrieve()` still only guards the reranker call, not
  `vectorstore.search()` — a Qdrant outage/misconfig still propagates as an
  unhandled exception on `/v1/retrieve` and `/v1/chat` (pass 1 P1
  recommendation, not yet implemented).
- Worker-count change (`--workers 2`) is prepared in the working tree but
  **not committed, not pushed, not deployed** — re-run the `/v1/retrieve`
  throughput ramp (`TESTING.md` §3) after it ships to confirm the ceiling
  actually moves before calling it done.
