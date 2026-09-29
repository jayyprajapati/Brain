# Brain — built in CI (GitHub Actions → GHCR) and PULLED by the Hetzner server; the server never
# compiles. fastembed downloads the embedding + reranker ONNX models (~500 MB); we bake them into the
# image at build time so startup is fast and needs no network access for models.
FROM python:3.12-slim

# onnxruntime (pulled in by fastembed) needs libgomp1 at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user; its HOME holds the fastembed model cache baked in below.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
RUN chown -R appuser:appuser /app

USER appuser

# Pre-download (bake) the models as appuser so the cache lands in this user's HOME. Brain's lifespan
# also warms lazily, but baking avoids the ~500 MB cold download on first boot.
RUN python -c "from app.embeddings import warmup; warmup()" \
    && python -c "from app.reranker import warmup; warmup()"

EXPOSE 8000
# --workers 2 (was 1): the 2026-08-15 load test found --workers 1 serializes CPU-bound
# fastembed embed + cross-encoder rerank calls on one event loop, capping /v1/retrieve
# throughput at ~7 rps and pushing p50 to 17s at 100 concurrent. Each worker loads its
# own copy of both ONNX models — measured ~0.9-1.2 GiB RSS per warmed worker (locally
# and on the production container). Kept at 2, NOT the DEPLOY.md "1 per GB, capped at
# vCPUs" rule of thumb literally (which would suggest 4) — the Hetzner box is 8GB RAM /
# 4 vCPU shared with ~6 other projects' containers, and Brain already runs the single
# largest per-container memory footprint on that box. See TEST_RESULTS.md for the
# measurements this is based on. UNCOMMITTED / pending review — see TEST_RESULTS.md.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--proxy-headers", "--forwarded-allow-ips=*"]
