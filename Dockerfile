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
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
