"""Brain — client-agnostic RAG service. FastAPI entrypoint."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from . import embeddings, pipeline, reranker, vectorstore
from .auth import require_api_key
from .config import settings
from .schemas import (
    ChatRequest,
    DeleteRequest,
    DeleteResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)

logger = logging.getLogger("brain")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the local models. Qdrant collections are created lazily per app on first
    # ingest, so there's nothing collection-specific to set up here.
    embeddings.warmup()
    reranker.warmup()
    yield


app = FastAPI(title="Brain", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(chat_model=settings.chat_model, embed_model=settings.embed_model)


@app.post("/v1/generate", response_model=GenerateResponse, dependencies=[Depends(require_api_key)])
async def generate(req: GenerateRequest) -> GenerateResponse:
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    data = req.data if isinstance(req.data, str) else json.dumps(req.data, indent=2, default=str)
    text = await pipeline_generate(req.prompt, data)
    return GenerateResponse(text=text)


async def pipeline_generate(prompt: str, data: str) -> str:
    from . import llm

    user = f"Structured data:\n{data}"
    return await llm.generate(prompt, user)


@app.post("/v1/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest(req: IngestRequest) -> IngestResponse:
    if not req.app_name.strip():
        raise HTTPException(status_code=400, detail="app_name is required")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    count = pipeline.ingest(req.app_name, req.doc_id, req.text)
    return IngestResponse(doc_id=req.doc_id, chunk_count=count)


@app.post("/v1/delete", response_model=DeleteResponse, dependencies=[Depends(require_api_key)])
async def delete(req: DeleteRequest) -> DeleteResponse:
    if not req.app_name.strip():
        raise HTTPException(status_code=400, detail="app_name is required")
    deleted = vectorstore.delete(req.app_name, req.doc_id)
    return DeleteResponse(ok=True, deleted=deleted)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/v1/chat", dependencies=[Depends(require_api_key)])
async def chat(req: ChatRequest) -> StreamingResponse:
    if not req.app_name or not req.app_name.strip():
        raise HTTPException(status_code=400, detail="app_name is required")
    # client_prompt is mandatory — no fallback. It always comes from the caller's admin.
    if not req.client_prompt or not req.client_prompt.strip():
        raise HTTPException(status_code=400, detail="client_prompt is required")
    if not any(m.role == "user" and m.content.strip() for m in req.messages):
        raise HTTPException(status_code=400, detail="messages must contain a user message")

    async def event_stream():
        try:
            async for event, data in pipeline.chat(
                app_name=req.app_name,
                messages=req.messages,
                client_prompt=req.client_prompt,
                doc_ids=req.doc_ids,
                model=req.model,
            ):
                yield _sse(event, data)
        except Exception as exc:  # noqa: BLE001 — surface a clean error event
            logger.exception("chat stream failed")
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
