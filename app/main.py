"""Brain — client-agnostic RAG + LLM service. FastAPI entrypoint."""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from . import embeddings, extract as extractor, llm, pipeline, reranker, vectorstore
from .auth import require_api_key
from .config import settings
from .schemas import (
    ChatRequest,
    DeleteRequest,
    DeleteResponse,
    ExtractResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    PingRequest,
    PingResponse,
    RetrievedChunk,
    RetrieveRequest,
    RetrieveResponse,
)

logger = logging.getLogger("brain")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the local models. Qdrant collections are created lazily per app on first
    # ingest, so there's nothing collection-specific to set up here.
    embeddings.warmup()
    reranker.warmup()
    yield


app = FastAPI(
    title="Brain",
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def _llm_dict(req_llm) -> dict | None:
    return req_llm.model_dump(exclude_none=True) if req_llm is not None else None


def _llm_http_error(exc: llm.LLMError) -> HTTPException:
    # Client-side credential/config problems surface as 4xx; upstream faults as 502.
    status = exc.status if exc.status and 400 <= exc.status < 500 else 502
    return HTTPException(status_code=status, detail=str(exc))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        chat_model=settings.chat_model,
        embed_model=settings.embed_model,
        providers=list(llm.SUPPORTED_PROVIDERS),
    )


@app.post("/v1/generate", response_model=GenerateResponse, dependencies=[Depends(require_api_key)])
async def generate(req: GenerateRequest) -> GenerateResponse:
    if not req.prompt.strip() and not (req.system and req.system.strip()):
        raise HTTPException(status_code=400, detail="prompt is required")

    data_text = None
    if req.data is not None:
        data_text = req.data if isinstance(req.data, str) else json.dumps(req.data, indent=2, default=str)

    if req.system is not None:
        system = req.system
        user = req.prompt
        if data_text is not None:
            user = f"{user}\n\n{data_text}" if user.strip() else data_text
    else:
        # Backwards-compatible path (Portfolio): prompt acts as the system instruction.
        system = req.prompt
        user = f"Structured data:\n{data_text}" if data_text is not None else "."

    try:
        text = await llm.generate(
            system,
            user,
            llm=_llm_dict(req.llm),
            response_format=req.response_format,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except llm.LLMError as exc:
        raise _llm_http_error(exc)

    if req.response_format == "json":
        try:
            parsed = llm.parse_json(text)
        except llm.LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return GenerateResponse(text=text, **{"json": parsed})
    return GenerateResponse(text=text)


@app.post("/v1/llm/ping", response_model=PingResponse, dependencies=[Depends(require_api_key)])
async def llm_ping(req: PingRequest) -> PingResponse:
    try:
        info = await llm.ping(_llm_dict(req.llm))
    except llm.LLMError as exc:
        raise _llm_http_error(exc)
    return PingResponse(ok=True, provider=info["provider"], model=info["model"])


@app.post("/v1/extract", response_model=ExtractResponse, dependencies=[Depends(require_api_key)])
async def extract(
    file: UploadFile = File(...),
    app_name: str = Form(""),
    doc_id: str = Form(""),
    namespace: str = Form(""),
    ingest: bool = Form(False),
    dedup: bool = Form(False),
) -> ExtractResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        text = extractor.extract_text(data, filename=file.filename or "", content_type=file.content_type or "")
    except extractor.ExtractionError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    result = ExtractResponse(text=text, char_count=len(text))
    if ingest:
        if not app_name.strip():
            raise HTTPException(status_code=400, detail="app_name is required to ingest")
        did = doc_id or str(uuid.uuid4())
        ing = pipeline.ingest(app_name, did, text, namespace=namespace or None, dedup=dedup)
        result.doc_id = did
        result.chunk_count = ing["chunk_count"]
        result.skipped_duplicates = ing["skipped_duplicates"]
        result.ingested = True
    return result


@app.post("/v1/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest(req: IngestRequest) -> IngestResponse:
    if not req.app_name.strip():
        raise HTTPException(status_code=400, detail="app_name is required")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    result = pipeline.ingest(
        req.app_name,
        req.doc_id,
        req.text,
        namespace=req.namespace,
        dedup=req.dedup,
        metadata=req.metadata,
    )
    return IngestResponse(
        doc_id=req.doc_id,
        chunk_count=result["chunk_count"],
        skipped_duplicates=result["skipped_duplicates"],
    )


@app.post("/v1/retrieve", response_model=RetrieveResponse, dependencies=[Depends(require_api_key)])
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    if not req.app_name.strip():
        raise HTTPException(status_code=400, detail="app_name is required")
    hits = pipeline.retrieve(
        req.app_name, req.query, req.doc_ids, namespace=req.namespace, top_k=req.top_k
    )
    return RetrieveResponse(
        chunks=[
            RetrievedChunk(text=h.text, heading=h.heading, score=h.score, doc_id=h.doc_id, urls=h.urls)
            for h in hits
        ]
    )


@app.post("/v1/delete", response_model=DeleteResponse, dependencies=[Depends(require_api_key)])
async def delete(req: DeleteRequest) -> DeleteResponse:
    if not req.app_name.strip():
        raise HTTPException(status_code=400, detail="app_name is required")
    if not req.doc_id and not req.namespace:
        raise HTTPException(status_code=400, detail="doc_id or namespace is required")
    deleted = vectorstore.delete(req.app_name, doc_id=req.doc_id, namespace=req.namespace)
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
