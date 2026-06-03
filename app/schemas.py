"""Request/response models for the Brain API."""
from typing import Any, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class LLMOverride(BaseModel):
    """Per-request BYOK provider override. All fields optional except provider."""
    provider: str = "ollama_cloud"
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None


class GenerateRequest(BaseModel):
    # Pure LLM call — touches no vectors, so app_name is accepted but optional.
    app_name: str = ""
    # Optional system/persona instruction. Falls back to a generic default.
    system: Optional[str] = None
    prompt: str
    # Optional structured data — a JSON object or an already-stringified blob.
    data: Any = None
    llm: Optional[LLMOverride] = None
    # "text" (default) or "json" — when "json", the reply is parsed and returned in `json`.
    response_format: str = "text"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class GenerateResponse(BaseModel):
    text: str
    json_value: Optional[Any] = Field(default=None, alias="json")

    model_config = {"populate_by_name": True}


class PingRequest(BaseModel):
    llm: Optional[LLMOverride] = None


class PingResponse(BaseModel):
    ok: bool = True
    provider: str
    model: str


class IngestRequest(BaseModel):
    # Selects the app's dedicated collection.
    app_name: str
    doc_id: str
    text: str
    # Optional tenant grouping (e.g. a user id) for isolation + dedup scope.
    namespace: Optional[str] = None
    # Skip chunks that near-duplicate existing chunks in the same namespace.
    dedup: bool = False
    # Extra payload fields stored on every chunk (reserved keys are ignored).
    metadata: Optional[dict] = None


class IngestResponse(BaseModel):
    doc_id: str
    chunk_count: int
    skipped_duplicates: int = 0


class ExtractResponse(BaseModel):
    text: str
    char_count: int
    doc_id: Optional[str] = None
    chunk_count: int = 0
    skipped_duplicates: int = 0
    ingested: bool = False


class RetrieveRequest(BaseModel):
    app_name: str
    query: str
    doc_ids: Optional[list[str]] = None
    namespace: Optional[str] = None
    top_k: Optional[int] = None


class RetrievedChunk(BaseModel):
    text: str
    heading: str = ""
    score: float = 0.0
    doc_id: str = ""
    urls: list[str] = Field(default_factory=list)


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class DeleteRequest(BaseModel):
    app_name: str
    doc_id: Optional[str] = None
    namespace: Optional[str] = None


class DeleteResponse(BaseModel):
    ok: bool = True
    deleted: int = 0


class ChatRequest(BaseModel):
    # Selects the app's dedicated collection to retrieve from.
    app_name: str
    messages: list[Message] = Field(default_factory=list)
    # Personality/identity prompt supplied by the caller's admin layer. Required.
    client_prompt: str
    doc_ids: Optional[list[str]] = None
    model: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    chat_model: str
    embed_model: str
    providers: list[str] = Field(default_factory=list)
