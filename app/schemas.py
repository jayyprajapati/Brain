"""Request/response models for the Brain API."""
from typing import Any, Literal, Optional

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


class LLMUsage(BaseModel):
    """Token usage as reported by the upstream provider (0 = not reported)."""
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


class GenerateResponse(BaseModel):
    text: str
    json_value: Optional[Any] = Field(default=None, alias="json")
    usage: Optional[LLMUsage] = None

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
    # Optional document title. When set, chunks are embedded and reranked with
    # a "title > heading" breadcrumb (contextual chunk headers).
    title: Optional[str] = Field(default=None, max_length=200)


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
    top_n: Optional[int] = Field(default=None, ge=1, le=30)
    # Dynamic K (opt-in, see pipeline.retrieve): cut the reranked list where
    # relevance falls off. Scores are raw cross-encoder logits.
    min_score: Optional[float] = None
    score_margin: Optional[float] = Field(default=None, ge=0.0)
    min_n: int = Field(default=0, ge=0, le=30)
    # Cap on chunks from any one document (diversity). Omitted = no cap.
    max_per_doc: Optional[int] = Field(default=None, ge=1, le=30)


class RetrievedChunk(BaseModel):
    text: str
    heading: str = ""
    score: float = 0.0
    doc_id: str = ""
    urls: list[str] = Field(default_factory=list)
    title: str = ""
    rerank_score: Optional[float] = None


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
    # When true, client_prompt is the COMPLETE system prompt: Brain skips its own
    # BASE_CHAT_PROMPT / RESPONSE_REMINDER and drops the retrieved notes into a
    # `{{context}}` placeholder (or appends them if the placeholder is absent).
    raw_system: bool = False
    # How many reranked chunks to hand the model (defaults to RERANK_TOP_N).
    top_n: Optional[int] = Field(default=None, ge=1, le=30)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    # Vector-search candidates handed to the reranker (defaults to RETRIEVE_TOP_K).
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    # Optional hint for the query rewriter (who "he"/"it" refers to, the domain).
    # When set, even a first message is rewritten before retrieval.
    query_context: Optional[str] = Field(default=None, max_length=1000)
    # Reasoning effort for reasoning models on Ollama, mapped onto whatever the
    # model accepts (effort string or on/off). Lower = faster first token.
    # Omitted = provider default.
    reasoning: Optional[Literal["off", "low", "medium", "high"]] = None
    # Split a multi-part question into one search query per part and merge the
    # results, so every part gets grounding. Opt-in.
    multi_query: bool = False
    # Dynamic K (opt-in, see pipeline.retrieve): cut the reranked list where
    # relevance falls off. Scores are raw cross-encoder logits.
    min_score: Optional[float] = None
    score_margin: Optional[float] = Field(default=None, ge=0.0)
    min_n: int = Field(default=0, ge=0, le=30)
    # Cap on chunks from any one document (diversity). Omitted = no cap.
    max_per_doc: Optional[int] = Field(default=None, ge=1, le=30)


class HealthResponse(BaseModel):
    status: str = "ok"
    chat_model: str
    embed_model: str
    providers: list[str] = Field(default_factory=list)
