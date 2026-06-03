"""Request/response models for the Brain API."""
from typing import Any, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    # Pure LLM call — touches no vectors, so app_name is accepted but optional.
    app_name: str = ""
    prompt: str
    # Structured data — a JSON object or an already-stringified blob.
    data: Any


class GenerateResponse(BaseModel):
    text: str


class IngestRequest(BaseModel):
    # Selects the app's dedicated collection.
    app_name: str
    doc_id: str
    text: str


class IngestResponse(BaseModel):
    doc_id: str
    chunk_count: int


class DeleteRequest(BaseModel):
    app_name: str
    doc_id: str


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
