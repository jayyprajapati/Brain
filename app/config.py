"""Centralized settings, loaded from the environment / .env file."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Auth
    brain_api_key: str = "change-me"

    # Ollama Cloud (LLM)
    ollama_api_key: str = ""
    ollama_base_url: str = "https://ollama.com"
    chat_model: str = "gpt-oss:120b"

    # Qdrant Cloud — one collection per app. The collection name is derived from
    # the request's `app_name` (optionally namespaced by this prefix), so each app's
    # vectors stay fully isolated and new apps never collide with existing ones.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_prefix: str = ""

    # Embeddings + reranking (local fastembed)
    embed_model: str = "BAAI/bge-base-en-v1.5"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Retrieval + chunking
    retrieve_top_k: int = 25
    rerank_top_n: int = 5
    # Ingest-time dedup: a candidate chunk whose cosine to an existing chunk in the
    # same namespace meets/exceeds this is treated as a duplicate and skipped.
    dedup_threshold: float = 0.92
    chunk_max_tokens: int = 320
    chunk_min_tokens: int = 80
    semantic_threshold: float = 0.5
    # A chunk must carry at least this many word tokens to be embedded (URLs are
    # always kept). Filters separators / stray punctuation that have no semantic
    # value but would otherwise become their own vectors.
    chunk_min_content_words: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
