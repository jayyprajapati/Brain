"""Ollama Cloud client (LLM only). Brain owns the key; callers may override the model."""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .config import settings


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.ollama_api_key}"}


def _model(override: str | None = None) -> str:
    return override or settings.chat_model


async def generate(system: str, user: str, model: str | None = None) -> str:
    """Single-shot, non-streaming completion. Used by /v1/generate."""
    payload = {
        "model": _model(model),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/chat", json=payload, headers=_headers()
        )
        resp.raise_for_status()
        data = resp.json()
    return (data.get("message") or {}).get("content", "")


async def chat_stream(
    system: str, messages: list[dict], model: str | None = None
) -> AsyncIterator[str]:
    """Streaming completion. Yields text deltas as they arrive (NDJSON from Ollama)."""
    payload = {
        "model": _model(model),
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            headers=_headers(),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = (obj.get("message") or {}).get("content", "")
                if delta:
                    yield delta
                if obj.get("done"):
                    break


async def list_models() -> list[str]:
    """Best-effort model listing — used by scripts/check_ollama.py."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{settings.ollama_base_url}/api/tags", headers=_headers()
        )
        resp.raise_for_status()
        data = resp.json()
    return [m.get("name", "") for m in data.get("models", [])]
