"""Multi-provider LLM client.

Brain owns its own default keys (Ollama Cloud) for callers like Portfolio, but
every call accepts an optional per-request ``llm`` override so callers can bring
their own key (BYOK). Supported providers:

  - ``openai``        → OpenAI Chat Completions
  - ``anthropic``     → Anthropic Messages
  - ``ollama_cloud``  → Ollama Cloud (Brain's key by default; override allowed)
  - ``ollama_local``  → a user-hosted Ollama (``base_url``, no key)

The override shape is ``{provider, api_key?, model?, base_url?}``. Provider keys
are never read from Brain's environment except the built-in Ollama Cloud default.
"""
from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator, Optional

import httpx

from .config import settings

_TIMEOUT = 300.0
DEFAULT_MAX_TOKENS = 2048

# Sensible default model per provider when the caller doesn't pin one.
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
    "ollama_local": "llama3.1",
}

SUPPORTED_PROVIDERS = ("openai", "anthropic", "ollama_cloud", "ollama_local")


class LLMError(Exception):
    """Raised when an upstream LLM provider call fails."""

    def __init__(self, message: str, *, status: int | None = None, provider: str | None = None):
        super().__init__(message)
        self.status = status
        self.provider = provider


def _normalize(llm: dict | None) -> dict:
    """Coerce a caller's ``llm`` override into a complete, defaulted dict."""
    llm = dict(llm or {})
    provider = (llm.get("provider") or "ollama_cloud").strip().lower()
    return {
        "provider": provider,
        "api_key": (llm.get("api_key") or "").strip(),
        "model": (llm.get("model") or "").strip(),
        "base_url": (llm.get("base_url") or "").rstrip("/"),
    }


def _resolve_model(cfg: dict, override: str | None) -> str:
    if override:
        return override
    if cfg["model"]:
        return cfg["model"]
    if cfg["provider"] == "ollama_cloud":
        return settings.chat_model
    return _DEFAULT_MODELS.get(cfg["provider"], settings.chat_model)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json(text: str) -> Any:
    """Best-effort parse of a model's reply into JSON.

    Tolerates markdown fences and leading/trailing prose by falling back to the
    outermost ``{...}`` / ``[...]`` span."""
    raw = (text or "").strip()
    if not raw:
        raise LLMError("model returned an empty response")
    cleaned = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced-looking object/array span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError("model did not return valid JSON")


def _json_nudge(system: str, response_format: str) -> str:
    """Ensure a JSON-mode prompt actually mentions JSON (OpenAI requires it)."""
    if response_format != "json":
        return system
    if "json" in (system or "").lower():
        return system
    suffix = "Respond with a single valid JSON value and nothing else."
    return f"{system}\n\n{suffix}" if system else suffix


# ── Per-provider request builders ────────────────────────────────────────────


async def _post(url: str, *, headers: dict, payload: dict, provider: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise LLMError(f"{provider} request timed out", status=504, provider=provider) from exc
    except httpx.RequestError as exc:
        # Connection refused / DNS / TLS — no HTTP response was received.
        raise LLMError(f"could not reach {provider}: {exc}", provider=provider) from exc
    if resp.status_code >= 400:
        detail = resp.text
        try:
            body = resp.json()
            detail = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("error") or body.get("detail") or detail
        except Exception:  # noqa: BLE001
            pass
        raise LLMError(f"{provider} request failed ({resp.status_code}): {detail}", status=resp.status_code, provider=provider)
    return resp.json()


async def _openai_generate(cfg, system, user, model, response_format, max_tokens, temperature) -> str:
    if not cfg["api_key"]:
        raise LLMError("openai requires an api_key", status=400, provider="openai")
    base = cfg["base_url"] or "https://api.openai.com"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}
    data = await _post(
        f"{base}/v1/chat/completions",
        headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
        payload=payload,
        provider="openai",
    )
    choices = data.get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content", "") or ""


async def _anthropic_generate(cfg, system, user, model, response_format, max_tokens, temperature) -> str:
    if not cfg["api_key"]:
        raise LLMError("anthropic requires an api_key", status=400, provider="anthropic")
    base = cfg["base_url"] or "https://api.anthropic.com"
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,  # Anthropic requires max_tokens
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    data = await _post(
        f"{base}/v1/messages",
        headers={
            "x-api-key": cfg["api_key"],
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        payload=payload,
        provider="anthropic",
    )
    parts = data.get("content") or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


def _ollama_payload(model, system, user, response_format, max_tokens, temperature, stream) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": stream,
    }
    if response_format == "json":
        payload["format"] = "json"
    options: dict[str, Any] = {}
    if max_tokens:
        options["num_predict"] = max_tokens
    if temperature is not None:
        options["temperature"] = temperature
    if options:
        payload["options"] = options
    return payload


def _ollama_target(cfg) -> tuple[str, dict]:
    if cfg["provider"] == "ollama_local":
        base = cfg["base_url"] or "http://localhost:11434"
        return base, {}
    # ollama_cloud — BYOK key if supplied, else Brain's own.
    base = cfg["base_url"] or settings.ollama_base_url
    key = cfg["api_key"] or settings.ollama_api_key
    return base, ({"Authorization": f"Bearer {key}"} if key else {})


async def _ollama_generate(cfg, system, user, model, response_format, max_tokens, temperature) -> str:
    base, headers = _ollama_target(cfg)
    headers = {**headers, "Content-Type": "application/json"}
    payload = _ollama_payload(model, system, user, response_format, max_tokens, temperature, stream=False)
    data = await _post(f"{base}/api/chat", headers=headers, payload=payload, provider=cfg["provider"])
    return (data.get("message") or {}).get("content", "") or ""


# ── Public API ───────────────────────────────────────────────────────────────


async def generate(
    system: str,
    user: str,
    *,
    model: str | None = None,
    llm: dict | None = None,
    response_format: str = "text",
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Single-shot, non-streaming completion across any supported provider.

    ``model`` is a convenience override that wins over ``llm['model']``."""
    cfg = _normalize(llm)
    provider = cfg["provider"]
    if provider not in SUPPORTED_PROVIDERS:
        raise LLMError(f"unsupported provider '{provider}'", status=400, provider=provider)
    resolved_model = _resolve_model(cfg, model)
    system = _json_nudge(system, response_format)

    if provider == "openai":
        return await _openai_generate(cfg, system, user, resolved_model, response_format, max_tokens, temperature)
    if provider == "anthropic":
        return await _anthropic_generate(cfg, system, user, resolved_model, response_format, max_tokens, temperature)
    return await _ollama_generate(cfg, system, user, resolved_model, response_format, max_tokens, temperature)


async def ping(llm: dict | None = None) -> dict:
    """Cheap liveness/credentials check — a single-token generation.

    Returns ``{provider, model}`` on success; raises ``LLMError`` otherwise."""
    cfg = _normalize(llm)
    if cfg["provider"] not in SUPPORTED_PROVIDERS:
        raise LLMError(f"unsupported provider '{cfg['provider']}'", status=400, provider=cfg["provider"])
    model = _resolve_model(cfg, None)
    await generate(
        "You are a connectivity probe. Reply with the single word OK.",
        "ping",
        llm=llm,
        max_tokens=5,
    )
    return {"provider": cfg["provider"], "model": model}


async def chat_stream(
    system: str, messages: list[dict], model: str | None = None, llm: dict | None = None
) -> AsyncIterator[str]:
    """Streaming completion (NDJSON deltas). Used by /v1/chat (Portfolio).

    Defaults to Brain's Ollama Cloud; honours an ``llm`` override for BYOK Ollama."""
    cfg = _normalize(llm)
    base, headers = _ollama_target(cfg)
    headers = {**headers, "Content-Type": "application/json"}
    resolved_model = _resolve_model(cfg, model)
    payload = {
        "model": resolved_model,
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        async with client.stream("POST", f"{base}/api/chat", json=payload, headers=headers) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise LLMError(
                    f"{cfg['provider']} stream failed ({resp.status_code}): {body.decode(errors='ignore')}",
                    status=resp.status_code,
                    provider=cfg["provider"],
                )
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
    """Best-effort model listing for Brain's default Ollama Cloud (scripts/check_ollama.py)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{settings.ollama_base_url}/api/tags",
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()
    return [m.get("name", "") for m in data.get("models", [])]
