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

import asyncio
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


def _usage(input_tokens, output_tokens, cached_input_tokens=None) -> dict:
    """Normalized token usage as reported by the provider (0 when it didn't say)."""
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "cached_input_tokens": int(cached_input_tokens or 0),
    }


# ── Per-provider request builders ────────────────────────────────────────────


# Ollama Cloud's free tier answers "429 too many concurrent requests" as soon
# as a few chats overlap. Those are worth a short wait and retry, and nothing
# has reached the caller yet when they happen.
_RETRY_STATUSES = {429, 503}
_RETRY_DELAYS = (1.5, 3.0)


async def _post(url: str, *, headers: dict, payload: dict, provider: str) -> dict:
    for delay in (*_RETRY_DELAYS, None):
        resp = await _post_once(url, headers=headers, payload=payload, provider=provider)
        if resp.status_code not in _RETRY_STATUSES or delay is None:
            break
        await asyncio.sleep(delay)
    return _parse_response(resp, provider)


async def _post_once(url: str, *, headers: dict, payload: dict, provider: str) -> httpx.Response:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            return await client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise LLMError(f"{provider} request timed out", status=504, provider=provider) from exc
    except httpx.RequestError as exc:
        # Connection refused / DNS / TLS — no HTTP response was received.
        raise LLMError(f"could not reach {provider}: {exc}", provider=provider) from exc


def _parse_response(resp: httpx.Response, provider: str) -> dict:
    if resp.status_code >= 400:
        detail = resp.text
        try:
            body = resp.json()
            detail = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("error") or body.get("detail") or detail
        except Exception:  # noqa: BLE001
            pass
        raise LLMError(f"{provider} request failed ({resp.status_code}): {detail}", status=resp.status_code, provider=provider)
    return resp.json()


async def _openai_generate(cfg, system, user, model, response_format, max_tokens, temperature) -> tuple[str, dict]:
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
    u = data.get("usage") or {}
    usage = _usage(u.get("prompt_tokens"), u.get("completion_tokens"), (u.get("prompt_tokens_details") or {}).get("cached_tokens"))
    return (choices[0].get("message") or {}).get("content", "") or "", usage


async def _anthropic_generate(cfg, system, user, model, response_format, max_tokens, temperature) -> tuple[str, dict]:
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
    u = data.get("usage") or {}
    usage = _usage(
        (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0),
        u.get("output_tokens"),
        u.get("cache_read_input_tokens"),
    )
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict)), usage


def _ollama_payload(model, system, user, response_format, max_tokens, temperature, stream, think=None) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": stream,
    }
    if think is not None:
        payload["think"] = think
    if response_format == "json":
        payload["format"] = "json"
    # Keep the model warm between requests to cut Ollama Cloud time-to-first-token.
    payload["keep_alive"] = settings.ollama_keep_alive
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


# Ollama models expose reasoning differently: gpt-oss and beam take an effort
# string ("low"/"medium"/...), nemotron and gemma take true/false, others take
# nothing. Callers ask for an effort; this maps it onto what the model accepts
# (read once per model from /api/show) so switching CHAT_MODEL never sends a
# value the provider rejects.
_THINK_VALUES: dict[str, list | None] = {}
_EFFORT_ORDER = ("off", "low", "medium", "high", "xhigh", "max")


async def _thinking_values(cfg: dict, model: str) -> list | None:
    key = f"{cfg['provider']}|{cfg['base_url']}|{model}"
    if key not in _THINK_VALUES:
        base, headers = _ollama_target(cfg)
        values = None
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{base}/api/show", json={"model": model}, headers=headers)
                if resp.status_code < 400:
                    data = resp.json()
                    if "thinking" in (data.get("capabilities") or []):
                        values = (data.get("thinking") or {}).get("values") or [True, False]
                    else:
                        values = []
        except (httpx.HTTPError, ValueError):
            return None  # unknown; retry on the next call
        _THINK_VALUES[key] = values
    return _THINK_VALUES[key]


async def _resolve_think(cfg: dict, model: str, think: str | None):
    """Map a requested effort onto the model's own ``think`` values (None = omit)."""
    if not think:
        return None
    values = await _thinking_values(cfg, model)
    if values is None:
        # /api/show unavailable: only gpt-oss is known to take effort strings.
        return think if model.startswith("gpt-oss") and think != "off" else None
    if not values:
        return None
    if all(isinstance(v, bool) for v in values):
        # On/off models: low effort means no thinking (rewrite, short answers).
        return think in ("medium", "high", "xhigh", "max")
    if think == "off":
        return values[0] if values else None
    if think in values:
        return think
    # Nearest supported effort at or below the request.
    wanted = _EFFORT_ORDER.index(think) if think in _EFFORT_ORDER else 2
    below = [v for v in values if v in _EFFORT_ORDER and _EFFORT_ORDER.index(v) <= wanted]
    return below[-1] if below else values[0]


async def _ollama_generate(cfg, system, user, model, response_format, max_tokens, temperature, think=None) -> tuple[str, dict]:
    base, headers = _ollama_target(cfg)
    headers = {**headers, "Content-Type": "application/json"}
    think = await _resolve_think(cfg, model, think)
    payload = _ollama_payload(model, system, user, response_format, max_tokens, temperature, stream=False, think=think)
    data = await _post(f"{base}/api/chat", headers=headers, payload=payload, provider=cfg["provider"])
    usage = _usage(data.get("prompt_eval_count"), data.get("eval_count"))
    return (data.get("message") or {}).get("content", "") or "", usage


# ── Public API ───────────────────────────────────────────────────────────────


async def generate_with_usage(
    system: str,
    user: str,
    *,
    model: str | None = None,
    llm: dict | None = None,
    response_format: str = "text",
    max_tokens: int | None = None,
    temperature: float | None = None,
    think: str | None = None,
) -> tuple[str, dict]:
    """Single-shot, non-streaming completion across any supported provider.

    Returns ``(text, usage)`` where usage is ``{provider, model, input_tokens,
    output_tokens, cached_input_tokens}`` as reported by the provider.
    ``model`` is a convenience override that wins over ``llm['model']``."""
    cfg = _normalize(llm)
    provider = cfg["provider"]
    if provider not in SUPPORTED_PROVIDERS:
        raise LLMError(f"unsupported provider '{provider}'", status=400, provider=provider)
    resolved_model = _resolve_model(cfg, model)
    system = _json_nudge(system, response_format)

    if provider == "openai":
        fn = _openai_generate
    elif provider == "anthropic":
        fn = _anthropic_generate
    else:
        fn = _ollama_generate
    # ``think`` (reasoning effort) only means something to Ollama reasoning models.
    extra = {"think": think} if think and fn is _ollama_generate else {}
    text, usage = await fn(cfg, system, user, resolved_model, response_format, max_tokens, temperature, **extra)
    return text, {"provider": provider, "model": resolved_model, **usage}


async def generate(
    system: str,
    user: str,
    *,
    model: str | None = None,
    llm: dict | None = None,
    response_format: str = "text",
    max_tokens: int | None = None,
    temperature: float | None = None,
    think: str | None = None,
) -> str:
    """Text-only convenience wrapper over :func:`generate_with_usage`."""
    text, _ = await generate_with_usage(
        system, user, model=model, llm=llm, response_format=response_format,
        max_tokens=max_tokens, temperature=temperature, think=think,
    )
    return text


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
    system: str,
    messages: list[dict],
    model: str | None = None,
    llm: dict | None = None,
    temperature: float | None = None,
    think: str | None = None,
) -> AsyncIterator[str]:
    """Streaming completion (NDJSON deltas). Used by /v1/chat (Portfolio).

    Defaults to Brain's Ollama Cloud; honours an ``llm`` override for BYOK Ollama.
    ``think`` is a reasoning effort ("off"/"low"/"medium"/"high"), mapped per model."""
    cfg = _normalize(llm)
    base, headers = _ollama_target(cfg)
    headers = {**headers, "Content-Type": "application/json"}
    resolved_model = _resolve_model(cfg, model)
    payload = {
        "model": resolved_model,
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": True,
        # Keep the model warm (top-level) and pin a grounded generation temperature.
        "keep_alive": settings.ollama_keep_alive,
        "options": {"temperature": settings.chat_temperature if temperature is None else temperature},
    }
    think = await _resolve_think(cfg, resolved_model, think)
    if think is not None:
        payload["think"] = think
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for delay in (*_RETRY_DELAYS, None):
            async with client.stream("POST", f"{base}/api/chat", json=payload, headers=headers) as resp:
                if resp.status_code in _RETRY_STATUSES and delay is not None:
                    await resp.aread()
                    await asyncio.sleep(delay)
                    continue
                async for delta in _stream_deltas(resp, cfg):
                    yield delta
                return


async def _stream_deltas(resp: httpx.Response, cfg: dict) -> AsyncIterator[str]:
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
