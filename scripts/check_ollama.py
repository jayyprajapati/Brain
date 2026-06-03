"""Verify the Ollama Cloud key works and that CHAT_MODEL is available.

Run: python scripts/check_ollama.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm  # noqa: E402
from app.config import settings  # noqa: E402


async def main() -> int:
    if not settings.ollama_api_key:
        print("✗ OLLAMA_API_KEY is not set in .env")
        return 1
    try:
        models = await llm.list_models()
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Could not reach Ollama Cloud: {exc}")
        return 1

    print(f"Base URL : {settings.ollama_base_url}")
    print(f"Configured CHAT_MODEL : {settings.chat_model}")
    print(f"Available models ({len(models)}):")
    for m in models:
        print(f"  - {m}")

    if any(settings.chat_model == m or settings.chat_model in m for m in models):
        print(f"\n✓ '{settings.chat_model}' looks available.")
        return 0
    print(
        f"\n⚠ '{settings.chat_model}' was not found in the list above. "
        f"Update CHAT_MODEL in .env to one of the available ids "
        f"(watch for a '-cloud' suffix)."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
