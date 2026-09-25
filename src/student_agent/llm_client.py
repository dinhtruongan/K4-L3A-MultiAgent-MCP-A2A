"""LLM client wrapper cho model < 10B tham số."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Đảm bảo .env đã được load trước khi đọc biến môi trường
_env_path = Path(__file__).resolve().parents[2].parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)


def _build_client() -> AsyncOpenAI:
    """Khởi tạo OpenAI-compatible client dựa trên LLM_PROVIDER."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    print(f"[LLM] Provider={provider}, Model={os.getenv('LLM_MODEL', 'N/A')}")

    if provider == "groq":
        return AsyncOpenAI(
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        )
    elif provider == "ollama":
        return AsyncOpenAI(
            api_key="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    elif provider == "together":
        return AsyncOpenAI(
            api_key=os.getenv("TOGETHER_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
        )
    elif provider == "openai":
        return AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def get_model() -> str:
    return os.getenv("LLM_MODEL", "llama-3.1-8b-instant")


async def ask_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    json_mode: bool = True,
) -> str:
    """Gọi LLM và trả về response text."""
    client = get_client()
    model = get_model()

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or "{}"


def parse_json_safe(text: str) -> dict[str, Any]:
    """Parse JSON từ LLM output, xử lý cả trường hợp LLM bọc trong markdown."""
    text = text.strip()
    # Xử lý trường hợp LLM wrap trong ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
