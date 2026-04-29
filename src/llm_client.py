"""Unified LLM client — thin async wrapper over an OpenAI-compatible API.

Целевой backend — LiteLLM-прокси (порт 4000), который мимикрирует под
OpenAI Chat Completions.  Клиент держит ту же ``chat()`` сигнатуру, что
была у предыдущей Ollama-обёртки, и возвращает словарь формы
``{"message": {"content": ...}}`` — call sites (judge, decomposer,
summary) на это завязаны.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMClient:
    """Thin wrapper: ``await client.chat(model, messages) → dict``."""

    provider: str = "litellm"  # kept for log output / health diagnostics
    _client: Any = None  # openai.AsyncOpenAI

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> dict:
        """Send a chat completion via OpenAI-compatible API.

        Returns ``{"message": {"content": ...}}`` to match the legacy
        Ollama shape — downstream code reaches ``resp["message"]["content"]``.
        """
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = resp.choices[0].message.content or ""
        return {"message": {"content": content}}

    async def list_models(self) -> list[str]:
        """For health checks — returns available model names."""
        resp = await self._client.models.list()
        return [m.id for m in (resp.data or [])]
