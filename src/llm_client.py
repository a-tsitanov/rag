"""Unified LLM client — thin async wrapper over Ollama's chat API.

Kept as a small dataclass so call sites (judge, decomposer, summaries)
don't hand-roll HTTP calls and so tests can swap in a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMClient:
    """Thin wrapper: ``await client.chat(model, messages) → dict``."""

    provider: str = "ollama"  # kept for log output / health diagnostics
    _client: Any = None

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> dict:
        """Send a chat completion and return ``{"message": {"content": ...}}``.

        Shape matches ``ollama.AsyncClient.chat`` — downstream code
        reaches ``resp["message"]["content"]``.
        """
        return await self._client.chat(model=model, messages=messages)

    async def list_models(self) -> list[str]:
        """For health checks — returns available model names."""
        resp = await self._client.list()
        return [m.model for m in (resp.models or [])]
