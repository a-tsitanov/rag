"""Query decomposition + RRF merge (Phase 2b).

``decompose_query`` asks the LLM to break a complex query into 1-3
independent sub-queries. Each sub-query is searched separately, and
results are merged via Reciprocal Rank Fusion (``rrf_merge``).
"""

from __future__ import annotations

import json

from loguru import logger

from src.config import settings
from src.llm_client import LLMClient

_DECOMPOSE_PROMPT = (
    "You are a search query decomposer. "
    "Break the user's query into 1-3 independent sub-queries that together "
    "cover the full intent. Each sub-query should be a standalone search query. "
    "Return ONLY a JSON array of strings, no explanation.\n"
    "Example: [\"sub-query 1\", \"sub-query 2\"]"
)


async def decompose_query(
    query: str,
    client: LLMClient,
) -> list[str]:
    """Ask LLM to split *query* into 1-3 sub-queries.

    Returns the original query as a single-element list on any failure.
    """
    try:
        resp = await client.chat(
            model=settings.effective_llm_model,
            messages=[
                {"role": "system", "content": _DECOMPOSE_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        raw = resp["message"]["content"].strip()
        # Strip markdown fences if LLM wraps response
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        if isinstance(result, list) and all(isinstance(s, str) for s in result):
            return result[:3]
    except Exception as exc:
        logger.warning("query decomposition failed: {err}", err=exc)
    return [query]


def rrf_merge(
    results_lists: list[list[tuple]],
    k: int = 60,
) -> list[tuple]:
    """Reciprocal Rank Fusion across multiple result lists.

    Each result list contains ``(Document, score)`` tuples from
    ``vectorstore.similarity_search_with_score``.

    Returns a single merged list sorted by RRF score (descending).
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, tuple] = {}

    for results in results_lists:
        for rank, pair in enumerate(results):
            doc, _vs_score = pair
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = pair

    ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in ranked_keys]
