"""Hybrid search: LightRAG answer + Milvus vectorstore sources + rerank.

Pipeline
--------
1. ``vectorstore.similarity_search_with_score`` — embeds query + searches
   Milvus (department filter via ``expr``), returns candidate chunks.
2. Rerank with cross-encoder, keep the best ``top_k``.
3. In parallel, call ``LightRAG.aquery`` to generate a natural-language
   answer.
4. Build :class:`SearchResponse` with answer + source citations + latency.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Callable, Literal, Protocol

from loguru import logger

from src.models.search import SearchResponse, SourceCitation

# ── type aliases ──────────────────────────────────────────────────────

SearchMode = Literal["naive", "local", "global", "hybrid"]
RerankerFn = Callable[[str, list[str]], list[float]]


class _RAGLike(Protocol):
    async def aquery(self, query: str, param: Any) -> str: ...


# ── chunk_id → position parser ────────────────────────────────────────

_POSITION_RE = re.compile(r"_(\d+)$")


def _position_from_chunk_id(chunk_id: str) -> int:
    m = _POSITION_RE.search(chunk_id)
    return int(m.group(1)) if m else 0


# ── default reranker ─────────────────────────────────────────────────


def _default_reranker() -> RerankerFn:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "BGE reranker needs sentence-transformers."
        ) from exc

    model = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def _rerank(query: str, candidates: list[str]) -> list[float]:
        pairs = [[query, c] for c in candidates]
        return model.predict(pairs).tolist()

    return _rerank


# ── searcher ──────────────────────────────────────────────────────────


class HybridSearcher:
    """vectorstore → rerank → RAG answer."""

    def __init__(
        self,
        *,
        rag: _RAGLike,
        vectorstore: Any,
        reranker_fn: RerankerFn | None = None,
        candidate_multiplier: int = 3,
    ):
        self._rag = rag
        self._vs = vectorstore
        self._reranker_fn = reranker_fn or _default_reranker()
        self._candidate_multiplier = candidate_multiplier

    async def _ask_rag(self, query: str, mode: SearchMode, top_k: int) -> str:
        from lightrag import QueryParam

        param = QueryParam(mode=mode, top_k=top_k)
        try:
            return await self._rag.aquery(query, param=param)
        except Exception as exc:
            logger.warning("RAG query failed: {err}", err=exc)
            return ""

    async def search(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        department: str | None = None,
        top_k: int = 10,
        user_id: str | None = None,
    ) -> SearchResponse:
        t0 = time.monotonic()

        # 1. vectorstore: embed query + search Milvus
        expr = f'department == "{department}"' if department else None
        fetch_k = top_k * self._candidate_multiplier

        results = await asyncio.to_thread(
            self._vs.similarity_search_with_score,
            query, k=fetch_k, expr=expr,
        )

        # 2. rerank + RAG answer — in parallel
        rag_task = asyncio.create_task(self._ask_rag(query, mode, top_k))
        sources = self._rerank(query, results, top_k)
        answer = await rag_task

        latency_ms = (time.monotonic() - t0) * 1000.0

        logger.info(
            "search  query={q!r}  mode={mode}  dept={dept}  "
            "sources={n}  latency_ms={ms:.1f}",
            q=query, mode=mode, dept=department,
            n=len(sources), ms=latency_ms,
        )
        return SearchResponse(
            query=query, answer=answer, mode=mode,
            sources=sources, latency_ms=latency_ms,
        )

    def _rerank(
        self,
        query: str,
        results: list[tuple],
        top_k: int,
    ) -> list[SourceCitation]:
        if not results:
            return []

        texts = [doc.page_content for doc, _score in results]
        scores = self._reranker_fn(query, texts)
        if asyncio.iscoroutine(scores):
            import asyncio as _aio
            scores = _aio.get_event_loop().run_until_complete(scores)

        ranked = sorted(
            zip(results, scores),
            key=lambda t: t[1],
            reverse=True,
        )[:top_k]

        citations = []
        for (doc, _vs_score), rerank_score in ranked:
            meta = doc.metadata or {}
            chunk_id = meta.get("pk", meta.get("id", ""))
            citations.append(SourceCitation(
                doc_id=meta.get("doc_id", ""),
                chunk_id=str(chunk_id),
                position=_position_from_chunk_id(str(chunk_id)),
                content=doc.page_content,
                score=float(rerank_score),
                department=meta.get("department", ""),
                doc_type=meta.get("doc_type", ""),
            ))
        return citations
