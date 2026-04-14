"""Hybrid search: LightRAG answer + vector-DB sources + cross-encoder rerank.

Pipeline
--------
1. Embed the query and call ``AsyncMilvusClient.search`` (department filter)
   to fetch ``top_k × 3`` candidate chunks.
2. Rerank with BGE-reranker-v2-m3 (cross-encoder), keep the best ``top_k``.
3. In parallel, call ``LightRAG.aquery(query, QueryParam(mode=..., top_k=...))``
   to produce the generated answer.
4. Build :class:`SearchResponse` with answer + source citations + latency.

All external collaborators are injected via the constructor so the
pipeline is testable without live Ollama / Neo4j / a downloaded reranker.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Callable, Literal, Protocol

import numpy as np

from src.models.search import SearchResponse, SourceCitation
from src.storage.milvus_client import AsyncMilvusClient, SearchResult

logger = logging.getLogger(__name__)

# ── type aliases ──────────────────────────────────────────────────────

SearchMode = Literal["naive", "local", "global", "hybrid"]
EmbedFn = Callable[[list[str]], np.ndarray]
# (query, candidate_texts) → score per candidate
RerankerFn = Callable[[str, list[str]], list[float]]


class _RAGLike(Protocol):
    async def aquery(self, query: str, param) -> str: ...


# ── LangFuse @observe (no-op if package missing) ──────────────────────

try:  # pragma: no cover
    from langfuse.decorators import observe as _langfuse_observe

    _OBSERVE = _langfuse_observe
except Exception:  # pragma: no cover
    def _OBSERVE(**_kwargs):
        def _no_op(fn):
            return fn
        return _no_op


# ── chunk_id → position parser ────────────────────────────────────────

_POSITION_RE = re.compile(r"_(\d+)$")


def _position_from_chunk_id(chunk_id: str) -> int:
    """Our ingestion worker builds chunk_id as ``{doc_id}_{position}``."""
    m = _POSITION_RE.search(chunk_id)
    return int(m.group(1)) if m else 0


# ── default reranker (lazy import of sentence-transformers) ───────────


def _default_reranker() -> RerankerFn:
    """BGE-reranker-v2-m3 via sentence-transformers CrossEncoder.

    Lazy-loaded the first time the closure is called so importing this
    module is free of model-download side effects.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "BGE reranker needs sentence-transformers.  "
            "Install with `pip install sentence-transformers` or pass a "
            "custom reranker callable to HybridSearcher."
        ) from exc

    model = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def _rerank(query: str, candidates: list[str]) -> list[float]:
        pairs = [[query, c] for c in candidates]
        return model.predict(pairs).tolist()

    return _rerank


# ── searcher ──────────────────────────────────────────────────────────


class HybridSearcher:
    """Orchestrates Milvus candidates → rerank → RAG answer."""

    def __init__(
        self,
        *,
        rag: _RAGLike,
        milvus: AsyncMilvusClient,
        embed_fn: EmbedFn,
        reranker_fn: RerankerFn | None = None,
        candidate_multiplier: int = 3,
    ):
        self._rag = rag
        self._milvus = milvus
        self._embed_fn = embed_fn
        self._reranker_fn = reranker_fn or _default_reranker()
        self._candidate_multiplier = candidate_multiplier

    # ── helpers ───────────────────────────────────────────────────────

    async def _embed_query(self, query: str) -> list[float]:
        vecs = self._embed_fn([query])
        if asyncio.iscoroutine(vecs):
            vecs = await vecs
        arr = np.asarray(vecs, dtype=np.float32)
        return arr[0].tolist()

    async def _ask_rag(self, query: str, mode: SearchMode, top_k: int) -> str:
        from lightrag import QueryParam

        param = QueryParam(mode=mode, top_k=top_k)
        try:
            return await self._rag.aquery(query, param=param)
        except Exception as exc:
            logger.warning("RAG query failed: %s", exc)
            return ""

    async def _rerank_candidates(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SourceCitation]:
        if not candidates:
            return []

        texts = [c.content for c in candidates]
        scores = self._reranker_fn(query, texts)
        if asyncio.iscoroutine(scores):
            scores = await scores

        ranked = sorted(
            zip(candidates, scores),
            key=lambda t: t[1],
            reverse=True,
        )[:top_k]

        return [
            SourceCitation(
                doc_id=sr.doc_id,
                chunk_id=sr.id,
                position=_position_from_chunk_id(sr.id),
                content=sr.content,
                score=float(score),
                department=sr.department,
                doc_type=sr.doc_type,
            )
            for sr, score in ranked
        ]

    # ── public API ────────────────────────────────────────────────────

    @_OBSERVE(name="hybrid_search")
    async def search(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        department: str | None = None,
        top_k: int = 10,
        user_id: str | None = None,
    ) -> SearchResponse:
        t0 = time.monotonic()

        # 1. embed + vector candidates (department filter applied in Milvus)
        query_vec = await self._embed_query(query)
        candidates = await self._milvus.search(
            query_vector=query_vec,
            top_k=top_k * self._candidate_multiplier,
            department=department,
        )

        # 2. rerank + RAG answer — in parallel
        rag_task = asyncio.create_task(self._ask_rag(query, mode, top_k))
        sources = await self._rerank_candidates(query, candidates, top_k)
        answer = await rag_task

        latency_ms = (time.monotonic() - t0) * 1000.0

        logger.info(
            "search  q=%r  mode=%s  dept=%s  user=%s  sources=%d  latency=%.1fms",
            query, mode, department, user_id, len(sources), latency_ms,
        )

        return SearchResponse(
            query=query,
            answer=answer,
            mode=mode,
            sources=sources,
            latency_ms=latency_ms,
        )


# ── convenience: module-level singleton + function ────────────────────

_searcher: HybridSearcher | None = None


def init_searcher(searcher: HybridSearcher) -> None:
    global _searcher
    _searcher = searcher


async def search(
    query: str,
    mode: SearchMode = "hybrid",
    department: str | None = None,
    top_k: int = 10,
    user_id: str | None = None,
) -> SearchResponse:
    if _searcher is None:
        raise RuntimeError(
            "HybridSearcher singleton not initialised.  "
            "Call init_searcher(HybridSearcher(...)) at app startup."
        )
    return await _searcher.search(
        query=query,
        mode=mode,
        department=department,
        top_k=top_k,
        user_id=user_id,
    )
