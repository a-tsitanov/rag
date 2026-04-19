"""Hybrid search: LightRAG answer + Milvus vectorstore sources + rerank.

Pipeline
--------
1. **Stage 1 (optional)** — pre-filter by PG ``documents.summary ILIKE``
   to narrow the search space to relevant doc_ids.
2. Build Milvus ``expr`` from metadata filters (department, doc_type,
   created_after/before) + doc_id whitelist from stage 1.
3. ``vectorstore.similarity_search_with_score`` — embeds query + searches
   Milvus with the ``expr``, returns candidate chunks.
4. Rerank with cross-encoder, keep the best ``top_k``.
5. In parallel, call ``LightRAG.aquery`` with full QueryParam knobs
   to generate a natural-language answer.
6. Build :class:`SearchResponse` with answer + source citations + latency.

When ``decompose=True`` (Phase 2b), the query is split into sub-queries
by LLM; each sub-query runs through steps 1-4 independently, and results
are merged via Reciprocal Rank Fusion before step 5.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Callable, Literal, Protocol

import psycopg
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from loguru import logger

from src.llm_client import LLMClient
from src.models.search import SearchResponse, SourceCitation
from src.retrieval.query_decomposer import decompose_query, rrf_merge
from src.storage.sparse_encoder import SparseEncoder

# ── type aliases ──────────────────────────────────────────────────────

SearchMode = Literal["naive", "local", "global", "hybrid", "mix", "bypass"]
RerankerFn = Callable[[str, list[str]], list[float]]


class _RAGLike(Protocol):
    async def aquery(self, query: str, param: Any) -> str: ...


# ── helpers ───────────────────────────────────────────────────────────

_POSITION_RE = re.compile(r"_(\d+)$")


def _position_from_chunk_id(chunk_id: str) -> int:
    m = _POSITION_RE.search(chunk_id)
    return int(m.group(1)) if m else 0


def _build_expr(
    *,
    department: str | None = None,
    doc_type_filter: str | None = None,
    created_after: int | None = None,
    created_before: int | None = None,
) -> str | None:
    parts: list[str] = []
    if department:
        parts.append(f'department == "{department}"')
    if doc_type_filter:
        parts.append(f'doc_type == "{doc_type_filter}"')
    if created_after is not None:
        parts.append(f"created_at >= {created_after}")
    if created_before is not None:
        parts.append(f"created_at <= {created_before}")
    return " and ".join(parts) if parts else None


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
        pg: psycopg.AsyncConnection | None = None,
        llm_client: LLMClient | None = None,
        embeddings: Embeddings | None = None,
        sparse_encoder: SparseEncoder | None = None,
        milvus_uri: str | None = None,
        collection_name: str | None = None,
        candidate_multiplier: int = 3,
        two_stage_limit: int = 20,
    ):
        self._rag = rag
        self._vs = vectorstore
        self._reranker_fn = reranker_fn or _default_reranker()
        self._pg = pg
        self._llm_client = llm_client
        self._embeddings = embeddings
        self._sparse_encoder = sparse_encoder
        self._milvus_uri = milvus_uri
        self._collection_name = collection_name
        self._candidate_multiplier = candidate_multiplier
        self._two_stage_limit = two_stage_limit

    # ── LightRAG: graph data without LLM (Phase 4) ────────────────────

    async def query_graph_data(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        *,
        top_k: int = 10,
        chunk_top_k: int = 20,
        max_entity_tokens: int = 6000,
        max_relation_tokens: int = 8000,
        max_total_tokens: int = 30000,
    ) -> dict:
        """Get entities, relations, chunks from KG without LLM call."""
        from lightrag import QueryParam

        param = QueryParam(
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            max_entity_tokens=max_entity_tokens,
            max_relation_tokens=max_relation_tokens,
            max_total_tokens=max_total_tokens,
        )
        try:
            return await self._rag.aquery_data(query, param=param)
        except Exception as exc:
            logger.warning("graph data query failed: {err}", err=exc)
            return {}

    # ── LightRAG call with full knobs ────────────────────────────────

    async def _ask_rag(
        self,
        query: str,
        mode: SearchMode,
        *,
        top_k: int = 10,
        chunk_top_k: int = 20,
        max_entity_tokens: int = 6000,
        max_relation_tokens: int = 8000,
        max_total_tokens: int = 30000,
        response_type: str = "Multiple Paragraphs",
        include_references: bool = False,
    ) -> str:
        from lightrag import QueryParam

        param = QueryParam(
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            max_entity_tokens=max_entity_tokens,
            max_relation_tokens=max_relation_tokens,
            max_total_tokens=max_total_tokens,
            response_type=response_type,
            include_references=include_references,
        )
        try:
            return await self._rag.aquery(query, param=param)
        except Exception as exc:
            logger.warning("RAG query failed: {err}", err=exc)
            return ""

    # ── two-stage: PG summary pre-filter (Phase 2a) ───────────────────

    async def _find_relevant_doc_ids(self, query: str) -> list[str]:
        """Stage 1: find doc_ids whose summary matches the query (ILIKE).

        Returns empty list when PG is unavailable or no matches found.
        """
        if not self._pg:
            return []
        try:
            cur = await self._pg.execute(
                "SELECT id::text FROM documents "
                "WHERE status = 'completed' AND summary IS NOT NULL "
                "AND summary ILIKE %(pattern)s "
                "LIMIT %(lim)s",
                {"pattern": f"%{query}%", "lim": self._two_stage_limit},
            )
            rows = await cur.fetchall()
            return [r[0] for r in rows]
        except Exception as exc:
            logger.warning("two-stage PG lookup failed: {err}", err=exc)
            return []

    # ── single-query vector search (used by search + decompose) ──────

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        expr: str | None,
    ) -> list[tuple]:
        """Embed query + search Milvus, return raw (doc, score) pairs."""
        fetch_k = top_k * self._candidate_multiplier

        # Phase 3: dense+sparse hybrid when sparse encoder is available
        if self._sparse_encoder and self._embeddings and self._milvus_uri:
            return await self._hybrid_vector_search(query, fetch_k, expr)

        return await asyncio.to_thread(
            self._vs.similarity_search_with_score,
            query, k=fetch_k, expr=expr,
        )

    # ── Phase 3: dense + sparse hybrid search via pymilvus ───────────

    async def _hybrid_vector_search(
        self,
        query: str,
        limit: int,
        expr: str | None,
    ) -> list[tuple]:
        """Hybrid search: dense HNSW + sparse BM25 via pymilvus RRFRanker."""
        from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

        # Compute both vectors
        dense_vec = await asyncio.to_thread(
            self._embeddings.embed_query, query,
        )
        sparse_vec = self._sparse_encoder.encode_query(query)

        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit,
            expr=expr or "",
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse_embedding",
            param={"metric_type": "IP"},
            limit=limit,
            expr=expr or "",
        )

        def _search() -> list[tuple]:
            client = MilvusClient(uri=self._milvus_uri)
            try:
                raw = client.hybrid_search(
                    collection_name=self._collection_name,
                    reqs=[dense_req, sparse_req],
                    ranker=RRFRanker(),
                    limit=limit,
                    output_fields=[
                        "content", "doc_id", "department",
                        "doc_type", "created_at",
                    ],
                )
            finally:
                client.close()

            # Convert pymilvus results → (LCDocument, score) tuples
            results = []
            for hit in raw[0] if raw else []:
                entity = hit.get("entity", {})
                doc = LCDocument(
                    page_content=entity.get("content", ""),
                    metadata={
                        "doc_id": entity.get("doc_id", ""),
                        "department": entity.get("department", ""),
                        "doc_type": entity.get("doc_type", ""),
                        "created_at": entity.get("created_at", 0),
                        "id": hit.get("id", ""),
                    },
                )
                results.append((doc, float(hit.get("distance", 0.0))))
            return results

        return await asyncio.to_thread(_search)

    # ── public API ────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        *,
        department: str | None = None,
        top_k: int = 10,
        user_id: str | None = None,
        # metadata filters (Phase 1a)
        doc_type_filter: str | None = None,
        created_after: int | None = None,
        created_before: int | None = None,
        # LightRAG knobs (Phase 1b)
        chunk_top_k: int = 20,
        max_entity_tokens: int = 6000,
        max_relation_tokens: int = 8000,
        max_total_tokens: int = 30000,
        response_type: str = "Multiple Paragraphs",
        include_references: bool = False,
        # Phase 2b
        decompose: bool = False,
        # Phase 4: agentic rounds skip RAG LLM call
        skip_rag: bool = False,
    ) -> SearchResponse:
        t0 = time.monotonic()
        sub_queries: list[str] = []

        # ── Phase 2b: query decomposition ────────────────────────────
        if decompose and self._llm_client:
            sub_queries = await decompose_query(query, self._llm_client)
            logger.info(
                "decomposed  query={q!r}  sub_queries={subs}",
                q=query, subs=sub_queries,
            )

        queries_to_search = sub_queries if sub_queries else [query]

        # ── Phase 2a: two-stage pre-filter (non-naive modes) ─────────
        doc_id_whitelist: list[str] = []
        if mode != "naive" and self._pg:
            doc_id_whitelist = await self._find_relevant_doc_ids(query)
            if doc_id_whitelist:
                logger.info(
                    "two-stage  matched_docs={n}",
                    n=len(doc_id_whitelist),
                )

        # 1. build base filter expression
        base_expr = _build_expr(
            department=department,
            doc_type_filter=doc_type_filter,
            created_after=created_after,
            created_before=created_before,
        )

        # add doc_id whitelist from stage-1
        if doc_id_whitelist:
            ids_literal = ", ".join(f'"{d}"' for d in doc_id_whitelist)
            doc_filter = f"doc_id in [{ids_literal}]"
            expr = f"{base_expr} and {doc_filter}" if base_expr else doc_filter
        else:
            expr = base_expr

        # 2. vector search (parallel for decomposed sub-queries)
        search_tasks = [
            self._vector_search(q, top_k, expr)
            for q in queries_to_search
        ]
        all_results = await asyncio.gather(*search_tasks)

        # 3. merge results (RRF if multiple sub-queries, plain if single)
        if len(all_results) > 1:
            merged = rrf_merge(all_results)[:top_k * self._candidate_multiplier]
        else:
            merged = all_results[0]

        # 4. rerank + RAG answer — in parallel (skip RAG in agentic rounds)
        if not skip_rag:
            rag_task = asyncio.create_task(self._ask_rag(
                query, mode,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                max_entity_tokens=max_entity_tokens,
                max_relation_tokens=max_relation_tokens,
                max_total_tokens=max_total_tokens,
                response_type=response_type,
                include_references=include_references,
            ))

        sources = self._rerank(query, merged, top_k)

        # 4b. fallback: if two-stage narrowed too much, retry without whitelist
        if doc_id_whitelist and not sources:
            logger.info("two-stage returned 0 sources, falling back to full search")
            fallback_results = await self._vector_search(query, top_k, base_expr)
            sources = self._rerank(query, fallback_results, top_k)

        answer = (await rag_task) if not skip_rag else ""

        latency_ms = (time.monotonic() - t0) * 1000.0

        logger.info(
            "search  query={q!r}  mode={mode}  dept={dept}  "
            "expr={expr}  sources={n}  decomposed={decomp}  "
            "sub_queries={subs}  latency_ms={ms:.1f}",
            q=query, mode=mode, dept=department,
            expr=expr, n=len(sources), decomp=decompose,
            subs=sub_queries, ms=latency_ms,
        )
        return SearchResponse(
            query=query, answer=answer, mode=mode,
            sources=sources, latency_ms=latency_ms,
            sub_queries=sub_queries if sub_queries else None,
        )

    # ── reranker ──────────────────────────────────────────────────────

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
