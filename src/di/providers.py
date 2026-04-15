"""Dishka providers — single source of truth for every client / service.

Topology
--------
* ``CommonProvider`` — infra shared by API and worker (Milvus, Neo4j,
  Postgres, Ollama, embed fn, reranker fn).
* ``ApiProvider``    — LightRAG + ``HybridSearcher`` for the HTTP service.
* ``WorkerProvider`` — LightRAG + worker callbacks (Milvus writer,
  LightRAG inserter, PG status updater) + ``AsyncDocumentWorker``.

All bindings are ``Scope.APP`` — created once per process, torn down via
``await container.close()``.  Each generator-style provider ``yield``s
the live resource and awaits its shutdown below the yield.

The RabbitMQ broker + taskiq tasks live in ``src.ingestion.tasks`` — it's
a module singleton (owned by FastAPI lifespan / ``taskiq worker`` CLI),
not a dishka binding, which avoids an API↔worker import cycle.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np
import ollama
import psycopg
from dishka import Provider, Scope, provide
from lightrag import LightRAG

from src.config import Settings, settings
from src.ingestion.worker import (
    AsyncDocumentWorker,
    LightRAGInserter,
    MilvusWriter,
    PGStatusUpdater,
)
from src.retrieval.hybrid_search import EmbedFn, HybridSearcher, RerankerFn
from src.retrieval.lightrag_setup import close_rag_graph, create_rag
from src.storage.milvus_client import AsyncMilvusClient
from src.storage.neo4j_client import AsyncNeo4jClient

logger = logging.getLogger(__name__)

# ── Postgres: documents table upsert ──────────────────────────────────

_UPSERT_DOC = """
    INSERT INTO documents (id, path, status, department, error, processed_at)
    VALUES (
        %(doc_id)s::uuid, %(path)s, %(status)s, %(department)s, %(error)s,
        CASE WHEN %(status)s IN ('completed', 'failed') THEN now() END
    )
    ON CONFLICT (id) DO UPDATE SET
        path         = EXCLUDED.path,
        status       = EXCLUDED.status,
        department   = EXCLUDED.department,
        error        = EXCLUDED.error,
        processed_at = EXCLUDED.processed_at
"""


# ── rerank stub ───────────────────────────────────────────────────────


def _stub_reranker(query: str, candidates: list[str]) -> list[float]:
    """Keeps Milvus' original ordering.  Swap for BGE-reranker-v2-m3 in
    production (``sentence_transformers.CrossEncoder``) when model
    weights are pre-downloaded."""
    return [1.0 - i * 0.01 for i in range(len(candidates))]


# ── common provider ──────────────────────────────────────────────────


class CommonProvider(Provider):
    """Infra clients shared by API + worker."""

    scope = Scope.APP

    @provide
    def settings(self) -> Settings:
        return settings

    @provide
    async def milvus(self, s: Settings) -> AsyncIterator[AsyncMilvusClient]:
        client = AsyncMilvusClient(
            uri=f"http://{s.milvus_host}:{s.milvus_port}",
        )
        await client.connect()
        logger.info("milvus connected")
        try:
            yield client
        finally:
            await client.disconnect()

    @provide
    async def neo4j(self, s: Settings) -> AsyncIterator[AsyncNeo4jClient]:
        client = AsyncNeo4jClient(
            uri=s.neo4j_uri, user=s.neo4j_user, password=s.neo4j_password,
        )
        await client.connect()
        logger.info("neo4j connected: %s", s.neo4j_uri)
        try:
            yield client
        finally:
            await client.disconnect()

    @provide
    async def postgres(
        self, s: Settings,
    ) -> AsyncIterator[psycopg.AsyncConnection]:
        conn = await psycopg.AsyncConnection.connect(
            s.postgres_dsn, autocommit=True,
        )
        logger.info("postgres connected")
        try:
            yield conn
        finally:
            await conn.close()

    @provide
    def pg_status_updater(
        self, pg: psycopg.AsyncConnection,
    ) -> PGStatusUpdater:
        async def _update(**kwargs) -> None:
            payload = {
                "doc_id": kwargs["doc_id"],
                "path": kwargs.get("path", ""),
                "status": kwargs["status"],
                "department": kwargs.get("department", "") or None,
                "error": kwargs.get("error", "") or None,
            }
            await pg.execute(_UPSERT_DOC, payload)

        return _update

    @provide
    def ollama_client(self, s: Settings) -> ollama.AsyncClient:
        return ollama.AsyncClient(host=s.ollama_host)

    @provide
    def embed_fn(
        self, s: Settings, ollama_client: ollama.AsyncClient,
    ) -> EmbedFn:
        async def _embed(texts: list[str]) -> np.ndarray:
            vecs = []
            for t in texts:
                resp = await ollama_client.embeddings(
                    model=s.embedding_model, prompt=t,
                )
                vecs.append(resp["embedding"])
            return np.array(vecs, dtype=np.float32)

        return _embed

    @provide
    def reranker_fn(self) -> RerankerFn:
        return _stub_reranker


# ── api provider ─────────────────────────────────────────────────────


class ApiProvider(Provider):
    """Bindings used only by the FastAPI process."""

    scope = Scope.APP

    @provide
    async def lightrag(self, s: Settings) -> AsyncIterator[LightRAG]:
        # API currently runs LightRAG with the in-process NetworkX graph
        # so that entity extraction doesn't race the worker.  Override
        # the config default explicitly.
        rag = await create_rag(graph_storage="NetworkXStorage")
        logger.info("lightrag ready (graph=NetworkXStorage)")
        try:
            yield rag
        finally:
            await close_rag_graph(rag)

    @provide
    def searcher(
        self,
        rag: LightRAG,
        milvus: AsyncMilvusClient,
        embed: EmbedFn,
        rerank: RerankerFn,
    ) -> HybridSearcher:
        return HybridSearcher(
            rag=rag, milvus=milvus,
            embed_fn=embed, reranker_fn=rerank,
        )


# ── worker provider ──────────────────────────────────────────────────


class WorkerProvider(Provider):
    """Bindings used only by the ingestion worker daemon."""

    scope = Scope.APP

    @provide
    async def lightrag(self, s: Settings) -> AsyncIterator[LightRAG]:
        # Worker writes the knowledge graph straight into Neo4j so
        # multiple worker replicas share the same graph.
        rag = await create_rag(graph_storage=s.lightrag_graph_storage)
        logger.info("lightrag ready (graph=%s)", s.lightrag_graph_storage)
        try:
            yield rag
        finally:
            await close_rag_graph(rag)

    @provide
    def lightrag_inserter(self, rag: LightRAG) -> LightRAGInserter:
        async def _insert(text: str) -> None:
            await rag.ainsert(text)

        return _insert

    @provide
    def milvus_writer(self, milvus: AsyncMilvusClient) -> MilvusWriter:
        async def _write(rows: list[dict]) -> None:
            await asyncio.to_thread(
                milvus._client.upsert,
                collection_name=milvus._collection,
                data=rows,
            )

        return _write

    @provide
    def document_worker(
        self,
        embed: EmbedFn,
        writer: MilvusWriter,
        inserter: LightRAGInserter,
        status: PGStatusUpdater,
    ) -> AsyncDocumentWorker:
        return AsyncDocumentWorker(
            embed_fn=embed,
            milvus_writer=writer,
            lightrag_inserter=inserter,
            pg_status_updater=status,
        )


# ── factories ────────────────────────────────────────────────────────


def build_api_container():
    from dishka import make_async_container

    return make_async_container(CommonProvider(), ApiProvider())


def build_worker_container():
    from dishka import make_async_container

    return make_async_container(CommonProvider(), WorkerProvider())
