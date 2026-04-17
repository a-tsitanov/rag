"""Dishka providers — single source of truth for every client / service.

Topology
--------
* ``CommonProvider`` — infra shared by API and worker: Milvus vectorstore
  (langchain), Neo4j, Postgres, OllamaEmbeddings, reranker.
* ``ApiProvider``    — LightRAG + ``HybridSearcher`` for the HTTP service.
* ``WorkerProvider`` — LightRAG + ``AsyncDocumentWorker`` for ingestion.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import ollama
import psycopg
from dishka import Provider, Scope, provide
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from lightrag import LightRAG

from src.config import Settings, settings
from src.ingestion.chunker import SemanticChunker
from src.ingestion.worker import (
    AsyncDocumentWorker,
    LightRAGInserter,
    PGStatusUpdater,
)
from src.retrieval.hybrid_search import HybridSearcher, RerankerFn
from src.retrieval.lightrag_setup import close_rag_graph, create_rag
from src.storage.neo4j_client import AsyncNeo4jClient

logger = logging.getLogger(__name__)

# ── Postgres: documents table upsert ──────────────────────────────────

_UPSERT_DOC = """
    INSERT INTO documents (id, path, status, department, error, processed_at)
    VALUES (
        %(doc_id)s::uuid,
        %(path)s,
        %(status)s::text,
        %(department)s,
        %(error)s,
        CASE WHEN %(status)s::text IN ('completed', 'failed')
             THEN now() END
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
    return [1.0 - i * 0.01 for i in range(len(candidates))]


# ── common provider ──────────────────────────────────────────────────


class CommonProvider(Provider):
    scope = Scope.APP

    @provide
    def settings(self) -> Settings:
        return settings

    # ── embeddings (langchain-ollama) ────────────────────────────────

    @provide
    def embeddings(self, s: Settings) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=s.ollama.embedding_model,
            base_url=s.ollama.host,
        )

    # ── vectorstore (langchain-milvus) ───────────────────────────────

    @provide
    def vectorstore(self, s: Settings, emb: OllamaEmbeddings) -> Milvus:
        return Milvus(
            embedding_function=emb,
            collection_name=s.milvus.collection,
            connection_args={
                "uri": f"http://{s.milvus.host}:{s.milvus.port}",
            },
            primary_field="id",
            text_field="content",
            vector_field="embedding",
            auto_id=False,
            drop_old=False,
            metadata_schema={
                "doc_id": {"dtype": "VARCHAR", "max_length": 128},
                "department": {"dtype": "VARCHAR", "max_length": 64},
                "doc_type": {"dtype": "VARCHAR", "max_length": 64},
                "created_at": {"dtype": "INT64"},
            },
        )

    # ── neo4j ────────────────────────────────────────────────────────

    @provide
    async def neo4j(self, s: Settings) -> AsyncIterator[AsyncNeo4jClient]:
        client = AsyncNeo4jClient(
            uri=s.neo4j.uri, user=s.neo4j.user, password=s.neo4j.password,
            connection_timeout=s.neo4j.timeout_s,
        )
        await client.connect()
        logger.info("neo4j connected: %s", s.neo4j.uri)
        try:
            yield client
        finally:
            await client.disconnect()

    # ── postgres ─────────────────────────────────────────────────────

    @provide
    async def postgres(
        self, s: Settings,
    ) -> AsyncIterator[psycopg.AsyncConnection]:
        conn = await psycopg.AsyncConnection.connect(
            s.postgres.dsn, autocommit=True,
            connect_timeout=s.postgres.connect_timeout_s,
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

    # ── ollama client (for health check + LightRAG host passing) ─────

    @provide
    def ollama_client(self, s: Settings) -> ollama.AsyncClient:
        return ollama.AsyncClient(host=s.ollama.host, timeout=s.ollama.timeout_s)

    @provide
    def reranker_fn(self) -> RerankerFn:
        return _stub_reranker


# ── api provider ─────────────────────────────────────────────────────


class ApiProvider(Provider):
    scope = Scope.APP

    @provide
    async def lightrag(self, s: Settings) -> AsyncIterator[LightRAG]:
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
        vs: Milvus,
        rerank: RerankerFn,
    ) -> HybridSearcher:
        return HybridSearcher(
            rag=rag, vectorstore=vs, reranker_fn=rerank,
        )


# ── worker provider ──────────────────────────────────────────────────


class WorkerProvider(Provider):
    scope = Scope.APP

    @provide
    async def lightrag(self, s: Settings) -> AsyncIterator[LightRAG]:
        rag = await create_rag(graph_storage=s.lightrag.graph_storage)
        logger.info("lightrag ready (graph=%s)", s.lightrag.graph_storage)
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
    def chunker(self, emb: OllamaEmbeddings, s: Settings) -> SemanticChunker:
        return SemanticChunker(
            embeddings=emb,
            max_tokens=s.ingestion.chunk_size,
            overlap=s.ingestion.chunk_overlap,
        )

    @provide
    def document_worker(
        self,
        vs: Milvus,
        inserter: LightRAGInserter,
        status: PGStatusUpdater,
        chunker: SemanticChunker,
    ) -> AsyncDocumentWorker:
        return AsyncDocumentWorker(
            vectorstore=vs,
            lightrag_inserter=inserter,
            pg_status_updater=status,
            chunker=chunker,
        )


# ── factories ────────────────────────────────────────────────────────


def build_api_container():
    from dishka import make_async_container
    return make_async_container(CommonProvider(), ApiProvider())


def build_worker_container():
    from dishka import make_async_container
    return make_async_container(CommonProvider(), WorkerProvider())
