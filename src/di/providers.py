"""Dishka providers — single source of truth for every client / service.

Topology
--------
* ``CommonProvider`` — infra shared by API and worker: Milvus vectorstore
  (langchain), Neo4j, Postgres, Ollama embeddings, reranker.
* ``ApiProvider``    — LightRAG + ``HybridSearcher`` for the HTTP service.
* ``WorkerProvider`` — LightRAG + ``AsyncDocumentWorker`` for ingestion.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import ollama
import psycopg
from dishka import Provider, Scope, provide
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from lightrag import LightRAG
from pymilvus import MilvusClient, connections as _pymilvus_connections


# ── MilvusClient ↔ ORM bridge ────────────────────────────────────────
#
# ``langchain_milvus.Milvus`` creates an internal ``MilvusClient``, reads its
# auto-generated ``_using`` alias, and then calls ``Collection(name,
# using=alias)`` — which goes through the pymilvus **ORM** registry
# (``pymilvus.orm.connections``).  MilvusClient never registers itself
# there, so the ORM-side ``Collection()`` raises
# ``ConnectionNotExistException`` the first time any new MilvusClient is
# instantiated (fresh process — taskiq worker, test, cold import).
#
# We patch ``MilvusClient.__init__`` once so every instance also registers
# its alias with the ORM.  Idempotent; safe to call from multiple
# providers / processes.

_MILVUS_PATCHED = False


def _patch_milvus_client_to_register_orm() -> None:
    global _MILVUS_PATCHED
    if _MILVUS_PATCHED:
        return
    _MILVUS_PATCHED = True

    _orig_init = MilvusClient.__init__

    def _patched_init(
        self,
        uri: str = "http://localhost:19530",
        user: str = "",
        password: str = "",
        db_name: str = "",
        token: str = "",
        timeout=None,
        **kwargs,
    ) -> None:
        _orig_init(
            self, uri=uri, user=user, password=password,
            db_name=db_name, token=token, timeout=timeout, **kwargs,
        )
        try:
            _pymilvus_connections.connect(
                alias=self._using,
                uri=uri, user=user, password=password,
                db_name=db_name, token=token,
            )
        except Exception as exc:  # pragma: no cover — best effort
            logging.getLogger(__name__).warning(
                "could not bridge MilvusClient alias %s to ORM: %s",
                self._using, exc,
            )

    MilvusClient.__init__ = _patched_init


_patch_milvus_client_to_register_orm()

from src.config import Settings, settings
from src.ingestion.chunker import SemanticChunker
from src.ingestion.worker import (
    AsyncDocumentWorker,
    LightRAGInserter,
    PGStatusUpdater,
)
from src.llm_client import LLMClient
from src.retrieval.hybrid_search import HybridSearcher, RerankerFn
from src.retrieval.lightrag_setup import close_rag_graph, create_rag
from src.storage.neo4j_client import AsyncNeo4jClient
from src.storage.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)

# ── Postgres: documents table upsert ──────────────────────────────────

_UPSERT_DOC = """
    INSERT INTO documents (id, path, status, department, error, summary, processed_at)
    VALUES (
        %(doc_id)s::uuid,
        %(path)s,
        %(status)s::text,
        %(department)s,
        %(error)s,
        %(summary)s,
        CASE WHEN %(status)s::text IN ('completed', 'failed')
             THEN now() END
    )
    ON CONFLICT (id) DO UPDATE SET
        path         = EXCLUDED.path,
        status       = EXCLUDED.status,
        department   = EXCLUDED.department,
        error        = EXCLUDED.error,
        summary      = COALESCE(EXCLUDED.summary, documents.summary),
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

    # ── embeddings (langchain — Ollama) ──────────────────────────────

    @provide
    def embeddings(self, s: Settings) -> Embeddings:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=s.ollama.embedding_model,
            base_url=s.ollama.host,
        )

    # ── vectorstore (langchain-milvus) ───────────────────────────────

    @provide
    def vectorstore(self, s: Settings, emb: Embeddings) -> Milvus:
        # MilvusClient ↔ ORM bridge is patched at module load
        # (see _patch_milvus_client_to_register_orm above) — every
        # MilvusClient the langchain wrapper creates registers its alias
        # with pymilvus.orm.connections, so Collection(using=alias) works.
        milvus_uri = f"http://{s.milvus.host}:{s.milvus.port}"

        return Milvus(
            embedding_function=emb,
            collection_name=s.milvus.collection,
            connection_args={"uri": milvus_uri},
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
                "summary": kwargs.get("summary") or None,
            }
            await pg.execute(_UPSERT_DOC, payload)
        return _update

    # ── LLM client (Ollama) ──────────────────────────────────────────

    @provide
    def llm_client(self, s: Settings) -> LLMClient:
        return LLMClient(
            provider="ollama",
            _client=ollama.AsyncClient(
                host=s.ollama.host, timeout=s.ollama.timeout_s,
            ),
        )

    # Keep raw ollama client for health checks
    @provide
    def ollama_client(self, s: Settings) -> ollama.AsyncClient:
        return ollama.AsyncClient(host=s.ollama.host, timeout=s.ollama.timeout_s)

    @provide
    def reranker_fn(self) -> RerankerFn:
        return _stub_reranker

    @provide
    def sparse_encoder(self) -> SparseEncoder:
        return SparseEncoder()


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
        s: Settings,
        rag: LightRAG,
        vs: Milvus,
        rerank: RerankerFn,
        pg: psycopg.AsyncConnection,
        llm: LLMClient,
        emb: Embeddings,
        sparse: SparseEncoder,
    ) -> HybridSearcher:
        return HybridSearcher(
            rag=rag, vectorstore=vs, reranker_fn=rerank,
            pg=pg, llm_client=llm,
            embeddings=emb, sparse_encoder=sparse,
            milvus_uri=f"http://{s.milvus.host}:{s.milvus.port}",
            collection_name=s.milvus.collection,
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
    def chunker(self, emb: Embeddings, s: Settings) -> SemanticChunker:
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
        llm: LLMClient,
        emb: Embeddings,
        sparse: SparseEncoder,
    ) -> AsyncDocumentWorker:
        return AsyncDocumentWorker(
            vectorstore=vs,
            lightrag_inserter=inserter,
            pg_status_updater=status,
            chunker=chunker,
            llm_client=llm,
            embeddings=emb,
            sparse_encoder=sparse,
        )


# ── factories ────────────────────────────────────────────────────────


def build_api_container():
    from dishka import make_async_container
    return make_async_container(CommonProvider(), ApiProvider())


def build_worker_container():
    from dishka import make_async_container
    return make_async_container(CommonProvider(), WorkerProvider())
