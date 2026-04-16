"""Async wrapper вокруг pymilvus.MilvusClient.

Особенности:
* Блокирующие pymilvus-вызовы уезжают в **выделенный** ThreadPoolExecutor
  с ограниченным ``pool_size`` — default-executor не засоряется зомби-
  threads при флапах Milvus (когда ``asyncio.wait_for`` стрельнул, но
  сам thread ещё держит соединение).
* Retry политика здесь НЕ выставлена: retries делает taskiq на уровне
  всей задачи ingestion, чтобы не было двойного ретрая (tenacity 3× ×
  taskiq 2×). Первая ошибка storage — сразу наверх.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from loguru import logger
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from src.config import settings

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200

_OUTPUT_FIELDS = ["content", "doc_id", "department", "doc_type", "created_at"]


@dataclass
class Document:
    id: str
    content: str
    embedding: list[float]
    doc_id: str
    department: str
    created_at: int
    doc_type: str


@dataclass
class SearchResult:
    id: str
    content: str
    doc_id: str
    department: str
    doc_type: str
    score: float


class AsyncMilvusClient:
    """Async wrapper around pymilvus MilvusClient with HNSW index."""

    def __init__(
        self,
        uri: str | None = None,
        collection: str | None = None,
        *,
        index_type: str = "HNSW",
        timeout: float | None = None,
        vector_dim: int | None = None,
        pool_size: int | None = None,
    ):
        self._uri = uri or f"http://{settings.milvus.host}:{settings.milvus.port}"
        self._collection = collection or settings.milvus.collection
        self._index_type = index_type
        self._timeout = timeout or settings.milvus.timeout_s or None
        self._vector_dim = vector_dim or settings.ollama.embedding_dim
        self._pool = ThreadPoolExecutor(
            max_workers=pool_size or settings.milvus.pool_size,
            thread_name_prefix="milvus",
        )
        self._client: MilvusClient | None = None

    async def _run(self, fn, *args, **kwargs):
        """Run blocking pymilvus call in our dedicated thread pool.

        ``asyncio.wait_for`` обеспечивает async-уровневый таймаут, но
        **не отменяет** сам thread — блокирующий вызов продолжит жить в
        pool'е до своего естественного завершения. Поэтому pool с
        фиксированным ``max_workers=pool_size`` — чтобы ошибка не
        уводила executor в зомби-состояние.
        """
        loop = asyncio.get_running_loop()
        call = lambda: fn(*args, **kwargs)  # noqa: E731
        coro = loop.run_in_executor(self._pool, call)
        if self._timeout:
            return await asyncio.wait_for(coro, timeout=self._timeout)
        return await coro

    async def connect(self):
        self._client = await self._run(MilvusClient, uri=self._uri)
        await self._ensure_collection()

    async def disconnect(self):
        if self._client:
            await self._run(self._client.close)
            self._client = None
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ── schema + index ────────────────────────────────────────────────

    async def _ensure_collection(self):
        has = await self._run(self._client.has_collection, self._collection)
        if has:
            return

        schema = CollectionSchema(fields=[
            FieldSchema(
                name="id", dtype=DataType.VARCHAR,
                is_primary=True, max_length=128,
            ),
            FieldSchema(
                name="content", dtype=DataType.VARCHAR, max_length=65535,
            ),
            FieldSchema(
                name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._vector_dim,
            ),
            FieldSchema(
                name="doc_id", dtype=DataType.VARCHAR, max_length=128,
            ),
            FieldSchema(
                name="department", dtype=DataType.VARCHAR, max_length=64,
            ),
            FieldSchema(
                name="created_at", dtype=DataType.INT64,
            ),
            FieldSchema(
                name="doc_type", dtype=DataType.VARCHAR, max_length=64,
            ),
        ])

        index_params = self._client.prepare_index_params()
        index_kwargs: dict = {
            "field_name": "embedding",
            "index_type": self._index_type,
            "metric_type": "COSINE",
        }
        if self._index_type == "HNSW":
            index_kwargs["params"] = {
                "M": HNSW_M,
                "efConstruction": HNSW_EF_CONSTRUCTION,
            }
        index_params.add_index(**index_kwargs)

        await self._run(
            self._client.create_collection,
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )
        logger.info(
            "Created collection {collection} with {index} index",
            collection=self._collection, index=self._index_type,
        )

    # ── writes ────────────────────────────────────────────────────────

    async def upsert_batch(self, documents: list[Document]) -> None:
        data = [
            {
                "id": doc.id,
                "content": doc.content,
                "embedding": doc.embedding,
                "doc_id": doc.doc_id,
                "department": doc.department,
                "created_at": doc.created_at,
                "doc_type": doc.doc_type,
            }
            for doc in documents
        ]
        await self._run(
            self._client.upsert,
            collection_name=self._collection,
            data=data,
        )

    # ── reads ─────────────────────────────────────────────────────────

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        department: str | None = None,
    ) -> list[SearchResult]:
        search_kwargs: dict = {
            "collection_name": self._collection,
            "data": [query_vector],
            "limit": top_k,
            "output_fields": _OUTPUT_FIELDS,
            "search_params": {"metric_type": "COSINE"},
        }
        if department:
            search_kwargs["filter"] = f'department == "{department}"'

        results = await self._run(
            lambda: self._client.search(**search_kwargs)
        )

        if not results or not results[0]:
            return []

        return [
            SearchResult(
                id=hit["id"],
                content=hit["entity"]["content"],
                doc_id=hit["entity"]["doc_id"],
                department=hit["entity"]["department"],
                doc_type=hit["entity"]["doc_type"],
                score=hit["distance"],
            )
            for hit in results[0]
        ]
