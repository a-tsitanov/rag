import asyncio
import logging
from dataclasses import dataclass

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from pymilvus.exceptions import MilvusException
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings

logger = logging.getLogger(__name__)

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200

_OUTPUT_FIELDS = ["content", "doc_id", "department", "doc_type", "created_at"]

_retry_milvus = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((MilvusException, ConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


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
    """Async wrapper around pymilvus MilvusClient with HNSW index and retry."""

    def __init__(
        self,
        uri: str | None = None,
        collection: str | None = None,
        *,
        index_type: str = "HNSW",
        timeout: float | None = None,
        vector_dim: int | None = None,
    ):
        self._uri = uri or f"http://{settings.milvus.host}:{settings.milvus.port}"
        self._collection = collection or settings.milvus.collection
        self._index_type = index_type
        # 0 / None → no wait_for wrapper
        self._timeout = timeout or settings.milvus.timeout_s or None
        # Размерность берём из OllamaSettings (тот же вектор в pipeline и
        # в search); коллекция создаётся один раз с этой dim и переживает
        # последующие запуски, пока её явно не дропнут.
        self._vector_dim = vector_dim or settings.ollama.embedding_dim
        self._client: MilvusClient | None = None

    async def _run(self, fn, *args, **kwargs):
        """Run a blocking pymilvus call in a thread with an async timeout."""
        coro = asyncio.to_thread(fn, *args, **kwargs)
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
            "Created collection '%s' with %s index",
            self._collection,
            self._index_type,
        )

    # ── writes ────────────────────────────────────────────────────────

    @_retry_milvus
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

    @_retry_milvus
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
