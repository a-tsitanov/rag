"""Milvus helpers: dataclasses + health-check probe.

Write и search делаются через ``langchain_milvus.Milvus`` vectorstore
(DI: ``CommonProvider.vectorstore``).  Здесь остаются только:
* ``Document`` / ``SearchResult`` dataclass'ы (используются в типах и тестах)
* ``check_milvus_health`` — лёгкая проба для ``/health`` endpoint'а
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger
from pymilvus import MilvusClient

from src.config import settings

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200


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


async def check_milvus_health(
    uri: str | None = None,
    timeout: float | None = None,
) -> str:
    """Return Milvus server version string or raise on failure."""
    _uri = uri or f"http://{settings.milvus.host}:{settings.milvus.port}"
    _timeout = timeout or settings.milvus.timeout_s

    def _probe() -> str:
        client = MilvusClient(uri=_uri)
        try:
            return client.get_server_version()
        finally:
            client.close()

    return await asyncio.wait_for(
        asyncio.to_thread(_probe), timeout=_timeout,
    )
